"""Hybrid inference state: GDN recurrent/conv state + GQA KV + stream bookkeeping.

The cache is the *working context* layer of §6's state model: exact within one
model version, recomputable from the event log, never shared across tenants.
Branching and MTP rollback restore whole snapshots so that recurrent, attention
and neural-memory state can never diverge (Appendix A).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch


@dataclass
class GDNLayerState:
    conv: torch.Tensor  # [B, 3*inner, K-1] pre-activation conv inputs
    recurrent: torch.Tensor  # [B, H, Dk, Dv] FP32

    def clone(self) -> "GDNLayerState":
        return GDNLayerState(self.conv.clone(), self.recurrent.clone())

    def nbytes(self) -> int:
        return self.conv.numel() * self.conv.element_size() + self.recurrent.numel() * self.recurrent.element_size()


@dataclass
class KVLayerState:
    k: torch.Tensor  # [B, Hkv, L, D] (post QK-norm, post-RoPE)
    v: torch.Tensor  # [B, Hkv, L, D]
    pos: torch.Tensor  # [B, L] stream positions
    seg: torch.Tensor  # [B, L] segment IDs

    def clone(self) -> "KVLayerState":
        return KVLayerState(self.k.clone(), self.v.clone(), self.pos.clone(), self.seg.clone())

    def append(self, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor, seg: torch.Tensor) -> "KVLayerState":
        return KVLayerState(
            torch.cat([self.k, k], dim=2), torch.cat([self.v, v], dim=2),
            torch.cat([self.pos, pos], dim=1), torch.cat([self.seg, seg], dim=1),
        )

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()


LayerState = Union[GDNLayerState, KVLayerState, None]


@dataclass
class HybridCache:
    layers: list[LayerState]
    next_pos: int = 0  # next stream position (lock-step across the batch)
    last_seg: Optional[torch.Tensor] = None  # [B] segment ID of the last cached token
    next_rope: Optional[torch.Tensor] = None  # [B] next RoPE position (restarts per document)
    mtp: Optional["HybridCache"] = None  # MTP module's own attention cache
    meta: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, num_layers: int) -> "HybridCache":
        return cls(layers=[None] * num_layers)

    def clone(self) -> "HybridCache":
        return HybridCache(
            layers=[s.clone() if s is not None else None for s in self.layers],
            next_pos=self.next_pos,
            last_seg=None if self.last_seg is None else self.last_seg.clone(),
            next_rope=None if self.next_rope is None else self.next_rope.clone(),
            mtp=None if self.mtp is None else self.mtp.clone(),
            meta=dict(self.meta),
        )

    def detach_(self) -> "HybridCache":
        for i, s in enumerate(self.layers):
            if isinstance(s, GDNLayerState):
                self.layers[i] = GDNLayerState(s.conv.detach(), s.recurrent.detach())
            elif isinstance(s, KVLayerState):
                self.layers[i] = KVLayerState(s.k.detach(), s.v.detach(), s.pos, s.seg)
        return self

    def kv_bytes(self) -> int:
        return sum(s.nbytes() for s in self.layers if isinstance(s, KVLayerState))

    def recurrent_bytes(self) -> int:
        return sum(s.nbytes() for s in self.layers if isinstance(s, GDNLayerState))

    def __len__(self) -> int:
        return self.next_pos


@dataclass
class StreamIndex:
    """Resolved per-token indices for one forward call."""

    stream_pos: torch.Tensor  # [B, T] absolute positions within the session stream
    seg: torch.Tensor  # [B, T] segment (document) IDs
    rope_pos: torch.Tensor  # [B, T] RoPE positions, restarting at each document
    reset: torch.Tensor  # [B, T] bool: token starts a new document (recurrent reset)


def resolve_stream(
    B: int,
    T: int,
    device: torch.device,
    cache: Optional[HybridCache] = None,
    reset_mask: Optional[torch.Tensor] = None,
    segment_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> StreamIndex:
    start = cache.next_pos if cache is not None else 0
    stream = torch.arange(start, start + T, device=device).unsqueeze(0).expand(B, T)
    has_prev = cache is not None and cache.last_seg is not None
    prev_seg = cache.last_seg if has_prev else torch.zeros(B, dtype=torch.long, device=device)
    if segment_ids is not None:
        seg = segment_ids.long()
        prev = torch.cat([prev_seg.view(B, 1), seg[:, :-1]], dim=1)
        derived = seg != prev
        if not has_prev:
            derived[:, 0] = True
        reset = derived if reset_mask is None else (reset_mask.bool() | derived)
    else:
        reset = torch.zeros(B, T, dtype=torch.bool, device=device) if reset_mask is None else reset_mask.bool()
        if not has_prev:
            reset = reset.clone()
            reset[:, 0] = True
        seg = prev_seg.view(B, 1) + torch.cumsum(reset.long(), dim=1) - (0 if has_prev else 1)
    if position_ids is not None:
        rope = position_ids.long()
    else:
        prev_rope = cache.next_rope if (cache is not None and cache.next_rope is not None) else torch.zeros(B, dtype=torch.long, device=device)
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        marker = torch.where(reset, idx, torch.full_like(idx, -1))
        last_reset = torch.cummax(marker, dim=1).values
        rope = torch.where(last_reset >= 0, idx - last_reset, prev_rope.view(B, 1) + idx)
    return StreamIndex(stream, seg, rope, reset)
