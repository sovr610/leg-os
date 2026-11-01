"""Gated grouped-query attention (blueprint §3).

Reference shape: 32 query heads, 8 KV heads, 128-dim heads, QK RMSNorm, partial
RoPE on 32 of 128 dims, and an input-dependent sigmoid output gate applied
before the output projection. Neural-memory read tokens (``extra_kv``) are
projected with this layer's own K/V weights, carry no RoPE, and are visible to
every query. The MTP module reuses this class with ``output_gate=False``
("standard GQA").
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import AttentionConfig
from ..kernels.attention import attention, attention_flex, attention_reference, build_attention_mask
from .cache import KVLayerState, StreamIndex
from .layers import PartialRotary, RMSNorm

ATTENTION_BACKENDS = ("sdpa", "reference", "flex")


class GatedGQA(nn.Module):
    def __init__(self, d_model: int, cfg: AttentionConfig, eps: float = 1e-6, output_gate: Optional[bool] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.Hq, self.Hkv, self.D = cfg.query_heads, cfg.kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(d_model, self.Hq * self.D, bias=False)
        self.k_proj = nn.Linear(d_model, self.Hkv * self.D, bias=False)
        self.v_proj = nn.Linear(d_model, self.Hkv * self.D, bias=False)
        self.o_proj = nn.Linear(self.Hq * self.D, d_model, bias=False)
        use_gate = cfg.output_gate if output_gate is None else output_gate
        self.gate_proj = nn.Linear(d_model, self.Hq * self.D, bias=False) if use_gate else None
        self.q_norm = RMSNorm(self.D, eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.D, eps) if cfg.qk_norm else None
        self.rope = PartialRotary(cfg.rotary_dims, cfg.rope_theta)
        self.backend = "sdpa"  # "sdpa" | "reference" | "flex" (kernel choice only; numerics must match)

    @property
    def use_reference_kernel(self) -> bool:  # backwards-compatible alias
        return self.backend == "reference"

    @use_reference_kernel.setter
    def use_reference_kernel(self, value: bool) -> None:
        self.backend = "reference" if value else "sdpa"

    def _kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        k = self.k_proj(x).view(B, T, self.Hkv, self.D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.Hkv, self.D).transpose(1, 2)
        if self.k_norm is not None:
            k = self.k_norm(k)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        idx: StreamIndex,
        state: Optional[KVLayerState] = None,
        extra_kv: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, KVLayerState]:
        """x: [B, T, d] (already normalised); extra_kv: [B, M, d] memory read tokens."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.Hq, self.D).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        k, v = self._kv(x)
        cos, sin = self.rope.cos_sin(idx.rope_pos, q.dtype)
        q, k = self.rope.apply(q, cos, sin), self.rope.apply(k, cos, sin)
        new_state = KVLayerState(k, v, idx.stream_pos, idx.seg) if state is None else state.append(k, v, idx.stream_pos, idx.seg)
        keys, values = new_state.k, new_state.v
        n_mem = 0
        if extra_kv is not None:
            mk, mv = self._kv(extra_kv)  # memory tokens: no RoPE
            keys, values = torch.cat([mk, keys], dim=2), torch.cat([mv, values], dim=2)
            n_mem = extra_kv.shape[1]
        if self.backend == "flex":
            o = attention_flex(q, keys, values, idx.stream_pos, idx.seg, new_state.pos, new_state.seg, n_mem)
        else:
            mask = build_attention_mask(idx.stream_pos, idx.seg, new_state.pos, new_state.seg, n_mem)
            attn_fn = attention_reference if self.backend == "reference" else attention
            o = attn_fn(q, keys, values, mask)  # [B, Hq, T, D]
        o = o.transpose(1, 2).reshape(B, T, self.Hq * self.D)
        if self.gate_proj is not None:
            o = o * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(o), new_state
