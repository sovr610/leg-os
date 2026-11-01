"""Optional multi-token-prediction (MTP) modules (blueprint §3, §14).

Each depth is one standard GQA layer plus one dense SwiGLU FFN (width 11,008).
Depths are *sequential* (DeepSeek-V3): depth ``j`` fuses depth ``j-1``'s output
at position ``t`` (depth 0 is the backbone's final hidden state) with the
embedding of token ``t+j`` — a parameter-free sum of two RMS-normalised streams
— and predicts token ``t+j+1`` through the tied LM head. ``mtp.layers`` sets the
trained depth; ``mtp.share_weights`` reuses one module at every depth (the
parameters are shared, the compute is not). Speculative decoding drafts at most
``mtp.layers`` tokens, each with the depth that was trained for it; the target
model always verifies them, and a target-only path is always retained.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import M63Config
from .attention import GatedGQA
from .cache import KVLayerState, StreamIndex
from .layers import RMSNorm, SwiGLU


class MTPModule(nn.Module):
    def __init__(self, cfg: M63Config) -> None:
        super().__init__()
        d, eps = cfg.model.hidden_size, cfg.model.norm_eps
        self.h_norm = RMSNorm(d, eps)
        self.e_norm = RMSNorm(d, eps)
        self.attn_norm = RMSNorm(d, eps)
        self.attn = GatedGQA(d, cfg.attention, eps, output_gate=False)
        self.ffn_norm = RMSNorm(d, eps)
        self.ffn = SwiGLU(d, cfg.mtp.dense_ffn_hidden)
        self.out_norm = RMSNorm(d, eps)

    def forward(
        self, hidden: torch.Tensor, next_emb: torch.Tensor, idx: StreamIndex, state: Optional[KVLayerState] = None
    ) -> tuple[torch.Tensor, KVLayerState]:
        x = self.h_norm(hidden) + self.e_norm(next_emb)
        a, new_state = self.attn(self.attn_norm(x), idx, state)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return self.out_norm(x), new_state


class MTPStack(nn.Module):
    """``mtp.layers`` sequential MTP depths (one shared module when ``mtp.share_weights``)."""

    def __init__(self, cfg: M63Config) -> None:
        super().__init__()
        self.depth = cfg.mtp.layers
        self.depths = nn.ModuleList([MTPModule(cfg) for _ in range(1 if cfg.mtp.share_weights else self.depth)])

    def at(self, j: int) -> MTPModule:
        """The module for depth ``j`` (1-based)."""
        if not 1 <= j <= self.depth:
            raise ValueError(f"MTP depth {j} outside 1..{self.depth} (the trained depth)")
        return self.depths[0 if len(self.depths) == 1 else j - 1]
