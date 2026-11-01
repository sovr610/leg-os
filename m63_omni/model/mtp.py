"""Optional multi-token-prediction (MTP) module (blueprint §3, §14).

One standard GQA layer plus one dense SwiGLU FFN (width 11,008). Input fuses the
backbone's final hidden state at position ``t`` with the embedding of token
``t+1`` (parameter-free sum of two RMS-normalised streams), and the module
predicts token ``t+2`` through the tied LM head. Applied recursively it drafts
``draft_tokens`` tokens; the target model always verifies them, and a
target-only path is always retained.
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
