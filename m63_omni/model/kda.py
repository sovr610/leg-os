"""Kimi Delta Attention sequence mixer (Kimi Linear, 2025) — a tournament alternative to GDN.

KDA shares GDN's state shape and decode cost (``[d_k, d_v]`` per head, short
causal convolutions, L2-normalised q/k, ``beta = sigmoid(W_beta x)``) and
differs in two places:

* the forget gate is **channel-wise**: ``g = -exp(A_log) * softplus(W_up W_down x
  + dt_bias)`` gives one decay per key channel, through a low-rank projection
  whose rank defaults to the head dimension;
* the output gate is a low-rank **sigmoid** gate:
  ``o = W_o (sigmoid(W_g_up W_g_down x) ⊙ RMSNorm(KDA(q, k, v, alpha, beta)))``.

The layer reads its shape from the ``gdn`` config section (heads, key/value
dims, conv kernel, chunk size) so a ``kda`` / ``gdn`` swap in
``model.hybrid_pattern`` is a like-for-like comparison of the mixing rule.
Recurrent state is a :class:`GDNLayerState` and every cache, fork, and
rollback path treats the two kinds identically.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GDNConfig
from ..kernels.gdn import causal_depthwise_conv
from ..kernels.kda import kda_chunked, kda_recurrent_reference
from ..kernels.precision import acc
from .cache import GDNLayerState


class KimiDeltaAttention(nn.Module):
    def __init__(self, d_model: int, cfg: GDNConfig, eps: float = 1e-6) -> None:
        super().__init__()
        self.cfg = cfg
        H, Dk, Dv = cfg.heads, cfg.key_dim, cfg.value_dim
        self.H, self.Dk, self.Dv = H, Dk, Dv
        rank = cfg.kda_low_rank or Dk
        inner = H * Dv
        self.q_proj = nn.Linear(d_model, H * Dk, bias=False)
        self.k_proj = nn.Linear(d_model, H * Dk, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.f_down = nn.Linear(d_model, rank, bias=False)  # channel-wise forget gate (low rank)
        self.f_up = nn.Linear(rank, H * Dk, bias=False)
        self.b_proj = nn.Linear(d_model, H, bias=False)
        if cfg.output_gate:
            self.g_down = nn.Linear(d_model, rank, bias=False)
            self.g_up = nn.Linear(rank, inner, bias=False)
        else:
            self.g_down = self.g_up = None
        self.o_proj = nn.Linear(inner, d_model, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(H * Dk * 2 + inner, cfg.conv_kernel))
        self.A_log = nn.Parameter(torch.empty(H))
        self.dt_bias = nn.Parameter(torch.empty(H * Dk))
        self.o_norm_weight = nn.Parameter(torch.ones(Dv))
        self.eps = eps
        self.backend = "chunked"  # decode always uses the recurrent path
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.conv_weight, -1.0 / math.sqrt(self.cfg.conv_kernel), 1.0 / math.sqrt(self.cfg.conv_kernel))
        with torch.no_grad():
            self.A_log.copy_(torch.log(torch.empty_like(self.A_log).uniform_(1.0, 16.0)))
            dt = torch.exp(torch.empty_like(self.dt_bias).uniform_(math.log(1e-3), math.log(1e-1)))
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> GDNLayerState:
        return GDNLayerState(
            conv=torch.zeros(batch, self.conv_weight.shape[0], self.cfg.conv_kernel - 1, device=device, dtype=dtype),
            recurrent=torch.zeros(batch, self.H, self.Dk, self.Dv, device=device, dtype=torch.float32),
        )

    def decay(self, x: torch.Tensor) -> torch.Tensor:
        """Channel-wise log-decay ``g [B, T, H, Dk]`` in >= FP32 (§4: decay arithmetic in FP32)."""
        B, T, _ = x.shape
        z = acc(self.f_up(self.f_down(x))) + acc(self.dt_bias)
        g = -torch.exp(acc(self.A_log)).repeat_interleave(self.Dk) * F.softplus(z)
        return g.view(B, T, self.H, self.Dk)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[GDNLayerState] = None,
        reset: Optional[torch.Tensor] = None,
        mode: str = "auto",
    ) -> tuple[torch.Tensor, GDNLayerState]:
        """x: [B, T, d]. ``mode``: "chunked", "recurrent", or "auto" (recurrent iff T == 1)."""
        B, T, _ = x.shape
        H, Dk, Dv = self.H, self.Dk, self.Dv
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        conv_state = None if state is None else state.conv
        qkv, new_conv = causal_depthwise_conv(qkv, self.conv_weight, conv_state=conv_state, reset=reset)
        qkv = F.silu(qkv)
        q, k, v = qkv.split([H * Dk, H * Dk, H * Dv], dim=-1)
        q = F.normalize(acc(q.view(B, T, H, Dk)), dim=-1, eps=1e-6)
        k = F.normalize(acc(k.view(B, T, H, Dk)), dim=-1, eps=1e-6)
        v = v.view(B, T, H, Dv)
        g = self.decay(x)
        beta = torch.sigmoid(acc(self.b_proj(x)))
        init = None if state is None else state.recurrent
        if mode == "auto":
            mode = "recurrent" if T == 1 else "chunked"
        if mode == "chunked":
            o, S = kda_chunked(q, k, v, g, beta, chunk_size=self.cfg.train_chunk, initial_state=init, reset=reset)
        elif mode == "recurrent":
            o, S = kda_recurrent_reference(q, k, v, g, beta, initial_state=init, reset=reset)
        else:
            raise ValueError(f"unknown KDA mode {mode!r}")
        o = acc(o)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps) * self.o_norm_weight.to(o.dtype)
        o = o.to(x.dtype).reshape(B, T, H * Dv)
        if self.g_up is not None:
            o = o * torch.sigmoid(self.g_up(self.g_down(x)))
        return self.o_proj(o), GDNLayerState(new_conv, S)
