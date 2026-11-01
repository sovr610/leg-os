"""Gated DeltaNet sequence mixer (blueprint §3, §4).

Reference shape: 16 heads, 128-dim keys/values, 2,048-dim inner width, short
causal convolutions (kernel 4) on q/k/v, L2-normalised q/k, Mamba-2-style
data-dependent decay ``g = -exp(A_log) * softplus(a(x) + dt_bias)`` computed in
FP32, ``beta = sigmoid(b(x))``, gated per-head RMSNorm on the output, and an
output projection. Chunkwise kernel for training/prefill, recurrent kernel for
decoding; both are cross-checked in ``tests/``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GDNConfig
from ..kernels.precision import acc
from ..kernels.gdn import causal_depthwise_conv, gdn_chunked, gdn_recurrent_reference
from .cache import GDNLayerState

GDN_BACKENDS = ("chunked", "fla")


class GatedDeltaNet(nn.Module):
    def __init__(self, d_model: int, cfg: GDNConfig, eps: float = 1e-6) -> None:
        super().__init__()
        self.cfg = cfg
        H, Dk, Dv = cfg.heads, cfg.key_dim, cfg.value_dim
        self.H, self.Dk, self.Dv = H, Dk, Dv
        inner = H * Dv
        self.q_proj = nn.Linear(d_model, H * Dk, bias=False)
        self.k_proj = nn.Linear(d_model, H * Dk, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.a_proj = nn.Linear(d_model, H, bias=False)
        self.b_proj = nn.Linear(d_model, H, bias=False)
        self.g_proj = nn.Linear(d_model, inner, bias=False) if cfg.output_gate else None
        self.o_proj = nn.Linear(inner, d_model, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(H * Dk * 2 + inner, cfg.conv_kernel))
        self.A_log = nn.Parameter(torch.empty(H))
        self.dt_bias = nn.Parameter(torch.empty(H))
        self.o_norm_weight = nn.Parameter(torch.ones(Dv))
        self.eps = eps
        self.backend = "chunked"  # "chunked" (reference) | "fla" (Triton, GPU); decode always uses the recurrent path
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.conv_weight, -1.0 / math.sqrt(self.cfg.conv_kernel), 1.0 / math.sqrt(self.cfg.conv_kernel))
        with torch.no_grad():
            # A in [1, 16], dt in [1e-3, 1e-1] (log-uniform), dt_bias = softplus^{-1}(dt)
            self.A_log.copy_(torch.log(torch.empty_like(self.A_log).uniform_(1.0, 16.0)))
            dt = torch.exp(torch.empty_like(self.dt_bias).uniform_(math.log(1e-3), math.log(1e-1)))
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> GDNLayerState:
        return GDNLayerState(
            conv=torch.zeros(batch, self.conv_weight.shape[0], self.cfg.conv_kernel - 1, device=device, dtype=dtype),
            recurrent=torch.zeros(batch, self.H, self.Dk, self.Dv, device=device, dtype=torch.float32),
        )

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
        # FP32 decay / gate arithmetic (§4, §7)
        g = -torch.exp(acc(self.A_log)) * F.softplus(acc(self.a_proj(x)) + acc(self.dt_bias))
        beta = torch.sigmoid(acc(self.b_proj(x)))
        init = None if state is None else state.recurrent
        if mode == "auto":
            mode = "recurrent" if T == 1 else "chunked"
        if mode == "chunked" and self.backend == "fla":
            from ..kernels.gdn_fla import gdn_fla

            o, S = gdn_fla(q, k, v, g, beta, initial_state=init, reset=reset)
        elif mode == "chunked":
            o, S = gdn_chunked(q, k, v, g, beta, chunk_size=self.cfg.train_chunk, initial_state=init, reset=reset)
        elif mode == "recurrent":
            o, S = gdn_recurrent_reference(q, k, v, g, beta, initial_state=init, reset=reset)
        else:
            raise ValueError(f"unknown GDN mode {mode!r}")
        o = acc(o)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps) * self.o_norm_weight.to(o.dtype)
        o = o.to(x.dtype).reshape(B, T, H * Dv)
        if self.g_proj is not None:
            o = o * F.silu(self.g_proj(x))
        return self.o_proj(o), GDNLayerState(new_conv, S)
