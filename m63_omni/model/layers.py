"""Shared layers: RMSNorm (FP32 statistics), partial RoPE, SwiGLU, init helpers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Pre-RMSNorm. Statistics are computed in FP32 regardless of input dtype (§7)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """Dense SwiGLU FFN: ``down(silu(gate(x)) * up(x))`` — 3·d·h parameters."""

    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class PartialRotary(nn.Module):
    """RoPE applied to the first ``rotary_dims`` channels of each head (partial RoPE)."""

    def __init__(self, rotary_dims: int, theta: float) -> None:
        super().__init__()
        self.rotary_dims = rotary_dims
        self.theta = theta

    def cos_sin(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [B, T] -> cos, sin: [B, 1, T, rotary_dims/2] (FP32 angles)."""
        half = self.rotary_dims // 2
        inv = 1.0 / (self.theta ** (torch.arange(half, device=positions.device, dtype=torch.float32) / half))
        ang = positions.float().unsqueeze(-1) * inv
        return ang.cos().unsqueeze(1).to(dtype), ang.sin().unsqueeze(1).to(dtype)

    def apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: [B, H, T, D]."""
        r = self.rotary_dims
        if r == 0:
            return x
        rot, rest = x[..., :r], x[..., r:]
        x1, x2 = rot[..., : r // 2], rot[..., r // 2 :]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return torch.cat([rotated, rest], dim=-1)


def sinusoidal_features(values: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal encoding of arbitrary scalar values (e.g. timestamps in seconds)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=values.device, dtype=torch.float32) / half)
    ang = values.float().unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def init_linear(module: nn.Module, std: float) -> None:
    """Truncated-normal init for every Linear/Embedding under ``module`` (meta-device safe)."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(m.weight, std=std, a=-3 * std, b=3 * std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
