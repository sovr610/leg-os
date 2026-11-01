"""Neural-memory kernels (blueprint §6).

The session memory is a two-layer MLP ``M(x) = W2 silu(W1 x)`` with shapes
``W1: [h, d_mem]``, ``W2: [d_mem, h]`` (``[256 -> 512 -> 256]`` in the
reference). Its associative loss over a segment is

    l(M; k, v) = mean_{t, j} (M(k_t) - v_t)_j^2

and the "surprise" signal is its gradient w.r.t. the memory weights.
:func:`assoc_grad_closed_form` is the optimised path (no autograd graph needed
at inference); :func:`assoc_grad_autograd` is the numerical reference.
Both are batched over sessions: ``W1: [B, h, d]``, ``k, v: [B, T, d]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mlp_forward(W1: torch.Tensor, W2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """x: [B, T, d] -> [B, T, d]."""
    return F.silu(x @ W1.transpose(-1, -2)) @ W2.transpose(-1, -2)


def assoc_loss(W1: torch.Tensor, W2: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Per-session associative loss ``[B]``. ``valid``: optional [B, T] mask."""
    err = (mlp_forward(W1, W2, k) - v).pow(2).mean(dim=-1)  # [B, T]
    if valid is None:
        return err.mean(dim=-1)
    w = valid.to(err.dtype)
    return (err * w).sum(-1) / w.sum(-1).clamp(min=1.0)


def assoc_grad_closed_form(
    W1: torch.Tensor, W2: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(loss [B], dW1 [B,h,d], dW2 [B,d,h])`` without building an autograd graph
    through the inner problem (the ops remain differentiable for outer training)."""
    B, T, d = k.shape
    a = k @ W1.transpose(-1, -2)  # [B,T,h]
    sig = torch.sigmoid(a)
    z = a * sig
    y = z @ W2.transpose(-1, -2)
    e = y - v
    if valid is None:
        w = torch.full((B, T), 1.0 / T, dtype=k.dtype, device=k.device)
    else:
        vm = valid.to(k.dtype)
        w = vm / vm.sum(-1, keepdim=True).clamp(min=1.0)
    loss = (e.pow(2).mean(-1) * w).sum(-1)
    dy = e * (2.0 / d) * w.unsqueeze(-1)  # [B,T,d]
    dW2 = dy.transpose(-1, -2) @ z  # [B,d,h]
    dz = dy @ W2  # [B,T,h]
    da = dz * (sig * (1 + a * (1 - sig)))
    dW1 = da.transpose(-1, -2) @ k  # [B,h,d]
    return loss, dW1, dW2


def assoc_grad_autograd(
    W1: torch.Tensor, W2: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        W1_ = W1.detach().requires_grad_(True)
        W2_ = W2.detach().requires_grad_(True)
        loss = assoc_loss(W1_, W2_, k.detach(), v.detach(), valid)
        g1, g2 = torch.autograd.grad(loss.sum(), (W1_, W2_))
    return loss.detach(), g1, g2


def clip_by_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Per-session (leading dim) norm clipping: x / max(1, ||x|| / max_norm)."""
    n = x.flatten(1).norm(dim=-1).clamp(min=1e-12)
    factor = torch.clamp(max_norm / n, max=1.0)
    return x * factor.view(-1, *([1] * (x.dim() - 1)))
