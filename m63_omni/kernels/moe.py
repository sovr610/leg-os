"""Dropless MoE expert computation (blueprint §4, §7).

Token dropping is forbidden: every (token, selected expert) pair is computed.
Three implementations share one contract and are cross-checked in tests:

* :func:`moe_reference`   — naive per-expert masking loop (numerical reference);
* :func:`moe_grouped`     — sort-by-expert permutation + per-group GEMMs
  (MegaBlocks-style grouped GEMM with variable group sizes);
* :func:`moe_padded_bmm`  — block-padded batched GEMM over all experts
  (fast when loads are balanced; padding is masked, never dropped).

Expert weights are stacked: ``w_gate, w_up: [E, d, h]``, ``w_down: [E, h, d]``;
an expert computes ``down(silu(x @ w_gate) * (x @ w_up))`` (SwiGLU).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TokenDropError(RuntimeError):
    """Raised if any routed token-slot would not be computed (a correctness bug)."""


def _swiglu(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor) -> torch.Tensor:
    return (F.silu(x @ wg) * (x @ wu)) @ wd


def moe_reference(
    x: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    """x: [N, d]; idx, weight: [N, k]. Returns [N, d]."""
    out = torch.zeros_like(x)
    for e in range(w_gate.shape[0]):
        hit = idx == e  # [N, k]
        rows = hit.any(dim=-1)
        if not rows.any():
            continue
        w = (weight * hit).sum(dim=-1)[rows].unsqueeze(-1)
        out[rows] += w.to(x.dtype) * _swiglu(x[rows], w_gate[e], w_up[e], w_down[e])
    return out


def expert_counts(idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(idx.reshape(-1), minlength=num_experts)


def moe_grouped(
    x: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    N, k = idx.shape
    E = w_gate.shape[0]
    flat_idx = idx.reshape(-1)
    order = torch.argsort(flat_idx, stable=True)
    token_of_slot = order // k
    counts = expert_counts(idx, E)
    if int(counts.sum()) != N * k:
        raise TokenDropError("grouped dispatch lost token slots")
    xs = x[token_of_slot]
    ys = torch.empty_like(xs)
    start = 0
    for e, c in enumerate(counts.tolist()):
        if c:
            ys[start : start + c] = _swiglu(xs[start : start + c], w_gate[e], w_up[e], w_down[e])
            start += c
    ws = weight.reshape(-1)[order].unsqueeze(-1).to(x.dtype)
    out = torch.zeros_like(x)
    out.index_add_(0, token_of_slot, ys * ws)
    return out


def moe_padded_bmm(
    x: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    N, k = idx.shape
    E, d, _ = w_gate.shape
    counts = expert_counts(idx, E)
    cap = int(counts.max()) if N else 0
    if cap == 0:
        return torch.zeros_like(x)
    flat_idx = idx.reshape(-1)
    order = torch.argsort(flat_idx, stable=True)
    sorted_e = flat_idx[order]
    offsets = torch.cumsum(counts, 0) - counts
    rank_in_group = torch.arange(N * k, device=x.device) - offsets[sorted_e]
    buf = x.new_zeros(E, cap, d)
    buf[sorted_e, rank_in_group] = x[order // k]
    h = F.silu(torch.bmm(buf, w_gate)) * torch.bmm(buf, w_up)
    yb = torch.bmm(h, w_down)  # [E, cap, d]
    ys = yb[sorted_e, rank_in_group]
    if ys.shape[0] != N * k:
        raise TokenDropError("padded dispatch lost token slots")
    ws = weight.reshape(-1)[order].unsqueeze(-1).to(x.dtype)
    out = torch.zeros_like(x)
    out.index_add_(0, order // k, ys * ws)
    return out


def grouped_mm_available() -> bool:
    return hasattr(torch, "_grouped_mm")


def moe_grouped_mm(
    x: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    """One grouped GEMM per projection via ``torch._grouped_mm`` (variable-size expert groups).

    Tokens are permuted so each expert's slots are contiguous and the three
    projections each run as a single grouped matmul over all experts — no Python
    loop over experts. On GPU this targets the CUTLASS grouped-GEMM kernels
    (BF16, dims aligned to 16); it is a private PyTorch API, so callers must gate on
    :func:`grouped_mm_available` and keep the reference as the fallback.
    """
    N, k = idx.shape
    E = w_gate.shape[0]
    flat_idx = idx.reshape(-1)
    order = torch.argsort(flat_idx, stable=True)
    counts = expert_counts(idx, E)
    if int(counts.sum()) != N * k:
        raise TokenDropError("grouped-mm dispatch lost token slots")
    offs = torch.cumsum(counts, 0).to(torch.int32)
    xs = x[order // k]
    h = F.silu(torch._grouped_mm(xs, w_gate, offs=offs)) * torch._grouped_mm(xs, w_up, offs=offs)
    ys = torch._grouped_mm(h, w_down, offs=offs)
    ws = weight.reshape(-1)[order].unsqueeze(-1).to(x.dtype)
    out = torch.zeros_like(x)
    out.index_add_(0, order // k, ys * ws)
    return out


IMPLEMENTATIONS = {"reference": moe_reference, "grouped": moe_grouped, "padded": moe_padded_bmm}
if grouped_mm_available():
    IMPLEMENTATIONS["grouped_mm"] = moe_grouped_mm
