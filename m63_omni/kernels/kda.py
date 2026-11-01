"""Kimi Delta Attention (KDA) kernels: recurrent reference + chunkwise-parallel form.

KDA (Kimi Linear, 2025) refines Gated DeltaNet's scalar per-head forget gate
into a *channel-wise* one. In the transposed state convention of
:mod:`m63_omni.kernels.gdn` (state ``S`` is ``[d_k, d_v]`` per head)::

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T (scale * q_t)

with ``alpha_t = exp(g_t)``, ``g_t in R^{d_k}``, ``g <= 0``. When every channel
of ``g_t`` is equal this is exactly GDN (tested). Document resets zero the
state before token ``t`` is processed, as in GDN.

Chunk derivation used by :func:`kda_chunked` (local positions ``t`` in a chunk,
chunk-start state ``S0``, segment-local ``gcum_t[c] = sum_{s<=t} g_s[c]``)::

    S_t      = Diag(exp(gcum_t)) S0 + sum_{i<=t} Diag(exp(gcum_t - gcum_i)) k_i u_i^T
    A[t, i]  = beta_t sum_c k_t[c] k_i[c] exp(gcum_t[c] - gcum_i[c])          (i < t)
    (I + A) U = diag(beta) V - diag(beta) (K ⊙ exp(gcum)) S0
    O        = (Q ⊙ exp(gcum)) S0 + P U,   P[t, i] = sum_c q_t[c] k_i[c] exp(gcum_t[c] - gcum_i[c])  (i <= t)
    S_C      = Diag(exp(gcum_last)) S0 + (K ⊙ exp(gcum_last - gcum))^T U

The pairwise decays ``exp(gcum_t - gcum_i)`` are formed explicitly per chunk
(``[C, C, d_k]``) and masked before exponentiation, so the reference never
divides by a decayed quantity and stays exact for arbitrarily strong decay.
Production kernels (flash-linear-attention's ``chunk_kda``) use secondary
chunking for the same stability; they must match this reference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .gdn import _segment_local_cumsum
from .precision import acc_dtype


def _check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> None:
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape [B, T, H, Dk]")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("v must be [B, T, H, Dv]")
    if g.shape != q.shape:
        raise ValueError("g must be [B, T, H, Dk] (channel-wise decay)")
    if beta.shape != q.shape[:3]:
        raise ValueError("beta must be [B, T, H]")


def kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    reset: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token KDA in >= FP32. Returns ``(o [B,T,H,Dv], final_state [B,H,Dk,Dv])``."""
    _check(q, k, v, g, beta)
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    scale = Dk**-0.5 if scale is None else scale
    out_dtype = v.dtype
    cd = acc_dtype(q, k, v, g, beta)
    q, k, v = q.to(cd) * scale, k.to(cd), v.to(cd)
    g, beta = g.to(cd), beta.to(cd)
    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state.to(cd).clone()
    outs = []
    for t in range(T):
        if reset is not None:
            S = S * (~reset[:, t]).to(cd).view(B, 1, 1, 1)
        S = S * torch.exp(g[:, t]).unsqueeze(-1)  # Diag(alpha_t) S_{t-1}
        kt, vt, bt = k[:, t], v[:, t], beta[:, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        u = bt.unsqueeze(-1) * (vt - pred)
        S = S + kt.unsqueeze(-1) * u.unsqueeze(-2)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    o = torch.stack(outs, dim=1) if outs else v.new_zeros(B, 0, H, Dv)
    return o.to(out_dtype), S


def kda_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    reset: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise-parallel KDA. Same signature/semantics as :func:`kda_recurrent_reference`."""
    _check(q, k, v, g, beta)
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    scale = Dk**-0.5 if scale is None else scale
    out_dtype = v.dtype
    cd = acc_dtype(q, k, v, g, beta)
    if T == 0:
        S0 = q.new_zeros(B, H, Dk, Dv, dtype=cd) if initial_state is None else initial_state.to(cd)
        return v.new_zeros(B, 0, H, Dv), S0
    C = chunk_size
    N = -(-T // C)
    pad = N * C - T
    if reset is None:
        reset = torch.zeros(B, T, dtype=torch.bool, device=q.device)

    def prep(x: torch.Tensor) -> torch.Tensor:  # [B,T,H,D] -> [B,H,N,C,D] in >= fp32, zero padded
        x = x.to(cd)
        if pad:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
        return x.view(B, N, C, H, -1).permute(0, 3, 1, 2, 4)

    qc, kc, vc = prep(q) * scale, prep(k), prep(v)
    gc = prep(g)  # [B,H,N,C,Dk]; padded positions: g = 0, beta = 0 -> state untouched
    bc = prep(beta.unsqueeze(-1)).squeeze(-1)  # [B,H,N,C]
    rc = F.pad(reset, (0, pad)).view(B, 1, N, C).expand(B, H, N, C)

    seg = torch.cumsum(rc.long(), dim=-1)  # resets at or before t within the chunk
    gcum = _segment_local_cumsum(gc.transpose(-1, -2), rc.unsqueeze(-2).expand(B, H, N, Dk, C)).transpose(-1, -2)
    first_seg = (seg == 0).unsqueeze(-1)
    gamma = torch.where(first_seg, torch.exp(gcum), torch.zeros_like(gcum))  # decay from chunk start, per channel
    last_same = seg[..., -1:].eq(seg).unsqueeze(-1)  # positions in the chunk's final segment
    decay_to_end = torch.where(last_same, torch.exp(gcum[..., -1:, :] - gcum), torch.zeros_like(gcum))
    gamma_last = gamma[..., -1, :]  # [B,H,N,Dk]
    same_seg = seg.unsqueeze(-1) == seg.unsqueeze(-2)  # [B,H,N,C,C]
    causal_incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    causal_strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    eye = torch.eye(C, device=q.device, dtype=cd)

    S = q.new_zeros(B, H, Dk, Dv, dtype=cd) if initial_state is None else initial_state.to(cd)
    outs = []
    for n in range(N):
        gn, kn, qn, bn = gcum[:, :, n], kc[:, :, n], qc[:, :, n], bc[:, :, n]
        valid = (same_seg[:, :, n] & causal_incl).unsqueeze(-1)  # [B,H,C,C,1]
        dec = torch.exp((gn.unsqueeze(-2) - gn.unsqueeze(-3)).masked_fill(~valid, float("-inf")))  # [B,H,C,C,Dk]
        kk = torch.einsum("bhtc,bhic,bhtic->bhti", kn, kn, dec)
        A = (bn.unsqueeze(-1) * kk).masked_fill(~causal_strict, 0.0)
        rhs = torch.cat([bn.unsqueeze(-1) * gamma[:, :, n] * kn, bn.unsqueeze(-1) * vc[:, :, n]], dim=-1)
        sol = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
        W, U_prime = sol[..., :Dk], sol[..., Dk:]
        U = U_prime - W @ S
        P = torch.einsum("bhtc,bhic,bhtic->bhti", qn, kn, dec)  # zero above the diagonal / across segments
        outs.append((gamma[:, :, n] * qn) @ S + P @ U)
        S = gamma_last[:, :, n].unsqueeze(-1) * S + (decay_to_end[:, :, n] * kn).transpose(-1, -2) @ U
    o = torch.stack(outs, dim=2)  # [B,H,N,C,Dv]
    o = o.permute(0, 2, 3, 1, 4).reshape(B, N * C, H, Dv)[:, :T]
    return o.to(out_dtype), S
