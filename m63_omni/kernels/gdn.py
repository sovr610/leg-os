"""Gated DeltaNet (GDN) kernels: naive recurrent reference + chunkwise-parallel form.

Recurrence (state ``S`` is ``[d_k, d_v]`` per head, blueprint §4 written in the
transposed convention used by most GDN kernels)::

    S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T (scale * q_t)

with ``alpha_t = exp(g_t)``, ``g_t <= 0``. A document reset at position ``t``
(``reset[b, t] = True``) zeroes the state *before* token ``t`` is processed,
which is exactly ``alpha_t = 0``.

* :func:`gdn_recurrent_reference` is the numerical reference (token loop, FP32).
  It is also the decode path.
* :func:`gdn_chunked` is the chunkwise-parallel training/prefill form. Within a
  chunk, the delta-rule updates are folded into a unit-lower-triangular solve
  (the WY / UT representation); across chunks the state is carried
  sequentially. Resets inside a chunk are handled exactly with segment-local
  cumulative decays instead of a large negative constant, so no precision is
  lost in the cumulative-decay arithmetic (computed in FP32 per §4).

Derivation used by :func:`gdn_chunked` (per chunk, local positions ``t``,
chunk-start state ``S0``, ``Gamma[t,i] = prod_{i<s<=t} alpha_s``,
``gamma[t] = prod_{s<=t} alpha_s``)::

    u_t = beta_t (v_t - alpha_t S_{t-1}^T k_t)           so that  S_t = alpha_t S_{t-1} + k_t u_t^T
    (I + A) U = diag(beta) V - diag(beta*gamma) K S0      A[t,i] = beta_t Gamma[t,i] k_t.k_i  (i < t)
    O = diag(gamma) Q S0 + (Gamma ⊙ Q K^T)_{i<=t} U
    S_C = gamma_last S0 + (Gamma[last, :] ⊙ K)^T U
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> None:
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape [B, T, H, Dk]")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("v must be [B, T, H, Dv]")
    if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("g and beta must be [B, T, H]")


def gdn_recurrent_reference(
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
    """Token-by-token GDN in FP32. Returns ``(o [B,T,H,Dv], final_state [B,H,Dk,Dv] fp32)``."""
    _check(q, k, v, g, beta)
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    scale = Dk**-0.5 if scale is None else scale
    out_dtype = v.dtype
    q, k, v = q.float() * scale, k.float(), v.float()
    g, beta = g.float(), beta.float()
    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state.float().clone()
    outs = []
    for t in range(T):
        if reset is not None:
            keep = (~reset[:, t]).float().view(B, 1, 1, 1)
            S = S * keep
        S = S * torch.exp(g[:, t]).view(B, H, 1, 1)
        kt, vt, bt = k[:, t], v[:, t], beta[:, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        u = bt.unsqueeze(-1) * (vt - pred)
        S = S + kt.unsqueeze(-1) * u.unsqueeze(-2)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    o = torch.stack(outs, dim=1) if outs else v.new_zeros(B, 0, H, Dv)
    return o.to(out_dtype), S


def _segment_local_cumsum(g: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
    """Cumulative sum of ``g`` along the last dim that restarts at reset positions.

    The value at a reset position ``r`` is ``g_r`` (decays *after* the reset are
    accumulated from ``r``); differences within a segment are exact.
    """
    C = g.shape[-1]
    cs = torch.cumsum(g, dim=-1)
    idx = torch.arange(C, device=g.device).expand_as(g)
    marker = torch.where(reset, idx, torch.full_like(idx, -1))
    last_reset = torch.cummax(marker, dim=-1).values
    base = torch.gather(cs - g, -1, last_reset.clamp(min=0))
    base = torch.where(last_reset >= 0, base, torch.zeros_like(base))
    return cs - base


def gdn_chunked(
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
    """Chunkwise-parallel GDN. Same signature/semantics as the recurrent reference."""
    _check(q, k, v, g, beta)
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    scale = Dk**-0.5 if scale is None else scale
    out_dtype = v.dtype
    if T == 0:
        S0 = q.new_zeros(B, H, Dk, Dv, dtype=torch.float32) if initial_state is None else initial_state.float()
        return v.new_zeros(B, 0, H, Dv), S0
    C = chunk_size
    N = -(-T // C)
    pad = N * C - T
    if reset is None:
        reset = torch.zeros(B, T, dtype=torch.bool, device=q.device)

    def prep(x: torch.Tensor) -> torch.Tensor:  # [B,T,H,D] -> [B,H,N,C,D] fp32, zero padded
        x = x.float()
        if pad:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
        return x.view(B, N, C, H, -1).permute(0, 3, 1, 2, 4)

    qc, kc, vc = prep(q) * scale, prep(k), prep(v)
    # Padded positions: beta = 0, g = 0 -> state untouched, outputs discarded.
    gc = prep(g.unsqueeze(-1)).squeeze(-1)  # [B,H,N,C]
    bc = prep(beta.unsqueeze(-1)).squeeze(-1)
    rc = F.pad(reset, (0, pad)).view(B, 1, N, C).expand(B, H, N, C)

    seg = torch.cumsum(rc.long(), dim=-1)  # resets at or before t within chunk
    gcum = _segment_local_cumsum(gc, rc)
    first_seg = seg == 0
    gamma = torch.where(first_seg, torch.exp(gcum), torch.zeros_like(gcum))  # decay from chunk start

    same_seg = seg.unsqueeze(-1) == seg.unsqueeze(-2)  # [B,H,N,C,C]
    causal_incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    causal_strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    diff = gcum.unsqueeze(-1) - gcum.unsqueeze(-2)
    gamma_pair = torch.exp(diff.masked_fill(~(same_seg & causal_incl), float("-inf")))  # Gamma[t, i]

    kk = kc @ kc.transpose(-1, -2)
    A = (bc.unsqueeze(-1) * gamma_pair * kk).masked_fill(~causal_strict, 0.0)
    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    L = eye + A
    rhs = torch.cat([(bc * gamma).unsqueeze(-1) * kc, bc.unsqueeze(-1) * vc], dim=-1)
    sol = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)
    W, U_prime = sol[..., :Dk], sol[..., Dk:]

    P = (qc @ kc.transpose(-1, -2)) * gamma_pair  # already zero above diagonal / across segments
    gamma_q = gamma.unsqueeze(-1) * qc
    last_same = seg[..., -1:].eq(seg)  # positions in the chunk's final segment
    decay_to_end = torch.where(last_same, torch.exp(gcum[..., -1:] - gcum), torch.zeros_like(gcum))
    k_to_end = decay_to_end.unsqueeze(-1) * kc
    gamma_last = gamma[..., -1]  # [B,H,N]

    S = q.new_zeros(B, H, Dk, Dv, dtype=torch.float32) if initial_state is None else initial_state.float()
    outs = []
    for n in range(N):
        U = U_prime[:, :, n] - W[:, :, n] @ S
        outs.append(gamma_q[:, :, n] @ S + P[:, :, n] @ U)
        S = gamma_last[:, :, n, None, None] * S + k_to_end[:, :, n].transpose(-1, -2) @ U
    o = torch.stack(outs, dim=2)  # [B,H,N,C,Dv]
    o = o.permute(0, 2, 3, 1, 4).reshape(B, N * C, H, Dv)[:, :T]
    return o.to(out_dtype), S


def causal_depthwise_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    conv_state: Optional[torch.Tensor] = None,
    reset: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Short causal depthwise convolution that never mixes across documents.

    ``x``: [B, T, C]; ``weight``: [C, K] (tap ``K-1`` multiplies the current
    token); ``conv_state``: [B, C, K-1] previous inputs (oldest first).
    Returns ``(y [B,T,C], new_state [B,C,K-1])``. An input at ``t-j`` contributes
    to output ``t`` only when no reset occurs in ``(t-j, t]``.
    """
    B, T, Cc = x.shape
    K = weight.shape[1]
    if conv_state is None:
        conv_state = x.new_zeros(B, Cc, K - 1)
    hist = torch.cat([conv_state.transpose(1, 2).to(x.dtype), x], dim=1)  # [B, K-1+T, C]
    if reset is None:
        doc = torch.zeros(B, T, dtype=torch.long, device=x.device)
    else:
        doc = torch.cumsum(reset.long(), dim=1)
    doc_hist = torch.cat([doc.new_zeros(B, K - 1), doc], dim=1)
    y = torch.zeros_like(x)
    for j in range(K):  # j = lag
        src = hist[:, K - 1 - j : K - 1 - j + T]
        same = (doc_hist[:, K - 1 - j : K - 1 - j + T] == doc).unsqueeze(-1).to(x.dtype)
        y = y + src * same * weight[:, K - 1 - j].to(x.dtype)
    new_state = hist[:, -(K - 1):].transpose(1, 2) if K > 1 else x.new_zeros(B, Cc, 0)
    if reset is not None and K > 1:
        # If the last document began within the final K-1 inputs, older inputs must not leak forward.
        last_doc = doc[:, -1:]
        keep = (doc_hist[:, -(K - 1):] == last_doc).unsqueeze(1).to(x.dtype)
        new_state = new_state * keep
    return y, new_state.contiguous()
