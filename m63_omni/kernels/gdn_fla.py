"""Adapter for flash-linear-attention's GDN kernels (optional GPU backend, blueprint §11).

``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` is the Triton
implementation of the chunkwise gated delta rule. It handles packed documents
through *variable-length* batches (``cu_seqlens``) rather than per-token reset
flags, so this adapter:

1. flattens ``[B, T]`` into one ``[1, B*T]`` stream and cuts it into sequences at
   every row start and every reset position (``cu_seqlens``);
2. gives each row's first sequence that row's initial state and every
   post-reset sequence a zero state;
3. runs the kernel, then unflattens outputs and returns, per row, the final
   state of that row's last sequence.

The conversion is validated on CPU against :func:`m63_omni.kernels.gdn.gdn_chunked`
by passing a reference varlen kernel (:func:`varlen_reference`); the real Triton
kernel is exercised only when a GPU and ``fla`` are present. Inputs follow this
repository's convention: q/k already L2-normalised, ``g`` is the log decay (≤ 0).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .gdn import gdn_chunked

VarlenKernel = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def fla_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    except Exception:
        return False
    return True


def _fla_kernel() -> VarlenKernel:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def varlen_reference(q, k, v, g, beta, *, scale, initial_state, output_final_state, cu_seqlens):
    """Reference implementation of the fla varlen contract (CPU, for adapter tests)."""
    outs, finals = [], []
    cu = cu_seqlens.tolist()
    for i in range(len(cu) - 1):
        a, b = cu[i], cu[i + 1]
        o, s = gdn_chunked(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], scale=scale,
                           initial_state=initial_state[i : i + 1])
        outs.append(o)
        finals.append(s)
    return torch.cat(outs, dim=1), torch.cat(finals, dim=0)


def to_varlen(B: int, T: int, reset: Optional[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Sequence boundaries for the flattened stream.

    Returns ``(cu_seqlens int32 [N+1], row_first_seq [B] long, last_seq_of_row)``.
    """
    starts = torch.zeros(B, T, dtype=torch.bool, device=device)
    starts[:, 0] = True
    if reset is not None:
        starts |= reset.bool()
    flat_starts = starts.reshape(-1).nonzero().flatten()
    cu = torch.cat([flat_starts, torch.tensor([B * T], device=device)]).to(torch.int32)
    seq_row = (flat_starts // T).tolist()
    row_first = torch.tensor([seq_row.index(b) for b in range(B)], device=device)
    last = [max(i for i, r in enumerate(seq_row) if r == b) for b in range(B)]
    return cu, row_first, last


def gdn_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    reset: Optional[torch.Tensor] = None,
    kernel: Optional[VarlenKernel] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same contract as :func:`gdn_chunked`; runs a varlen chunk kernel (fla by default)."""
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    scale = Dk**-0.5 if scale is None else scale
    kernel = kernel or _fla_kernel()
    cu, row_first, last = to_varlen(B, T, reset, q.device)
    n_seq = cu.numel() - 1
    init = torch.zeros(n_seq, H, Dk, Dv, device=q.device, dtype=torch.float32)
    if initial_state is not None:
        first_has_state = ~(reset[:, 0].bool()) if reset is not None else torch.ones(B, dtype=torch.bool, device=q.device)
        init[row_first] = torch.where(first_has_state.view(B, 1, 1, 1), initial_state.float(), torch.zeros_like(initial_state.float()))

    def flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(1, B * T, *x.shape[2:])

    o, finals = kernel(flat(q), flat(k), flat(v), flat(g), flat(beta), scale=scale, initial_state=init,
                       output_final_state=True, cu_seqlens=cu)
    return o.reshape(B, T, H, Dv).to(v.dtype), finals[last].float()
