"""Attention kernels: explicit-softmax reference and SDPA fast path.

Masks combine three rules (Appendix B "GQA / memory masks"):

* causal: a query at absolute position ``p`` sees keys at positions ``<= p``;
* document isolation: keys must share the query's segment (packed sequences);
* memory prefix: ``n_mem`` neural-memory read tokens are visible to every query
  and carry no position (they come from the *previous completed* memory state,
  so they cannot leak future information).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .precision import acc_dtype


def build_attention_mask(
    q_pos: torch.Tensor,
    q_seg: torch.Tensor,
    k_pos: torch.Tensor,
    k_seg: torch.Tensor,
    n_mem: int = 0,
) -> torch.Tensor:
    """Boolean mask ``[B, 1, Tq, n_mem + Tk]`` (True = may attend).

    ``q_pos/q_seg``: [B, Tq] stream positions and segment IDs of queries.
    ``k_pos/k_seg``: [B, Tk] for (cached + new) keys. Stream positions are
    monotonically increasing token indices within the session (not RoPE
    positions, which restart per document).
    """
    causal = k_pos.unsqueeze(1) <= q_pos.unsqueeze(2)
    same = k_seg.unsqueeze(1) == q_seg.unsqueeze(2)
    mask = causal & same
    if n_mem:
        mem = torch.ones(mask.shape[0], mask.shape[1], n_mem, dtype=torch.bool, device=mask.device)
        mask = torch.cat([mem, mask], dim=-1)
    return mask.unsqueeze(1)


def _expand_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    return x if groups == 1 else x.repeat_interleave(groups, dim=1)


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: Optional[float] = None
) -> torch.Tensor:
    """Explicit FP32 softmax attention. q: [B,Hq,Tq,D]; k,v: [B,Hkv,Tk,D]."""
    groups = q.shape[1] // k.shape[1]
    k, v = _expand_kv(k, groups), _expand_kv(v, groups)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    cd = acc_dtype(q, k, v)
    scores = (q.to(cd) @ k.to(cd).transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)  # fully-masked rows produce zeros
    return (probs @ v.to(cd)).to(q.dtype)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: Optional[float] = None
) -> torch.Tensor:
    """Fast path via ``scaled_dot_product_attention`` (FlashAttention-class on GPU)."""
    groups = q.shape[1] // k.shape[1]
    k, v = _expand_kv(k, groups), _expand_kv(v, groups)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


# ----------------------------------------------------------------- FlexAttention
_COMPILED_FLEX = None


def flex_available() -> bool:
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa: F401
    except Exception:
        return False
    return True


def attention_flex(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pos: torch.Tensor,
    q_seg: torch.Tensor,
    k_pos: torch.Tensor,
    k_seg: torch.Tensor,
    n_mem: int = 0,
    scale: Optional[float] = None,
    compile: bool = False,
) -> torch.Tensor:
    """FlexAttention with a *block-sparse* mask instead of a dense ``[B, Tq, Tk]`` boolean.

    The rule is the same as :func:`build_attention_mask` (memory prefix visible,
    causal, same segment), expressed as a ``mask_mod``; ``create_block_mask``
    marks fully-masked tiles so GPU kernels skip them — packed documents and
    long contexts no longer pay for the dense mask. ``compile=True`` wraps the
    kernel in ``torch.compile`` (required for speed; eager mode is a reference).
    """
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    global _COMPILED_FLEX
    B, Hq, Tq, _ = q.shape
    Tk = k.shape[2]

    def mask_mod(b, h, qi, ki):
        is_mem = ki < n_mem
        kk = (ki - n_mem).clamp(min=0)
        visible = (k_pos[b, kk] <= q_pos[b, qi]) & (k_seg[b, kk] == q_seg[b, qi])
        return is_mem | visible

    block_mask = create_block_mask(mask_mod, B, None, Tq, Tk, device=q.device)
    fn = flex_attention
    if compile:
        if _COMPILED_FLEX is None:
            _COMPILED_FLEX = torch.compile(flex_attention)
        fn = _COMPILED_FLEX
    return fn(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
