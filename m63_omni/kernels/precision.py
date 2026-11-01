"""Accumulation-precision policy shared by kernels and layers (blueprint §4, §7).

Sensitive arithmetic (cumulative decays, gates, router logits, normalisation
statistics) runs in **at least FP32**: BF16/FP16 inputs are upcast to FP32, and
FP64 inputs stay FP64 so float64 gradient checks exercise the real code path.
"""

from __future__ import annotations

import torch


def acc_dtype(*tensors: torch.Tensor) -> torch.dtype:
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def acc(t: torch.Tensor) -> torch.Tensor:
    """Upcast to FP32 unless already FP64."""
    return t if t.dtype == torch.float64 else t.float()
