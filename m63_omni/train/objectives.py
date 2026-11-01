"""Training objectives (blueprint §9).

    L = L_AR + λ_ctc L_CTC + λ_align L_align + λ_time L_time + λ_bal L_balance + λ_MTP L_MTP + λ_mem L_memory

Every term is normalised by its own valid-target count (never by raw batch
items), so packing density and modality mix do not silently reweight losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from ..config import M63Config
from ..model.m63 import ModelOutput


def _f(x: torch.Tensor) -> float:
    return float(x.detach())


@dataclass
class LossBreakdown:
    total: torch.Tensor
    terms: dict[str, float] = field(default_factory=dict)


def ctc_loss(logits: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor,
             blank: int = 0) -> torch.Tensor:
    """CTC/ASR auxiliary loss normalised by the number of target labels. logits: [B, L, V]."""
    logp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    loss = F.ctc_loss(logp, targets, input_lengths, target_lengths, blank=blank, reduction="sum", zero_infinity=True)
    return loss / target_lengths.sum().clamp(min=1)


def contrastive_alignment(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE between paired embeddings ``a[i] <-> b[i]`` (cross-modal retrieval)."""
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    logits = a @ b.t() / temperature
    target = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def timestamp_loss(logits: torch.Tensor, target_bins: torch.Tensor) -> torch.Tensor:
    """Timestamp / temporal-ordering prediction as classification over time bins."""
    return F.cross_entropy(logits.float(), target_bins, ignore_index=-100)


def compute_loss(
    cfg: M63Config,
    out: ModelOutput,
    *,
    ctc: Optional[torch.Tensor] = None,
    align: Optional[torch.Tensor] = None,
    time: Optional[torch.Tensor] = None,
) -> LossBreakdown:
    t = cfg.training
    if out.lm_loss is None:
        raise ValueError("model output has no LM loss; pass labels")
    total = out.lm_loss
    terms = {"lm": _f(out.lm_loss)}
    aux = out.router_aux_loss()
    total = total + t.lambda_balance * cfg.moe.aux_loss_coef * aux
    terms["balance"] = _f(aux)
    if cfg.moe.z_loss_coef:
        z = out.router_z_loss()
        total = total + cfg.moe.z_loss_coef * z
        terms["router_z"] = _f(z)
    if out.mtp_loss is not None:
        total = total + t.lambda_mtp * out.mtp_loss
        terms["mtp"] = _f(out.mtp_loss)
    if out.memory_loss is not None:
        total = total + t.lambda_memory * out.memory_loss
        terms["memory"] = _f(out.memory_loss)
    for name, val, lam in (("ctc", ctc, t.lambda_ctc), ("align", align, t.lambda_align), ("time", time, t.lambda_time)):
        if val is not None:
            total = total + lam * val
            terms[name] = _f(val)
    terms["total"] = _f(total)
    return LossBreakdown(total, terms)
