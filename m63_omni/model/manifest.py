"""Parameter manifest generated from code (blueprint §3, §11, Step 1 gate).

The manifest instantiates the real :class:`M63Omni` on PyTorch's ``meta``
device (no memory is allocated, so the full 62B-parameter shape is cheap),
groups every parameter into the blueprint's budget categories, and reports
resident and active-per-text-token counts. An independent analytic formula
cross-checks the backbone. Documentation (model card) is generated from the
same source of truth so the two cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from ..common import content_hash
from ..config import GDN, KDA, M63Config

# Blueprint §3 reference table (resident parameters, billions).
BLUEPRINT_REFERENCE = {
    "routed_experts": 57.982e9,
    "shared_experts": 0.906e9,
    "gdn_mixers": 1.516e9,
    "gqa_mixers": 0.705e9,
    "embedding": 0.537e9,
    "multimodal": 0.630e9,
    "mtp": 0.180e9,
    "routers_memory": 0.026e9,
    "total": 62.481e9,
    "active_excl_head": 4.96e9,
    "active_incl_head": 5.501e9,
}

CATEGORIES = (
    "routed_experts",
    "shared_experts",
    "gdn_mixers",
    "kda_mixers",
    "gqa_mixers",
    "norms",
    "embedding",
    "multimodal",
    "mtp",
    "routers_memory",
)


@dataclass
class ParameterManifest:
    config_fingerprint: str
    resident: dict[str, int]
    active: dict[str, float]  # per text token, excluding the tied head
    head: int
    tensors: dict[str, list[int]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.resident.values())

    @property
    def active_excl_head(self) -> float:
        return sum(self.active.values())

    @property
    def active_incl_head(self) -> float:
        return self.active_excl_head + self.head

    def to_dict(self) -> dict:
        return {
            "config_fingerprint": self.config_fingerprint,
            "resident": self.resident,
            "active_per_text_token": self.active,
            "tied_head": self.head,
            "total": self.total,
            "active_excl_head": self.active_excl_head,
            "active_incl_head": self.active_incl_head,
            "tensor_count": len(self.tensors),
            "manifest_hash": content_hash({k: v for k, v in self.tensors.items()}),
        }

    def to_markdown(self) -> str:
        rows = ["| Component | Resident | Active per text token |", "|---|---:|---:|"]
        for c in CATEGORIES:
            if not self.resident[c] and c not in BLUEPRINT_REFERENCE:
                continue  # e.g. no KDA layers in the blueprint pattern
            rows.append(f"| {c} | {self.resident[c] / 1e9:.3f}B | {self.active[c] / 1e9:.3f}B |")
        rows.append(f"| tied LM head (shares embedding) | — | {self.head / 1e9:.3f}B when included |")
        rows.append(f"| **total** | **{self.total / 1e9:.3f}B** | **{self.active_excl_head / 1e9:.3f}B** "
                    f"(**{self.active_incl_head / 1e9:.3f}B** incl. head) |")
        return "\n".join(rows)


def categorize(name: str, cfg: M63Config) -> str:
    parts = name.split(".")
    if parts[0] == "embed" or parts[0] == "lm_head":
        return "embedding"
    if parts[0] == "final_norm":
        return "norms"
    if parts[0] == "memory":
        return "routers_memory"
    if parts[0] == "mtp":
        return "mtp"
    if parts[0] == "multimodal":
        return "multimodal"
    if parts[0] == "blocks":
        layer = int(parts[1])
        sub = parts[2]
        if sub in ("mixer_norm", "moe_norm"):
            return "norms"
        if sub == "mixer":
            kind = cfg.layer_kinds()[layer]
            return "gdn_mixers" if kind == GDN else "kda_mixers" if kind == KDA else "gqa_mixers"
        if sub == "moe":
            if parts[3] == "router":
                return "routers_memory"
            if parts[3] == "shared":
                return "shared_experts"
            return "routed_experts"
    raise ValueError(f"uncategorised parameter {name}")


def build_manifest(cfg: M63Config, model: Optional[torch.nn.Module] = None) -> ParameterManifest:
    from .m63 import M63Omni

    if model is None:
        with torch.device("meta"):
            model = M63Omni(cfg)
    resident = {c: 0 for c in CATEGORIES}
    tensors: dict[str, list[int]] = {}
    for name, p in model.named_parameters():
        resident[categorize(name, cfg)] += p.numel()
        tensors[name] = list(p.shape)
    frac = cfg.moe.top_k / cfg.moe.routed_experts
    active = {c: float(resident[c]) for c in CATEGORIES}
    active["routed_experts"] = resident["routed_experts"] * frac
    active["embedding"] = 0.0  # lookup, not compute; the tied head is reported separately
    active["multimodal"] = 0.0  # input-dependent
    active["mtp"] = 0.0  # zero during ordinary decoding
    head = cfg.model.vocab_size * cfg.model.hidden_size
    return ParameterManifest(cfg.fingerprint(), resident, active, head, tensors)


def analytic_backbone(cfg: M63Config) -> dict[str, int]:
    """Closed-form counts for the backbone categories (independent of module code)."""
    d, L = cfg.model.hidden_size, cfg.model.layers
    g, a, moe = cfg.gdn, cfg.attention, cfg.moe
    kinds = cfg.layer_kinds()
    n_gdn = sum(k == GDN for k in kinds)
    n_kda = sum(k == KDA for k in kinds)
    n_gqa = L - n_gdn - n_kda
    expert = 3 * d * moe.expert_hidden
    gdn = (2 * d * g.heads * g.key_dim + d * g.inner_width  # q, k, v
           + 2 * d * g.heads  # a, b
           + (d * g.inner_width if g.output_gate else 0)  # output gate
           + g.inner_width * d  # o
           + (2 * g.heads * g.key_dim + g.inner_width) * g.conv_kernel  # short convs
           + 2 * g.heads + g.value_dim)  # A_log, dt_bias, output norm
    rank = g.kda_low_rank or g.key_dim
    kda = (2 * d * g.heads * g.key_dim + d * g.inner_width  # q, k, v
           + d * rank + rank * g.heads * g.key_dim  # channel-wise forget gate (low rank)
           + d * g.heads  # beta
           + ((d + g.inner_width) * rank if g.output_gate else 0)  # low-rank sigmoid output gate
           + g.inner_width * d  # o
           + (2 * g.heads * g.key_dim + g.inner_width) * g.conv_kernel  # short convs
           + g.heads + g.heads * g.key_dim + g.value_dim)  # A_log, per-channel dt_bias, output norm
    qd, kvd = a.query_heads * a.head_dim, a.kv_heads * a.head_dim
    gqa = d * qd + 2 * d * kvd + qd * d + (d * qd if a.output_gate else 0) + (2 * a.head_dim if a.qk_norm else 0)
    return {
        "routed_experts": L * moe.routed_experts * expert,
        "shared_experts": L * moe.shared_experts * expert,
        "gdn_mixers": n_gdn * gdn,
        "kda_mixers": n_kda * kda,
        "gqa_mixers": n_gqa * gqa,
        "embedding": cfg.model.vocab_size * d * (1 if cfg.model.tie_embeddings else 2),
        "routers": L * moe.routed_experts * d,
    }
