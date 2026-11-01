"""Stage A-D training curriculum and compute-normalised mixture weights (blueprint §8).

The §8 table (kept verbatim as :data:`BLUEPRINT_STAGES`):

======  ========  =================  =========================================
Stage   Positions Sequence           Sampling start (text/img/video/audio/AV)
======  ========  =================  =========================================
A       3.50T     predominantly 4K   60 / 10 / 10 / 10 / 10
B       0.75T     8K-32K             50 / 12 / 14 / 12 / 12
C       0.35T     32K->128K          55 / 10 / 15 / 10 / 10
D       0.15T     128K->256K         45 /  5 / 20 / 10 / 20
Total   4.75T
======  ========  =================  =========================================

The default curriculum (:data:`STAGES`) makes Stage A text-heavy, as
recommended in ``docs/IMPROVEMENTS.md`` §4: a 0.35T-position text-only warm-up,
then 70 / 12 / 5 / 8 / 5 for the remaining 3.15T. Peer omni models all started
from a pretrained text model and still regressed on instruction following,
which is this orchestrator's core skill; the warm-up length is a starting point
for the proxy ablation, not a measured optimum. Stages B-D are unchanged.

"A position is counted after text tokenization or modality projection ...
a video position incurs encoder cost that a text position does not. The
sampler therefore starts from the sampling percentages below and converts them
into measured compute-normalized weights after the pilot." The table column is
the *compute-normalised starting point*, so each percentage is read as the
target share of compute. :func:`compute_normalized_weights` converts it into
per-position sampling weights ``w_m ∝ p_m / cost_m`` using measured cost per
position; :func:`compute_shares` is the inverse and is used to verify it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

__all__ = [
    "MIXTURE_KEYS",
    "StageMixture",
    "STAGES",
    "BLUEPRINT_STAGES",
    "BLUEPRINT_STAGE_A",
    "STAGE_A",
    "STAGE_B",
    "STAGE_C",
    "STAGE_D",
    "TOTAL_POSITIONS",
    "get_stage",
    "stage_at",
    "mixture_at",
    "compute_normalized_weights",
    "compute_shares",
    "stage_position_budget",
    "text_control_budget",
    "validate_curriculum",
    "MixtureError",
]

#: Mixture categories, in table order.
MIXTURE_KEYS: tuple[str, ...] = ("text_code", "image", "video", "audio", "av_sync")

_T = 1_000_000_000_000


class MixtureError(ValueError):
    pass


@dataclass(frozen=True)
class StageMixture:
    """One curriculum stage from the §8 table."""

    name: str
    title: str
    positions: int
    seq_len_start: int
    seq_len_end: int
    sequence_curriculum: str
    sampling_percent: Mapping[str, float]  # the stage's mix (after any warm-up)
    purpose: str
    warmup_positions: int = 0  # leading positions of the stage sampled from ``warmup_percent`` instead
    warmup_percent: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampling_percent", MappingProxyType(dict(self.sampling_percent)))
        object.__setattr__(self, "warmup_percent", MappingProxyType(dict(self.warmup_percent)))

    @property
    def effective_percent(self) -> dict[str, float]:
        """Position-weighted share of each category over the whole stage (warm-up included)."""
        if not self.warmup_positions:
            return dict(self.sampling_percent)
        f = self.warmup_positions / self.positions
        return {k: f * self.warmup_percent.get(k, 0.0) + (1 - f) * self.sampling_percent[k] for k in self.sampling_percent}

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "title": self.title,
            "positions": self.positions,
            "seq_len_start": self.seq_len_start,
            "seq_len_end": self.seq_len_end,
            "sequence_curriculum": self.sequence_curriculum,
            "sampling_percent": dict(self.sampling_percent),
            "warmup_positions": self.warmup_positions,
            "warmup_percent": dict(self.warmup_percent),
            "purpose": self.purpose,
        }


def _mix(text: float, image: float, video: float, audio: float, av: float) -> dict[str, float]:
    return dict(zip(MIXTURE_KEYS, (text, image, video, audio, av)))


BLUEPRINT_STAGE_A = StageMixture(
    "A", "joint foundation", 3_500_000_000_000, 4096, 4096,
    "Predominantly 4K; native multimodal batches from the start",
    _mix(60, 10, 10, 10, 10),
    "Language, world knowledge, perception, basic cross-modal alignment",
)
STAGE_A = StageMixture(
    "A", "joint foundation (text-heavy)", 3_500_000_000_000, 4096, 4096,
    "Predominantly 4K; text-only warm-up, then native multimodal batches",
    _mix(70, 12, 5, 8, 5),
    "Language, world knowledge, perception, basic cross-modal alignment; protect instruction following",
    warmup_positions=350_000_000_000,
    warmup_percent=_mix(100, 0, 0, 0, 0),
)
STAGE_B = StageMixture(
    "B", "capability mid-training", 750_000_000_000, 8192, 32768,
    "8K-32K",
    _mix(50, 12, 14, 12, 12),
    "STEM/code, document understanding, richer temporal and tool-adjacent data",
)
STAGE_C = StageMixture(
    "C", "long-context training", 350_000_000_000, 32768, 131072,
    "32K->128K",
    _mix(55, 10, 15, 10, 10),
    "Long documents, long videos, retrieval, aggregation, session tracking",
)
STAGE_D = StageMixture(
    "D", "memory and ultra-long adaptation", 150_000_000_000, 131072, 262144,
    "128K->256K",
    _mix(45, 5, 20, 10, 20),
    "Multi-segment memory, branching sessions, long-horizon tool traces",
)

STAGES: tuple[StageMixture, ...] = (STAGE_A, STAGE_B, STAGE_C, STAGE_D)
#: The §8 table exactly as the blueprint states it (Stage A 60 / 10 / 10 / 10 / 10, no warm-up).
BLUEPRINT_STAGES: tuple[StageMixture, ...] = (BLUEPRINT_STAGE_A, STAGE_B, STAGE_C, STAGE_D)
#: §8: "The starting full-run budget is 4.75T text-equivalent token positions".
TOTAL_POSITIONS: int = 4_750_000_000_000


def get_stage(stage: str | StageMixture, stages: Sequence[StageMixture] = STAGES) -> StageMixture:
    if isinstance(stage, StageMixture):
        return stage
    for s in stages:
        if s.name == stage.upper():
            return s
    raise MixtureError(f"unknown stage {stage!r}")


def stage_at(position: int, stages: Sequence[StageMixture] = STAGES) -> StageMixture:
    """The stage active after ``position`` cumulative training positions."""
    if position < 0:
        raise MixtureError("position must be >= 0")
    cum = 0
    for s in stages:
        cum += s.positions
        if position < cum:
            return s
    raise MixtureError(f"position {position} is beyond the {cum}-position budget")


def mixture_at(position: int, stages: Sequence[StageMixture] = STAGES) -> tuple[StageMixture, dict[str, float]]:
    """The stage and the sampling percentages in force after ``position`` cumulative positions."""
    st = stage_at(position, stages)
    start = 0
    for s in stages:
        if s is st:
            break
        start += s.positions
    in_warmup = position - start < st.warmup_positions
    return st, dict(st.warmup_percent if in_warmup else st.sampling_percent)


def compute_normalized_weights(
    stage: str | StageMixture, measured_cost_per_position: Mapping[str, float]
) -> dict[str, float]:
    """Per-position sampling weights that realise the stage's compute shares.

    ``w_m = (p_m / c_m) / sum_j (p_j / c_j)`` where ``p_m`` is the table
    percentage and ``c_m`` the measured cost per position (any consistent unit,
    e.g. FLOPs including encoder cost). With uniform costs the weights equal
    the percentages / 100. Weights sum to 1.
    """
    st = get_stage(stage)
    missing = [k for k in st.sampling_percent if k not in measured_cost_per_position]
    if missing:
        raise MixtureError(f"missing measured cost for {missing}")
    raw: dict[str, float] = {}
    for k, pct in st.sampling_percent.items():
        cost = float(measured_cost_per_position[k])
        if not cost > 0:
            raise MixtureError(f"cost per position for {k} must be positive, got {cost}")
        raw[k] = pct / 100.0 / cost
    z = sum(raw.values())
    if z <= 0:
        raise MixtureError("all sampling percentages are zero")
    return {k: v / z for k, v in raw.items()}


def compute_shares(position_weights: Mapping[str, float], cost_per_position: Mapping[str, float]) -> dict[str, float]:
    """Fraction of compute each category receives under ``position_weights``."""
    raw = {k: float(w) * float(cost_per_position[k]) for k, w in position_weights.items()}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


def stage_position_budget(stage: str | StageMixture, weights: Mapping[str, float] | None = None) -> dict[str, int]:
    """Positions per category for a stage (defaults to its effective percentages, warm-up included)."""
    st = get_stage(stage)
    w = dict(weights) if weights is not None else {k: v / 100.0 for k, v in st.effective_percent.items()}
    budget = {k: int(round(st.positions * w[k])) for k in w}
    # put rounding residue on text so the stage total is exact
    budget["text_code"] = budget.get("text_code", 0) + st.positions - sum(budget.values())
    return budget


def text_control_budget(stages: Sequence[StageMixture] = STAGES) -> dict[str, int]:
    """Text/code positions per stage: the text-only control for the parity gate trains on exactly these.

    The control is the candidate's configuration with the multimodal front end
    disabled, fed the candidate's own text/code stream (same documents, order,
    and stage boundaries) with the media batches removed, so the comparison
    isolates what multimodal training does to instruction following.
    """
    return {s.name: stage_position_budget(s)["text_code"] for s in stages}


def validate_curriculum(stages: Sequence[StageMixture] = STAGES, total: int = TOTAL_POSITIONS) -> None:
    """Check the curriculum invariants stated in §8.

    * every stage's sampling percentages sum to 100 over the five categories;
    * stage positions sum to ``total`` (4.75T);
    * sequence lengths never decrease across stages;
    * synchronized AV data is present in every stage ("throughout, not as a
      late adapter stage") and text stays substantial (>= 40%) to limit
      modality-induced forgetting;
    * a warm-up is shorter than its stage, its percentages sum to 100, and its
      text share is at least the stage's.
    """
    errs: list[str] = []
    prev_end = 0
    for s in stages:
        keys = set(s.sampling_percent)
        if keys != set(MIXTURE_KEYS):
            errs.append(f"stage {s.name}: categories {sorted(keys)} != {list(MIXTURE_KEYS)}")
        total_pct = sum(s.sampling_percent.values())
        if abs(total_pct - 100.0) > 1e-9:
            errs.append(f"stage {s.name}: sampling percentages sum to {total_pct}, not 100")
        if any(v < 0 for v in s.sampling_percent.values()):
            errs.append(f"stage {s.name}: negative percentage")
        if s.sampling_percent.get("av_sync", 0) <= 0:
            errs.append(f"stage {s.name}: synchronized AV data must be present in every stage")
        if s.sampling_percent.get("text_code", 0) < 40:
            errs.append(f"stage {s.name}: text/code share below 40%")
        if s.seq_len_start > s.seq_len_end or s.seq_len_start < prev_end:
            errs.append(f"stage {s.name}: sequence curriculum must be non-decreasing")
        prev_end = s.seq_len_end
        if s.positions <= 0:
            errs.append(f"stage {s.name}: positions must be positive")
        if s.warmup_positions:
            wp = s.warmup_percent
            if not 0 < s.warmup_positions < s.positions:
                errs.append(f"stage {s.name}: warm-up must be shorter than the stage")
            if set(wp) != set(MIXTURE_KEYS) or abs(sum(wp.values()) - 100.0) > 1e-9 or any(v < 0 for v in wp.values()):
                errs.append(f"stage {s.name}: warm-up percentages must cover {list(MIXTURE_KEYS)} and sum to 100")
            elif wp["text_code"] < s.sampling_percent.get("text_code", 0):
                errs.append(f"stage {s.name}: a warm-up must not be less text-heavy than its stage")
    got = sum(s.positions for s in stages)
    if got != total:
        errs.append(f"stage positions sum to {got / _T:.2f}T, expected {total / _T:.2f}T")
    if errs:
        raise MixtureError("; ".join(errs))
