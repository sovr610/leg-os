"""Capacity planning arithmetic (blueprint §4, §14).

Every number here is *derived from configuration*, never a benchmark. Two
levels are provided:

* the blueprint's own arithmetic (:func:`training_flops`, the default
  :func:`session_footprint`), kept so the §4/§14 tables stay reproducible; and
* the corrected estimates from ``docs/IMPROVEMENTS.md`` §9:
  :func:`training_compute` adds the MTP, attention, recurrent-mixer, and
  encoder terms that ``6·N·D`` omits and pairs them with named hardware
  throughput profiles (which already include utilisation losses, so no extra
  allowance is stacked on top); :func:`session_footprint` takes separate KV,
  recurrent-state, and memory dtypes and can add MTP KV, recurrent-state
  checkpoints, and page padding.

Production capacity planning must still add multimodal caches, fragmentation,
and activations, and must be measured (§14).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..config import GATED_GQA, RECURRENT_KINDS, M63Config

GiB = 1024**3
MiB = 1024**2
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "fp8": 1, "int4": 0.5}


def kv_bytes_per_token(cfg: M63Config, dtype_bytes: float = 2) -> float:
    """GQA KV bytes per token: layers x (K and V) x kv_heads x head_dim x bytes (48 KiB in the reference)."""
    n_attn = sum(k == GATED_GQA for k in cfg.layer_kinds())
    a = cfg.attention
    return n_attn * 2 * a.kv_heads * a.head_dim * dtype_bytes


def kv_bytes(cfg: M63Config, tokens: int, dtype_bytes: float = 2) -> float:
    return kv_bytes_per_token(cfg, dtype_bytes) * tokens


def full_attention_kv_bytes_per_token(cfg: M63Config, dtype_bytes: float = 2) -> float:
    """Matched control: every layer is GQA with the same shape (4x the hybrid's KV)."""
    a = cfg.attention
    return cfg.model.layers * 2 * a.kv_heads * a.head_dim * dtype_bytes


def mtp_kv_bytes_per_token(cfg: M63Config, dtype_bytes: float = 2) -> float:
    """KV of the MTP attention layer(s), kept per session when drafting with an incremental MTP cache."""
    if not cfg.mtp.enabled:
        return 0
    a = cfg.attention
    return cfg.mtp.layers * 2 * a.kv_heads * a.head_dim * dtype_bytes


def gdn_state_bytes(cfg: M63Config, dtype_bytes: float = 2, include_conv: bool = False) -> float:
    """Recurrent (GDN and KDA) state per session (18 MiB at BF16 in the reference)."""
    n_gdn = sum(k in RECURRENT_KINDS for k in cfg.layer_kinds())
    g = cfg.gdn
    b = n_gdn * g.heads * g.key_dim * g.value_dim * dtype_bytes
    if include_conv:
        b += n_gdn * (2 * g.heads * g.key_dim + g.inner_width) * (g.conv_kernel - 1) * dtype_bytes
    return b


def memory_state_bytes(cfg: M63Config, dtype_bytes: float = 2) -> float:
    """Neural-memory weights + momentum per session (~3 MiB at BF16 in the reference)."""
    if not cfg.memory.enabled:
        return 0
    d, h, _ = cfg.memory.mlp_shape
    return len(cfg.memory.anchors) * 2 * (2 * d * h) * dtype_bytes


def weight_bytes(total_params: float, dtype: str) -> float:
    return total_params * DTYPE_BYTES[dtype]


@dataclass
class SessionFootprint:
    context_tokens: int
    kv: float
    gdn: float  # recurrent (GDN/KDA) state, plus conv state when requested
    memory: float
    mtp_kv: float = 0.0
    checkpoints: float = 0.0  # recurrent-state snapshots for prefix reuse and branches

    @property
    def total(self) -> float:
        return self.kv + self.gdn + self.memory + self.mtp_kv + self.checkpoints


def session_footprint(
    cfg: M63Config,
    context_tokens: int,
    dtype_bytes: float = 2,
    *,
    kv_dtype_bytes: Optional[float] = None,
    state_dtype_bytes: Optional[float] = None,
    memory_dtype_bytes: Optional[float] = None,
    include_conv: bool = False,
    include_mtp: bool = False,
    state_checkpoints: int = 0,
    page_tokens: int = 0,
) -> SessionFootprint:
    """Per-session bytes. With only ``dtype_bytes`` this is the blueprint's lower bound (KV + state + memory).

    The keyword arguments give the serving view: KV may be FP8 while recurrent
    and memory state stay at >= 2 bytes (no engine stores them in FP8),
    ``include_mtp`` adds the MTP layers' KV, ``state_checkpoints`` adds that many
    recurrent-state snapshots (prefix-cache alignment points, branch heads), and
    ``page_tokens`` rounds the KV up to whole pages.
    """
    kv_b = dtype_bytes if kv_dtype_bytes is None else kv_dtype_bytes
    st_b = dtype_bytes if state_dtype_bytes is None else state_dtype_bytes
    mem_b = dtype_bytes if memory_dtype_bytes is None else memory_dtype_bytes
    explicit_state = state_dtype_bytes is not None or memory_dtype_bytes is not None
    if explicit_state and min(st_b, mem_b) < 2:
        raise ValueError("recurrent and memory state need >= 2 bytes per element (no FP8 state in any engine)")
    tokens = -(-context_tokens // page_tokens) * page_tokens if page_tokens else context_tokens
    state = gdn_state_bytes(cfg, st_b, include_conv=include_conv)
    return SessionFootprint(
        context_tokens,
        kv=kv_bytes(cfg, tokens, kv_b),
        gdn=state,
        memory=memory_state_bytes(cfg, mem_b),
        mtp_kv=mtp_kv_bytes_per_token(cfg, kv_b) * tokens if include_mtp else 0.0,
        checkpoints=state_checkpoints * gdn_state_bytes(cfg, st_b, include_conv=True),
    )


def serving_footprint(cfg: M63Config, context_tokens: int, *, kv_dtype: str = "fp8", state_dtype: str = "bfloat16",
                      checkpoint_every: int = 8192) -> SessionFootprint:
    """The serving view used in ``docs/IMPROVEMENTS.md`` §8: FP8 KV, BF16 state, MTP KV, periodic checkpoints."""
    return session_footprint(
        cfg, context_tokens, kv_dtype_bytes=DTYPE_BYTES[kv_dtype], state_dtype_bytes=DTYPE_BYTES[state_dtype],
        memory_dtype_bytes=DTYPE_BYTES[state_dtype], include_conv=True, include_mtp=True,
        state_checkpoints=context_tokens // checkpoint_every if checkpoint_every else 0,
        page_tokens=cfg.runtime.kv_block_tokens,
    )


def sessions_per_deployment(
    cfg: M63Config, total_params: float, gpus: int, gpu_bytes: float, context_tokens: int,
    weight_dtype: str = "bfloat16", reserve_fraction: float = 0.2, footprint: Optional[SessionFootprint] = None,
) -> int:
    """Upper bound on concurrent full-context sessions after weights and a scratch reserve.

    Uses the blueprint lower-bound footprint unless ``footprint`` is given (e.g.
    :func:`serving_footprint`). Ignores encoder buffers, activations, and
    fragmentation; real capacity must be measured (§14).
    """
    usable = gpus * gpu_bytes * (1 - reserve_fraction) - weight_bytes(total_params, weight_dtype)
    if usable <= 0:
        return 0
    per = footprint if footprint is not None else session_footprint(cfg, context_tokens)
    return int(usable // per.total)


def training_flops(active_params: float, positions: float, overhead: float = 0.30) -> tuple[float, float]:
    """The blueprint's §14 estimate: dense 6·N·D and the version with a fractional overhead allowance.

    Kept to reproduce the blueprint table. It omits MTP and attention compute
    and, combined with a *sustained* throughput, double-counts inefficiency;
    prefer :func:`training_compute` with a :data:`HARDWARE` profile.
    """
    base = 6.0 * active_params * positions
    return base, base * (1.0 + overhead)


def training_days(flops: float, gpus: int, sustained_tflops_per_gpu: float) -> float:
    """Idealised wall-clock days at a *sustained effective* per-GPU throughput."""
    return flops / (gpus * sustained_tflops_per_gpu * 1e12) / 86400.0


#: Sustained *effective* dense TFLOP/s per GPU for a ~5B-active MoE at this shape,
#: as (low, typical, high). Planning assumptions from docs/IMPROVEMENTS.md §2
#: (public Megatron-style MoE throughput tables), not measurements of this code.
#: They already include utilisation losses: do not add an overhead allowance on top.
HARDWARE: dict[str, tuple[float, float, float]] = {
    "h100_bf16": (150.0, 180.0, 210.0),
    "h100_fp8": (220.0, 250.0, 280.0),
    "b200_bf16": (300.0, 350.0, 400.0),
    "b200_mxfp8": (420.0, 480.0, 540.0),
}


@dataclass
class TrainingCompute:
    """Training FLOPs by term. Every term is 6 x (forward multiply-adds) per position unless noted."""

    positions: float
    backbone: float  # 6 · active parameters (incl. the tied head) · positions
    mtp: float  # MTP depth(s): module + tied head per depth
    attention: float  # causal score/value products of every GQA layer (backbone + MTP) at the stage's packed length
    recurrent: float  # chunked GDN/KDA intra-chunk and state terms (not in the parameter count)
    encoders: float  # vision tower over patches (assumption-driven; 0 when frozen and cached)

    @property
    def total(self) -> float:
        return self.backbone + self.mtp + self.attention + self.recurrent + self.encoders

    @property
    def overhead_vs_6nd(self) -> float:
        """Fraction the omitted terms add on top of 6·N·D."""
        return self.total / self.backbone - 1.0

    def days(self, gpus: int, hardware: str = "h100_bf16", which: int = 1) -> float:
        """Wall-clock days on ``gpus`` GPUs at the low (0), typical (1), or high (2) profile throughput."""
        return training_days(self.total, gpus, HARDWARE[hardware][which])

    def gpu_hours(self, hardware: str = "h100_bf16", which: int = 1) -> float:
        return self.total / (HARDWARE[hardware][which] * 1e12) / 3600.0

    def to_dict(self) -> dict[str, float]:
        return {"positions": self.positions, "backbone_6ND": self.backbone, "mtp": self.mtp,
                "attention": self.attention, "recurrent": self.recurrent, "encoders": self.encoders,
                "total": self.total, "overhead_vs_6ND": self.overhead_vs_6nd}


def _stage_rows(stages: Optional[Sequence]) -> list[tuple[float, float, float]]:
    """(positions, mean packed length, visual-position fraction) per curriculum stage."""
    if stages is None:
        from ..data.mixture import STAGES

        stages = STAGES
    rows = []
    for st in stages:
        sp = st.effective_percent  # includes any text-only warm-up
        visual = (sp.get("image", 0.0) + sp.get("video", 0.0) + 0.5 * sp.get("av", 0.0)) / 100.0
        rows.append((float(st.positions), (st.seq_len_start + st.seq_len_end) / 2.0, visual))
    return rows


def training_compute(
    cfg: M63Config,
    stages: Optional[Sequence] = None,
    *,
    manifest=None,
    patches_per_visual_position: float = 4.0,
    encoders_trainable: bool = True,
) -> TrainingCompute:
    """Corrected training-compute estimate for the curriculum ``stages`` (default: the §8 Stage A–D table).

    * backbone: ``6 · N_active · D`` with the tied head included (the head runs on every position);
    * MTP: each depth runs its module and the tied head on every position (weights may be shared, compute is not);
    * attention: ``6 · n_ctx · d_q`` per GQA layer per position (Kaplan et al.'s causal term, x3 for training),
      with ``n_ctx`` the stage's mean packed length — an upper bound when document masking skips blocks;
    * recurrent: per GDN/KDA layer ``6 · H · (2·C·d_k + 3·d_k·d_v)`` (intra-chunk scores plus state reads/writes);
    * encoders: the vision tower over ``patches_per_visual_position`` patches per visual position, where visual
      positions are image + video + half of synchronized-AV positions; forward-only (2x) when frozen.
    """
    from ..model.manifest import build_manifest

    m = manifest if manifest is not None else build_manifest(cfg)
    rows = _stage_rows(stages)
    D = sum(r[0] for r in rows)
    backbone = 6.0 * m.active_incl_head * D
    depth = cfg.mtp.layers if cfg.mtp.enabled else 0
    mtp_modules = 1 if cfg.mtp.share_weights else max(depth, 1)
    mtp_per_depth = m.resident["mtp"] / mtp_modules + m.head
    mtp = 6.0 * depth * mtp_per_depth * D
    a, g = cfg.attention, cfg.gdn
    n_attn = sum(k == GATED_GQA for k in cfg.layer_kinds()) + depth
    n_rec = sum(k in RECURRENT_KINDS for k in cfg.layer_kinds())
    qd = a.query_heads * a.head_dim
    attention = sum(6.0 * n_attn * ctx * qd * pos for pos, ctx, _ in rows)
    recurrent = 6.0 * n_rec * g.heads * (2 * g.train_chunk * g.key_dim + 3 * g.key_dim * g.value_dim) * D
    encoders = 0.0
    if cfg.multimodal.enabled:
        vision = sum(_numel(shape) for name, shape in m.tensors.items() if name.startswith("multimodal.vision."))
        visual_positions = sum(pos * vis for pos, _, vis in rows)
        encoders = (6.0 if encoders_trainable else 2.0) * vision * patches_per_visual_position * visual_positions
    return TrainingCompute(D, backbone, mtp, attention, recurrent, encoders)


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n
