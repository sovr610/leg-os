"""Configuration schema for M63-Omni.

The dataclass defaults ARE the blueprint reference configuration (§11). Every
YAML file under ``configs/`` is applied as a strict override on top of these
defaults: unknown keys are rejected, so a typo cannot silently fall back to a
default. A run's configuration is immutable and identified by
:meth:`M63Config.fingerprint`.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .common import content_hash

GDN = "gdn"
KDA = "kda"  # Kimi Delta Attention: channel-wise-gated GDN (tournament alternative)
GATED_GQA = "gated_gqa"
MIXER_KINDS = (GDN, KDA, GATED_GQA)
RECURRENT_KINDS = (GDN, KDA)  # constant-size state; share the ``gdn`` shape section


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BackboneConfig:
    name: str = "m63_omni"
    hidden_size: int = 4096
    layers: int = 48
    hybrid_pattern: tuple[str, ...] = (GDN, GDN, GDN, GATED_GQA)  # repeated layers/len times
    vocab_size: int = 131072
    special_tokens: int = 256  # reserved special IDs at the top of the vocabulary
    tie_embeddings: bool = True
    norm: str = "rmsnorm_pre"
    norm_eps: float = 1e-6
    context_targets: tuple[int, ...] = (131072, 262144)
    init_std: float = 0.02


@dataclass(frozen=True)
class GDNConfig:
    heads: int = 16
    key_dim: int = 128
    value_dim: int = 128
    inner_width: int = 2048
    conv_kernel: int = 4
    train_chunk: int = 64
    decode_mode: str = "recurrent"
    decay_gate_dtype: str = "float32"
    output_gate: bool = True
    kda_low_rank: int = 0  # KDA forget/output-gate projection rank; 0 = key_dim (Kimi Linear)


@dataclass(frozen=True)
class AttentionConfig:
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    rotary_dims: int = 32  # partial RoPE
    rope_theta: float = 1_000_000.0
    qk_norm: bool = True
    output_gate: bool = True


@dataclass(frozen=True)
class MoEConfig:
    routed_experts: int = 64
    shared_experts: int = 1
    top_k: int = 2
    expert_hidden: int = 1536
    activation: str = "swiglu"
    router: str = "normalized_sigmoid_with_load_bias"
    max_nodes_per_token: int = 2
    experts_per_node: int = 8  # 64 experts over 8 expert-parallel nodes
    token_dropping: bool = False
    bias_update_rate: float = 1e-3  # auxiliary-loss-free balancing step (DeepSeek-V3 style)
    aux_loss_coef: float = 1e-4  # small global-batch balancing loss for warm-up / fallback
    z_loss_coef: float = 0.0  # enabled if router logits drift


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = True
    anchors: tuple[int, ...] = (16, 32, 48)  # 1-indexed layer numbers; must be attention layers
    read_tokens_per_anchor: int = 8
    update_segment_tokens: int = 2048
    mlp_shape: tuple[int, ...] = (256, 512, 256)
    freeze_during_decode: bool = True
    untrusted_persistent_writes: bool = False
    max_eta: float = 0.9  # momentum gate bound
    max_theta: float = 0.1  # inner learning-rate bound
    max_alpha: float = 0.1  # forgetting gate bound
    grad_clip: float = 1.0  # per-tensor inner-gradient norm clip
    state_dtype: str = "bfloat16"  # storage dtype for session snapshots
    decay_target: str = "zero"  # forgetting pulls M toward "zero" (blueprint §6) or the learned "init" M0 (ablation)


@dataclass(frozen=True)
class MTPConfig:
    enabled: bool = True
    layers: int = 1  # trained sequential MTP depth D (DeepSeek-V3 uses D = 1)
    mixer: str = "standard_gqa"
    dense_ffn_hidden: int = 11008
    draft_tokens: int = 1  # <= layers: each draft uses the depth trained for it (blueprint §11 says 2; see IMPLEMENTATION.md)
    share_weights: bool = False  # one module reused at every depth (parameters shared, compute per depth)


@dataclass(frozen=True)
class MultimodalConfig:
    enabled: bool = True
    # vision (shared by image and video)
    vision_layers: int = 24
    vision_width: int = 1024
    vision_heads: int = 16
    vision_mlp_ratio: int = 4
    patch_size: int = 14
    max_patch_grid: int = 64  # factorised 2-D learned positions up to 64x64 patches (896 px)
    resampler_layers: int = 1
    resampler_max_queries: int = 256
    # video
    temporal_layers: int = 4
    max_frames: int = 1024
    # audio
    audio_layers: int = 20
    audio_width: int = 1024
    audio_heads: int = 16
    audio_mlp_ratio: int = 4
    audio_sample_rate: int = 16000
    mel_bins: int = 128
    window_ms: float = 25.0
    hop_ms: float = 10.0
    audio_downsample: int = 16
    audio_tokens_per_second: float = 6.25
    audio_chunk_tokens: int = 25  # chunk-aware attention window (4 s at 6.25 tok/s)
    ctc_vocab: int = 257  # byte-level CTC targets + blank
    explicit_timestamps: bool = True
    # AV interleaving (§5). The blueprint: 4-s windows, <|av_chunk|> + <|ts|> carrying a learned time embedding.
    av_chunk_ms: int = 0  # AV window length; 0 = the audio encoder chunk (4 s). 2000 needs audio_attention: causal
    timestamp_mode: str = "embedding"  # "embedding" (blueprint) | "text": <|av_chunk|> + the window start in seconds as text
    audio_attention: str = "chunk"  # "chunk": block-causal over current + previous chunk | "causal": token-causal, 2-chunk lookback


@dataclass(frozen=True)
class TrainingConfig:
    precision: str = "bf16"
    optimizer: str = "adamw"
    peak_lr: float = 2e-4
    min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    clip_grad_norm: float = 1.0
    global_batch_tokens: int = 8_388_608
    batch_ramp_start_tokens: int = 2_097_152
    batch_ramp_positions: int = 100_000_000_000
    warmup_positions: int = 1_000_000_000
    decay_fraction: float = 0.1  # final fraction of a stage spent in the decay phase
    schedule: str = "warmup_stable_decay"
    token_dropping: bool = False
    seq_len: int = 4096
    seed: int = 1234
    # loss weights (L = L_AR + λ_ctc L_CTC + λ_align L_align + λ_time L_time + λ_bal L_bal + λ_MTP L_MTP + λ_mem L_mem)
    lambda_ctc: float = 0.1
    lambda_align: float = 0.05
    lambda_time: float = 0.05
    lambda_balance: float = 1.0  # multiplies the (already small) router aux loss
    lambda_mtp: float = 0.3
    lambda_memory: float = 0.01


@dataclass(frozen=True)
class RuntimeConfig:
    paged_kv: bool = True
    kv_block_tokens: int = 256
    kv_dtype: str = "bfloat16"
    recurrent_state_manager: bool = True
    multimodal_prefix_hash: bool = True
    copy_on_write_branches: bool = True
    memory_off_fallback: bool = True
    session_ttl_seconds: int = 7200


@dataclass(frozen=True)
class M63Config:
    model: BackboneConfig = field(default_factory=BackboneConfig)
    gdn: GDNConfig = field(default_factory=GDNConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        validate(self)

    # ------------------------------------------------------------------ helpers
    def layer_kinds(self) -> list[str]:
        p = self.model.hybrid_pattern
        return [p[i % len(p)] for i in range(self.model.layers)]

    def memory_anchor_indices(self) -> list[int]:
        """0-indexed layer indices that host a neural-memory anchor."""
        if not self.memory.enabled:
            return []
        return [a - 1 for a in self.memory.anchors]

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    def fingerprint(self) -> str:
        return content_hash(self.to_dict())

    def replace(self, **sections: Mapping[str, Any]) -> "M63Config":
        """Return a new config with per-section overrides, e.g. ``cfg.replace(moe={"top_k": 1})``."""
        return from_dict(_deep_merge(self.to_dict(), sections))


# ---------------------------------------------------------------------- loading
def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(cls: type, data: Mapping[str, Any], path: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}")
    kwargs = {}
    for name, value in data.items():
        f = known[name]
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()  # type: ignore[misc]
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{path}.{name}")
        elif isinstance(default, tuple):
            if not isinstance(value, (list, tuple)):
                raise ConfigError(f"{path}.{name}: expected a list")
            kwargs[name] = tuple(value)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{path}.{name}: expected bool")
            kwargs[name] = value
        elif isinstance(default, int) and not isinstance(value, bool):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if not isinstance(value, int):
                raise ConfigError(f"{path}.{name}: expected int")
            kwargs[name] = value
        elif isinstance(default, float):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(f"{path}.{name}: expected number")
            kwargs[name] = float(value)
        else:
            kwargs[name] = value
    return cls(**kwargs) if cls is not M63Config else kwargs


def from_dict(data: Mapping[str, Any]) -> M63Config:
    kwargs = _build(M63Config, data, "config")
    return M63Config(**kwargs)


def load_config(path: str | Path) -> M63Config:
    """Load a YAML/JSON config as a strict override of the reference defaults.

    A top-level ``extends: other.yaml`` key (relative to the file) is applied first.
    """
    path = Path(path)
    data = _read(path)
    base: dict[str, Any] = {}
    parent = data.pop("extends", None)
    if parent is not None:
        base = load_config(path.parent / parent).to_dict()
    return from_dict(_deep_merge(base, data))


def _read(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    import yaml

    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


# ------------------------------------------------------------------- validation
def validate(cfg: M63Config) -> None:
    m, g, a, moe, mem = cfg.model, cfg.gdn, cfg.attention, cfg.moe, cfg.memory
    errs: list[str] = []
    if any(k not in MIXER_KINDS for k in m.hybrid_pattern):
        errs.append(f"hybrid_pattern entries must be in {MIXER_KINDS}")
    if m.layers % len(m.hybrid_pattern) != 0:
        errs.append("layers must be a multiple of the hybrid pattern length")
    if m.special_tokens >= m.vocab_size:
        errs.append("special_tokens must be smaller than vocab_size")
    if g.inner_width != g.heads * g.value_dim:
        errs.append("gdn.inner_width must equal gdn.heads * gdn.value_dim")
    if g.key_dim != g.value_dim:
        errs.append("gdn.key_dim must equal gdn.value_dim (shared conv/projection width)")
    if g.kda_low_rank < 0:
        errs.append("gdn.kda_low_rank must be >= 0 (0 = key_dim)")
    if g.decay_gate_dtype != "float32":
        errs.append("gdn.decay_gate_dtype must be float32 (§4: cumulative decay in FP32)")
    if a.query_heads % a.kv_heads != 0:
        errs.append("attention.query_heads must be divisible by kv_heads")
    if a.rotary_dims % 2 or a.rotary_dims > a.head_dim:
        errs.append("attention.rotary_dims must be even and <= head_dim")
    if moe.top_k > moe.routed_experts or moe.top_k < 1:
        errs.append("moe.top_k must be in [1, routed_experts]")
    if moe.routed_experts % moe.experts_per_node != 0:
        errs.append("moe.routed_experts must be divisible by experts_per_node")
    if moe.max_nodes_per_token * moe.experts_per_node < moe.top_k:
        errs.append("node-limited routing cannot satisfy top_k")
    if moe.token_dropping or cfg.training.token_dropping:
        errs.append("token dropping is forbidden in training and inference (§4, §7)")
    if mem.enabled:
        kinds = [m.hybrid_pattern[i % len(m.hybrid_pattern)] for i in range(m.layers)]
        for anchor in mem.anchors:
            if not 1 <= anchor <= m.layers:
                errs.append(f"memory anchor {anchor} outside 1..{m.layers}")
            elif kinds[anchor - 1] != GATED_GQA:
                errs.append(f"memory anchor {anchor} is not an attention layer")
        if len(mem.mlp_shape) != 3 or mem.mlp_shape[0] != mem.mlp_shape[2]:
            errs.append("memory.mlp_shape must be [d_mem, d_hidden, d_mem]")
        if mem.decay_target not in ("zero", "init"):
            errs.append("memory.decay_target must be 'zero' or 'init'")
        if mem.untrusted_persistent_writes:
            errs.append("memory.untrusted_persistent_writes must be false (§6 memory governance)")
    if cfg.mtp.enabled and cfg.mtp.mixer != "standard_gqa":
        errs.append("mtp.mixer must be standard_gqa")
    if cfg.mtp.enabled and cfg.mtp.layers < 1:
        errs.append("mtp.layers (trained depth) must be >= 1")
    if cfg.mtp.enabled and not 0 <= cfg.mtp.draft_tokens <= cfg.mtp.layers:
        errs.append(f"mtp.draft_tokens must be in [0, mtp.layers={cfg.mtp.layers}]: a depth-{cfg.mtp.layers} MTP "
                    "was never trained to draft further (raise mtp.layers, optionally with share_weights)")
    mm = cfg.multimodal
    if mm.enabled:
        frames_per_sec = 1000.0 / mm.hop_ms
        derived = frames_per_sec / mm.audio_downsample
        if abs(derived - mm.audio_tokens_per_second) > 1e-6:
            errs.append(f"audio_tokens_per_second {mm.audio_tokens_per_second} != derived {derived}")
        if mm.audio_downsample & (mm.audio_downsample - 1):
            errs.append("audio_downsample must be a power of two")
        if mm.timestamp_mode not in ("embedding", "text"):
            errs.append("multimodal.timestamp_mode must be 'embedding' or 'text'")
        if mm.audio_attention not in ("chunk", "causal"):
            errs.append("multimodal.audio_attention must be 'chunk' or 'causal'")
        if mm.av_chunk_ms < 0:
            errs.append("multimodal.av_chunk_ms must be >= 0 (0 = the audio encoder chunk)")
        elif mm.av_chunk_ms and mm.audio_attention == "chunk":
            chunk_ms = mm.audio_chunk_tokens / mm.audio_tokens_per_second * 1000.0
            ratio = mm.av_chunk_ms / chunk_ms
            if abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 1:
                errs.append(f"multimodal.av_chunk_ms={mm.av_chunk_ms} is not a multiple of the {chunk_ms:g} ms audio chunk: "
                            "chunked audio attention would see later windows; set audio_attention: causal")
    if errs:
        raise ConfigError("; ".join(errs))


# ---------------------------------------------------------------------- presets
def reference() -> M63Config:
    """The full 62.48B-resident reference configuration from the blueprint."""
    return M63Config()


def tiny(**sections: Mapping[str, Any]) -> M63Config:
    """A CPU-sized configuration that preserves every structural feature.

    8 layers (two hybrid groups), memory anchors on both attention layers,
    small MoE with node-limited routing, small encoders.
    """
    base = {
        "model": {
            "hidden_size": 64,
            "layers": 8,
            "vocab_size": 512,
            "special_tokens": 64,
            "context_targets": [1024, 2048],
        },
        "gdn": {"heads": 2, "key_dim": 16, "value_dim": 16, "inner_width": 32, "train_chunk": 8},
        "attention": {"query_heads": 4, "kv_heads": 2, "head_dim": 16, "rotary_dims": 8, "rope_theta": 10000.0},
        "moe": {"routed_experts": 8, "top_k": 2, "expert_hidden": 32, "experts_per_node": 2, "max_nodes_per_token": 2},
        "memory": {"anchors": [4, 8], "read_tokens_per_anchor": 4, "update_segment_tokens": 16, "mlp_shape": [16, 32, 16]},
        "mtp": {"dense_ffn_hidden": 96},
        "multimodal": {
            "vision_layers": 2,
            "vision_width": 32,
            "vision_heads": 2,
            "patch_size": 4,
            "max_patch_grid": 8,
            "resampler_max_queries": 16,
            "temporal_layers": 1,
            "max_frames": 32,
            "audio_layers": 2,
            "audio_width": 32,
            "audio_heads": 2,
            "mel_bins": 16,
            "audio_chunk_tokens": 4,
        },
        "training": {"seq_len": 64, "global_batch_tokens": 512, "batch_ramp_start_tokens": 512,
                     "warmup_positions": 2048, "batch_ramp_positions": 1, "peak_lr": 3e-3},
        "runtime": {"kv_block_tokens": 8},
    }
    return from_dict(_deep_merge(base, sections))
