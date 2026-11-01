"""Capacity planning arithmetic (blueprint §4, §14).

Every number here is *derived from configuration*, never a benchmark. Production
capacity planning must add multimodal caches, memory pages, fragmentation,
activations, and speculative-decoding state (§4).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import GDN, GATED_GQA, M63Config

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


def gdn_state_bytes(cfg: M63Config, dtype_bytes: float = 2, include_conv: bool = False) -> float:
    """Recurrent state per session (18 MiB at BF16 in the reference)."""
    n_gdn = sum(k == GDN for k in cfg.layer_kinds())
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
    gdn: float
    memory: float

    @property
    def total(self) -> float:
        return self.kv + self.gdn + self.memory


def session_footprint(cfg: M63Config, context_tokens: int, dtype_bytes: float = 2) -> SessionFootprint:
    return SessionFootprint(context_tokens, kv_bytes(cfg, context_tokens, dtype_bytes),
                            gdn_state_bytes(cfg, dtype_bytes), memory_state_bytes(cfg, dtype_bytes))


def sessions_per_deployment(
    cfg: M63Config, total_params: float, gpus: int, gpu_bytes: float, context_tokens: int,
    weight_dtype: str = "bfloat16", reserve_fraction: float = 0.2,
) -> int:
    """Upper bound on concurrent full-context sessions after weights and a scratch reserve.

    Ignores encoder buffers, activations, fragmentation, and MTP state; real
    capacity must be measured (§14).
    """
    usable = gpus * gpu_bytes * (1 - reserve_fraction) - weight_bytes(total_params, weight_dtype)
    if usable <= 0:
        return 0
    return int(usable // session_footprint(cfg, context_tokens).total)


def training_flops(active_params: float, positions: float, overhead: float = 0.30) -> tuple[float, float]:
    """Dense 6·N·D approximation and the version with a fractional overhead allowance (§14)."""
    base = 6.0 * active_params * positions
    return base, base * (1.0 + overhead)


def training_days(flops: float, gpus: int, sustained_tflops_per_gpu: float) -> float:
    """Idealised wall-clock days at a *sustained effective* per-GPU throughput."""
    return flops / (gpus * sustained_tflops_per_gpu * 1e12) / 86400.0
