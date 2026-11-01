"""Learning-rate and batch-size schedules (blueprint §9).

Warmup–stable–decay (WSD) allows intermediate evaluation and continuation:
linear warmup over ``warmup_positions``, a constant plateau, then a linear decay
to ``min_lr_ratio * peak`` over the final ``decay_fraction`` of the stage.
Global batch size ramps linearly from 2M to 8M positions.
"""

from __future__ import annotations

from ..config import TrainingConfig


def wsd_lr(positions: float, total_positions: float, cfg: TrainingConfig, peak_lr: float | None = None) -> float:
    peak = cfg.peak_lr if peak_lr is None else peak_lr
    floor = peak * cfg.min_lr_ratio
    if positions < cfg.warmup_positions:
        return peak * (positions + 1) / cfg.warmup_positions
    decay_start = total_positions * (1.0 - cfg.decay_fraction)
    if positions < decay_start:
        return peak
    frac = min(1.0, (positions - decay_start) / max(total_positions - decay_start, 1.0))
    return peak + (floor - peak) * frac


def batch_tokens(positions: float, cfg: TrainingConfig) -> int:
    """Global batch size in positions, rounded down to a multiple of ``seq_len``."""
    start, end = cfg.batch_ramp_start_tokens, cfg.global_batch_tokens
    frac = min(1.0, positions / max(cfg.batch_ramp_positions, 1))
    tokens = start + (end - start) * frac
    return max(cfg.seq_len, int(tokens) // cfg.seq_len * cfg.seq_len)


# Long-context stages use a lower peak (§9): 2e-5 .. 5e-5.
LONG_CONTEXT_PEAK_RANGE = (2e-5, 5e-5)
