"""Training health monitors (blueprint §7, §12, §15).

* Router: dead-expert alarm (an expert receives no tokens over a large window),
  overload alarm (busiest/mean load repeatedly above a measured multiple), both
  segmented by modality so a video-heavy microbatch cannot hide imbalance.
* Numerics: NaN/Inf guards on loss and gradients, gradient-norm tracking.
* Loss spikes: z-score against an exponential moving average; a spike triggers
  replay from the last clean checkpoint with the exact batch manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..common import Modality
from ..model.moe import RouterStats


@dataclass
class Alarm:
    kind: str
    layer: int
    detail: str
    modality: Optional[str] = None


class RouterMonitor:
    def __init__(self, num_layers: int, num_experts: int, *, window_tokens: int = 100_000,
                 overload_ratio: float = 4.0, overload_patience: int = 3, min_modality_tokens: int = 1_000) -> None:
        self.L, self.E = num_layers, num_experts
        self.window_tokens = window_tokens
        self.overload_ratio = overload_ratio
        self.patience = overload_patience
        self.min_modality_tokens = min_modality_tokens
        self._reset_window()
        self._overload_streak = [0] * num_layers
        self.alarms: list[Alarm] = []

    def _reset_window(self) -> None:
        self.counts = torch.zeros(self.L, len(Modality), self.E, dtype=torch.long)
        self.tokens = 0

    def update(self, stats: list[RouterStats]) -> list[Alarm]:
        for l, s in enumerate(stats):
            self.counts[l] += s.modality_counts.cpu()
        self.tokens += stats[0].tokens if stats else 0
        if self.tokens < self.window_tokens:
            return []
        alarms = self._evaluate()
        self.alarms.extend(alarms)
        self._reset_window()
        return alarms

    def _evaluate(self) -> list[Alarm]:
        alarms: list[Alarm] = []
        for l in range(self.L):
            total = self.counts[l].sum(0)
            dead = (total == 0).nonzero().flatten().tolist()
            if dead:
                alarms.append(Alarm("dead_expert", l, f"experts {dead} received no tokens in {self.tokens} tokens"))
            mean = total.float().mean()
            ratio = float(total.max() / mean) if mean > 0 else 0.0
            if ratio > self.overload_ratio:
                self._overload_streak[l] += 1
                if self._overload_streak[l] >= self.patience:
                    alarms.append(Alarm("overload", l, f"max/mean load {ratio:.2f} for {self._overload_streak[l]} windows"))
            else:
                self._overload_streak[l] = 0
            for m in Modality:
                mc = self.counts[l, int(m)]
                if mc.sum() < self.min_modality_tokens:
                    continue
                mm = mc.float().mean()
                r = float(mc.max() / mm) if mm > 0 else 0.0
                if r > self.overload_ratio:
                    alarms.append(Alarm("modality_skew", l, f"max/mean {r:.2f}", m.name.lower()))
        return alarms

    @staticmethod
    def load_summary(stats: list[RouterStats]) -> dict[str, float]:
        ratios = []
        for s in stats:
            m = s.counts.float().mean()
            ratios.append(float(s.counts.max() / m) if m > 0 else 0.0)
        return {"max_over_mean": max(ratios) if ratios else 0.0,
                "entropy": float(sum(float(s.entropy.detach()) for s in stats) / max(len(stats), 1))}


class NonFiniteError(FloatingPointError):
    pass


def check_finite(loss: torch.Tensor, model: torch.nn.Module) -> float:
    """Raise on NaN/Inf loss or gradients; return the global gradient norm."""
    if not torch.isfinite(loss):
        raise NonFiniteError(f"non-finite loss {float(loss.detach())}")
    sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            raise NonFiniteError(f"non-finite gradient in {name}")
        sq += float(g.float().pow(2).sum())
    return math.sqrt(sq)


@dataclass
class SpikeDetector:
    beta: float = 0.98
    z_threshold: float = 6.0
    warmup_steps: int = 20
    mean: float = 0.0
    var: float = 0.0
    n: int = 0
    spikes: list[int] = field(default_factory=list)

    def update(self, step: int, loss: float) -> bool:
        self.n += 1
        if self.n == 1:
            self.mean = loss
            return False
        std = math.sqrt(max(self.var, 1e-12))
        spike = self.n > self.warmup_steps and (loss - self.mean) / std > self.z_threshold
        if spike:
            self.spikes.append(step)
        else:
            d = loss - self.mean
            self.mean += (1 - self.beta) * d
            self.var = self.beta * (self.var + (1 - self.beta) * d * d)
        return spike
