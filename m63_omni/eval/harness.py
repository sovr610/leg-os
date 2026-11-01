"""Evaluation harness primitives (blueprint §13, Appendix B "Evaluation harness").

* Bootstrap confidence intervals and paired comparisons.
* Per-cohort reporting — aggregate scores never hide a critical failure in one
  modality, language, or cohort.
* Scorer self-tests with known-good / known-bad fixtures, so a broken scorer
  cannot certify a broken model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

COHORTS = (
    "short_text", "long_text", "single_image", "document", "short_video", "long_proxy_video",
    "speech", "noisy_audio", "mixed_av", "tool_only", "adversarial",
)


def bootstrap_ci(scores: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high) percentile bootstrap interval."""
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no scores")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05,
                          seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over paired items with a bootstrap interval."""
    a_, b_ = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a_.shape != b_.shape:
        raise ValueError("paired comparison needs equal-length score lists")
    return bootstrap_ci(a_ - b_, n_boot=n_boot, alpha=alpha, seed=seed)


@dataclass
class CohortResult:
    cohort: str
    n: int
    mean: float
    low: float
    high: float
    latency_ms_p50: Optional[float] = None
    critical_failures: int = 0


@dataclass
class EvalReport:
    suite: str
    evaluator_version: str
    cohorts: dict[str, CohortResult] = field(default_factory=dict)
    config: dict = field(default_factory=dict)  # prompts, decoding params, context construction

    def worst_cohort(self) -> CohortResult:
        return min(self.cohorts.values(), key=lambda c: c.mean)

    def has_critical_failure(self) -> bool:
        return any(c.critical_failures > 0 for c in self.cohorts.values())

    def to_markdown(self) -> str:
        rows = [f"### {self.suite} (evaluator {self.evaluator_version})", "",
                "| cohort | n | mean | 95% CI | p50 latency | critical failures |", "|---|---:|---:|---|---:|---:|"]
        for c in self.cohorts.values():
            lat = "—" if c.latency_ms_p50 is None else f"{c.latency_ms_p50:.1f} ms"
            rows.append(f"| {c.cohort} | {c.n} | {c.mean:.3f} | [{c.low:.3f}, {c.high:.3f}] | {lat} | {c.critical_failures} |")
        return "\n".join(rows)


def evaluate_cohorts(
    items: Iterable[dict[str, Any]],
    predict: Callable[[dict[str, Any]], Any],
    score: Callable[[dict[str, Any], Any], float],
    *,
    suite: str,
    evaluator_version: str,
    critical: Optional[Callable[[dict[str, Any], Any], bool]] = None,
    seed: int = 0,
) -> EvalReport:
    """Run ``predict`` on each item (which must carry a ``cohort`` key) and score per cohort."""
    import time

    per: dict[str, list[float]] = {}
    lat: dict[str, list[float]] = {}
    crit: dict[str, int] = {}
    for item in items:
        t0 = time.perf_counter()
        pred = predict(item)
        lat.setdefault(item["cohort"], []).append((time.perf_counter() - t0) * 1000)
        per.setdefault(item["cohort"], []).append(float(score(item, pred)))
        if critical is not None and critical(item, pred):
            crit[item["cohort"]] = crit.get(item["cohort"], 0) + 1
    report = EvalReport(suite, evaluator_version)
    for cohort, scores in per.items():
        m, lo, hi = bootstrap_ci(scores, seed=seed)
        report.cohorts[cohort] = CohortResult(cohort, len(scores), m, lo, hi, float(np.median(lat[cohort])), crit.get(cohort, 0))
    return report


class ScorerSelfTestError(AssertionError):
    pass


def verify_scorer(score: Callable[[dict, Any], float], good: Sequence[tuple[dict, Any]], bad: Sequence[tuple[dict, Any]],
                  *, threshold: float = 0.5) -> None:
    """Known-good fixtures must score >= threshold and known-bad fixtures below it."""
    for item, pred in good:
        if score(item, pred) < threshold:
            raise ScorerSelfTestError(f"scorer rejected known-good fixture {item.get('id')}")
    for item, pred in bad:
        if score(item, pred) >= threshold:
            raise ScorerSelfTestError(f"scorer accepted known-bad fixture {item.get('id')}")


def exact_match(item: dict, pred: Any) -> float:
    return float(pred == item["answer"])
