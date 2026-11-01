"""Calibrated escalation router (blueprint §10, §13 "Escalation" row).

"Escalation is an economic and reliability decision, not a personality trait"
(§10). A small logistic model estimates ``p_fail`` from verifier outcomes,
constraint complexity, domain shift, previous repair failures, and model
disagreement, and the router escalates iff::

    p_fail * cost_failure + expected_quality_gain > cost_escalation

subject to hard rules (unsupported tool, unsafe action, user-requested
flagship review) that force escalation. The language model's own uncertainty
text is reduced to one feature (:func:`uncertainty_from_text`) and its
``escalation.required`` flag is recorded but never acts as a veto in either
direction.

The :class:`EscalationPolicy` (version string, feature names, weights, Platt
calibration, cost thresholds) is a JSON-serialisable value with a content
fingerprint, so "the threshold and feature definitions are versioned with the
serving policy" and can be rolled back independently of the checkpoint
(:class:`PolicyRegistry`). :func:`fit_policy` fits weights with Newton/IRLS
logistic regression (numpy only) plus optional Platt calibration on a held-out
split; :func:`evaluate_policy` reports hard-task recall, unnecessary-escalation
rate, expected calibration error and cost per resolved task (§13).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..common import canonical_json, content_hash

__all__ = [
    "FEATURE_NAMES",
    "HARD_RULES",
    "EscalationFeatures",
    "EscalationContext",
    "EscalationPolicy",
    "EscalationDecision",
    "EscalationRouter",
    "PolicyRegistry",
    "DEFAULT_POLICY",
    "uncertainty_from_text",
    "constraint_complexity",
    "fit_logistic",
    "fit_platt",
    "fit_policy",
    "expected_calibration_error",
    "RouterMetrics",
    "evaluate_policy",
]

FEATURE_NAMES: tuple[str, ...] = (
    "verifier_failure_rate",  # fraction of verifier checks that failed so far in the task, [0, 1]
    "constraint_complexity",  # normalised count of operations + verification constraints, [0, 1]
    "domain_shift",  # out-of-distribution score from an upstream detector, [0, 1]
    "repair_failures",  # number of failed repair attempts so far (count)
    "model_disagreement",  # disagreement between samples / MTP drafts / sidecars, [0, 1]
    "lm_uncertainty",  # hedging in the model's own text — a feature only, never a veto
)
HARD_RULES: tuple[str, ...] = ("unsupported_tool", "unsafe_action", "user_requested_review")


@dataclass(frozen=True)
class EscalationFeatures:
    verifier_failure_rate: float = 0.0
    constraint_complexity: float = 0.0
    domain_shift: float = 0.0
    repair_failures: float = 0.0
    model_disagreement: float = 0.0
    lm_uncertainty: float = 0.0

    def as_array(self, names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
        return np.array([float(getattr(self, n)) for n in names], dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return {n: float(getattr(self, n)) for n in FEATURE_NAMES}

    @classmethod
    def from_mapping(cls, m: Mapping[str, float]) -> "EscalationFeatures":
        unknown = set(m) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown escalation features {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in m.items()})


@dataclass(frozen=True)
class EscalationContext:
    """Inputs to one routing decision."""

    features: EscalationFeatures = EscalationFeatures()
    unsupported_tool: bool = False
    unsafe_action: bool = False
    user_requested_review: bool = False
    expected_quality_gain: Optional[float] = None
    lm_requested_escalation: bool = False  # recorded for audit/training only


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class EscalationPolicy:
    """Versioned, serialisable routing policy (weights, calibration, cost thresholds)."""

    version: str
    weights: tuple[float, ...]
    bias: float
    feature_names: tuple[str, ...] = FEATURE_NAMES
    platt_a: float = 1.0
    platt_b: float = 0.0
    cost_failure: float = 10.0
    cost_escalation: float = 3.0
    default_quality_gain: float = 0.0
    hard_rules: tuple[str, ...] = HARD_RULES
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", tuple(float(w) for w in self.weights))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "hard_rules", tuple(self.hard_rules))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if len(self.weights) != len(self.feature_names):
            raise ValueError("weights and feature_names must have the same length")
        unknown = set(self.feature_names) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown feature names {sorted(unknown)}")
        if set(self.hard_rules) - set(HARD_RULES):
            raise ValueError(f"unknown hard rules {sorted(set(self.hard_rules) - set(HARD_RULES))}")
        nums = (*self.weights, self.bias, self.platt_a, self.platt_b, self.cost_failure, self.cost_escalation, self.default_quality_gain)
        if not all(math.isfinite(float(x)) for x in nums):
            raise ValueError("policy parameters must be finite")
        if self.cost_failure <= 0 or self.cost_escalation <= 0:
            raise ValueError("costs must be positive")

    # ------------------------------------------------------------- scoring
    def score(self, features: EscalationFeatures | np.ndarray) -> float:
        x = features.as_array(self.feature_names) if isinstance(features, EscalationFeatures) else np.asarray(features, float)
        return float(np.dot(np.asarray(self.weights), x) + self.bias)

    def p_fail(self, features: EscalationFeatures | np.ndarray) -> float:
        return float(_sigmoid(self.platt_a * self.score(features) + self.platt_b))

    def p_fail_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(_sigmoid(self.platt_a * (X @ np.asarray(self.weights) + self.bias) + self.platt_b))

    def economic_escalate(self, p_fail: float | np.ndarray, quality_gain: Optional[float] = None) -> Any:
        gain = self.default_quality_gain if quality_gain is None else quality_gain
        return np.asarray(p_fail) * self.cost_failure + gain > self.cost_escalation

    @property
    def p_fail_threshold(self) -> float:
        """``p_fail`` above which the economic rule escalates (with the default quality gain)."""
        return (self.cost_escalation - self.default_quality_gain) / self.cost_failure

    # ------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "bias": self.bias,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "cost_failure": self.cost_failure,
            "cost_escalation": self.cost_escalation,
            "default_quality_gain": self.default_quality_gain,
            "hard_rules": list(self.hard_rules),
            "metadata": json.loads(canonical_json(dict(self.metadata))),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EscalationPolicy":
        allowed = {
            "version", "feature_names", "weights", "bias", "platt_a", "platt_b", "cost_failure",
            "cost_escalation", "default_quality_gain", "hard_rules", "metadata",
        }
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown policy fields {sorted(unknown)}")
        kw = dict(d)
        for k in ("feature_names", "weights", "hard_rules"):
            if k in kw:
                kw[k] = tuple(kw[k])
        return cls(**kw)

    @classmethod
    def from_json(cls, text: str) -> "EscalationPolicy":
        return cls.from_dict(json.loads(text))

    @property
    def fingerprint(self) -> str:
        return content_hash(self.to_dict())


#: Hand-set prior used before pilot data exists (to be replaced by :func:`fit_policy`).
DEFAULT_POLICY = EscalationPolicy(
    version="m63.escalation/prior-0",
    weights=(2.5, 1.0, 1.5, 0.8, 1.5, 0.25),
    bias=-3.0,
    metadata={"source": "hand-set prior; replace with a fitted policy after the pilot"},
)


@dataclass(frozen=True)
class EscalationDecision:
    escalate: bool
    p_fail: float
    expected_failure_cost: float
    expected_quality_gain: float
    cost_escalation: float
    economic: bool
    hard_rules: tuple[str, ...]
    policy_version: str
    policy_fingerprint: str
    features: Mapping[str, float]
    lm_requested_escalation: bool = False

    @property
    def reasons(self) -> tuple[str, ...]:
        out = tuple(f"hard_rule:{r}" for r in self.hard_rules)
        if self.economic:
            out += (
                f"economic:p_fail={self.p_fail:.3f}*cost_failure+gain={self.expected_failure_cost + self.expected_quality_gain:.3f}"
                f">cost_escalation={self.cost_escalation:.3f}",
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "escalate": self.escalate,
            "p_fail": self.p_fail,
            "expected_failure_cost": self.expected_failure_cost,
            "expected_quality_gain": self.expected_quality_gain,
            "cost_escalation": self.cost_escalation,
            "economic": self.economic,
            "hard_rules": list(self.hard_rules),
            "reasons": list(self.reasons),
            "policy_version": self.policy_version,
            "policy_fingerprint": self.policy_fingerprint,
            "features": dict(self.features),
            "lm_requested_escalation": self.lm_requested_escalation,
        }


class PolicyRegistry:
    """Versioned policy store with activation and rollback (independent of model checkpoints)."""

    def __init__(self, initial: EscalationPolicy = DEFAULT_POLICY) -> None:
        self._policies: dict[str, EscalationPolicy] = {initial.version: initial}
        self._history: list[str] = [initial.version]

    def register(self, policy: EscalationPolicy, *, activate: bool = True) -> None:
        existing = self._policies.get(policy.version)
        if existing is not None and existing.fingerprint != policy.fingerprint:
            raise ValueError(f"policy version {policy.version!r} already registered with different content")
        self._policies[policy.version] = policy
        if activate:
            self.activate(policy.version)

    def activate(self, version: str) -> EscalationPolicy:
        if version not in self._policies:
            raise KeyError(f"unknown policy version {version!r}")
        if self._history[-1] != version:
            self._history.append(version)
        return self._policies[version]

    def rollback(self) -> EscalationPolicy:
        if len(self._history) < 2:
            raise RuntimeError("no earlier policy to roll back to")
        self._history.pop()
        return self.active

    @property
    def active(self) -> EscalationPolicy:
        return self._policies[self._history[-1]]

    def versions(self) -> tuple[str, ...]:
        return tuple(self._policies)

    def to_json(self) -> str:
        return canonical_json(
            {"policies": [p.to_dict() for p in self._policies.values()], "history": list(self._history)}
        )

    @classmethod
    def from_json(cls, text: str) -> "PolicyRegistry":
        d = json.loads(text)
        policies = [EscalationPolicy.from_dict(p) for p in d["policies"]]
        reg = cls(policies[0])
        for p in policies[1:]:
            reg.register(p, activate=False)
        reg._history = list(d["history"])
        return reg


class EscalationRouter:
    """Applies the active policy: hard rules first, then the economic rule (§10)."""

    def __init__(self, policy: EscalationPolicy | PolicyRegistry = DEFAULT_POLICY) -> None:
        self._source = policy

    @property
    def policy(self) -> EscalationPolicy:
        return self._source.active if isinstance(self._source, PolicyRegistry) else self._source

    def decide(self, ctx: EscalationContext) -> EscalationDecision:
        pol = self.policy
        p = pol.p_fail(ctx.features)
        gain = pol.default_quality_gain if ctx.expected_quality_gain is None else float(ctx.expected_quality_gain)
        economic = bool(pol.economic_escalate(p, gain))
        fired = tuple(
            r
            for r, flag in (
                ("unsupported_tool", ctx.unsupported_tool),
                ("unsafe_action", ctx.unsafe_action),
                ("user_requested_review", ctx.user_requested_review),
            )
            if flag and r in pol.hard_rules
        )
        return EscalationDecision(
            escalate=bool(fired) or economic,
            p_fail=p,
            expected_failure_cost=p * pol.cost_failure,
            expected_quality_gain=gain,
            cost_escalation=pol.cost_escalation,
            economic=economic,
            hard_rules=fired,
            policy_version=pol.version,
            policy_fingerprint=pol.fingerprint,
            features=ctx.features.to_dict(),
            lm_requested_escalation=ctx.lm_requested_escalation,
        )


# ------------------------------------------------------------------ features
_HEDGES = (
    r"\bnot (?:sure|certain|confident)\b",
    r"\bunsure\b",
    r"\buncertain\b",
    r"\bmight\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bi think\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\bi(?:'| a)m not able\b",
    r"\bcan(?:no|')t (?:verify|guarantee)\b",
)


def uncertainty_from_text(text: Optional[str]) -> float:
    """Crude hedge density in the model's own message, in [0, 1]. A feature, never a decision."""
    if not text:
        return 0.0
    t = text.lower()
    hits = sum(len(re.findall(p, t)) for p in _HEDGES)
    return min(1.0, hits / 3.0)


def constraint_complexity(n_operations: int, n_verifiers: int = 0, n_calls: int = 0) -> float:
    """Normalised constraint count in [0, 1] (log scale, saturating at ~64 constraints)."""
    n = max(0, n_operations) + max(0, n_verifiers) + max(0, n_calls)
    return min(1.0, math.log1p(n) / math.log1p(64))


# ------------------------------------------------------------------- fitting
def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-10,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression by Newton-Raphson (IRLS); bias is unpenalised.

    ``y`` may be soft targets in [0, 1] (used by Platt scaling). Returns
    ``(weights, bias)``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError("X must be (n, d) and y must be (n,)")
    n, d = X.shape
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    Xb = np.hstack([X, np.ones((n, 1))])
    reg = np.full(d + 1, float(l2))
    reg[-1] = 0.0
    theta = np.zeros(d + 1)
    for _ in range(max_iter):
        p = np.asarray(_sigmoid(Xb @ theta))
        grad = Xb.T @ (sw * (p - y)) + reg * theta
        w = sw * p * (1.0 - p)
        H = (Xb * w[:, None]).T @ Xb + np.diag(reg) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        # damped Newton: halve the step while the penalised loss increases
        loss = _penalised_nll(Xb, y, sw, theta, reg)
        t = 1.0
        while t > 1e-4 and _penalised_nll(Xb, y, sw, theta - t * step, reg) > loss + 1e-12:
            t *= 0.5
        theta = theta - t * step
        if np.linalg.norm(t * step) < tol:
            break
    return theta[:-1].copy(), float(theta[-1])


def _penalised_nll(Xb: np.ndarray, y: np.ndarray, sw: np.ndarray, theta: np.ndarray, reg: np.ndarray) -> float:
    z = Xb @ theta
    # log(1 + e^z) - y z, computed stably
    nll = np.logaddexp(0.0, z) - y * z
    return float(np.sum(sw * nll) + 0.5 * np.sum(reg * theta * theta))


def fit_platt(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling ``p = sigmoid(a * score + b)`` with Platt's smoothed targets."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_pos = float(np.sum(y > 0.5))
    n_neg = float(len(y) - n_pos)
    t = np.where(y > 0.5, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    w, b = fit_logistic(scores[:, None], t, l2=0.0)
    return float(w[0]), float(b)


def fit_policy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    version: str,
    base: EscalationPolicy = DEFAULT_POLICY,
    l2: float = 1e-3,
    calibrate: bool = True,
    calibration_fraction: float = 0.3,
    seed: int = 0,
) -> EscalationPolicy:
    """Fit a new policy version from pilot labels (``y=1``: the mid-tier failed the task).

    Costs, hard rules, and feature definitions are inherited from ``base``;
    with ``calibrate`` a held-out split fits Platt parameters.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if calibrate and n >= 20:
        perm = np.random.default_rng(seed).permutation(n)
        n_cal = max(1, int(round(calibration_fraction * n)))
        cal, train = perm[:n_cal], perm[n_cal:]
    else:
        cal, train = np.array([], dtype=int), np.arange(n)
    w, b = fit_logistic(X[train], y[train], l2=l2)
    a, c = 1.0, 0.0
    if len(cal):
        a, c = fit_platt(X[cal] @ w + b, y[cal])
    return EscalationPolicy(
        version=version,
        weights=tuple(w.tolist()),
        bias=b,
        feature_names=base.feature_names,
        platt_a=a,
        platt_b=c,
        cost_failure=base.cost_failure,
        cost_escalation=base.cost_escalation,
        default_quality_gain=base.default_quality_gain,
        hard_rules=base.hard_rules,
        metadata={
            "base_version": base.version,
            "fit": "newton-irls",
            "l2": l2,
            "n_train": int(len(train)),
            "n_calibration": int(len(cal)),
            "base_rate": float(np.mean(y)) if n else 0.0,
        },
    )


# ------------------------------------------------------------------- metrics
def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE: ``sum_b (n_b / n) * |mean(p_b) - mean(y_b)|``."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(p) == 0:
        return 0.0
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


@dataclass(frozen=True)
class RouterMetrics:
    """§13 escalation metrics. ``unnecessary_escalation_rate`` = escalated easy tasks / easy tasks."""

    n: int
    hard_tasks: int
    escalation_rate: float
    hard_task_recall: float
    unnecessary_escalation_rate: float
    escalation_precision: float
    ece: float
    resolved_rate: float
    cost_per_resolved_task: float
    total_cost: float

    def to_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def evaluate_policy(
    policy: EscalationPolicy,
    X: np.ndarray,
    y: np.ndarray,
    *,
    quality_gain: Optional[float] = None,
    base_cost: float = 1.0,
    flagship_success: float = 1.0,
    hard_rule_mask: Optional[np.ndarray] = None,
) -> RouterMetrics:
    """Score a policy on labelled tasks (``y=1`` = the mid-tier alone would fail).

    Cost model: every task pays ``base_cost``; escalations pay
    ``cost_escalation`` and resolve with probability ``flagship_success``;
    unescalated hard tasks fail and pay ``cost_failure``.
    """
    X = np.asarray(X, dtype=np.float64)
    hard = np.asarray(y).astype(bool)
    p = policy.p_fail_batch(X)
    esc = np.asarray(policy.economic_escalate(p, quality_gain), dtype=bool)
    if hard_rule_mask is not None:
        esc = esc | np.asarray(hard_rule_mask, dtype=bool)
    n = len(hard)
    n_hard = int(hard.sum())
    n_easy = n - n_hard
    resolved = esc * float(flagship_success) + (~esc & ~hard)
    total_cost = n * base_cost + esc.sum() * policy.cost_escalation + (~esc & hard).sum() * policy.cost_failure
    n_resolved = float(resolved.sum())
    return RouterMetrics(
        n=n,
        hard_tasks=n_hard,
        escalation_rate=float(esc.mean()) if n else 0.0,
        hard_task_recall=float((esc & hard).sum() / n_hard) if n_hard else 1.0,
        unnecessary_escalation_rate=float((esc & ~hard).sum() / n_easy) if n_easy else 0.0,
        escalation_precision=float((esc & hard).sum() / esc.sum()) if esc.sum() else 0.0,
        ece=expected_calibration_error(p, hard.astype(float)),
        resolved_rate=n_resolved / n if n else 0.0,
        cost_per_resolved_task=float(total_cost / n_resolved) if n_resolved > 0 else math.inf,
        total_cost=float(total_cost),
    )
