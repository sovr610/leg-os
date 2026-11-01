"""Calibrated escalation router: hard rules, economic rule, versioned policy, fitting (§10, §13)."""

from __future__ import annotations

import numpy as np
import pytest

from m63_omni.agent import (
    DEFAULT_POLICY,
    FEATURE_NAMES,
    EscalationContext,
    EscalationFeatures,
    EscalationPolicy,
    EscalationRouter,
    PolicyRegistry,
    evaluate_policy,
    expected_calibration_error,
    fit_logistic,
    fit_policy,
    uncertainty_from_text,
)

TRUE_W = np.array([4.0, 1.5, 3.0, 1.2, 2.0, 0.0])  # lm_uncertainty carries no signal
TRUE_B = -3.5


def synth(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.beta(1.2, 4, n),  # verifier failure rate
            rng.uniform(0, 1, n),  # constraint complexity
            rng.beta(1, 5, n),  # domain shift
            rng.poisson(0.4, n).astype(float),  # repair failures
            rng.beta(1, 4, n),  # model disagreement
            rng.uniform(0, 1, n),  # LM self-reported uncertainty (pure noise here)
        ]
    )
    p = 1.0 / (1.0 + np.exp(-(X @ TRUE_W + TRUE_B)))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y


def _policy(bias: float, **kw) -> EscalationPolicy:
    return EscalationPolicy("test", weights=(0.0,) * len(FEATURE_NAMES), bias=bias, **kw)


def test_hard_rules_force_escalation_even_when_p_fail_is_tiny() -> None:
    router = EscalationRouter(_policy(-20.0))
    calm = EscalationContext(EscalationFeatures(), lm_requested_escalation=False)
    assert not router.decide(calm).escalate
    for flag in ("unsupported_tool", "unsafe_action", "user_requested_review"):
        d = router.decide(EscalationContext(EscalationFeatures(), **{flag: True}))
        assert d.escalate and d.hard_rules == (flag,) and not d.economic
        assert f"hard_rule:{flag}" in d.reasons


def test_economic_rule_boundary() -> None:
    # p_fail = 0.5 exactly (bias 0): 0.5 * cost_failure + gain > cost_escalation ?
    pol = _policy(0.0, cost_failure=10.0, cost_escalation=5.0)
    router = EscalationRouter(pol)
    assert not router.decide(EscalationContext()).escalate  # 5.0 > 5.0 is false
    assert router.decide(EscalationContext(expected_quality_gain=0.01)).escalate
    cheap = EscalationRouter(_policy(0.0, cost_failure=10.0, cost_escalation=4.9))
    d = cheap.decide(EscalationContext())
    assert d.escalate and d.economic and d.p_fail == pytest.approx(0.5)
    assert pol.p_fail_threshold == pytest.approx(0.5)


def test_lm_uncertainty_is_only_a_feature_never_a_veto() -> None:
    router = EscalationRouter(DEFAULT_POLICY)
    text = "I'm not sure, maybe this might possibly work, I think. I am uncertain."
    assert uncertainty_from_text(text) == 1.0 and uncertainty_from_text("Done.") == 0.0
    # maximal self-reported uncertainty + an explicit LM escalation request is not enough on its own
    d = router.decide(
        EscalationContext(EscalationFeatures(lm_uncertainty=uncertainty_from_text(text)), lm_requested_escalation=True)
    )
    assert not d.escalate and d.lm_requested_escalation
    # and a confident LM cannot veto a hard rule or strong failure evidence
    assert router.decide(EscalationContext(EscalationFeatures(lm_uncertainty=0.0), unsupported_tool=True)).escalate
    strong = EscalationFeatures(verifier_failure_rate=1.0, repair_failures=2.0, model_disagreement=0.8)
    assert router.decide(EscalationContext(strong, lm_requested_escalation=False)).escalate


def test_policy_is_versioned_serialisable_and_rollbackable() -> None:
    js = DEFAULT_POLICY.to_json()
    again = EscalationPolicy.from_json(js)
    assert again == DEFAULT_POLICY and again.fingerprint == DEFAULT_POLICY.fingerprint
    with pytest.raises(ValueError):
        EscalationPolicy.from_dict({**DEFAULT_POLICY.to_dict(), "surprise": 1})
    with pytest.raises(ValueError):
        EscalationPolicy("bad", weights=(1.0,), bias=0.0)
    with pytest.raises(ValueError):
        EscalationPolicy("bad", weights=(0.0,) * 6, bias=0.0, hard_rules=("never_escalate",))

    reg = PolicyRegistry()
    strict = EscalationPolicy("v2-strict", weights=DEFAULT_POLICY.weights, bias=5.0)
    reg.register(strict)
    router = EscalationRouter(reg)
    assert router.decide(EscalationContext()).escalate and router.policy.version == "v2-strict"
    reg.rollback()
    assert router.policy.version == DEFAULT_POLICY.version and not router.decide(EscalationContext()).escalate
    restored = PolicyRegistry.from_json(reg.to_json())
    assert restored.active == reg.active and set(restored.versions()) == {DEFAULT_POLICY.version, "v2-strict"}
    with pytest.raises(ValueError):
        reg.register(EscalationPolicy("v2-strict", weights=DEFAULT_POLICY.weights, bias=1.0))  # same version, new content


def test_fit_logistic_recovers_parameters() -> None:
    X, y = synth(20000, 7)
    w, b = fit_logistic(X, y, l2=1e-6)
    assert np.allclose(w, TRUE_W, atol=0.45), w
    assert b == pytest.approx(TRUE_B, abs=0.35)
    assert abs(w[FEATURE_NAMES.index("lm_uncertainty")]) < 0.3


def test_fitted_policy_improves_calibration_recall_and_cost() -> None:
    X, y = synth(6000, 0)
    Xt, yt = synth(4000, 1)
    fitted = fit_policy(X, y, version="m63.escalation/pilot-1")
    assert fitted.metadata["base_version"] == DEFAULT_POLICY.version
    assert fitted.cost_failure == DEFAULT_POLICY.cost_failure and fitted.hard_rules == DEFAULT_POLICY.hard_rules

    prior = evaluate_policy(DEFAULT_POLICY, Xt, yt)
    naive = evaluate_policy(EscalationPolicy("naive", weights=(0, 0, 0, 0, 0, 4.0), bias=-2.0), Xt, yt)
    always = evaluate_policy(_policy(20.0), Xt, yt)
    never = evaluate_policy(_policy(-20.0), Xt, yt)
    fit = evaluate_policy(fitted, Xt, yt)

    assert fit.ece < 0.03 and fit.ece < prior.ece / 2 and fit.ece < naive.ece / 4
    assert fit.hard_task_recall > prior.hard_task_recall + 0.05
    assert fit.hard_task_recall > naive.hard_task_recall
    assert fit.unnecessary_escalation_rate < naive.unnecessary_escalation_rate
    assert fit.cost_per_resolved_task < min(prior.cost_per_resolved_task, naive.cost_per_resolved_task,
                                            always.cost_per_resolved_task, never.cost_per_resolved_task)
    # the extremes behave as their names say
    assert always.hard_task_recall == 1.0 and always.unnecessary_escalation_rate == 1.0
    assert never.hard_task_recall == 0.0 and never.escalation_rate == 0.0
    # deterministic given the seed
    assert fit_policy(X, y, version="m63.escalation/pilot-1").fingerprint == fitted.fingerprint


def test_expected_calibration_error_known_values() -> None:
    assert expected_calibration_error(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0
    assert expected_calibration_error(np.array([0.9] * 10), np.array([0.0] * 10)) == pytest.approx(0.9)
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=float)
    assert expected_calibration_error(p, y) == pytest.approx(0.0)
