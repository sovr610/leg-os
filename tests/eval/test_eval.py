"""Evaluation harness, release gates, security suite, long-context probes (§13, Appendix B)."""

import pytest

from m63_omni.eval.efficiency import measure
from m63_omni.eval.gates import NON_NEGOTIABLE, MemoryPromotionGate, evaluate_release
from m63_omni.eval.harness import (ScorerSelfTestError, bootstrap_ci, evaluate_cohorts, exact_match,
                                   paired_bootstrap_diff, verify_scorer)
from m63_omni.eval.long_context import greedy_answer, multikey_task, passkey_task
from m63_omni.eval.security import run_security_suite


def test_bootstrap_and_paired_diff():
    m, lo, hi = bootstrap_ci([0, 1] * 50)
    assert lo < m == 0.5 < hi
    d, lo, hi = paired_bootstrap_diff([1] * 40, [0] * 40)
    assert d == lo == hi == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1], [1, 2])


def test_scorer_self_test_catches_broken_scorer():
    good = [({"id": 1, "answer": "a"}, "a")]
    bad = [({"id": 2, "answer": "a"}, "b")]
    verify_scorer(exact_match, good, bad)
    with pytest.raises(ScorerSelfTestError):
        verify_scorer(lambda item, pred: 1.0, good, bad)  # always-pass scorer must be rejected


def test_per_cohort_report_surfaces_critical_failures():
    items = [{"cohort": "short_text", "answer": 1, "x": 1}] * 5 + [{"cohort": "adversarial", "answer": 1, "x": 0}] * 5
    rep = evaluate_cohorts(items, lambda it: it["x"], exact_match, suite="toy", evaluator_version="v1",
                           critical=lambda it, p: it["cohort"] == "adversarial" and p != it["answer"])
    assert rep.cohorts["short_text"].mean == 1.0 and rep.worst_cohort().cohort == "adversarial"
    assert rep.has_critical_failure() and "adversarial" in rep.to_markdown()


def test_release_gates():
    ok = {g: True for g in NON_NEGOTIABLE}
    assert evaluate_release(ok).passed
    missing = dict(ok)
    del missing["verified_deletion"]
    rep = evaluate_release(missing)
    assert not rep.passed and "verified_deletion" in rep.failures[0]
    good = dict(session_on=60, session_off=55, latency_on_ms=105, latency_off_ms=100, short_on=70, short_off=70.2,
                security_unauthorized_actions=0)
    assert evaluate_release(ok, good).memory_promoted is True
    weak = dict(good, session_on=57)
    slow = dict(good, latency_on_ms=115)
    regress = dict(good, short_on=69)
    for m in (weak, slow, regress):
        rep = evaluate_release(ok, m)
        assert rep.passed and rep.memory_promoted is False  # memory-off core still ships
    assert MemoryPromotionGate().evaluate(**dict(good, security_unauthorized_actions=1))


def test_runtime_security_suite_passes(model):
    rep = run_security_suite(model)
    assert rep.passed, [r for r in rep.results if not r.passed]
    assert all(rep.gate_inputs().values())


def test_long_context_task_generators():
    t = passkey_task(400, 128, 0.5, seed=1)
    assert len(t.prompt) == 128 and t.prompt[-2:] == [2, 1]
    i = t.prompt.index(1)
    assert t.prompt[i + 1 : i + 5] == t.answer
    mk = multikey_task(400, 200, 5, seed=2)
    assert mk.kind == "multikey_5" and len(mk.answer) == 2


def test_greedy_answer_paths_agree_without_memory_updates(model):
    t = passkey_task(400, 40, 0.3, seed=0)
    assert len(greedy_answer(model, t.prompt, 3)) == 3
    assert len(greedy_answer(model, t.prompt, 3, segmented=True)) == 3


def test_efficiency_report_kv_matches_analytic(model):
    r = measure(model, prompt_tokens=24, new_tokens=4)
    assert r.kv_bytes_per_token_measured == r.kv_bytes_per_token_analytic
    assert r.ttft_ms > 0 and r.decode_tok_s > 0
