"""Instruction-following parity gate: IFEval/BFCL scorers, paired non-inferiority, release blocking."""

import json
import random

import pytest

from m63_omni.cli import main
from m63_omni.eval import bfcl, ifeval
from m63_omni.eval.gates import NON_NEGOTIABLE, evaluate_release
from m63_omni.eval.harness import ScorerSelfTestError
from m63_omni.eval.parity import InstructionParityGate, ParityMargins, ParityReport

# ------------------------------------------------------------------ IFEval


def test_ifeval_checks_self_test_and_strict_vs_loose():
    ifeval.self_test()
    resp = "Sure! Here it is:\nall lowercase text"
    assert not ifeval.check("change_case:english_lowercase", resp)
    assert ifeval.check("change_case:english_lowercase", resp, loose=True)  # first line removed
    assert ifeval.check("length_constraints:number_words", "a b c", {"num_words": 3, "relation": "at least"})
    assert not ifeval.check("punctuation:no_comma", "")  # empty responses never follow an instruction
    with pytest.raises(ifeval.IFEvalError, match="relation"):
        ifeval.check("length_constraints:number_words", "a", {"num_words": 1, "relation": "about"})


def test_ifeval_loader_scoring_and_summary(tmp_path):
    rows = [
        {"key": 1, "prompt": "Reply without commas.", "instruction_id_list": ["punctuation:no_comma"],
         "kwargs": [{"num_words": None}]},
        {"key": 2, "prompt": "Two rules.", "instruction_id_list": ["punctuation:no_comma", "keywords:existence"],
         "kwargs": [{}, {"keywords": ["clip"]}]},
        {"key": 3, "prompt": "Antworte auf Deutsch.", "instruction_id_list": ["language:response_language"],
         "kwargs": [{"language": "de"}]},
    ]
    data = tmp_path / "input.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ifeval.IFEvalError, match="unsupported"):
        ifeval.load_ifeval(data)
    items, skipped = ifeval.load_ifeval(data, skip_unsupported=True)
    assert [i.key for i in items] == ["1", "2"] and skipped == ["3"] and items[0].kwargs == ({},)
    resp = tmp_path / "resp.jsonl"
    resp.write_text(json.dumps({"prompt": "Reply without commas.", "response": "No commas"}) + "\n")
    res = ifeval.score_ifeval(items, ifeval.load_responses(resp))
    assert res[0].prompt_strict and res[1].missing_response and not any(res[1].strict)
    s = ifeval.summarize(res)
    assert s["prompt_strict"] == 0.5 and s["instruction_strict"] == pytest.approx(1 / 3) and s["missing_responses"] == 1


# -------------------------------------------------------------------- BFCL
def test_bfcl_parsing_forms():
    assert bfcl.parse_calls('get_weather(city="SF", days=3)') == [bfcl.Call("get_weather", {"city": "SF", "days": 3})]
    assert bfcl.parse_calls('```python\n[a.b(x=1), c(y=[1, 2])]\n```')[0].name == "a.b"
    oa = json.dumps([{"type": "function", "function": {"name": "f", "arguments": json.dumps({"x": 1})}}])
    assert bfcl.parse_calls(oa) == [bfcl.Call("f", {"x": 1})]
    assert bfcl.parse_calls('{"tool_calls": [{"name": "f", "arguments": {"x": 2}}]}')[0].args == {"x": 2}
    assert bfcl.parse_calls("[]") == []
    with pytest.raises(bfcl.BFCLError, match="positional"):
        bfcl.parse_calls("f(1)")
    with pytest.raises(bfcl.BFCLError):
        bfcl.parse_calls("I would call the weather tool.")
    assert bfcl.category_of("live_parallel_multiple_3-1-0") == "parallel_multiple"
    assert bfcl.category_of("simple_python_12") == "simple"


def test_bfcl_scoring_and_files(tmp_path):
    bfcl.self_test()
    fn = {"name": "trim", "parameters": {"type": "dict", "required": ["clip_id"], "properties": {
        "clip_id": {"type": "string"}, "end_ms": {"type": "integer"}, "gain": {"type": "float"}}}}
    q = tmp_path / "q.jsonl"
    a = tmp_path / "a.jsonl"
    q.write_text("\n".join(json.dumps({"id": i, "question": [[{"role": "user", "content": "x"}]], "function": [fn]})
                           for i in ("simple_0", "parallel_0")))
    a.write_text("\n".join([
        json.dumps({"id": "simple_0", "ground_truth": [{"trim": {"clip_id": ["c 17", "c_17"], "end_ms": [4000, ""],
                                                                  "gain": [1.0, ""]}}]}),
        json.dumps({"id": "parallel_0", "ground_truth": [{"trim": {"clip_id": ["a"], "end_ms": [""], "gain": [""]}},
                                                          {"trim": {"clip_id": ["b"], "end_ms": [""], "gain": [""]}}]}),
    ]))
    items = bfcl.load_bfcl(q, a)
    assert [i.category for i in items] == ["simple", "parallel"]
    ok = bfcl.score_bfcl(items, {"simple_0": 'trim(clip_id="C-17", gain=1)', "parallel_0": '[trim(clip_id="b"), trim(clip_id="a")]'})
    assert all(r.correct for r in ok)  # normalised string, int accepted for float, any call order
    bad = bfcl.score_bfcl(items, {"simple_0": 'trim(clip_id="c_17", end_ms=5)'})
    assert not bad[0].correct and "end_ms" in bad[0].reason and bad[1].missing_output
    assert bfcl.summarize(ok)["overall"] == 1.0 and bfcl.summarize(bad)["overall"] == 0.0


# -------------------------------------------------------------------- gate
def _ifeval_arm(n, seed, p=0.75):
    r = random.Random(seed)
    return [ifeval.IFEvalResult(str(i), ("punctuation:no_comma", "keywords:existence"),
                                (r.random() < p, r.random() < p), (True, True)) for i in range(n)]


def _bfcl_arm(n, seed, p=0.85):
    r = random.Random(seed)
    return [bfcl.BFCLResult(f"{bfcl.CATEGORIES[i % 5]}_{i}", bfcl.CATEGORIES[i % 5], r.random() < p) for i in range(n)]


def _swap(results, k, seed, ife=True):
    """Flip k successes to failures and k failures to successes: same mean, k*2 disagreements."""
    r = random.Random(seed)
    out = list(results)
    if ife:
        ok = [i for i, x in enumerate(out) if all(x.strict)]
        bad = [i for i, x in enumerate(out) if not any(x.strict)]
        for i in r.sample(ok, k):
            out[i] = ifeval.IFEvalResult(out[i].key, out[i].instruction_ids, (False, False), out[i].loose)
        for i in r.sample(bad, k):
            out[i] = ifeval.IFEvalResult(out[i].key, out[i].instruction_ids, (True, True), out[i].loose)
        return out
    for cat in bfcl.CATEGORIES:
        ok = [i for i, x in enumerate(out) if x.category == cat and x.correct]
        bad = [i for i, x in enumerate(out) if x.category == cat and not x.correct]
        for i in r.sample(ok, k) + r.sample(bad, k):
            out[i] = bfcl.BFCLResult(out[i].id, out[i].category, not out[i].correct)
    return out


def _run(ic, ik, bc, bk, **kw):
    return InstructionParityGate(**kw).evaluate(ifeval_candidate=ic, ifeval_control=ik, bfcl_candidate=bc, bfcl_control=bk)


def test_equivalent_models_pass_at_benchmark_scale():
    ik, bk = _ifeval_arm(541, 0), _bfcl_arm(1700, 1)
    rep = _run(_swap(ik, 12, 2), ik, _swap(bk, 6, 3, ife=False), bk)
    assert rep.verdict == "pass", rep.to_markdown()
    assert all(abs(m.diff) < 1e-9 for m in rep.metrics) and all(m.lower < 0 for m in rep.metrics)


def test_small_samples_are_inconclusive_not_passed():
    ik, bk = _ifeval_arm(60, 0), _bfcl_arm(150, 1, p=0.7)
    rep = _run(_swap(ik, 4, 2), ik, _swap(bk, 2, 3, ife=False), bk)
    assert rep.verdict == "inconclusive" and not rep.passed
    assert "add prompts" in rep.to_markdown()


def test_a_real_regression_fails():
    ik, bk = _ifeval_arm(541, 0), _bfcl_arm(1700, 1)
    worse = [ifeval.IFEvalResult(x.key, x.instruction_ids, (False, False), x.loose) if i % 12 == 0 else x
             for i, x in enumerate(ik)]  # ~6-point drop on prompt-level strict
    rep = _run(worse, ik, bk, bk)
    by = {m.name: m for m in rep.metrics}
    assert by["IFEval prompt-level strict"].verdict == "fail" and rep.verdict == "fail"
    assert by["BFCL overall (mean of categories)"].verdict == "pass"


def test_one_collapsed_bfcl_category_fails_even_within_the_overall_margin():
    ik, bk = _ifeval_arm(541, 0), _bfcl_arm(5000, 1)
    cand = [bfcl.BFCLResult(x.id, x.category, False) if x.category == "irrelevance" and (i // 5) % 10 == 0 else x
            for i, x in enumerate(bk)]  # every 10th irrelevance item fails: ~9 points there, ~2 overall
    rep = _run(ik, ik, cand, bk, margins=ParityMargins(bfcl_overall=5.0))
    assert any("irrelevance" in f for f in rep.failures) and rep.verdict == "fail"
    assert rep.category_diffs["irrelevance"] < -5 and rep.category_diffs["simple"] == 0


def test_comparison_must_be_paired_and_complete():
    ik, bk = _ifeval_arm(50, 0), _bfcl_arm(50, 1)
    with pytest.raises(ValueError, match="paired"):
        _run(ik[1:], ik, bk, bk)
    with pytest.raises(ValueError, match="paired"):
        _run(ik, ik, bk[1:], bk)
    rep = InstructionParityGate().evaluate(ifeval_candidate=ik, ifeval_control=ik)
    assert rep.verdict == "fail" and any("BFCL" in f for f in rep.failures)


def test_a_broken_scorer_cannot_certify(monkeypatch):
    monkeypatch.setitem(ifeval.CHECKS, "punctuation:no_comma", lambda r: True)
    with pytest.raises(ScorerSelfTestError):
        _run(_ifeval_arm(10, 0), _ifeval_arm(10, 0), _bfcl_arm(10, 1), _bfcl_arm(10, 1))


def test_release_is_blocked_unless_parity_passes():
    ok = {g: True for g in NON_NEGOTIABLE}
    assert evaluate_release(ok).passed and evaluate_release(ok).parity is None
    ik, bk = _ifeval_arm(541, 0), _bfcl_arm(1700, 1)
    passing = _run(ik, ik, bk, bk)
    assert evaluate_release(ok, parity=passing).passed
    for verdict_rep in (_run(_ifeval_arm(30, 0), _ifeval_arm(30, 5), _bfcl_arm(30, 1), _bfcl_arm(30, 6)),
                        ParityReport(failures=["x"])):
        g = evaluate_release(ok, parity=verdict_rep)
        assert not g.passed and g.parity in ("fail", "inconclusive") and "parity" in g.failures[-1]
    assert ParityReport.from_dict(passing.to_dict()).verdict == "pass"


# --------------------------------------------------------------------- CLI
def test_parity_cli_end_to_end(tmp_path):
    prompts = [{"key": i, "prompt": f"Prompt {i} without commas.", "instruction_id_list": ["punctuation:no_comma"],
                "kwargs": [{}]} for i in range(8)]
    (tmp_path / "if.jsonl").write_text("\n".join(json.dumps(p) for p in prompts))
    for arm in ("cand", "ctrl"):
        (tmp_path / f"if_{arm}.jsonl").write_text("\n".join(json.dumps({"key": p["key"], "response": "No commas here"})
                                                            for p in prompts))
    fn = {"name": "f", "parameters": {"type": "dict", "required": ["x"], "properties": {"x": {"type": "integer"}}}}
    (tmp_path / "q.jsonl").write_text(json.dumps({"id": "simple_0", "question": [], "function": [fn]}))
    (tmp_path / "a.jsonl").write_text(json.dumps({"id": "simple_0", "ground_truth": [{"f": {"x": [1]}}]}))
    (tmp_path / "irr.jsonl").write_text(json.dumps({"id": "irrelevance_0", "question": [], "function": [fn]}))
    for arm in ("cand", "ctrl"):
        (tmp_path / f"b_{arm}.jsonl").write_text(json.dumps({"id": "simple_0", "result": "f(x=1)"}) + "\n"
                                                 + json.dumps({"id": "irrelevance_0", "result": "No tool fits."}))
    args = ["parity", "--ifeval-data", str(tmp_path / "if.jsonl"), "--ifeval-candidate", str(tmp_path / "if_cand.jsonl"),
            "--ifeval-control", str(tmp_path / "if_ctrl.jsonl"), "--bfcl-questions", str(tmp_path / "q.jsonl"),
            str(tmp_path / "irr.jsonl"), "--bfcl-answers", str(tmp_path / "a.jsonl"), "none",
            "--bfcl-candidate", str(tmp_path / "b_cand.jsonl"), "--bfcl-control", str(tmp_path / "b_ctrl.jsonl"),
            "--json", str(tmp_path / "parity.json")]
    assert main(args) == 0
    assert json.loads((tmp_path / "parity.json").read_text())["verdict"] == "pass"
    (tmp_path / "if_cand.jsonl").write_text("\n".join(json.dumps({"key": p["key"], "response": "Commas, everywhere"})
                                                      for p in prompts))
    assert main(args) == 1
