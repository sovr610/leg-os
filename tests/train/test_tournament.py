"""Proxy tournament runner: matched budgets, systems gates, Pareto selection (§12 step 6)."""

from pathlib import Path

import pytest

from m63_omni.config import tiny
from m63_omni.train.tournament import CandidateResult, _pareto, build_candidates, run_tournament, write_report

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ROOT / "configs" / "tournament"


def _cands(*names):
    return build_candidates(tiny(), [VARIANTS / f"{n}.yaml" for n in names])


def test_every_tournament_variant_maps_onto_the_tiny_base():
    cands = build_candidates(tiny(), sorted(VARIANTS.glob("*.yaml")))
    names = {c.name for c in cands}
    assert {"ratio_1to1", "ratio_3to1", "ratio_7to1", "full_attention_control", "memory_off", "top1", "top4",
            "no_shared_expert", "memory_decay_to_init", "kda", "kv_heads_half", "text_timestamps", "av_chunk_2s",
            "recommended_defaults", "loop_second_pass"} <= names
    seven = next(c for c in cands if c.name == "ratio_7to1")
    assert seven.cfg.model.hybrid_pattern.count("gdn") == 7 and seven.cfg.memory.anchors == (8,)  # re-fitted
    assert next(c for c in cands if c.name == "memory_decay_to_init").cfg.memory.decay_target == "init"
    assert all(c.cfg.model.hidden_size == 64 for c in cands)  # the base shape is preserved
    by = {c.name: c.cfg for c in cands}
    assert by["kda"].layer_kinds().count("kda") == 6 and by["kv_heads_half"].attention.kv_heads == 1  # tiny: 2 -> 1
    rec = by["recommended_defaults"]
    assert rec.model.hybrid_pattern[0] == "kda" and not rec.memory.enabled and rec.multimodal.timestamp_mode == "text"
    assert rec.multimodal.av_chunk_ms == 2000 and rec.multimodal.audio_attention == "causal"
    loop = by["loop_second_pass"].loop  # proxy range 5-12 does not fit 8 layers: re-fitted to the middle groups
    assert loop.enabled and (loop.first_layer, loop.last_layer) == (1, 4)


def test_matched_positions_metrics_and_pareto(tmp_path):
    rep = run_tournament(_cands("ratio_3to1", "full_attention_control", "memory_off"), steps=3, batch=2, eval_batches=2)
    by = {r.name: r for r in rep.results}
    assert len({r.positions for r in rep.results}) == 1  # identical data budget and order
    assert all(r.eval_loss == r.eval_loss and r.eval_ci[0] <= r.eval_loss <= r.eval_ci[1] for r in rep.results)
    assert by["full_attention_control"].kv_bytes_per_token == 4 * by["ratio_3to1"].kv_bytes_per_token
    assert rep.winners and all(by[w].pareto for w in rep.winners)
    write_report(rep, tmp_path / "t.json", tmp_path / "t.md")
    assert "full_attention_control" in (tmp_path / "t.md").read_text()


def test_systems_gate_excludes_candidates_from_winning():
    cands = _cands("ratio_3to1", "full_attention_control")
    kv_hybrid = 2 * 2 * 2 * 16 * 4  # 2 attention layers x K/V x kv_heads x head_dim x fp32
    rep = run_tournament(cands, steps=2, batch=2, eval_batches=2, max_kv_bytes_per_token=kv_hybrid)
    by = {r.name: r for r in rep.results}
    assert by["ratio_3to1"].systems_pass and not by["full_attention_control"].systems_pass
    assert "full_attention_control" not in rep.winners and not by["full_attention_control"].pareto


def test_flops_matched_budget_scales_steps_by_active_compute():
    rep = run_tournament(_cands("top1", "top4"), steps=4, batch=2, eval_batches=1, budget="flops", reference="top1")
    by = {r.name: r for r in rep.results}
    assert by["top1"].steps == 4 and by["top4"].steps < 4
    assert by["top4"].train_flops == pytest.approx(by["top1"].train_flops, rel=0.35)


def test_vocabulary_must_match():
    a = build_candidates(tiny(), [VARIANTS / "ratio_3to1.yaml"])
    b = build_candidates(tiny(model={"vocab_size": 640}), [VARIANTS / "top1.yaml"])
    with pytest.raises(ValueError, match="vocabulary"):
        run_tournament(a + b, steps=1)


def _row(name, loss, tpot, kv, ok=True):
    return CandidateResult(name, {}, 0, 0.0, 1, 1, 0.0, loss, loss, loss, (loss, loss), 0.0, 0.0, tpot, kv, 0, 1.0,
                           systems_pass=ok)


def test_pareto_treats_differences_within_tolerance_as_ties():
    rows = [_row("a", 5.00, 12.0, 512), _row("b", 5.00, 13.1, 512),  # same model, 9% timing jitter
            _row("c", 5.30, 12.0, 512),  # worse loss beyond tolerance -> dominated
            _row("d", 4.90, 30.0, 2048),  # best loss, worst systems -> still Pareto
            _row("e", 4.00, 1.0, 1, ok=False)]  # failed systems gate -> never Pareto, never dominates
    _pareto(rows, loss_tol=0.05, tpot_rel_tol=0.15)
    assert [r.name for r in rows if r.pareto] == ["a", "b", "d"]


def test_segmented_memory_mode_makes_memory_ablations_matter():
    cands = _cands("ratio_3to1", "memory_decay_to_init")
    packed = run_tournament(cands, steps=3, batch=2, eval_batches=2)
    seg = run_tournament(cands, steps=3, batch=2, eval_batches=2, segmented_memory=True)
    assert packed.results[0].eval_loss == packed.results[1].eval_loss
    assert seg.results[0].eval_loss != seg.results[1].eval_loss
    assert seg.mode == "segmented memory" and packed.mode == "packed"
    assert "segmented memory training" in seg.to_markdown()


def test_multimodal_only_variants_are_skipped_not_ranked():
    cands = _cands("ratio_3to1", "kda", "text_timestamps", "av_chunk_2s", "recommended_defaults", "memory_decay_to_init")
    rep = run_tournament(cands, steps=2, batch=2, eval_batches=1)
    ran = {r.name for r in rep.results}
    assert ran == {"ratio_3to1", "kda", "recommended_defaults", "memory_decay_to_init"}
    assert set(rep.skipped) == {"text_timestamps", "av_chunk_2s"} and "media data" in rep.skipped["text_timestamps"]
    assert rep.notes["recommended_defaults"] == ["multimodal overrides (needs media data)"]
    assert "segmented-memory" in rep.notes["memory_decay_to_init"][0]
    md = rep.to_markdown()
    assert "Skipped" in md and "text_timestamps" in md and "Not exercised" in md
    with pytest.raises(ValueError, match="skipped"):
        run_tournament(_cands("text_timestamps"), steps=1)


def test_looped_candidate_is_scored_at_every_pass_count_under_a_flops_budget():
    rep = run_tournament(_cands("ratio_3to1", "loop_second_pass"), steps=4, batch=2, eval_batches=1, budget="flops",
                         reference="ratio_3to1")
    by = {r.name: r for r in rep.results}
    assert set(by["loop_second_pass"].loop_eval) == {1, 2} and not by["ratio_3to1"].loop_eval
    assert by["loop_second_pass"].steps < 4  # the expected extra pass costs compute
    assert "Eval loss by loop pass count" in rep.to_markdown()
