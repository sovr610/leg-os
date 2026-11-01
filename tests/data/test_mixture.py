"""Stage A-D curriculum and compute-normalised weights (§8)."""

from __future__ import annotations

import dataclasses

import pytest

from m63_omni.data.mixture import (
    BLUEPRINT_STAGES,
    MIXTURE_KEYS,
    STAGES,
    TOTAL_POSITIONS,
    MixtureError,
    compute_normalized_weights,
    compute_shares,
    get_stage,
    mixture_at,
    stage_at,
    stage_position_budget,
    validate_curriculum,
)


def test_table_matches_blueprint():
    expected = {
        "A": (3.50e12, (60, 10, 10, 10, 10), (4096, 4096)),
        "B": (0.75e12, (50, 12, 14, 12, 12), (8192, 32768)),
        "C": (0.35e12, (55, 10, 15, 10, 10), (32768, 131072)),
        "D": (0.15e12, (45, 5, 20, 10, 20), (131072, 262144)),
    }
    for s in BLUEPRINT_STAGES:
        pos, pct, seq = expected[s.name]
        assert s.positions == pos
        assert tuple(s.sampling_percent[k] for k in MIXTURE_KEYS) == pct
        assert (s.seq_len_start, s.seq_len_end) == seq
        assert sum(s.sampling_percent.values()) == 100 and not s.warmup_positions
    assert sum(s.positions for s in BLUEPRINT_STAGES) == TOTAL_POSITIONS == 4_750_000_000_000
    validate_curriculum(BLUEPRINT_STAGES)


def test_default_stage_a_is_text_heavy_after_a_text_warmup():
    a = STAGES[0]
    assert STAGES[1:] == BLUEPRINT_STAGES[1:]  # only Stage A changes
    assert tuple(a.sampling_percent[k] for k in MIXTURE_KEYS) == (70, 12, 5, 8, 5)
    assert a.warmup_positions == 350_000_000_000 and a.warmup_percent["text_code"] == 100
    assert a.effective_percent == pytest.approx({"text_code": 73.0, "image": 10.8, "video": 4.5, "audio": 7.2,
                                                 "av_sync": 4.5})
    assert mixture_at(0)[1]["text_code"] == 100 and mixture_at(349_999_999_999)[1]["image"] == 0
    st, mix = mixture_at(350_000_000_000)
    assert st.name == "A" and mix["text_code"] == 70 and mix["av_sync"] == 5
    assert mixture_at(3_500_000_000_000)[1]["text_code"] == 50  # Stage B has no warm-up
    totals = {k: 0 for k in MIXTURE_KEYS}
    for s in STAGES:
        for k, v in stage_position_budget(s).items():
            totals[k] += v
    assert totals == {"text_code": 3_190_000_000_000, "image": 510_500_000_000, "video": 345_000_000_000,
                      "audio": 392_000_000_000, "av_sync": 312_500_000_000}
    validate_curriculum()


def test_warmup_validation():
    a = STAGES[0]
    with pytest.raises(MixtureError, match="shorter than the stage"):
        validate_curriculum((dataclasses.replace(a, warmup_positions=a.positions),) + STAGES[1:])
    with pytest.raises(MixtureError, match="sum to 100"):
        validate_curriculum((dataclasses.replace(a, warmup_percent={**a.warmup_percent, "image": 5}),) + STAGES[1:])
    with pytest.raises(MixtureError, match="less text-heavy"):
        mixed = {"text_code": 60, "image": 10, "video": 10, "audio": 10, "av_sync": 10}
        validate_curriculum((dataclasses.replace(a, warmup_percent=mixed),) + STAGES[1:])


def test_validation_rejects_broken_curricula():
    bad_sum = dataclasses.replace(STAGES[0], sampling_percent={**STAGES[0].sampling_percent, "image": 13})
    with pytest.raises(MixtureError, match="sum to 101"):
        validate_curriculum((bad_sum,) + STAGES[1:])
    no_av = dataclasses.replace(STAGES[3], sampling_percent={**STAGES[3].sampling_percent, "av_sync": 0, "video": 40})
    with pytest.raises(MixtureError, match="synchronized AV"):
        validate_curriculum(STAGES[:3] + (no_av,))
    with pytest.raises(MixtureError, match="4.75T"):
        validate_curriculum(STAGES[:3])
    with pytest.raises(MixtureError, match="non-decreasing"):
        validate_curriculum((STAGES[1], STAGES[0], STAGES[2], STAGES[3]))
    with pytest.raises(TypeError):
        STAGES[0].sampling_percent["text_code"] = 1  # read-only mapping


def test_compute_normalized_weights():
    uniform = compute_normalized_weights("A", {k: 1.0 for k in MIXTURE_KEYS})
    assert uniform == pytest.approx({"text_code": 0.7, "image": 0.12, "video": 0.05, "audio": 0.08, "av_sync": 0.05})
    cost = {"text_code": 1.0, "image": 2.0, "video": 4.0, "audio": 1.5, "av_sync": 5.0}
    for stage in STAGES:
        w = compute_normalized_weights(stage, cost)
        assert sum(w.values()) == pytest.approx(1.0)
        # expensive modalities get fewer positions than their compute share
        assert w["video"] < stage.sampling_percent["video"] / 100
        assert w["text_code"] > stage.sampling_percent["text_code"] / 100
        # converting back to compute recovers the table's starting point exactly
        shares = compute_shares(w, cost)
        assert shares == pytest.approx({k: v / 100 for k, v in stage.sampling_percent.items()})
    with pytest.raises(MixtureError, match="missing"):
        compute_normalized_weights("B", {"text_code": 1.0})
    with pytest.raises(MixtureError, match="positive"):
        compute_normalized_weights("B", {**cost, "audio": 0.0})


def test_stage_lookup_and_budgets():
    assert get_stage("c").name == "C"
    with pytest.raises(MixtureError):
        get_stage("E")
    assert stage_at(0).name == "A"
    assert stage_at(3_500_000_000_000 - 1).name == "A"
    assert stage_at(3_500_000_000_000).name == "B"
    assert stage_at(TOTAL_POSITIONS - 1).name == "D"
    with pytest.raises(MixtureError):
        stage_at(TOTAL_POSITIONS)
    budget = stage_position_budget("A")
    assert budget["text_code"] == 2_555_000_000_000 and sum(budget.values()) == STAGES[0].positions
    assert stage_position_budget(BLUEPRINT_STAGES[0])["text_code"] == 2_100_000_000_000


def test_text_control_budget_matches_the_candidate_text_stream():
    from m63_omni.data.mixture import text_control_budget

    b = text_control_budget()
    assert b == {"A": 2_555_000_000_000, "B": 375_000_000_000, "C": 192_500_000_000, "D": 67_500_000_000}
    assert sum(b.values()) == 3_190_000_000_000
