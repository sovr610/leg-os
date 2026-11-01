"""Stage A-D curriculum and compute-normalised weights (§8)."""

from __future__ import annotations

import dataclasses

import pytest

from m63_omni.data.mixture import (
    MIXTURE_KEYS,
    STAGES,
    TOTAL_POSITIONS,
    MixtureError,
    compute_normalized_weights,
    compute_shares,
    get_stage,
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
    for s in STAGES:
        pos, pct, seq = expected[s.name]
        assert s.positions == pos
        assert tuple(s.sampling_percent[k] for k in MIXTURE_KEYS) == pct
        assert (s.seq_len_start, s.seq_len_end) == seq
        assert sum(s.sampling_percent.values()) == 100
    assert sum(s.positions for s in STAGES) == TOTAL_POSITIONS == 4_750_000_000_000
    validate_curriculum()


def test_validation_rejects_broken_curricula():
    bad_sum = dataclasses.replace(STAGES[0], sampling_percent={**STAGES[0].sampling_percent, "image": 11})
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
    assert uniform == pytest.approx({"text_code": 0.6, "image": 0.1, "video": 0.1, "audio": 0.1, "av_sync": 0.1})
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
    assert budget["text_code"] == 2_100_000_000_000 and sum(budget.values()) == STAGES[0].positions
