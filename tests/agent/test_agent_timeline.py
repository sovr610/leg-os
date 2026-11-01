"""Deterministic timeline domain: pure edits, contradictory-edit detection, probes (§10)."""

from __future__ import annotations

import pytest
from agent_helpers import make_timeline

from m63_omni.agent import (
    Timeline,
    TimelineError,
    apply_operations,
    find_conflicts,
    render_probe,
    schema_probe,
    summarize_timeline,
    transition_probe,
    validate_operations,
    validate_timeline,
)

BLUEPRINT_OPS = [
    {"op": "trim", "clip_id": "c_17", "start_ms": 12400, "end_ms": 26800},
    {"op": "transition", "after_clip_id": "c_17", "kind": "dissolve", "duration_ms": 400},
]


def test_blueprint_ops_apply_purely_and_deterministically() -> None:
    tl = make_timeline()
    before = tl.to_dict()
    new, h = apply_operations(tl, BLUEPRINT_OPS)
    assert tl.to_dict() == before  # input untouched
    assert new.version == tl.version + 1
    assert new.clip("c_17").start_ms == 12400 and new.clip("c_17").end_ms == 26800
    assert new.transition_after("c_17").duration_ms == 400
    assert new.duration_ms == 8000 + 14400 + 12000 - 400
    assert h == new.content_hash == apply_operations(make_timeline(), BLUEPRINT_OPS)[1]
    assert Timeline.from_dict(new.to_dict()) == new
    assert validate_timeline(new) == []


def test_insert_and_delete_semantics() -> None:
    tl = make_timeline()
    new, _ = apply_operations(
        tl,
        [
            {"op": "insert", "clip_id": "c_new", "asset_id": "a0", "start_ms": 0, "end_ms": 1000, "after_clip_id": None},
            {"op": "transition", "after_clip_id": "c_new", "kind": "fade", "duration_ms": 200},
            {"op": "delete", "clip_id": "c_18"},
        ],
    )
    assert [c.clip_id for c in new.clips] == ["c_new", "c_16", "c_17"]
    # deleting the last clip drops the predecessor's now-dangling transition
    t2, _ = apply_operations(make_timeline(), [{"op": "transition", "after_clip_id": "c_17", "kind": "wipe", "duration_ms": 300}])
    t3, _ = apply_operations(t2, [{"op": "delete", "clip_id": "c_18"}])
    assert t3.transition_after("c_17") is None and validate_timeline(t3) == []


@pytest.mark.parametrize(
    "ops, needle",
    [
        ([{"op": "trim", "clip_id": "c_17", "start_ms": 5000, "end_ms": 5000}], "must be greater than start_ms"),
        ([{"op": "trim", "clip_id": "c_17", "start_ms": 9000, "end_ms": 1000}], "must be greater than start_ms"),
        ([{"op": "trim", "clip_id": "c_17", "start_ms": 0, "end_ms": 70000}], "exceeds asset a1 duration"),
        ([{"op": "trim", "clip_id": "c_99", "start_ms": 0, "end_ms": 10}], "clip c_99 does not exist"),
        ([{"op": "transition", "after_clip_id": "c_99", "kind": "dissolve", "duration_ms": 400}], "clip c_99 does not exist"),
        ([{"op": "transition", "after_clip_id": "c_18", "kind": "dissolve", "duration_ms": 400}], "is the last clip"),
        ([{"op": "transition", "after_clip_id": "c_16", "kind": "dissolve", "duration_ms": 9000}], "longer than adjacent clip c_16"),
        ([{"op": "transition", "after_clip_id": "c_16", "kind": "cut", "duration_ms": 100}], "a cut must have duration_ms 0"),
        # the trim makes the (otherwise valid) transition longer than the adjacent clip in the final state
        ([{"op": "transition", "after_clip_id": "c_17", "kind": "dissolve", "duration_ms": 2000},
          {"op": "trim", "clip_id": "c_18", "start_ms": 0, "end_ms": 1500}], "longer than adjacent clip c_18"),
        ([{"op": "insert", "clip_id": "c_17", "asset_id": "a0", "start_ms": 0, "end_ms": 10, "after_clip_id": None}], "already exists"),
        ([{"op": "insert", "clip_id": "c_x", "asset_id": "zz", "start_ms": 0, "end_ms": 10, "after_clip_id": "c_16"}], "unknown asset"),
        ([{"op": "trim", "clip_id": "c_17", "start_ms": 0, "end_ms": 10, "extra": True}], "unknown field"),
        ([{"op": "explode", "clip_id": "c_17"}], "is not one of"),
    ],
)
def test_invalid_edits_rejected(ops, needle) -> None:
    errors = validate_operations(make_timeline(), ops)
    assert any(needle in e for e in errors), errors
    with pytest.raises(TimelineError):
        apply_operations(make_timeline(), ops)


def test_conflicting_ops_on_same_clip_detected() -> None:
    contradictory = [
        {"op": "trim", "clip_id": "c_17", "start_ms": 0, "end_ms": 1000},
        {"op": "trim", "clip_id": "c_17", "start_ms": 500, "end_ms": 2000},
    ]
    assert any("contradicts operations[0]" in e for e in find_conflicts(contradictory))
    duplicate = [BLUEPRINT_OPS[1], dict(BLUEPRINT_OPS[1])]
    assert any("duplicates operations[0]" in e for e in find_conflicts(duplicate))
    delete_then_edit = [{"op": "delete", "clip_id": "c_17"}, BLUEPRINT_OPS[0]]
    assert any("which operations[0] deletes" in e for e in find_conflicts(delete_then_edit))
    assert validate_operations(make_timeline(), delete_then_edit)
    assert find_conflicts(BLUEPRINT_OPS) == []


def test_probes_pass_on_valid_and_fail_on_corrupted_timelines() -> None:
    before = make_timeline().to_dict()
    after, _ = apply_operations(make_timeline(), BLUEPRINT_OPS)
    ok, ev = schema_probe(after.to_dict(), before, BLUEPRINT_OPS)
    assert ok and ev["matches_replay"]
    ok, ev = render_probe(after.to_dict())
    assert ok and ev["gap_frames"] == 0 and ev["overlap_frames"] == 0 and ev["duration_ms"] == after.duration_ms
    assert render_probe(after.to_dict())[1]["probe_hash"] == ev["probe_hash"]  # deterministic
    ok, ev = transition_probe(after.to_dict(), BLUEPRINT_OPS)
    assert ok and ev["transitions"][0]["window_ms"] == [8000 + 14400 - 400, 8000 + 14400]

    # a tool that silently changed an unreferenced clip fails the schema comparison
    tampered = after.to_dict()
    tampered["clips"][0]["end_ms"] = 7000
    ok, ev = schema_probe(tampered, before, BLUEPRINT_OPS)
    assert not ok and ev["matches_replay"] is False

    # overlong transition -> clips overlap outside the window / start before predecessor
    bad = after.to_dict()
    bad["transitions"] = [{"after_clip_id": "c_16", "kind": "dissolve", "duration_ms": 9000}]
    assert not render_probe(bad)[0]
    assert not transition_probe(bad)[0]
    # requested transition missing from output
    assert not transition_probe(make_timeline().to_dict(), BLUEPRINT_OPS)[0]
    # empty programme
    empty = Timeline("asset_91", (), (), {"a0": 10}).to_dict()
    ok, ev = render_probe(empty)
    assert not ok and "total duration must be > 0" in ev["problems"]
    # source out of range (bypassing the constructor's checks)
    oob = make_timeline().to_dict()
    oob["clips"][2]["end_ms"] = 31000
    assert not render_probe(oob)[0]


def test_summary_is_compact_and_typed() -> None:
    s = summarize_timeline(make_timeline().to_dict())
    assert s["duration_ms"] == 8000 + 30000 + 12000 and len(s["clips"]) == 3
    assert summarize_timeline({"nope": 1})["invalid"] is True
