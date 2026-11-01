"""The ``m63.actions/v1`` envelope and §10 interface dataclasses."""

from __future__ import annotations

import copy
import json

import pytest
from agent_helpers import BLUEPRINT_ENVELOPE, blueprint_envelope

from m63_omni.agent import (
    ActionEnvelope,
    EnvelopeError,
    FinalResponse,
    ToolReceipt,
    ToolRefError,
    parse_envelope,
    parse_tool_ref,
)
from m63_omni.common import content_hash


def test_blueprint_example_envelope_parses_and_round_trips() -> None:
    env = parse_envelope(json.dumps(BLUEPRINT_ENVELOPE, ensure_ascii=False))
    assert isinstance(env, ActionEnvelope)
    assert env.schema_version == "m63.actions/v1"
    assert env.plan_id == "session_42/branch_7/plan_19"
    (call,) = env.calls
    assert (call.tool_name, call.tool_version) == ("timeline.apply_edit", 2)
    assert call.verify == ("timeline_schema", "render_probe", "transition_probe")
    assert call.timeout_class == "standard_media_edit"
    assert env.budget.max_calls == 4 and env.budget.max_repair_attempts == 1
    assert env.escalation.required is False and env.escalation.reason is None
    assert env.to_dict() == BLUEPRINT_ENVELOPE
    # dict input and bytes input are equivalent
    assert parse_envelope(BLUEPRINT_ENVELOPE) == env
    assert parse_envelope(json.dumps(BLUEPRINT_ENVELOPE).encode()) == env
    assert env.envelope_hash == content_hash(BLUEPRINT_ENVELOPE)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda e: e.update(extra=1), "$.extra: unknown field"),
        (lambda e: e["calls"][0].update(note="hi"), "$.calls[0].note: unknown field"),
        (lambda e: e["budget"].update(max_tokens=10), "$.budget.max_tokens: unknown field"),
        (lambda e: e["escalation"].update(priority="high"), "$.escalation.priority: unknown field"),
    ],
)
def test_unknown_fields_rejected_at_every_level(mutate, needle) -> None:
    env = blueprint_envelope()
    mutate(env)
    with pytest.raises(EnvelopeError) as ei:
        parse_envelope(env)
    assert any(needle in e for e in ei.value.errors), ei.value.errors


@pytest.mark.parametrize("version", ["m63.actions/v2", "m63.actions/v0", "", None, 1])
def test_bad_schema_version_rejected(version) -> None:
    with pytest.raises(EnvelopeError, match="unsupported schema version"):
        parse_envelope(blueprint_envelope(schema_version=version))
    env = blueprint_envelope()
    del env["schema_version"]
    with pytest.raises(EnvelopeError, match="unsupported schema version"):
        parse_envelope(env)


@pytest.mark.parametrize("ref", ["timeline.apply_edit", "timeline.apply_edit@v2", "Timeline.apply_edit@2",
                                 "timeline.apply_edit@0", "timeline..apply@2", "timeline.apply_edit@2@3", "@2"])
def test_malformed_tool_refs_rejected(ref) -> None:
    env = blueprint_envelope()
    env["calls"][0]["tool"] = ref
    with pytest.raises(EnvelopeError, match=r"\$\.calls\[0\]\.tool"):
        parse_envelope(env)
    with pytest.raises(ToolRefError):
        parse_tool_ref(ref)


def test_duplicate_idempotency_keys_rejected() -> None:
    env = blueprint_envelope()
    env["calls"].append(copy.deepcopy(env["calls"][0]))
    with pytest.raises(EnvelopeError, match="duplicate idempotency key"):
        parse_envelope(env)


def test_calls_exceeding_budget_rejected() -> None:
    env = blueprint_envelope(budget={"max_calls": 0, "max_repair_attempts": 0})
    with pytest.raises(EnvelopeError, match="exceed budget.max_calls=0"):
        parse_envelope(env)


def test_strict_json_and_types() -> None:
    text = json.dumps(BLUEPRINT_ENVELOPE)
    dup = text.replace('"plan_id"', '"plan_id": "x", "plan_id"', 1)
    with pytest.raises(EnvelopeError, match="duplicate JSON key"):
        parse_envelope(dup)
    with pytest.raises(EnvelopeError, match="not valid JSON"):
        parse_envelope(text[:-2])
    nan = text.replace('"max_calls": 4', '"max_calls": NaN')
    with pytest.raises(EnvelopeError, match="non-finite"):
        parse_envelope(nan)
    with pytest.raises(EnvelopeError, match="expected integer"):
        parse_envelope(blueprint_envelope(budget={"max_calls": 4.0, "max_repair_attempts": 1}))
    with pytest.raises(EnvelopeError, match="expected boolean"):
        parse_envelope(blueprint_envelope(escalation={"required": 0, "reason": None}))
    with pytest.raises(EnvelopeError, match="reason is required"):
        parse_envelope(blueprint_envelope(escalation={"required": True, "reason": None}))
    bad_hash = blueprint_envelope()
    bad_hash["calls"][0]["expected_input_hashes"] = {"asset_91": "md5:abc"}
    with pytest.raises(EnvelopeError, match="expected_input_hashes"):
        parse_envelope(bad_hash)


def test_envelope_reports_all_errors_at_once() -> None:
    env = blueprint_envelope(extra=1, plan_id="")
    env["calls"][0]["timeout_class"] = "Not A Class"
    with pytest.raises(EnvelopeError) as ei:
        parse_envelope(env)
    assert len(ei.value.errors) >= 3


def test_receipt_round_trip_and_final_response_invariants() -> None:
    fields = dict(
        receipt_id="rc_1", tenant_id="t", session_id="s", branch_id="main", principal_id="p", plan_id="p1",
        tool="x.y@1", idempotency_key="k", args_hash="sha256:" + "0" * 64, status="ok", error_class=None,
        error_message=None, artifact_refs=({"artifact_id": "a", "kind": "timeline", "hash": "h"},),
        input_hashes={}, output_hashes={"a": "h"}, output={"n": 1}, metrics={"elapsed_ms": 1.0},
        trust="untrusted", timeout_class="interactive", created_at=1.0, key_id="k1",
    )
    r = ToolReceipt(**fields)
    assert ToolReceipt.from_dict(r.to_dict()) == r
    assert "receipt_hash" not in r.body() and "signature" not in r.body()
    with pytest.raises(ValueError):
        ToolReceipt.from_dict({**r.to_dict(), "surprise": 1})
    with pytest.raises(ValueError):
        FinalResponse(user_message="done", status="succeeded", unresolved_issues=("x",))
    with pytest.raises(ValueError):
        FinalResponse(user_message="done", status="great")
