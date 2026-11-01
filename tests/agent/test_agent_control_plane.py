"""Deterministic control plane: validation before execution, idempotency, sandboxing,
signed receipts, tenant isolation, verification/commit, and undo-by-branch (§2, §6, §10, App. B)."""

from __future__ import annotations

import dataclasses
import json
import time

import pytest
from agent_helpers import (
    INJECTION,
    SECRET,
    BLUEPRINT_ENVELOPE,
    ToolKit,
    edit_envelope,
    make_timeline,
    single_call_envelope,
)

from m63_omni.agent import (
    REDACTED,
    ArtifactNotFound,
    BranchError,
    BranchRequest,
    ControlPlane,
    ErrorClass,
    Principal,
    RunBudget,
    Timeline,
    apply_operations,
    parse_envelope,
)
from m63_omni.common import Trust, canonical_json
from m63_omni.runtime.event_log import MAIN_BRANCH, Event, TenantIsolationError

BP_OPS = BLUEPRINT_ENVELOPE["calls"][0]["arguments"]["operations"]


def _call(env: dict, i: int = 0):
    e = parse_envelope(env)
    return e, e.calls[i]


def _events(session, kind: str | None = None, branch: str | None = None) -> list[Event]:
    evs = list(session.event_log.replay(branch or session.branch_id))
    return [e for e in evs if kind is None or e.kind == kind]


# --------------------------------------------------------------- happy path
def test_blueprint_call_executes_verifies_and_commits(cp, session, toolkit) -> None:
    env, call = _call(BLUEPRINT_ENVELOPE)
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    assert out.receipt.ok and out.succeeded and not out.cached
    assert [v.probe for v in out.verifications] == ["timeline_schema", "render_probe", "transition_probe"]
    assert all(v.deterministic for v in out.verifications)
    expected, h = apply_operations(make_timeline(), BP_OPS)
    assert out.receipt.output_hashes == {"asset_91": h}
    assert out.receipt.input_hashes == {"asset_91": make_timeline().content_hash}
    assert out.receipt.trust == Trust.TOOL_VERIFIED.value
    assert cp.artifacts(session)["asset_91"].hash == h
    assert cp.read_artifact(session, "asset_91") == expected.to_dict()
    kinds = [e.kind for e in _events(session)]
    assert kinds == ["session_start", "ingest", "receipt", "verification"]
    assert cp.verify_receipt(out.receipt)
    assert cp.audit(session).ok


# ------------------------------------------------------------- idempotency
def test_idempotent_retry_returns_cached_receipt_without_reexecuting(cp, session, toolkit) -> None:
    env, call = _call(BLUEPRINT_ENVELOPE)
    first = cp.execute_call(session, call, plan_id=env.plan_id)
    second = cp.execute_call(session, call, plan_id=env.plan_id)
    assert toolkit.calls("timeline.apply_edit@2") == 1
    assert second.cached and second.receipt == first.receipt
    assert second.receipt.receipt_hash == first.receipt.receipt_hash
    assert second.succeeded and second.committed[0].hash == first.committed[0].hash
    assert len(_events(session, "receipt")) == 1
    # the cache survives a control-plane restart because the event log is the source of truth
    cp2 = ControlPlane(toolkit.registry, secret=SECRET)
    resumed = cp2.open_session(session.principal, session.session_id, event_log=session.event_log)
    third = cp2.execute_call(resumed, call, plan_id=env.plan_id)
    assert third.cached and third.receipt.receipt_hash == first.receipt.receipt_hash
    assert toolkit.calls("timeline.apply_edit@2") == 1


def test_same_key_different_arguments_is_rejected(cp, session, toolkit) -> None:
    env, call = _call(BLUEPRINT_ENVELOPE)
    cp.execute_call(session, call, plan_id=env.plan_id)
    altered = dataclasses.replace(call, arguments={"timeline_id": "asset_91", "operations": [BP_OPS[0] | {"end_ms": 20000}]})
    out = cp.execute_call(session, altered, plan_id=env.plan_id)
    assert out.rejected and out.receipt.error_class == ErrorClass.IDEMPOTENCY_CONFLICT.value
    assert toolkit.calls("timeline.apply_edit@2") == 1


def test_failed_execution_is_also_idempotent(cp, session, toolkit) -> None:
    env, call = _call(single_call_envelope("p/crash", "flaky.crash@1", {}))
    a = cp.execute_call(session, call, plan_id=env.plan_id)
    b = cp.execute_call(session, call, plan_id=env.plan_id)
    assert a.receipt.status == "failed" and b.cached and toolkit.calls("flaky.crash@1") == 1


# ------------------------------------------- static validation before run
def test_permission_denied_before_execution(cp, session, toolkit) -> None:
    viewer = Principal("bob", "acme", frozenset({"web:read"}))
    env, call = _call(BLUEPRINT_ENVELOPE)
    out = cp.execute_call(session, call, plan_id=env.plan_id, principal=viewer)
    assert out.rejected and out.receipt.error_class == ErrorClass.PERMISSION_DENIED.value
    assert "timeline:write" in out.receipt.error_message
    assert toolkit.calls("timeline.apply_edit@2") == 0
    assert cp.artifacts(session)["asset_91"].hash == make_timeline().content_hash
    # the rejection itself is a signed, logged receipt
    (rec,) = _events(session, "receipt")
    assert rec.payload["status"] == "rejected" and cp.verify_receipt(rec.payload)


@pytest.mark.parametrize(
    "mutate, error_class",
    [
        (lambda c: c.update(tool="timeline.apply_edit@9"), ErrorClass.UNKNOWN_VERSION),
        (lambda c: c.update(tool="timeline.reticulate@1"), ErrorClass.UNKNOWN_TOOL),
        (lambda c: c["arguments"].update(force=True), ErrorClass.SCHEMA_INVALID),
        (lambda c: c["arguments"]["operations"][0].update(start_ms="12400"), ErrorClass.SCHEMA_INVALID),
        (lambda c: c.update(timeout_class="whenever"), ErrorClass.INVALID_TIMEOUT_CLASS),
        (lambda c: c.update(timeout_class="long_render"), ErrorClass.RESOURCE_LIMIT),
        (lambda c: c.update(verify=["vibes_check"]), ErrorClass.UNKNOWN_VERIFIER),
        (lambda c: c["arguments"].update(timeline_id="asset_404"), ErrorClass.UNKNOWN_ARTIFACT),
        (lambda c: c.update(expected_input_hashes={"asset_91": "sha256:" + "0" * 64}), ErrorClass.STALE_INPUT),
        (lambda c: c.update(expected_input_hashes={"asset_7": "sha256:" + "0" * 64}), ErrorClass.SCHEMA_INVALID),
    ],
)
def test_static_rejections_never_execute(cp, session, toolkit, mutate, error_class) -> None:
    env = json.loads(json.dumps(BLUEPRINT_ENVELOPE))
    mutate(env["calls"][0])
    e, call = _call(env)
    out = cp.execute_call(session, call, plan_id=e.plan_id)
    assert out.rejected and out.receipt.error_class == error_class.value, out.receipt.error_message
    assert toolkit.calls("timeline.apply_edit@2") == 0


@pytest.mark.parametrize(
    "ops",
    [
        [{"op": "trim", "clip_id": "c_17", "start_ms": 26800, "end_ms": 12400}],
        [{"op": "transition", "after_clip_id": "c_404", "kind": "dissolve", "duration_ms": 400}],
        [{"op": "transition", "after_clip_id": "c_16", "kind": "dissolve", "duration_ms": 8500}],
        [{"op": "trim", "clip_id": "c_17", "start_ms": 0, "end_ms": 1000}, {"op": "trim", "clip_id": "c_17", "start_ms": 0, "end_ms": 2000}],
    ],
)
def test_contradictory_edits_rejected_statically(cp, session, toolkit, ops) -> None:
    env, call = _call(edit_envelope("p/bad", ops))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    assert out.rejected and out.receipt.error_class == ErrorClass.PRECONDITION_FAILED.value
    assert toolkit.calls("timeline.apply_edit@2") == 0
    pre = cp.preflight(session, env)
    assert not pre.ok  # whole-envelope preflight catches it too, before anything runs
    assert toolkit.calls("timeline.apply_edit@2") == 0


def test_preflight_rejects_whole_envelope_before_any_execution(cp, session, toolkit) -> None:
    env = json.loads(json.dumps(BLUEPRINT_ENVELOPE))
    bad = json.loads(json.dumps(env["calls"][0]))
    bad["idempotency_key"] = "k2"
    bad["tool"] = "assets.delete_all@1"
    bad["arguments"] = {"scope": "tenant"}
    env["calls"].append(bad)
    report = cp.preflight(session, parse_envelope(env))
    assert not report.ok and report.rejections[0] is None
    assert report.rejections[1].receipt.error_class == ErrorClass.PERMISSION_DENIED.value
    over = parse_envelope({**BLUEPRINT_ENVELOPE, "budget": {"max_calls": 4, "max_repair_attempts": 9}})
    assert "exceeds policy limit" in cp.preflight(session, over).envelope_errors[0]
    assert toolkit.calls("timeline.apply_edit@2") == 0


def test_run_budget_enforced(cp, session, toolkit) -> None:
    budget = RunBudget(max_calls=1, max_repair_attempts=0, max_cost=10)
    env, call = _call(BLUEPRINT_ENVELOPE)
    assert cp.execute_call(session, call, plan_id=env.plan_id, budget=budget).succeeded
    env2, call2 = _call(edit_envelope("p/2", [{"op": "trim", "clip_id": "c_16", "start_ms": 0, "end_ms": 5000}]))
    out = cp.execute_call(session, call2, plan_id=env2.plan_id, budget=budget)
    assert out.receipt.error_class == ErrorClass.BUDGET_EXCEEDED.value
    cheap = RunBudget(max_calls=5, max_repair_attempts=0, max_cost=0.5)
    out = cp.execute_call(session, call2, plan_id=env2.plan_id, budget=cheap)
    assert out.receipt.error_class == ErrorClass.BUDGET_EXCEEDED.value and "cost budget" in out.receipt.error_message
    assert toolkit.calls("timeline.apply_edit@2") == 1


# ---------------------------------------------------------------- sandbox
def test_timeout_becomes_failed_receipt(cp, session, toolkit) -> None:
    env, call = _call(single_call_envelope("p/slow", "slow.probe@1", {}, timeout_class="interactive"))
    over = cp.execute_call(session, call, plan_id=env.plan_id)  # asks for more time than the tool allows
    assert over.rejected and over.receipt.error_class == ErrorClass.RESOURCE_LIMIT.value
    env, call = _call(single_call_envelope("p/slow2", "slow.probe@1", {}, timeout_class="probe_fast"))
    t0 = time.perf_counter()
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    assert time.perf_counter() - t0 < 2.0
    assert out.receipt.status == "failed" and out.receipt.error_class == ErrorClass.TIMEOUT.value
    assert not out.succeeded and out.committed == ()
    assert cp.verify_receipt(out.receipt)
    assert toolkit.calls("slow.probe@1") == 1


def test_tool_exception_becomes_failed_receipt_with_redacted_message(cp, session) -> None:
    env, call = _call(single_call_envelope("p/crash", "flaky.crash@1", {}))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    assert out.receipt.status == "failed" and out.receipt.error_class == ErrorClass.TOOL_EXCEPTION.value
    assert "ValueError" in out.receipt.error_message
    assert "sk-live" not in canonical_json([e.payload for e in _events(session)])


def test_tool_cannot_write_ungranted_artifacts(cp, session) -> None:
    env, call = _call(single_call_envelope("p/rogue", "rogue.writer@1", {"timeline_id": "asset_91"}))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    assert out.receipt.error_class == ErrorClass.INVALID_OUTPUT.value
    assert "asset_other" not in cp.artifacts(session)


def test_failed_verification_does_not_commit(cp, session) -> None:
    env, call = _call(single_call_envelope("p/corrupt", "timeline.corrupt@1", {"timeline_id": "asset_91"},
                                           verify=("render_probe", "transition_probe")))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    assert out.receipt.ok and not out.succeeded and out.committed == ()
    failed = {v.probe for v in out.verifications if not v.passed}
    assert {"render_probe", "transition_probe"} <= failed
    assert cp.artifacts(session)["asset_91"].hash == make_timeline().content_hash
    (ver,) = _events(session, "verification")
    assert ver.payload["passed"] is False and ver.payload["committed"] == []


def test_deterministic_verifiers_run_before_model_review(cp, session) -> None:
    seen: list[str] = []
    cp.verifiers.register("model_review", lambda ctx: (seen.append("model") or True, {}), deterministic=False)
    env, call = _call(edit_envelope("p/mr", BP_OPS, verify=("model_review", "render_probe")))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    assert [v.probe for v in out.verifications] == ["timeline_schema", "render_probe", "model_review"]
    assert out.succeeded and seen == ["model"]
    # after a deterministic failure, model review is skipped rather than run
    env, call = _call(single_call_envelope("p/c2", "timeline.corrupt@1", {"timeline_id": "asset_91"},
                                           verify=("model_review", "render_probe")))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    mr = next(v for v in out.verifications if v.probe == "model_review")
    assert mr.skipped and not mr.passed and seen == ["model"]


# ------------------------------------------------------- receipts & secrets
def test_receipt_signature_verification_and_tamper_detection(cp, session, toolkit) -> None:
    env, call = _call(BLUEPRINT_ENVELOPE)
    receipt = cp.execute_call(session, call, plan_id=env.plan_id).receipt
    d = receipt.to_dict()
    assert cp.verify_receipt(d)
    for field, value in [("status", "failed"), ("output_hashes", {"asset_91": "sha256:" + "f" * 64}),
                         ("trust", "system"), ("tenant_id", "globex"), ("created_at", 0.0)]:
        assert not cp.verify_receipt({**d, field: value}), field
    assert not cp.verify_receipt({**d, "signature": d["signature"][:-1] + ("0" if d["signature"][-1] != "0" else "1")})
    forged = {**d, "status": "failed"}
    forged["receipt_hash"] = __import__("m63_omni.common", fromlist=["content_hash"]).content_hash(
        {k: v for k, v in forged.items() if k not in ("receipt_hash", "signature")}
    )
    assert not cp.verify_receipt(forged)  # re-hashing without the key cannot re-sign
    other = ControlPlane(toolkit.registry, secret=b"another-secret-key-0123456789")
    assert not other.verify_receipt(d)
    assert not cp.verify_receipt({k: v for k, v in d.items() if k != "signature"})

    # tampering with the stored event is caught by the audit (chain + signature)
    log = session.event_log
    ev = next(e for e in _events(session) if e.kind == "receipt")
    log._events[ev.event_id] = dataclasses.replace(ev, payload={**ev.payload, "status": "failed"})
    report = cp.audit(session)
    assert not report.ok and not report.chain_ok and ev.event_id in report.bad_receipts


def test_secrets_redacted_from_receipts_and_log(cp, session) -> None:
    env, call = _call(single_call_envelope("p/leak", "leaky.echo@1", {}))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    assert out.receipt.output["api_key"] == REDACTED
    assert "abcdefghijklmnopqrstuvwxyz0123" not in out.receipt.output["note"]
    assert out.receipt.output["ok"] is True
    dump = canonical_json([e.payload for e in _events(session)])
    assert "plain-value-123456" not in dump and SECRET.decode() not in dump
    assert cp.verify_receipt(out.receipt)  # redaction happens before signing


# -------------------------------------------------------- tenant isolation
def test_cross_tenant_access_rejected(cp, session, toolkit) -> None:
    intruder = Principal("mallory", "globex", frozenset({"timeline:write", "web:read", "assets:delete"}))
    env, call = _call(BLUEPRINT_ENVELOPE)
    n_events = len(session.event_log)
    out = cp.execute_call(session, call, plan_id=env.plan_id, principal=intruder)
    assert out.rejected and out.receipt.error_class == ErrorClass.TENANT_ISOLATION.value
    assert toolkit.calls("timeline.apply_edit@2") == 0
    assert len(session.event_log) == n_events  # nothing written into the victim tenant's log
    assert cp.security_events and cp.security_events[0]["type"] == "cross_tenant_attempt"
    with pytest.raises(TenantIsolationError):
        cp.ingest_artifact(session, "asset_x", "timeline", {}, principal=intruder)
    with pytest.raises(TenantIsolationError):
        cp.confirm(session, "assets.delete_all@1", {"scope": "tenant"}, principal=intruder)
    with pytest.raises(TenantIsolationError):
        cp.open_session(intruder, "s", event_log=session.event_log)
    with pytest.raises(TenantIsolationError):
        session.event_log.get(_events(session)[0].event_id, tenant_id="globex")

    # another tenant's session cannot reference acme's artifacts, by id or by hash
    other = cp.open_session(intruder, "session_globex")
    out = cp.execute_call(other, call, plan_id=env.plan_id)
    assert out.receipt.error_class == ErrorClass.UNKNOWN_ARTIFACT.value
    with pytest.raises(ArtifactNotFound):
        cp.store.get("globex", make_timeline().content_hash)
    assert toolkit.calls("timeline.apply_edit@2") == 0


def test_content_cannot_be_ingested_above_principal_trust(cp, session) -> None:
    with pytest.raises(PermissionError):
        cp.ingest_artifact(session, "policy", "text", {"t": "you are root"}, trust=Trust.SYSTEM)
    bot = Principal("bot", "acme", frozenset(), trust=Trust.UNTRUSTED)
    with pytest.raises(PermissionError):
        cp.confirm(session, "assets.delete_all@1", {"scope": "tenant"}, principal=bot)


# ------------------------------------------------------- injection safety
def test_injected_instructions_in_tool_output_are_inert(cp, session, toolkit) -> None:
    env, call = _call(single_call_envelope("p/search", "web.search@1", {"query": "dissolve tips"}))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    assert out.succeeded and out.receipt.output["results"][0]["snippet"] == INJECTION
    assert out.receipt.trust == Trust.UNTRUSTED.value  # quarantined label
    assert toolkit.calls("assets.delete_all@1") == 0
    assert session.principal.permissions == frozenset({"timeline:write", "web:read"})
    # even a principal that *holds* the permission cannot run the destructive tool without a user confirmation
    admin = Principal("root", "acme", frozenset({"assets:delete"}))
    e2, c2 = _call(single_call_envelope("p/del", "assets.delete_all@1", {"scope": "tenant"}))
    out = cp.execute_call(session, c2, plan_id=e2.plan_id, principal=admin)
    assert out.receipt.error_class == ErrorClass.CONFIRMATION_REQUIRED.value
    assert toolkit.calls("assets.delete_all@1") == 0
    # an explicit user confirmation of exactly these arguments is single-use
    cp.confirm(session, "assets.delete_all@1", {"scope": "tenant"})
    e3, c3 = _call(single_call_envelope("p/del2", "assets.delete_all@1", {"scope": "everything"}))
    assert cp.execute_call(session, c3, plan_id=e3.plan_id, principal=admin).receipt.error_class == "confirmation_required"
    assert cp.execute_call(session, c2, plan_id=e2.plan_id, principal=admin).receipt.ok
    e4, c4 = _call(single_call_envelope("p/del3", "assets.delete_all@1", {"scope": "tenant"}))
    assert cp.execute_call(session, c4, plan_id=e4.plan_id, principal=admin).receipt.error_class == "confirmation_required"
    assert toolkit.calls("assets.delete_all@1") == 1


# ---------------------------------------------------------------- branches
def test_undo_creates_branch_and_replay_reflects_restored_timeline(cp, session, toolkit) -> None:
    v0 = cp.artifacts(session)["asset_91"].hash
    e1, c1 = _call(edit_envelope("p/1", [BP_OPS[0]]))
    v1 = cp.execute_call(session, c1, plan_id=e1.plan_id).committed[0].hash
    e2, c2 = _call(edit_envelope("p/2", [BP_OPS[1]]))
    v2 = cp.execute_call(session, c2, plan_id=e2.plan_id).committed[0].hash
    assert len({v0, v1, v2}) == 3 and session.branch_id == MAIN_BRANCH

    branch = cp.undo(session, name="undo-transition", memory_snapshot_id="mem_123")
    assert branch == "undo-transition" and session.branch_id == branch
    replayed = list(session.event_log.replay(branch))
    assert replayed[-1].kind == "branch" and replayed[-1].payload["meta"]["memory_snapshot_id"] == "mem_123"
    assert cp.artifacts(session)["asset_91"].hash == v1  # derived purely from the branch replay
    restored = Timeline.from_dict(cp.read_artifact(session, "asset_91"))
    assert restored.transition_after("c_17") is None and restored.clip("c_17").end_ms == 26800
    # nothing was destroyed: main still holds v2 and the full history
    assert cp.artifacts(session, MAIN_BRANCH)["asset_91"].hash == v2
    assert session.event_log.verify_chain()

    # edits continue on the undo branch; redo is restoring the old branch
    e3, c3 = _call(edit_envelope("p/3", [{"op": "transition", "after_clip_id": "c_17", "kind": "wipe", "duration_ms": 300}]))
    assert cp.execute_call(session, c3, plan_id=e3.plan_id).succeeded
    assert cp.artifacts(session, MAIN_BRANCH)["asset_91"].hash == v2
    cp.restore_branch(session, MAIN_BRANCH)
    assert cp.artifacts(session)["asset_91"].hash == v2

    two_back = cp.undo(session, steps=2, name="undo-two")
    assert cp.artifacts(session, two_back)["asset_91"].hash == v0
    with pytest.raises(BranchError):
        cp.undo(session, steps=5)


def test_branch_request_checks_expected_state(cp, session) -> None:
    head = session.event_log.head(session.branch_id)
    good = BranchRequest(head.event_id, "explore", name="alt", expected_state={"asset_91": make_timeline().content_hash})
    assert cp.create_branch(session, good, switch=False) == "alt"
    bad = BranchRequest(head.event_id, "explore", name="alt2", expected_state={"asset_91": "sha256:" + "1" * 64})
    with pytest.raises(BranchError, match="expected state"):
        cp.create_branch(session, bad)
    with pytest.raises(BranchError):
        cp.create_branch(session, BranchRequest("ev_missing", "x"))
    assert session.branch_id == MAIN_BRANCH


def test_idempotency_is_scoped_to_branch_lineage(cp, session, toolkit) -> None:
    env, call = _call(BLUEPRINT_ENVELOPE)
    cp.execute_call(session, call, plan_id=env.plan_id)
    cp.undo(session, name="before-edit")
    again = cp.execute_call(session, call, plan_id=env.plan_id)  # undone history: a fresh application
    assert not again.cached and again.succeeded and toolkit.calls("timeline.apply_edit@2") == 2


def test_toolkit_is_isolated_per_test() -> None:
    assert ToolKit().calls("timeline.apply_edit@2") == 0


def test_json_schema_verifier_and_fail_closed_verification(cp, session) -> None:
    env, call = _call(edit_envelope("p/js", BP_OPS, verify=("json_schema",)))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    js = next(v for v in out.verifications if v.probe == "json_schema")
    assert js.passed and set(js.evidence["checked"]) == {"data", "asset_91"}
    # requesting verification that has nothing to check fails closed instead of passing vacuously
    env, call = _call(single_call_envelope("p/js2", "web.search@1", {"query": "q"}, verify=("json_schema", "render_probe")))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    assert out.receipt.ok and not out.succeeded
    assert {v.probe: v.passed for v in out.verifications} == {"json_schema": False, "render_probe": False}


def test_verifier_exceptions_become_failures(cp, session) -> None:
    def broken(ctx):
        raise RuntimeError("verifier bug")

    cp.verifiers.register("broken_probe", broken)
    env, call = _call(edit_envelope("p/vb", BP_OPS, verify=("broken_probe",)))
    out = cp.execute_call(session, call, plan_id=env.plan_id)
    bp = next(v for v in out.verifications if v.probe == "broken_probe")
    assert not bp.passed and "verifier bug" in bp.evidence["error"]
    assert not out.succeeded and out.committed == ()
