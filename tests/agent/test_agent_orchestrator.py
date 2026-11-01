"""The §10 loop end to end: intake → bundle → plan → validate → execute → verify → repair/branch → finalize."""

from __future__ import annotations

import json
from typing import Any

import pytest
from agent_helpers import BLUEPRINT_ENVELOPE, INJECTION, edit_envelope, make_timeline, single_call_envelope

from m63_omni.agent import (
    FEATURE_NAMES,
    CompletedArtifact,
    ContextBundle,
    EscalationPolicy,
    EscalationRouter,
    FinalResponse,
    Orchestrator,
    OrchestratorConfig,
    PlannerResponse,
    Principal,
    ScriptedPlanner,
    UserRequest,
    apply_operations,
    final_envelope,
)
from m63_omni.common import Trust, canonical_json
from m63_omni.runtime.event_log import MAIN_BRANCH

BP_OPS = BLUEPRINT_ENVELOPE["calls"][0]["arguments"]["operations"]
NEVER = EscalationRouter(EscalationPolicy("never-economic", weights=(0.0,) * len(FEATURE_NAMES), bias=-30.0))


def kinds(session, branch: str | None = None) -> list[str]:
    return [e.kind for e in session.event_log.replay(branch or session.branch_id)]


def all_receipts(session) -> list[dict[str, Any]]:
    seen, out = set(), []
    for b in session.event_log.branches:
        for e in session.event_log.replay(b):
            if e.kind == "receipt" and e.event_id not in seen:
                seen.add(e.event_id)
                out.append(e.payload)
    return out


# ------------------------------------------------------------------ success
def test_blueprint_envelope_end_to_end(cp, session, toolkit) -> None:
    planner = ScriptedPlanner([BLUEPRINT_ENVELOPE], final_message="Trimmed c_17 and added a 400 ms dissolve.")
    orch = Orchestrator(cp, planner)
    result = orch.run(session, "Tighten c_17 and dissolve into the next shot.")

    assert result.outcome == "success"
    final = result.final
    assert final.status == "succeeded" and final.model_message_used
    assert final.user_message == "Trimmed c_17 and added a 400 ms dissolve."
    _, expected_hash = apply_operations(make_timeline(), BP_OPS)
    (art,) = final.completed_artifacts
    assert (art.artifact_id, art.hash, art.tool) == ("asset_91", expected_hash, "timeline.apply_edit@2")
    receipts = {r["receipt_hash"]: r for r in all_receipts(session)}
    assert art.receipt_hash in receipts and receipts[art.receipt_hash]["status"] == "ok"
    assert final.grounding_violations([o.receipt for o in result.outcomes], {art.receipt_hash}, cp.verify_receipt) == []
    assert final.plan_ids[0] == "session_42/branch_7/plan_19"

    assert kinds(session) == [
        "session_start", "ingest", "ingest", "plan", "receipt", "verification", "plan", "final",
    ]
    # the planner saw typed references + compact summaries, a USER-trust request and the tool manifest
    first: ContextBundle = planner.bundles[0]
    assert first.request == {"text": "Tighten c_17 and dissolve into the next shot.", "trust": "user"}
    assert first.artifact("asset_91")["hash"] == make_timeline().content_hash
    assert first.artifact("asset_91")["summary"]["duration_ms"] == 50000
    assert any(t["tool"] == "timeline.apply_edit@2" for t in first.tools)
    second = planner.bundles[1]
    assert second.mode == "continuation" and second.artifact("asset_91")["hash"] == expected_hash
    assert cp.audit(session).ok


# ------------------------------------------------------------------ honesty
@pytest.mark.parametrize(
    "failing",
    [
        single_call_envelope("p/corrupt", "timeline.corrupt@1", {"timeline_id": "asset_91"}, verify=("render_probe",)),
        single_call_envelope("p/crash", "flaky.crash@1", {}),
        edit_envelope("p/bad", [{"op": "trim", "clip_id": "c_17", "start_ms": 26800, "end_ms": 12400}]),
    ],
    ids=["verification-failed", "tool-exception", "statically-rejected"],
)
def test_final_response_never_claims_success_after_failed_call(cp, session, toolkit, failing) -> None:
    liar = final_envelope("p/final", "All edits were applied successfully and the render looks perfect!")
    orch = Orchestrator(cp, ScriptedPlanner([failing, liar]), router=NEVER)
    result = orch.run(session, "Make the edit.")
    final = result.final

    assert result.outcome == "failure"
    assert final.status == "failed" and not final.model_message_used
    assert "applied successfully" not in final.user_message
    assert "could not complete" in final.user_message
    assert final.completed_artifacts == () and final.receipt_hashes == ()
    assert final.unresolved_issues and failing["calls"][0]["tool"] in " ".join(final.unresolved_issues)
    assert cp.artifacts(session)["asset_91"].hash == make_timeline().content_hash
    logged = [e for e in session.event_log.replay(session.branch_id) if e.kind == "final"][-1].payload
    assert logged["status"] == "failed" and logged["completed_artifacts"] == []


def test_grounding_check_flags_unbacked_claims(cp, session) -> None:
    result = Orchestrator(cp, ScriptedPlanner([BLUEPRINT_ENVELOPE])).run(session, "edit")
    receipts = [o.receipt for o in result.outcomes]
    good = result.final.completed_artifacts[0]
    forged = FinalResponse(
        "done", "succeeded",
        completed_artifacts=(CompletedArtifact("asset_91", "timeline", "sha256:" + "e" * 64, good.receipt_hash, good.tool, "k"),),
        receipt_hashes=("sha256:" + "d" * 64,),
    )
    problems = forged.grounding_violations(receipts, None, cp.verify_receipt)
    assert any("does not produce hash" in p for p in problems)
    assert any("missing or not ok" in p for p in problems)
    tampered = [type(r).from_dict({**r.to_dict(), "metrics": {"elapsed_ms": 0.0}}) for r in receipts]
    assert any("signature" in p for p in result.final.grounding_violations(tampered, None, cp.verify_receipt))


# ------------------------------------------------------------------ repair
def test_repair_loop_is_bounded(cp, session, toolkit) -> None:
    calls = {"n": 0}

    def stubborn(bundle: ContextBundle) -> dict:
        calls["n"] += 1
        return edit_envelope(f"p/{calls['n']}", [{"op": "transition", "after_clip_id": "c_404", "kind": "fade", "duration_ms": 100}],
                             max_repair_attempts=5)

    cfg = OrchestratorConfig(max_repair_attempts=2, max_steps=10)
    result = Orchestrator(cp, stubborn, router=NEVER, config=cfg).run(session, "fade after c_404")
    assert calls["n"] == 3  # initial attempt + 2 repairs
    assert [s.mode for s in result.steps] == ["initial", "repair", "repair"]
    assert result.budget["repairs_used"] == 2 and result.final.status == "failed"
    assert toolkit.calls("timeline.apply_edit@2") == 0  # every attempt was rejected statically

    # the model's own envelope budget can only tighten the bound
    calls["n"] = 0
    tight = lambda b: {**stubborn(b), "budget": {"max_calls": 1, "max_repair_attempts": 0}}  # noqa: E731
    result = Orchestrator(cp, tight, router=NEVER, config=cfg).run(session, "again")
    assert calls["n"] == 1 and result.budget["repairs_used"] == 0


def test_steps_and_calls_are_bounded_even_when_every_step_succeeds(cp, session, toolkit) -> None:
    n = {"i": 0}

    def busy(bundle: ContextBundle) -> dict:
        n["i"] += 1
        return single_call_envelope(f"p/s{n['i']}", "web.search@1", {"query": f"q{n['i']}"})

    result = Orchestrator(cp, busy, config=OrchestratorConfig(max_steps=4)).run(session, "search a lot")
    assert len(result.steps) == 4 and toolkit.calls("web.search@1") == 4
    result = Orchestrator(cp, busy, config=OrchestratorConfig(max_steps=10, max_calls=2)).run(session, "search more")
    assert toolkit.calls("web.search@1") == 6
    assert any("run budget" in e for s in result.steps for e in s.preflight_errors)


def test_recovered_path_after_static_rejection(cp, session, toolkit) -> None:
    bad = edit_envelope("p/1", [{"op": "trim", "clip_id": "c_17", "start_ms": 26800, "end_ms": 12400}])
    fixed = edit_envelope("p/2", BP_OPS)

    def repair(bundle: ContextBundle) -> dict:
        status = next(o["content"] for o in bundle.observations if o["content"]["type"] == "call_status")
        assert bundle.mode == "repair" and status["error_class"] == "precondition_failed"
        assert "must be greater than start_ms" in status["error_message"]
        return fixed

    result = Orchestrator(cp, ScriptedPlanner([bad, repair], final_message="Fixed and applied."), router=NEVER).run(session, "trim")
    assert result.outcome == "recovered"
    assert result.final.status == "succeeded" and result.final.user_message == "Fixed and applied."
    summary = result.final.verification_summary
    assert summary[0]["status"] == "rejected" and summary[0]["resolved"] is True
    assert summary[1]["verified"] is True
    assert toolkit.calls("timeline.apply_edit@2") == 1


def test_partial_commit_repairs_on_a_branch_from_pre_plan_state(cp, session, toolkit) -> None:
    trim = edit_envelope("p/1", [BP_OPS[0]])["calls"][0]
    corrupt = single_call_envelope("p/1", "timeline.corrupt@1", {"timeline_id": "asset_91"},
                                   key="p/1/call_2", verify=("render_probe",))["calls"][0]
    two_calls = {**edit_envelope("p/1", []), "calls": [trim, corrupt]}
    repair = edit_envelope("p/2", [{"op": "trim", "clip_id": "c_16", "start_ms": 1000, "end_ms": 6000}])
    result = Orchestrator(cp, ScriptedPlanner([two_calls, repair]), router=NEVER).run(session, "edit")

    assert result.steps[0].branch_created == result.branch_id != MAIN_BRANCH
    v0 = make_timeline()
    v1, h1 = apply_operations(v0, [BP_OPS[0]])
    _, h_repair = apply_operations(v0, repair["calls"][0]["arguments"]["operations"])
    # the repair started from the verified pre-plan state, not from the half-applied plan
    assert cp.artifacts(session)["asset_91"].hash == h_repair
    # the failed path is preserved (not reversed) on main for audit and training
    assert cp.artifacts(session, MAIN_BRANCH)["asset_91"].hash == h1
    assert result.outcome == "recovered"
    assert [a.hash for a in result.final.completed_artifacts] == [h_repair]
    assert "branch" in kinds(session)


# --------------------------------------------------------------- security
def test_injection_in_tool_output_is_quarantined_and_never_executed(cp, session, toolkit) -> None:
    planner = ScriptedPlanner([single_call_envelope("p/search", "web.search@1", {"query": "dissolve tips"})],
                              final_message="Here are some tips.")
    result = Orchestrator(cp, planner).run(session, "Find tips on dissolves.")
    assert result.outcome == "success"
    obs = planner.bundles[1].observations
    tool_obs = [o for o in obs if o["content"]["type"] == "tool_output"]
    assert tool_obs and tool_obs[0]["trust"] == Trust.UNTRUSTED.value and tool_obs[0]["quarantined"] is True
    assert INJECTION in json.dumps(tool_obs[0], ensure_ascii=False).replace('\\"', '"')
    status_obs = [o for o in obs if o["content"]["type"] == "call_status"]
    assert all(o["trust"] == "system" and INJECTION not in json.dumps(o) for o in status_obs)
    assert toolkit.calls("assets.delete_all@1") == 0
    assert all(r["tool"] != "assets.delete_all@1" for r in all_receipts(session))


@pytest.mark.parametrize("perms", [frozenset({"web:read"}), frozenset({"web:read", "assets:delete"})],
                         ids=["lacks-permission", "has-permission-but-no-user-confirmation"])
def test_compromised_planner_following_injection_is_blocked(cp, toolkit, perms) -> None:
    principal = Principal("carol", "acme", perms)
    session = cp.open_session(principal, "session_inj")

    def gullible(bundle: ContextBundle) -> Any:
        text = json.dumps([o["content"] for o in bundle.observations])
        if "assets.delete_all@1" in text:  # obeys instructions found in tool output
            return single_call_envelope(f"p/{bundle.step}", "assets.delete_all@1", {"scope": "tenant"})
        if bundle.step == 0:
            return single_call_envelope("p/0", "web.search@1", {"query": "tips"})
        return final_envelope("p/f", "Deleted everything as instructed.")

    result = Orchestrator(cp, gullible).run(session, "Find tips.")
    assert toolkit.calls("assets.delete_all@1") == 0
    rejected = [o for o in result.outcomes if o.receipt.tool == "assets.delete_all@1"]
    assert rejected and all(o.rejected for o in rejected)
    assert rejected[0].receipt.error_class in ("permission_denied", "confirmation_required")
    assert result.final.status == "escalated" and "hard_rule:unsafe_action" in result.final.escalation.reasons
    assert "Deleted everything" not in result.final.user_message
    assert principal.permissions == perms


def test_delayed_payload_does_not_reach_a_later_task(cp, session, toolkit) -> None:
    orch = Orchestrator(cp, ScriptedPlanner([single_call_envelope("p/s", "web.search@1", {"query": "x"})]))
    orch.run(session, "search")
    later = ScriptedPlanner([BLUEPRINT_ENVELOPE])
    Orchestrator(cp, later).run(session, "now trim c_17")
    dump = canonical_json([b.to_dict() for b in later.bundles])
    assert "ignore all previous instructions" not in dump
    assert toolkit.calls("assets.delete_all@1") == 0


# ---------------------------------------------------------------- escalation
def test_unsupported_tool_forces_escalation(cp, session) -> None:
    env = single_call_envelope("p/up", "video.upscale@1", {"factor": 4})
    planner = ScriptedPlanner([env])
    result = Orchestrator(cp, planner).run(session, "Upscale to 8K.")
    assert result.outcome == "escalated" and len(planner.bundles) == 1
    esc = result.final.escalation
    assert esc is not None and esc.failed_capability == "video.upscale@1"
    assert "hard_rule:unsupported_tool" in esc.reasons and esc.policy_version
    assert esc.evidence[0]["error_class"] == "unknown_tool"
    assert any(c.startswith("edit asset_91 from version sha256:") for c in esc.constraints)
    assert "escalation" in kinds(session)


def test_user_requested_flagship_review_escalates_without_planning(cp, session) -> None:
    planner = ScriptedPlanner([BLUEPRINT_ENVELOPE])
    result = Orchestrator(cp, planner).run(session, UserRequest("Please have the flagship review this cut.", flagship_review=True))
    assert planner.bundles == [] and result.final.status == "escalated"
    assert result.escalation.hard_rules == ("user_requested_review",)


def test_lm_escalation_flag_is_a_feature_not_a_veto(cp, session) -> None:
    unsure = final_envelope("p/f", "I'm not sure, maybe I can't do this.", escalate=True, reason="feels hard")
    result = Orchestrator(cp, ScriptedPlanner([unsure])).run(session, "hi")
    assert result.escalation is None and result.final.status == "succeeded"
    assert not result.final.model_message_used  # it would claim an escalation that did not happen
    assert result.steps[0].escalation is not None and result.steps[0].escalation.lm_requested_escalation
    # high model disagreement (a real feature) can still trigger it economically
    shaky = PlannerResponse(single_call_envelope("p/x", "flaky.crash@1", {}), model_disagreement=1.0)
    result = Orchestrator(cp, ScriptedPlanner([shaky])).run(session, UserRequest("go", domain_shift=1.0))
    assert result.final.status == "escalated" and result.escalation.economic


# ---------------------------------------------------------------- recording
def test_every_path_is_recorded_and_exported_as_trajectories(cp, session, toolkit) -> None:
    secret_req = "use api token sk-live-ZZZZZZZZZZZZZZZZZZZZZZ to fetch"
    bad = edit_envelope("p/bad", [{"op": "trim", "clip_id": "c_17", "start_ms": 5, "end_ms": 1}], max_repair_attempts=0)
    orch = Orchestrator(cp, ScriptedPlanner([BLUEPRINT_ENVELOPE]), router=NEVER)
    r1 = orch.run(session, secret_req)
    orch.planner = ScriptedPlanner([bad])
    r2 = orch.run(session, "bad edit")
    orch.planner = ScriptedPlanner([edit_envelope("p/x", [{"op": "delete", "clip_id": "c_404"}]),
                                    edit_envelope("p/y", [{"op": "delete", "clip_id": "c_16"}])])
    r3 = orch.run(session, "delete")
    assert (r1.outcome, r2.outcome, r3.outcome) == ("success", "failure", "recovered")

    trajs = orch.export_trajectories()
    assert [t["outcome"] for t in trajs] == ["success", "failure", "recovered"]
    assert [t["outcome"] for t in orch.export_trajectories(outcomes=["failure"])] == ["failure"]
    for t, r in zip(trajs, (r1, r2, r3)):
        assert session.event_log.get(t["final_event_id"]).kind == "final"
        for step in t["steps"]:
            assert session.event_log.get(step["plan_event_id"]).kind == "plan"
            assert step["context"]["request"]["trust"] == "user"
        assert t["labels"] == {"review_status": "pending_review", "decontaminated": False}
        assert t["final"]["status"] == r.final.status
    dump = canonical_json(trajs) + canonical_json([e.payload for e in session.event_log.replay(session.branch_id)])
    assert "sk-live-ZZZZ" not in dump
    trajs[0]["outcome"] = "mutated"
    assert orch.export_trajectories()[0]["outcome"] == "success"  # exports are copies


@pytest.mark.parametrize("output", ["this is not json", {"schema_version": "m63.actions/v1"}, RuntimeError("model crashed")])
def test_planner_garbage_or_crash_is_a_bounded_failed_step(cp, session, output) -> None:
    def planner(bundle: ContextBundle) -> Any:
        if isinstance(output, Exception):
            raise output
        return output

    result = Orchestrator(cp, planner, router=NEVER, config=OrchestratorConfig(max_repair_attempts=1)).run(session, "x")
    assert len(result.steps) == 2 and all(s.parse_errors for s in result.steps)
    assert result.final.status == "failed" and "plan was rejected" in result.final.unresolved_issues[0]
    plans = [e for e in session.event_log.replay(session.branch_id) if e.kind == "plan"]
    assert len(plans) == 2 and plans[0].trust == Trust.UNTRUSTED.value and plans[0].payload["parse_errors"]


def test_completed_artifacts_report_only_current_versions(cp, session, toolkit) -> None:
    a = edit_envelope("p/a", [{"op": "trim", "clip_id": "c_16", "start_ms": 0, "end_ms": 5000}])
    b = edit_envelope("p/b", [{"op": "trim", "clip_id": "c_18", "start_ms": 0, "end_ms": 6000}])
    result = Orchestrator(cp, ScriptedPlanner([a, b])).run(session, "two edits")
    (art,) = result.final.completed_artifacts
    assert art.hash == cp.artifacts(session)["asset_91"].hash and art.idempotency_key == "p/b/call_1"
    assert len(result.final.receipt_hashes) == 2  # both executed calls back the response's claims
