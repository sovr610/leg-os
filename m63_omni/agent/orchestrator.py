"""The orchestration loop (blueprint §10, Appendix A ``generate``).

``intake → context bundle → plan → static validation → execute → verify →
repair/branch → finalize``

:class:`Orchestrator` drives one user request through that loop with a
pluggable *planner* — the M63 model in production, or a scripted planner in
tests — that maps a :class:`ContextBundle` to an ``m63.actions/v1`` envelope.
The orchestrator itself is deterministic runtime code; every power it
exercises goes through the :class:`~m63_omni.agent.control_plane.ControlPlane`.

Guarantees:

* **bounded** — at most ``max_steps`` planner turns, ``max_calls`` calls,
  ``max_repair_attempts`` repairs (the model's envelope budget may only lower
  these) and ``max_cost`` cost units per run;
* **untrusted output stays data** — tool output reaches the planner only as a
  trust-labelled, quarantined observation of the *current* run; it is never
  parsed for actions, never grants permissions or confirmations, and never
  carries over into later runs' context (§6 request-scoped quarantine);
* **repair/branch** — a failed step is fed back for a bounded repair; if the
  failed plan had already committed some calls, the repair starts on a new
  named branch from the pre-plan state instead of reversing anything;
* **escalation** — on failure (or a hard rule) the calibrated router decides
  between repair and handoff; the model's own flag is only a feature;
* **honest finalisation** — :class:`~m63_omni.agent.schemas.FinalResponse`
  status, completed artifacts and receipts are derived from receipts and
  verification results; the model's message is used only when everything it
  could be claiming is backed by a successful, verified receipt;
* **recorded** — every path (success, failure, recovered, escalated) is in
  the event log, and :meth:`Orchestrator.export_trajectories` returns redacted
  trajectories for later training after review (§8, §10).
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from ..common import Trust, canonical_json, content_hash, short_id
from .control_plane import BranchError, CallOutcome, ControlPlane, PreflightReport, RunBudget, Session
from .escalation import (
    EscalationContext,
    EscalationDecision,
    EscalationFeatures,
    EscalationRouter,
    constraint_complexity,
    uncertainty_from_text,
)
from .schemas import (
    SCHEMA_VERSION,
    ActionEnvelope,
    BranchRequest,
    CompletedArtifact,
    EnvelopeError,
    ErrorClass,
    EscalationRequest,
    FinalResponse,
    ReceiptStatus,
    parse_envelope,
)
from .timeline import TIMELINE_KIND, summarize_timeline

__all__ = [
    "UserRequest",
    "ContextBundle",
    "PlannerResponse",
    "Planner",
    "ScriptedPlanner",
    "final_envelope",
    "OrchestratorConfig",
    "StepRecord",
    "RunResult",
    "Orchestrator",
]

_UNSUPPORTED = {ErrorClass.UNKNOWN_TOOL.value, ErrorClass.UNKNOWN_VERSION.value}
_UNSAFE = {
    ErrorClass.PERMISSION_DENIED.value,
    ErrorClass.CONFIRMATION_REQUIRED.value,
    ErrorClass.TENANT_ISOLATION.value,
}
_TOOL_ORIGIN_ERRORS = {ErrorClass.TOOL_EXCEPTION.value, ErrorClass.INVALID_OUTPUT.value}


@dataclass(frozen=True)
class UserRequest:
    """A user turn (``USER`` trust).

    ``confirmations`` lists ``(tool_ref, arguments)`` pairs the user explicitly
    approved (needed for destructive tools). ``domain_shift`` comes from an
    upstream out-of-distribution detector; ``flagship_review`` is the user
    asking for flagship review (a hard escalation rule).
    """

    text: str
    flagship_review: bool = False
    domain_shift: float = 0.0
    expected_quality_gain: Optional[float] = None
    confirmations: tuple[tuple[str, Mapping[str, Any]], ...] = ()


@dataclass(frozen=True)
class ContextBundle:
    """What the planner sees: typed references and compact, trust-labelled summaries."""

    run_id: str
    step: int
    mode: str  # "initial" | "repair" | "continuation"
    request: Mapping[str, Any]
    tenant_id: str
    session_id: str
    branch_id: str
    plan_id_hint: str
    artifacts: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    verifiers: tuple[str, ...]
    observations: tuple[Mapping[str, Any], ...]
    remaining_budget: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            canonical_json(
                {
                    "run_id": self.run_id,
                    "step": self.step,
                    "mode": self.mode,
                    "request": dict(self.request),
                    "tenant_id": self.tenant_id,
                    "session_id": self.session_id,
                    "branch_id": self.branch_id,
                    "plan_id_hint": self.plan_id_hint,
                    "artifacts": [dict(a) for a in self.artifacts],
                    "tools": [dict(t) for t in self.tools],
                    "verifiers": list(self.verifiers),
                    "observations": [dict(o) for o in self.observations],
                    "remaining_budget": dict(self.remaining_budget),
                }
            )
        )

    @property
    def bundle_hash(self) -> str:
        return content_hash(self.to_dict())

    def artifact(self, artifact_id: str) -> Optional[Mapping[str, Any]]:
        return next((a for a in self.artifacts if a["artifact_id"] == artifact_id), None)


@dataclass(frozen=True)
class PlannerResponse:
    """Optional wrapper so a planner can report sample/draft disagreement (a router feature)."""

    output: Any
    model_disagreement: float = 0.0


Planner = Callable[[ContextBundle], Any]


def final_envelope(plan_id: str, message: Optional[str], *, escalate: bool = False, reason: Optional[str] = None) -> dict[str, Any]:
    """A zero-call envelope: the planner's way of saying "finalize"."""
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan_id,
        "assistant_message": message,
        "calls": [],
        "budget": {"max_calls": 0, "max_repair_attempts": 0},
        "escalation": {"required": escalate, "reason": reason},
    }


class ScriptedPlanner:
    """Deterministic planner for tests and demos.

    Each script item is an envelope (dict / JSON text / ``ActionEnvelope``),
    a :class:`PlannerResponse`, or a callable ``bundle -> item``. When the
    script is exhausted it finalises with ``final_message``. Every bundle it
    receives is kept in :attr:`bundles` for inspection.
    """

    def __init__(self, script: Sequence[Any], *, final_message: Optional[str] = "Done.") -> None:
        self.script = list(script)
        self.final_message = final_message
        self.bundles: list[ContextBundle] = []
        self._i = 0

    def __call__(self, bundle: ContextBundle) -> Any:
        self.bundles.append(bundle)
        if self._i < len(self.script):
            item = self.script[self._i]
            self._i += 1
            return item(bundle) if callable(item) else copy.deepcopy(item)
        return final_envelope(bundle.plan_id_hint, self.final_message)


@dataclass(frozen=True)
class OrchestratorConfig:
    max_steps: int = 8
    max_calls: int = 16
    max_repair_attempts: int = 2
    max_cost: float = 32.0
    artifact_summarizers: Mapping[str, Callable[[Any], Mapping[str, Any]]] = field(
        default_factory=lambda: {TIMELINE_KIND: summarize_timeline}
    )


@dataclass(frozen=True)
class StepRecord:
    step: int
    mode: str
    context: Mapping[str, Any]
    envelope: Optional[ActionEnvelope]
    raw_output_hash: str
    parse_errors: tuple[str, ...]
    preflight_errors: tuple[str, ...]
    outcomes: tuple[CallOutcome, ...]
    plan_event_id: str
    is_final: bool = False
    escalation: Optional[EscalationDecision] = None
    branch_created: Optional[str] = None
    model_disagreement: float = 0.0

    @property
    def succeeded(self) -> bool:
        """All proposed calls executed and verified (False for rejected/unparseable plans)."""
        if self.envelope is None or self.parse_errors or self.preflight_errors:
            return False
        return len(self.outcomes) == len(self.envelope.calls) and all(o.succeeded for o in self.outcomes)

    @property
    def failed(self) -> bool:
        return not self.is_final and not self.succeeded

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "mode": self.mode,
            "context": dict(self.context),
            "envelope": self.envelope.to_dict() if self.envelope else None,
            "raw_output_hash": self.raw_output_hash,
            "parse_errors": list(self.parse_errors),
            "preflight_errors": list(self.preflight_errors),
            "calls": [o.to_dict() for o in self.outcomes],
            "plan_event_id": self.plan_event_id,
            "is_final": self.is_final,
            "succeeded": self.succeeded,
            "escalation": self.escalation.to_dict() if self.escalation else None,
            "branch_created": self.branch_created,
            "model_disagreement": self.model_disagreement,
        }


@dataclass(frozen=True)
class RunResult:
    run_id: str
    final: FinalResponse
    outcome: str  # "success" | "recovered" | "failure" | "escalated"
    steps: tuple[StepRecord, ...]
    budget: Mapping[str, float]
    branch_id: str
    escalation: Optional[EscalationDecision]
    final_event_id: str

    @property
    def outcomes(self) -> list[CallOutcome]:
        return [o for s in self.steps for o in s.outcomes]


class Orchestrator:
    """Runs user requests through the §10 loop against a control plane and a planner."""

    def __init__(
        self,
        control_plane: ControlPlane,
        planner: Planner,
        *,
        router: Optional[EscalationRouter] = None,
        config: OrchestratorConfig = OrchestratorConfig(),
    ) -> None:
        self.cp = control_plane
        self.planner = planner
        self.router = router if router is not None else EscalationRouter()
        self.config = config
        self._trajectories: list[dict[str, Any]] = []

    # ================================================================== run
    def run(self, session: Session, request: UserRequest | str) -> RunResult:
        req = UserRequest(request) if isinstance(request, str) else request
        cfg = self.config
        run_id = "run_" + short_id(session.tenant_id, session.session_id, len(session.event_log), req.text, n=12)
        budget = RunBudget(cfg.max_calls, min(cfg.max_repair_attempts, self.cp.limits.max_repair_attempts), cfg.max_cost)
        start_branch = session.branch_id

        # ---- intake (USER trust); confirmations come only from the user turn
        intake = session.event_log.append_ingest(
            segment_hash=content_hash(req.text),
            asset_refs=[],
            trust=Trust.USER,
            branch_id=session.branch_id,
            extra=self.cp.redactor.redact(
                {"type": "user_request", "run_id": run_id, "text": req.text, "flagship_review": req.flagship_review}
            ),
        )
        for tool_ref, args in req.confirmations:
            self.cp.confirm(session, tool_ref, args)

        steps: list[StepRecord] = []
        observations: list[dict[str, Any]] = []
        decision: Optional[EscalationDecision] = None
        final_message: Optional[str] = None

        if req.flagship_review:
            decision = self.router.decide(
                EscalationContext(
                    EscalationFeatures(domain_shift=req.domain_shift),
                    user_requested_review=True,
                    expected_quality_gain=req.expected_quality_gain,
                )
            )
        else:
            for step_no in range(cfg.max_steps):
                mode = "initial" if not steps else ("repair" if steps[-1].failed else "continuation")
                bundle = self._bundle(session, run_id, step_no, mode, req, observations, budget)
                record = self._plan_and_execute(session, run_id, step_no, mode, bundle, budget)
                steps.append(record)
                env = record.envelope
                if record.is_final:
                    final_message = env.assistant_message if env else None
                    if env is not None and env.escalation.required:
                        decision = self._decide(req, steps, record, budget)
                        steps[-1] = _with(record, escalation=decision)
                    break
                observations = self._observations(record)
                if record.succeeded and not (env is not None and env.escalation.required):
                    continue
                decision = self._decide(req, steps, record, budget)
                steps[-1] = record = _with(record, escalation=decision)
                if decision.escalate:
                    break
                if record.succeeded:  # model asked to escalate but the policy declined; carry on
                    decision = None
                    continue
                if budget.repairs_left <= 0:
                    break
                budget.repairs_used += 1
                branch = self._maybe_branch_for_repair(session, run_id, record, budget)
                if branch:
                    steps[-1] = _with(record, branch_created=branch)
                decision = None

        escalation_req = None
        if decision is not None and decision.escalate:
            escalation_req = self._escalation_request(session, req, steps, decision, budget)
            self.cp.record(session, "escalation", {"run_id": run_id, **escalation_req.to_dict(), "decision": decision.to_dict()})
        final = self._finalize(session, steps, final_message, decision, escalation_req)
        final_ev = self.cp.record(session, "final", {"run_id": run_id, **final.to_dict()})

        if final.status == "escalated":
            outcome = "escalated"
        elif final.status == "succeeded":
            outcome = "recovered" if any(s.failed for s in steps) else "success"
        else:
            outcome = "failure"
        result = RunResult(
            run_id, final, outcome, tuple(steps), budget.to_dict(), session.branch_id,
            decision if decision is not None and decision.escalate else None, final_ev.event_id,
        )
        self._trajectories.append(self._trajectory(session, req, result, start_branch, intake.event_id))
        return result

    # ============================================================ one step
    def _plan_and_execute(
        self, session: Session, run_id: str, step_no: int, mode: str, bundle: ContextBundle, budget: RunBudget
    ) -> StepRecord:
        raw: Any = None
        disagreement = 0.0
        parse_errors: list[str] = []
        env: Optional[ActionEnvelope] = None
        try:
            raw = self.planner(bundle)
            if isinstance(raw, PlannerResponse):
                disagreement = float(raw.model_disagreement)
                raw = raw.output
            env = parse_envelope(raw)
        except EnvelopeError as exc:
            parse_errors = list(exc.errors)
        except Exception as exc:  # a crashing planner is a failed step, never a crashed loop
            parse_errors = [f"planner raised {type(exc).__name__}: {exc}"]
        try:
            raw_hash = content_hash(raw if isinstance(raw, (str, bytes)) else json.loads(canonical_json(raw)))
        except (TypeError, ValueError):
            raw_hash = content_hash(repr(raw))

        plan_ev = self.cp.record(
            session,
            "plan",
            {
                "type": "plan",
                "run_id": run_id,
                "step": step_no,
                "mode": mode,
                "bundle_hash": bundle.bundle_hash,
                "raw_output_hash": raw_hash,
                "envelope": env.to_dict() if env else None,
                "parse_errors": parse_errors,
            },
            trust=Trust.UNTRUSTED,  # a model proposal carries no authority until validated
        )
        common = dict(
            step=step_no, mode=mode, context=bundle.to_dict(), envelope=env, raw_output_hash=raw_hash,
            parse_errors=tuple(parse_errors), plan_event_id=plan_ev.event_id, model_disagreement=disagreement,
        )
        if env is None:
            return StepRecord(preflight_errors=(), outcomes=(), **common)
        if not env.calls:
            return StepRecord(preflight_errors=(), outcomes=(), is_final=True, **common)

        budget.tighten(max_repair_attempts=env.budget.max_repair_attempts)
        report: PreflightReport = self.cp.preflight(session, env, budget=budget)
        outcomes: list[CallOutcome] = [r for r in report.rejections if r is not None]
        if report.ok:
            for call in env.calls:
                outcome = self.cp.execute_call(session, call, plan_id=env.plan_id, budget=budget)
                outcomes.append(outcome)
                if not outcome.succeeded:
                    break  # later calls may depend on this one; stop and repair
        return StepRecord(preflight_errors=tuple(report.errors()), outcomes=tuple(outcomes), **common)

    # ============================================================== context
    def _bundle(
        self,
        session: Session,
        run_id: str,
        step_no: int,
        mode: str,
        req: UserRequest,
        observations: Sequence[Mapping[str, Any]],
        budget: RunBudget,
    ) -> ContextBundle:
        artifacts = []
        for aid, ref in sorted(self.cp.artifacts(session).items()):
            kind, content = self.cp.store.get(session.tenant_id, ref.hash)
            summarizer = self.config.artifact_summarizers.get(kind)
            summary = summarizer(content) if summarizer else {"note": "content available by reference only"}
            artifacts.append(
                {**ref.to_dict(), "summary": summary, "quarantined": ref.trust == Trust.UNTRUSTED.value}
            )
        return ContextBundle(
            run_id=run_id,
            step=step_no,
            mode=mode,
            request={"text": req.text, "trust": Trust.USER.value},
            tenant_id=session.tenant_id,
            session_id=session.session_id,
            branch_id=session.branch_id,
            plan_id_hint=f"{session.session_id}/{session.branch_id}/{run_id}/plan_{step_no}",
            artifacts=tuple(artifacts),
            tools=tuple(self.cp.tool_manifest()),
            verifiers=self.cp.verifiers.names(),
            observations=tuple(copy.deepcopy(list(observations))),
            remaining_budget=budget.remaining(),
        )

    def _observations(self, record: StepRecord) -> list[dict[str, Any]]:
        """Feedback for the next step. Tool-originated content is labelled and quarantined."""
        obs: list[dict[str, Any]] = []
        sys_label = {"trust": Trust.SYSTEM.value, "quarantined": False}
        if record.parse_errors:
            obs.append({"source": "control_plane", **sys_label, "content": {"type": "plan_rejected", "errors": list(record.parse_errors)}})
        elif record.preflight_errors:
            obs.append({"source": "control_plane", **sys_label, "content": {"type": "plan_rejected", "errors": list(record.preflight_errors)}})
        for o in record.outcomes:
            r = o.receipt
            executed = r.status != ReceiptStatus.REJECTED.value
            tool_origin_error = executed and r.error_class in _TOOL_ORIGIN_ERRORS
            obs.append(
                {
                    "source": f"control_plane:{r.receipt_hash}",
                    **sys_label,
                    "content": {
                        "type": "call_status",
                        "tool": r.tool,
                        "idempotency_key": r.idempotency_key,
                        "status": r.status,
                        "error_class": r.error_class,
                        "error_message": None if tool_origin_error else r.error_message,
                        "verified": o.succeeded,
                        "cached": o.cached,
                        "checks": [{"probe": v.probe, "passed": v.passed, "evidence": dict(v.evidence)} for v in o.verifications],
                        "artifact_refs": [dict(a) for a in r.artifact_refs],
                    },
                }
            )
            if executed and (r.output or tool_origin_error):
                obs.append(
                    {
                        "source": f"tool:{r.tool}:{r.receipt_hash}",
                        "trust": r.trust,
                        "quarantined": r.trust == Trust.UNTRUSTED.value,
                        "content": {
                            "type": "tool_output",
                            "data": dict(r.output),
                            "error_message": r.error_message if tool_origin_error else None,
                        },
                    }
                )
        return obs

    # ===================================================== repair / branch
    def _maybe_branch_for_repair(self, session: Session, run_id: str, record: StepRecord, budget: RunBudget) -> Optional[str]:
        """If the failed plan committed some calls, repair from the pre-plan state on a new branch."""
        if not any(o.committed and not o.cached for o in record.outcomes):
            return None
        plan_ev = session.event_log.get(record.plan_event_id, tenant_id=session.tenant_id)
        if plan_ev.parent_id is None:
            return None
        expected = {aid: ref.hash for aid, ref in self.cp.artifacts_at(session, plan_ev.parent_id).items()}
        try:
            return self.cp.create_branch(
                session,
                BranchRequest(
                    parent_event=plan_ev.parent_id,
                    reason=f"repair {budget.repairs_used} after failed plan "
                    f"{record.envelope.plan_id if record.envelope else record.plan_event_id}",
                    name=f"{session.branch_id}.{run_id}-repair{budget.repairs_used}",
                    expected_state=expected,
                ),
            )
        except BranchError:
            return None

    def undo(self, session: Session, *, steps: int = 1, name: Optional[str] = None, reason: str = "user undo") -> str:
        """Undo the last ``steps`` committed edits by switching to a new named branch."""
        return self.cp.undo(session, steps=steps, name=name, reason=reason)

    # ========================================================== escalation
    def _decide(self, req: UserRequest, steps: Sequence[StepRecord], record: StepRecord, budget: RunBudget) -> EscalationDecision:
        checks = failed = 0
        for s in steps:
            if s.parse_errors or (s.preflight_errors and not s.outcomes):
                checks += 1
                failed += 1
            for o in s.outcomes:
                if not o.receipt.ok:
                    checks += 1
                    failed += 1
                for v in o.verifications:
                    if not v.skipped:
                        checks += 1
                        failed += int(not v.passed)
        env = record.envelope
        n_ops = n_verifiers = n_calls = 0
        if env is not None:
            n_calls = len(env.calls)
            for c in env.calls:
                ops = c.arguments.get("operations")
                n_ops += len(ops) if isinstance(ops, list) else 0
                n_verifiers += len(c.verify)
        text = " ".join(t for t in ((env.assistant_message if env else None), (env.escalation.reason if env else None)) if t)
        classes = {o.receipt.error_class for o in record.outcomes if o.receipt.error_class}
        features = EscalationFeatures(
            verifier_failure_rate=failed / checks if checks else 0.0,
            constraint_complexity=constraint_complexity(n_ops, n_verifiers, n_calls),
            domain_shift=req.domain_shift,
            repair_failures=float(sum(1 for s in steps if s.mode == "repair" and s.failed)),
            model_disagreement=record.model_disagreement,
            lm_uncertainty=uncertainty_from_text(text),
        )
        return self.router.decide(
            EscalationContext(
                features,
                unsupported_tool=bool(classes & _UNSUPPORTED),
                unsafe_action=bool(classes & _UNSAFE),
                user_requested_review=req.flagship_review,
                expected_quality_gain=req.expected_quality_gain,
                lm_requested_escalation=bool(env and env.escalation.required),
            )
        )

    def _escalation_request(
        self, session: Session, req: UserRequest, steps: Sequence[StepRecord], decision: EscalationDecision, budget: RunBudget
    ) -> EscalationRequest:
        evidence: list[dict[str, Any]] = []
        failed_capability = "user_requested_review" if "user_requested_review" in decision.hard_rules else "planning"
        for s in steps:
            if s.parse_errors:
                evidence.append({"step": s.step, "plan_rejected": list(s.parse_errors)[:5]})
            for o in s.outcomes:
                if o.succeeded:
                    continue
                if failed_capability == "planning":
                    failed_capability = o.receipt.tool
                evidence.append(
                    {
                        "step": s.step,
                        "tool": o.receipt.tool,
                        "receipt_hash": o.receipt.receipt_hash,
                        "status": o.receipt.status,
                        "error_class": o.receipt.error_class,
                        "failed_checks": [v.probe for v in o.verifications if not v.passed],
                    }
                )
        refs = [ref.to_dict() for _, ref in sorted(self.cp.artifacts(session).items())]
        constraints = [f"edit {r['artifact_id']} from version {r['hash']}" for r in refs]
        constraints += ["use only tools in the registered manifest", "do not repeat a failed call with identical arguments"]
        return EscalationRequest(
            failed_capability=failed_capability,
            evidence=tuple(evidence),
            attempted_repairs=budget.repairs_used,
            remaining_budget=budget.remaining(),
            constraints=tuple(constraints),
            reasons=decision.reasons,
            p_fail=decision.p_fail,
            policy_version=decision.policy_version,
            context_refs=tuple([{"request": req.text}] + refs),
        )

    # ============================================================ finalize
    def _finalize(
        self,
        session: Session,
        steps: Sequence[StepRecord],
        model_message: Optional[str],
        decision: Optional[EscalationDecision],
        escalation_req: Optional[EscalationRequest],
    ) -> FinalResponse:
        action_steps = [s for s in steps if not s.is_final]
        last_ok = max((i for i, s in enumerate(action_steps) if s.succeeded), default=-1)
        lineage = {e.event_id for e in session.event_log.replay(session.branch_id)}
        current = {aid: ref.hash for aid, ref in self.cp.artifacts(session).items()}

        summary: list[dict[str, Any]] = []
        issues: list[str] = []
        completed: dict[str, CompletedArtifact] = {}
        backing: list[str] = []
        for i, s in enumerate(action_steps):
            resolved = s.succeeded or i < last_ok
            if s.parse_errors:
                summary.append({"step": s.step, "plan_rejected": True, "errors": list(s.parse_errors)[:5], "resolved": resolved})
                if not resolved:
                    issues.append(f"step {s.step}: the plan was rejected ({s.parse_errors[0]})")
            elif s.preflight_errors and not s.outcomes:
                summary.append({"step": s.step, "plan_rejected": True, "errors": list(s.preflight_errors)[:5], "resolved": resolved})
                if not resolved:
                    issues.append(f"step {s.step}: the plan was rejected ({s.preflight_errors[0]})")
            for o in s.outcomes:
                r = o.receipt
                summary.append(
                    {
                        "step": s.step,
                        "plan_id": r.plan_id,
                        "tool": r.tool,
                        "idempotency_key": r.idempotency_key,
                        "receipt_hash": r.receipt_hash,
                        "status": r.status,
                        "error_class": r.error_class,
                        "verified": o.succeeded,
                        "cached": o.cached,
                        "checks": {v.probe: v.passed for v in o.verifications},
                        "resolved": o.succeeded or resolved,
                    }
                )
                if o.succeeded:
                    in_lineage = [c for c in o.committed if c.event_id in lineage]
                    if in_lineage or not o.committed:
                        backing.append(r.receipt_hash)
                    for c in in_lineage:
                        if current.get(c.artifact_id) == c.hash:  # only versions still current on this branch
                            completed[c.artifact_id] = CompletedArtifact(
                                c.artifact_id, c.kind, c.hash, r.receipt_hash, r.tool, r.idempotency_key
                            )
                elif not resolved:
                    issues.append(self._describe_failure(o))
            if s.envelope is not None and not s.parse_errors and not s.preflight_errors:
                skipped = len(s.envelope.calls) - len(s.outcomes)
                if skipped > 0 and not resolved:
                    issues.append(f"step {s.step}: {skipped} later call(s) were not executed after a failure")

        escalated = decision is not None and decision.escalate
        if escalated:
            issues.append("escalated for flagship review: " + ", ".join(decision.reasons))
        if escalated:
            status = "escalated"
        elif not issues:
            status = "succeeded"
        elif completed:
            status = "partial"
        else:
            status = "failed"

        last_env = steps[-1].envelope if steps else None
        model_asked_escalation = bool(last_env and last_env.escalation.required)
        use_model = status == "succeeded" and bool(model_message) and not model_asked_escalation
        message = model_message if use_model else self._grounded_message(status, completed, issues)

        final = FinalResponse(
            user_message=message or "",
            status=status,
            completed_artifacts=tuple(completed.values()),
            verification_summary=tuple(summary),
            unresolved_issues=tuple(issues),
            plan_ids=tuple(dict.fromkeys(s.envelope.plan_id for s in steps if s.envelope)),
            receipt_hashes=tuple(dict.fromkeys(backing)),
            escalation=escalation_req,
            model_message_used=use_model,
        )
        # Belt and braces: re-check grounding against the signed receipts themselves.
        receipts = {o.receipt.receipt_hash: o.receipt for s in steps for o in s.outcomes}
        verified = {o.receipt.receipt_hash for s in steps for o in s.outcomes if o.succeeded}
        violations = final.grounding_violations(receipts, verified, self.cp.verify_receipt)
        if violations:
            final = FinalResponse(
                user_message=self._grounded_message("failed", {}, list(issues) + violations),
                status="escalated" if escalated else "failed",
                verification_summary=final.verification_summary,
                unresolved_issues=tuple(issues + violations),
                plan_ids=final.plan_ids,
                escalation=escalation_req,
            )
        return final

    @staticmethod
    def _describe_failure(o: CallOutcome) -> str:
        r = o.receipt
        if r.status != ReceiptStatus.OK.value:
            detail = f"{r.status} ({r.error_class})"
            if r.status == ReceiptStatus.REJECTED.value and r.error_message:
                detail += f": {r.error_message}"
            return f"{r.tool} [{r.idempotency_key}] {detail}"
        failed = [v.probe for v in o.verifications if not v.passed]
        return f"{r.tool} [{r.idempotency_key}] executed but failed verification: {', '.join(failed)}"

    @staticmethod
    def _grounded_message(status: str, completed: Mapping[str, CompletedArtifact], issues: Sequence[str]) -> str:
        done = ", ".join(f"{a.artifact_id} (version {a.hash[:19]}…, via {a.tool})" for a in completed.values())
        if status == "succeeded":
            return f"Completed and verified: {done}." if done else "Completed; no changes were required."
        head = {
            "partial": "I completed only part of the request.",
            "failed": "I could not complete the request.",
            "escalated": "I have escalated this request for flagship review.",
        }.get(status, "Request finished.")
        parts = [head]
        if done:
            parts.append(f"Verified and applied: {done}.")
        else:
            parts.append("No changes were applied.")
        if issues:
            parts.append("Unresolved: " + "; ".join(issues) + ".")
        return " ".join(parts)

    # ========================================================= trajectories
    def _trajectory(
        self, session: Session, req: UserRequest, result: RunResult, start_branch: str, intake_event: str
    ) -> dict[str, Any]:
        traj = {
            "run_id": result.run_id,
            "tenant_id": session.tenant_id,
            "session_id": session.session_id,
            "request": {"text": req.text, "flagship_review": req.flagship_review, "domain_shift": req.domain_shift},
            "intake_event_id": intake_event,
            "branch_start": start_branch,
            "branch_end": result.branch_id,
            "outcome": result.outcome,
            "steps": [s.to_dict() for s in result.steps],
            "final": result.final.to_dict(),
            "final_event_id": result.final_event_id,
            "escalation": result.escalation.to_dict() if result.escalation else None,
            "budget": dict(result.budget),
            "policy_version": self.router.policy.version,
            "labels": {"review_status": "pending_review", "decontaminated": False},
        }
        return self.cp.redactor.redact(json.loads(canonical_json(traj)))

    def export_trajectories(self, *, outcomes: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
        """Redacted trajectories (success, failure, recovered, escalated) for later training data."""
        return [copy.deepcopy(t) for t in self._trajectories if outcomes is None or t["outcome"] in outcomes]


def _with(record: StepRecord, **changes: Any) -> StepRecord:
    return dataclasses.replace(record, **changes)
