"""The deterministic control plane (blueprint §2, §6, §10, §13).

The model "must not directly execute code, mutate an asset store, authorize a
purchase, or bypass a policy check. A deterministic control plane owns those
powers" (§2). :class:`ControlPlane` is that component. For every proposed
:class:`~m63_omni.agent.schemas.ActionCall` it:

1. **isolates tenants** — the calling principal must belong to the session's
   tenant; a cross-tenant attempt is rejected and is *not* written into the
   victim tenant's log (it is recorded in :attr:`ControlPlane.security_events`);
2. **validates statically, before execution** — tool ``name@version`` exists,
   arguments match the tool's JSON schema (unknown fields rejected), the
   principal holds the tool's permissions, the timeout class and argument size
   are within limits, requested verifiers exist, destructive tools carry a
   user-issued confirmation bound to the exact arguments, referenced
   artifacts exist *in this tenant's session*, ``expected_input_hashes`` match
   the current state, the tool's deterministic precheck (e.g. contradictory
   timeline edits) passes, and the run budget can pay for the call;
3. **enforces idempotency** — the same key with the same tool+arguments in the
   current branch lineage returns the recorded receipt without re-executing;
   the same key with different arguments is rejected;
4. **executes in a sandbox** — a daemon worker thread with a timeout taken
   from the call's timeout class, a deep copy of the arguments, and a
   read-only :class:`~m63_omni.agent.tools.ToolContext`; exceptions and
   timeouts become *failed receipts* with an error class and never propagate
   (a production deployment would use process/container isolation; the
   contract is the same);
5. **issues receipts** — secrets redacted, hashed, HMAC-signed, labelled with
   the tool's output trust (``untrusted`` unless the spec says otherwise) and
   appended to the event log;
6. **verifies** — tool-mandated plus requested verifiers, deterministic first;
   only if every check passes are the outputs *committed* (a ``verification``
   event that moves the artifact head on the current branch);
7. **undoes by branching** — :meth:`undo`/:meth:`create_branch` create or
   restore a named event-log branch instead of destructively reversing state.

Instructions that appear in tool output are data: nothing here parses tool
output for actions, and permissions/confirmations can only be granted through
:class:`Principal` configuration and :meth:`confirm` by a user-trust principal.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from ..common import Trust, canonical_json, content_hash, short_id
from ..runtime.event_log import MAIN_BRANCH, Event, EventLog, EventLogError, TenantIsolationError
from .artifacts import ArtifactNotFound, ArtifactRef, ArtifactStore
from .jsonschema_lite import json_value_errors, validate
from .schemas import (
    ActionCall,
    ActionEnvelope,
    BranchRequest,
    ErrorClass,
    ReceiptStatus,
    ToolReceipt,
    ToolRefError,
    VerificationResult,
)
from .security import Redactor, ReceiptSigner
from .tools import ArtifactOutput, ToolContext, ToolRegistry, ToolResult, ToolSpec, UnknownToolError, UnknownToolVersionError
from .verifiers import VerificationContext, VerifierRegistry, default_verifier_registry

__all__ = [
    "DEFAULT_TIMEOUT_CLASSES",
    "Principal",
    "ControlPlaneLimits",
    "RunBudget",
    "Session",
    "CallOutcome",
    "PreflightReport",
    "AuditReport",
    "BranchError",
    "ControlPlane",
]

#: Timeout class → seconds. Calls may request a class no longer than their tool's.
DEFAULT_TIMEOUT_CLASSES: dict[str, float] = {
    "interactive": 2.0,
    "standard_media_edit": 10.0,
    "render_probe": 30.0,
    "long_render": 600.0,
}


_BRANCH_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}")


class BranchError(RuntimeError):
    """A branch/undo request is invalid (unknown parent, state mismatch, nothing to undo)."""


@dataclass(frozen=True)
class Principal:
    """An authenticated caller. Permissions come from control-plane configuration only."""

    principal_id: str
    tenant_id: str
    permissions: frozenset[str] = frozenset()
    trust: Trust = Trust.USER

    def __post_init__(self) -> None:
        object.__setattr__(self, "permissions", frozenset(self.permissions))
        object.__setattr__(self, "trust", Trust(self.trust))


@dataclass(frozen=True)
class ControlPlaneLimits:
    """Policy ceilings. Model-proposed budgets may only be tighter than these."""

    max_calls_per_envelope: int = 8
    max_repair_attempts: int = 3
    max_argument_bytes: int = 64_000
    max_output_bytes: int = 2_000_000
    max_error_chars: int = 400


class RunBudget:
    """Mutable per-run budget: calls, repair attempts, and cost units (§10 "fixed cost budget")."""

    def __init__(self, max_calls: int, max_repair_attempts: int, max_cost: float) -> None:
        self.max_calls = int(max_calls)
        self.max_repair_attempts = int(max_repair_attempts)
        self.max_cost = float(max_cost)
        self.calls_used = 0
        self.repairs_used = 0
        self.cost_used = 0.0
        self._lock = threading.Lock()

    def tighten(self, *, max_calls: Optional[int] = None, max_repair_attempts: Optional[int] = None) -> None:
        """Lower (never raise) limits, e.g. to the model's own envelope budget."""
        with self._lock:
            if max_calls is not None:
                self.max_calls = min(self.max_calls, int(max_calls))
            if max_repair_attempts is not None:
                self.max_repair_attempts = min(self.max_repair_attempts, int(max_repair_attempts))

    def denial(self, cost: float) -> Optional[str]:
        """Reason the next call cannot be paid for, or None."""
        if self.calls_used >= self.max_calls:
            return f"call budget exhausted ({self.calls_used}/{self.max_calls})"
        if self.cost_used + cost > self.max_cost + 1e-9:
            return f"cost budget exceeded ({self.cost_used:g} + {cost:g} > {self.max_cost:g})"
        return None

    def charge(self, cost: float) -> None:
        with self._lock:
            self.calls_used += 1
            self.cost_used += float(cost)

    @property
    def repairs_left(self) -> int:
        return max(0, self.max_repair_attempts - self.repairs_used)

    def remaining(self) -> dict[str, float]:
        return {
            "calls": max(0, self.max_calls - self.calls_used),
            "repair_attempts": self.repairs_left,
            "cost": max(0.0, self.max_cost - self.cost_used),
        }

    def to_dict(self) -> dict[str, float]:
        return {
            "max_calls": self.max_calls,
            "max_repair_attempts": self.max_repair_attempts,
            "max_cost": self.max_cost,
            "calls_used": self.calls_used,
            "repairs_used": self.repairs_used,
            "cost_used": self.cost_used,
        }


class Session:
    """A tenant-scoped editing session: owner principal, event log, current branch."""

    def __init__(self, principal: Principal, event_log: EventLog, branch_id: str = MAIN_BRANCH) -> None:
        if principal.tenant_id != event_log.tenant_id:
            raise TenantIsolationError(
                f"principal of tenant {principal.tenant_id!r} cannot own a session of tenant {event_log.tenant_id!r}"
            )
        self.principal = principal
        self.event_log = event_log
        self.branch_id = branch_id
        self.lock = threading.RLock()
        self._confirmations: set[str] = set()

    @property
    def tenant_id(self) -> str:
        return self.event_log.tenant_id

    @property
    def session_id(self) -> str:
        return self.event_log.session_id


@dataclass(frozen=True)
class CallOutcome:
    """Receipt + verification results + committed artifact versions for one call."""

    receipt: ToolReceipt
    verifications: tuple[VerificationResult, ...] = ()
    committed: tuple[ArtifactRef, ...] = ()
    cached: bool = False
    receipt_event_id: Optional[str] = None
    verification_event_id: Optional[str] = None

    @property
    def verified(self) -> bool:
        return self.receipt.ok and all(v.passed for v in self.verifications)

    @property
    def succeeded(self) -> bool:
        """Executed OK *and* every verification passed — the only basis for claiming success."""
        return self.verified

    @property
    def rejected(self) -> bool:
        return self.receipt.status == ReceiptStatus.REJECTED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.to_dict(),
            "verifications": [v.to_dict() for v in self.verifications],
            "committed": [c.to_dict() for c in self.committed],
            "cached": self.cached,
            "succeeded": self.succeeded,
            "receipt_event_id": self.receipt_event_id,
            "verification_event_id": self.verification_event_id,
        }


@dataclass(frozen=True)
class PreflightReport:
    """Static validation of a whole envelope before any call executes."""

    envelope_errors: tuple[str, ...] = ()
    rejections: tuple[Optional[CallOutcome], ...] = ()

    @property
    def ok(self) -> bool:
        return not self.envelope_errors and all(r is None for r in self.rejections)

    def errors(self) -> list[str]:
        out = list(self.envelope_errors)
        for r in self.rejections:
            if r is not None:
                out.append(f"{r.receipt.idempotency_key}: {r.receipt.error_class}: {r.receipt.error_message}")
        return out


@dataclass(frozen=True)
class AuditReport:
    chain_ok: bool
    receipts_checked: int
    bad_receipts: tuple[str, ...] = ()
    foreign_receipts: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.chain_ok and not self.bad_receipts and not self.foreign_receipts


class _Rejection(Exception):
    def __init__(self, error_class: ErrorClass, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.message = message


@dataclass
class _Prepared:
    spec: ToolSpec
    arguments: dict
    args_hash: str
    readable: dict[str, tuple[str, str, Any]] = field(default_factory=dict)
    write_ids: frozenset[str] = frozenset()


class ControlPlane:
    """Validates, executes, signs, logs, verifies and commits tool calls (see module doc)."""

    def __init__(
        self,
        registry: ToolRegistry,
        verifiers: Optional[VerifierRegistry] = None,
        *,
        secret: bytes,
        key_id: str = "cp-k1",
        limits: ControlPlaneLimits = ControlPlaneLimits(),
        timeout_classes: Optional[Mapping[str, float]] = None,
        store: Optional[ArtifactStore] = None,
        clock: Callable[[], float] = time.time,
        redactor: Optional[Redactor] = None,
    ) -> None:
        self.registry = registry
        self.verifiers = verifiers if verifiers is not None else default_verifier_registry()
        self.limits = limits
        self.timeout_classes = dict(timeout_classes if timeout_classes is not None else DEFAULT_TIMEOUT_CLASSES)
        self.store = store if store is not None else ArtifactStore()
        self._clock = clock
        self._signer = ReceiptSigner(secret, key_id)
        self.redactor = redactor if redactor is not None else Redactor()
        for s in self._signer.secrets:
            self.redactor.register_secret(s)
        self._counter = itertools.count()
        self.security_events: list[dict[str, Any]] = []

    # ================================================================ sessions
    def open_session(
        self,
        principal: Principal,
        session_id: str,
        *,
        path: str | Path | None = None,
        event_log: Optional[EventLog] = None,
    ) -> Session:
        """Create (or resume, when ``event_log`` is given) a session owned by ``principal``."""
        log = event_log if event_log is not None else EventLog(principal.tenant_id, session_id, path=path, clock=self._clock)
        session = Session(principal, log)
        if len(log) == 0:
            log.append(
                "session_start",
                {"principal_id": principal.principal_id, "tools": list(self.registry.refs())},
                trust=Trust.SYSTEM,
            )
        return session

    def record(self, session: Session, kind: str, payload: Mapping[str, Any], *, trust: Trust | str = Trust.SYSTEM) -> Event:
        """Append a redacted event on the session's current branch (for the orchestrator loop)."""
        return session.event_log.append(kind, self.redactor.redact(dict(payload)), branch_id=session.branch_id, trust=trust)

    # =============================================================== artifacts
    def ingest_artifact(
        self,
        session: Session,
        artifact_id: str,
        kind: str,
        content: Any,
        *,
        trust: Trust | str = Trust.USER,
        principal: Optional[Principal] = None,
    ) -> ArtifactRef:
        """Store content and append an ``ingest`` event making it the current version on the branch.

        The trust label may not exceed the ingesting principal's own trust: content
        can never be promoted into a higher-trust role (§6).
        """
        principal = principal or session.principal
        self._require_same_tenant(session, principal, action="ingest")
        trust = Trust(trust)
        if not principal.trust.at_least(trust):
            raise PermissionError(f"principal with trust {principal.trust.value} cannot ingest {trust.value} content")
        errs = json_value_errors(content)
        if errs:
            raise ValueError("artifact content is not JSON: " + "; ".join(errs))
        h = self.store.put(session.tenant_id, kind, content)
        ev = session.event_log.append_ingest(
            segment_hash=h,
            asset_refs=[artifact_id],
            trust=trust,
            branch_id=session.branch_id,
            extra={"artifact_id": artifact_id, "artifact_kind": kind, "artifact_hash": h},
        )
        return ArtifactRef(artifact_id, kind, h, trust.value, ev.event_id)

    @staticmethod
    def _heads_from(events: Iterable[Event]) -> dict[str, ArtifactRef]:
        heads: dict[str, ArtifactRef] = {}
        for ev in events:
            p = ev.payload
            if ev.kind == "ingest" and "artifact_id" in p:
                heads[p["artifact_id"]] = ArtifactRef(p["artifact_id"], p["artifact_kind"], p["artifact_hash"], ev.trust, ev.event_id)
            elif ev.kind == "verification" and p.get("passed") and p.get("committed"):
                for c in p["committed"]:
                    heads[c["artifact_id"]] = ArtifactRef(c["artifact_id"], c["kind"], c["hash"], c["trust"], ev.event_id)
        return heads

    def artifacts(self, session: Session, branch_id: Optional[str] = None) -> dict[str, ArtifactRef]:
        """Current (committed) artifact versions on a branch, derived from the event log."""
        return self._heads_from(session.event_log.replay(branch_id or session.branch_id))

    def artifacts_at(self, session: Session, event_id: str) -> dict[str, ArtifactRef]:
        """Artifact versions as of a committed event (used to check branch expectations)."""
        session.event_log.get(event_id, tenant_id=session.tenant_id)
        lineage = [e for e in session.event_log.lineage(event_id) if not session.event_log.is_deleted(e.event_id)]
        return self._heads_from(lineage)

    def read_artifact(self, session: Session, artifact_id: str, branch_id: Optional[str] = None) -> Any:
        ref = self.artifacts(session, branch_id).get(artifact_id)
        if ref is None:
            raise KeyError(f"artifact {artifact_id!r} does not exist on branch {branch_id or session.branch_id!r}")
        return self.store.get(session.tenant_id, ref.hash)[1]

    # ============================================================ confirmation
    def confirm(
        self, session: Session, tool_ref: str, arguments: Mapping[str, Any], *, principal: Optional[Principal] = None
    ) -> str:
        """Record a single-use user confirmation for one exact destructive call.

        Only a principal of the session's tenant with at least ``USER`` trust
        may confirm; tool output and model proposals have no path to this method.
        """
        principal = principal or session.principal
        self._require_same_tenant(session, principal, action="confirm")
        if not principal.trust.at_least(Trust.USER):
            raise PermissionError(f"principal trust {principal.trust.value} cannot confirm actions")
        self.registry.resolve(tool_ref)
        token = content_hash({"tool": tool_ref, "arguments": dict(arguments)})
        session._confirmations.add(token)
        session.event_log.append(
            "note",
            {"type": "confirmation", "tool": tool_ref, "args_hash": token, "principal_id": principal.principal_id},
            branch_id=session.branch_id,
            trust=Trust.USER,
        )
        return token

    # ============================================================== validation
    def _require_same_tenant(self, session: Session, principal: Principal, *, action: str) -> None:
        if principal.tenant_id != session.tenant_id:
            self.security_events.append(
                {
                    "type": "cross_tenant_attempt",
                    "action": action,
                    "principal_id": principal.principal_id,
                    "principal_tenant": principal.tenant_id,
                    "target_tenant": session.tenant_id,
                    "at": float(self._clock()),
                }
            )
            raise TenantIsolationError(
                f"principal {principal.principal_id!r} of tenant {principal.tenant_id!r} may not access tenant {session.tenant_id!r}"
            )

    def _resolve(self, ref: str) -> ToolSpec:
        try:
            return self.registry.resolve(ref)
        except ToolRefError as exc:
            raise _Rejection(ErrorClass.SCHEMA_INVALID, str(exc)) from None
        except UnknownToolVersionError as exc:
            raise _Rejection(ErrorClass.UNKNOWN_VERSION, str(exc)) from None
        except UnknownToolError as exc:
            raise _Rejection(ErrorClass.UNKNOWN_TOOL, str(exc)) from None

    def _static_check(self, call: ActionCall, principal: Principal) -> _Prepared:
        """State-independent checks (schema, permissions, limits, verifiers)."""
        spec = self._resolve(call.tool)
        arguments = copy.deepcopy(dict(call.arguments))
        errs = json_value_errors(arguments, "$.arguments")
        if errs:
            raise _Rejection(ErrorClass.SCHEMA_INVALID, "; ".join(errs))
        size = len(canonical_json(arguments).encode("utf-8"))
        if size > self.limits.max_argument_bytes:
            raise _Rejection(ErrorClass.RESOURCE_LIMIT, f"arguments are {size} bytes; limit is {self.limits.max_argument_bytes}")
        errs = validate(arguments, spec.argument_schema, "$.arguments")
        if errs:
            raise _Rejection(ErrorClass.SCHEMA_INVALID, "; ".join(errs[:20]))
        missing = sorted(spec.permissions - principal.permissions)
        if missing:
            raise _Rejection(
                ErrorClass.PERMISSION_DENIED,
                f"principal {principal.principal_id!r} lacks permission(s) {missing} required by {spec.ref}",
            )
        if call.timeout_class not in self.timeout_classes:
            raise _Rejection(
                ErrorClass.INVALID_TIMEOUT_CLASS,
                f"unknown timeout class {call.timeout_class!r}; known: {sorted(self.timeout_classes)}",
            )
        spec_limit = self.timeout_classes.get(spec.timeout_class, math.inf)
        if self.timeout_classes[call.timeout_class] > spec_limit:
            raise _Rejection(
                ErrorClass.RESOURCE_LIMIT,
                f"timeout class {call.timeout_class!r} exceeds the maximum {spec.timeout_class!r} for {spec.ref}",
            )
        unknown = self.verifiers.unknown([*spec.default_verifiers, *call.verify])
        if unknown:
            raise _Rejection(ErrorClass.UNKNOWN_VERIFIER, f"unknown verifier(s) {unknown}")
        artifact_fields = {arguments[f] for f in (*spec.reads, *spec.writes) if isinstance(arguments.get(f), str)}
        for label, mapping in (("expected_input_hashes", call.expected_input_hashes), ("expected_output_hashes", call.expected_output_hashes)):
            for aid in (mapping or {}):
                if aid not in artifact_fields:
                    raise _Rejection(ErrorClass.SCHEMA_INVALID, f"{label} names artifact {aid!r} that this call does not touch")
        write_ids = frozenset(arguments[f] for f in spec.writes if isinstance(arguments.get(f), str))
        return _Prepared(spec, arguments, call.args_hash, write_ids=write_ids)

    def _check_confirmation(self, session: Session, prepared: _Prepared) -> None:
        if prepared.spec.requires_confirmation and prepared.args_hash not in session._confirmations:
            raise _Rejection(
                ErrorClass.CONFIRMATION_REQUIRED,
                f"{prepared.spec.ref} is destructive and requires an explicit user confirmation of these exact arguments",
            )

    def _stateful_check(self, session: Session, call: ActionCall, prepared: _Prepared, branch_id: str) -> None:
        """Checks against the current branch state: artifacts, expected hashes, tool precheck."""
        heads = self.artifacts(session, branch_id)
        spec, args = prepared.spec, prepared.arguments
        readable: dict[str, tuple[str, str, Any]] = {}
        for f in spec.reads:
            aid = args.get(f)
            if not isinstance(aid, str):
                continue
            ref = heads.get(aid)
            if ref is None:
                raise _Rejection(ErrorClass.UNKNOWN_ARTIFACT, f"artifact {aid!r} does not exist in this session/branch")
            readable[aid] = self._load(session, ref)
        for aid in prepared.write_ids:
            if aid in heads and aid not in readable:
                readable[aid] = self._load(session, heads[aid])
        for aid, expected in (call.expected_input_hashes or {}).items():
            current = heads.get(aid)
            if current is None or current.hash != expected:
                raise _Rejection(
                    ErrorClass.STALE_INPUT,
                    f"expected {aid} at {expected} but current version is {current.hash if current else 'absent'}",
                )
        prepared.readable = readable
        if spec.precheck is not None:
            ctx = self._context(session, prepared, branch_id)
            try:
                errs = list(spec.precheck(copy.deepcopy(args), ctx))
            except Exception as exc:  # a crashing precheck must not allow execution
                errs = [f"precheck raised {type(exc).__name__}: {exc}"]
            if errs:
                raise _Rejection(ErrorClass.PRECONDITION_FAILED, "; ".join(str(e) for e in errs[:20]))

    def _load(self, session: Session, ref: ArtifactRef) -> tuple[str, str, Any]:
        try:
            kind, content = self.store.get(session.tenant_id, ref.hash)
        except ArtifactNotFound as exc:
            raise _Rejection(ErrorClass.UNKNOWN_ARTIFACT, str(exc)) from None
        return kind, ref.hash, content

    def _context(
        self, session: Session, prepared: _Prepared, branch_id: str, cancel: Optional[threading.Event] = None
    ) -> ToolContext:
        return ToolContext(
            tenant_id=session.tenant_id,
            session_id=session.session_id,
            branch_id=branch_id,
            principal_id=session.principal.principal_id,
            artifacts=prepared.readable,
            cancel_event=cancel,
        )

    def preflight(
        self,
        session: Session,
        envelope: ActionEnvelope,
        *,
        principal: Optional[Principal] = None,
        budget: Optional[RunBudget] = None,
    ) -> PreflightReport:
        """Validate a whole envelope before executing anything (§10 "static validation").

        Envelope-level problems (budget above policy, run budget too small) and
        per-call problems are reported; invalid calls get a signed ``rejected``
        receipt in the log. State-dependent checks (artifacts, expected hashes,
        contradictory-edit prechecks) run here for every call whose artifacts
        are not changed by an earlier call of the same envelope, and run again
        immediately before each call executes.
        """
        principal = principal or session.principal
        env_errors: list[str] = []
        if envelope.budget.max_calls > self.limits.max_calls_per_envelope:
            env_errors.append(
                f"budget.max_calls={envelope.budget.max_calls} exceeds policy limit {self.limits.max_calls_per_envelope}"
            )
        if envelope.budget.max_repair_attempts > self.limits.max_repair_attempts:
            env_errors.append(
                f"budget.max_repair_attempts={envelope.budget.max_repair_attempts} exceeds policy limit "
                f"{self.limits.max_repair_attempts}"
            )
        if budget is not None and len(envelope.calls) > budget.remaining()["calls"]:
            env_errors.append(f"{len(envelope.calls)} calls exceed the remaining run budget {budget.remaining()['calls']}")
        try:
            self._require_same_tenant(session, principal, action="preflight")
        except TenantIsolationError as exc:
            return PreflightReport(
                tuple(env_errors),
                tuple(self._reject(session, c, envelope.plan_id, principal, ErrorClass.TENANT_ISOLATION, str(exc), log=False)
                      for c in envelope.calls),
            )
        rejections: list[Optional[CallOutcome]] = []
        written: set[str] = set()  # artifacts an earlier call in this envelope will change
        for call in envelope.calls:
            try:
                prepared = self._static_check(call, principal)
                if self._find_prior(session, call.idempotency_key, session.branch_id) is None:
                    self._check_confirmation(session, prepared)
                    args = prepared.arguments
                    touched = {args[f] for f in (*prepared.spec.reads, *prepared.spec.writes) if isinstance(args.get(f), str)}
                    if not touched & written:  # its pre-state is the current state: check it now
                        self._stateful_check(session, call, prepared, session.branch_id)
                written |= prepared.write_ids
                rejections.append(None)
            except _Rejection as rej:
                rejections.append(self._reject(session, call, envelope.plan_id, principal, rej.error_class, rej.message))
        return PreflightReport(tuple(env_errors), tuple(rejections))

    # =============================================================== execution
    def execute_call(
        self,
        session: Session,
        call: ActionCall,
        *,
        plan_id: str,
        principal: Optional[Principal] = None,
        budget: Optional[RunBudget] = None,
    ) -> CallOutcome:
        """Validate → (idempotency) → execute → sign/log receipt → verify → commit. Never raises for tool faults."""
        principal = principal or session.principal
        try:
            self._require_same_tenant(session, principal, action="execute")
        except TenantIsolationError as exc:
            return self._reject(session, call, plan_id, principal, ErrorClass.TENANT_ISOLATION, str(exc), log=False)

        with session.lock:
            branch_id = session.branch_id
            try:
                prepared = self._static_check(call, principal)
            except _Rejection as rej:
                return self._reject(session, call, plan_id, principal, rej.error_class, rej.message)

            try:
                prior = self._find_prior(session, call.idempotency_key, branch_id)
            except _Rejection as rej:
                return self._reject(session, call, plan_id, principal, rej.error_class, rej.message)
            if prior is not None:
                if prior.receipt.args_hash == prepared.args_hash and prior.receipt.tool == call.tool:
                    session.event_log.append(
                        "note",
                        {"type": "idempotent_replay", "idempotency_key": call.idempotency_key,
                         "receipt_hash": prior.receipt.receipt_hash},
                        branch_id=branch_id,
                    )
                    return CallOutcome(prior.receipt, prior.verifications, prior.committed, True,
                                       prior.receipt_event_id, prior.verification_event_id)
                return self._reject(
                    session, call, plan_id, principal, ErrorClass.IDEMPOTENCY_CONFLICT,
                    f"idempotency key {call.idempotency_key!r} was already used with different arguments "
                    f"(receipt {prior.receipt.receipt_hash})",
                )

            try:
                self._check_confirmation(session, prepared)
                self._stateful_check(session, call, prepared, branch_id)
                if budget is not None:
                    denial = budget.denial(prepared.spec.cost)
                    if denial:
                        raise _Rejection(ErrorClass.BUDGET_EXCEEDED, denial)
            except _Rejection as rej:
                return self._reject(session, call, plan_id, principal, rej.error_class, rej.message)

            if budget is not None:
                budget.charge(prepared.spec.cost)
            session._confirmations.discard(prepared.args_hash)  # single use
            return self._execute(session, call, prepared, plan_id, principal, branch_id)

    validate_execute_verify = execute_call  # name used by the blueprint's Appendix A pseudocode

    def _execute(
        self, session: Session, call: ActionCall, prepared: _Prepared, plan_id: str, principal: Principal, branch_id: str
    ) -> CallOutcome:
        spec = prepared.spec
        timeout_s = self.timeout_classes[call.timeout_class]
        cancel = threading.Event()
        ctx = self._context(session, prepared, branch_id, cancel)
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["result"] = spec.handler(copy.deepcopy(prepared.arguments), ctx)
            except BaseException as exc:  # noqa: BLE001 - sandbox boundary: every fault becomes a receipt
                box["error"] = exc

        started = time.perf_counter()
        worker = threading.Thread(target=target, name=f"tool:{spec.ref}", daemon=True)
        worker.start()
        worker.join(timeout_s)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        metrics: dict[str, float] = {"elapsed_ms": elapsed_ms, "cost": float(spec.cost), "timeout_s": float(timeout_s)}
        input_hashes = {aid: h for aid, (_, h, _) in sorted(prepared.readable.items())}

        status, error_class, error_message = ReceiptStatus.OK, None, None
        data: dict[str, Any] = {}
        outputs: dict[str, tuple[str, Any]] = {}
        if worker.is_alive():
            cancel.set()
            status, error_class = ReceiptStatus.FAILED, ErrorClass.TIMEOUT
            error_message = f"{spec.ref} exceeded timeout class {call.timeout_class!r} ({timeout_s:g}s)"
        elif "error" in box:
            exc = box["error"]
            status, error_class = ReceiptStatus.FAILED, ErrorClass.TOOL_EXCEPTION
            error_message = f"{type(exc).__name__}: {exc}"
        else:
            try:
                data, outputs, tool_metrics = self._normalise_result(box.get("result"), spec, prepared)
                metrics.update(tool_metrics)
            except _Rejection as rej:
                status, error_class, error_message = ReceiptStatus.FAILED, rej.error_class, rej.message

        artifact_refs: list[dict[str, Any]] = []
        output_hashes: dict[str, str] = {}
        if status is ReceiptStatus.OK:
            for aid, (kind, content) in sorted(outputs.items()):
                h = self.store.put(session.tenant_id, kind, content)
                output_hashes[aid] = h
                artifact_refs.append({"artifact_id": aid, "kind": kind, "hash": h})

        receipt = self._make_receipt(
            session, call, plan_id, principal, branch_id,
            status=status, error_class=error_class, error_message=error_message,
            artifact_refs=artifact_refs, input_hashes=input_hashes, output_hashes=output_hashes,
            output=data, metrics=metrics, trust=spec.output_trust, args_hash=prepared.args_hash,
        )
        rev = session.event_log.append_receipt(receipt.to_dict(), branch_id=branch_id)
        if not receipt.ok:
            return CallOutcome(receipt, (), (), False, rev.event_id, None)
        return self._verify_and_commit(session, call, prepared, receipt, rev.event_id, outputs, data, branch_id)

    def _normalise_result(
        self, result: Any, spec: ToolSpec, prepared: _Prepared
    ) -> tuple[dict[str, Any], dict[str, tuple[str, Any]], dict[str, float]]:
        if result is None:
            result = ToolResult()
        elif isinstance(result, Mapping) and not isinstance(result, ToolResult):
            result = ToolResult(data=dict(result))
        if not isinstance(result, ToolResult):
            raise _Rejection(ErrorClass.INVALID_OUTPUT, f"tool returned {type(result).__name__}, expected ToolResult")
        data = dict(result.data)
        errs = json_value_errors(data, "$.data")
        if errs:
            raise _Rejection(ErrorClass.INVALID_OUTPUT, "; ".join(errs[:5]))
        data = json.loads(canonical_json(data))
        if len(canonical_json(data)) > self.limits.max_output_bytes:
            raise _Rejection(ErrorClass.INVALID_OUTPUT, "tool output exceeds the output size limit")
        if spec.output_schema is not None:
            errs = validate(data, spec.output_schema, "$.data")
            if errs:
                raise _Rejection(ErrorClass.INVALID_OUTPUT, "; ".join(errs[:10]))
        outputs: dict[str, tuple[str, Any]] = {}
        for aid, out in dict(result.outputs).items():
            if not isinstance(out, ArtifactOutput):
                raise _Rejection(ErrorClass.INVALID_OUTPUT, f"output {aid!r} is not an ArtifactOutput")
            if not spec.side_effect or aid not in prepared.write_ids:
                raise _Rejection(ErrorClass.INVALID_OUTPUT, f"{spec.ref} attempted to write artifact {aid!r} it was not granted")
            errs = json_value_errors(out.content, f"$.outputs.{aid}")
            if errs:
                raise _Rejection(ErrorClass.INVALID_OUTPUT, "; ".join(errs[:5]))
            outputs[aid] = (out.kind, json.loads(canonical_json(out.content)))
        metrics: dict[str, float] = {}
        for k, v in dict(result.metrics).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                metrics[f"tool.{k}"] = float(v)
        return data, outputs, metrics

    def _verify_and_commit(
        self,
        session: Session,
        call: ActionCall,
        prepared: _Prepared,
        receipt: ToolReceipt,
        receipt_event_id: str,
        outputs: Mapping[str, tuple[str, Any]],
        data: Mapping[str, Any],
        branch_id: str,
    ) -> CallOutcome:
        names = list(dict.fromkeys([*prepared.spec.default_verifiers, *call.verify]))
        target = ",".join(f"{r['artifact_id']}@{r['hash']}" for r in receipt.artifact_refs) or f"receipt:{receipt.receipt_hash}"
        vctx = VerificationContext(
            receipt=receipt,
            spec=prepared.spec,
            arguments=copy.deepcopy(prepared.arguments),
            inputs={aid: copy.deepcopy(c) for aid, (_, _, c) in prepared.readable.items()},
            input_kinds={aid: k for aid, (k, _, _) in prepared.readable.items()},
            outputs={aid: copy.deepcopy(c) for aid, (_, c) in outputs.items()},
            output_kinds={aid: k for aid, (k, _) in outputs.items()},
            output_data=copy.deepcopy(dict(data)),
        )
        results = self.verifiers.run(names, vctx, target_artifact=target, evidence_location=f"event:{receipt_event_id}")
        if call.expected_output_hashes:
            mismatches = {
                aid: {"expected": exp, "actual": receipt.output_hashes.get(aid)}
                for aid, exp in call.expected_output_hashes.items()
                if receipt.output_hashes.get(aid) != exp
            }
            results.append(
                VerificationResult(
                    "expected_output_hashes", not mismatches, {"mismatches": mismatches},
                    target_artifact=target, receipt_hash=receipt.receipt_hash,
                )
            )
        passed = all(r.passed for r in results)
        committed = tuple(
            ArtifactRef(r["artifact_id"], r["kind"], r["hash"], receipt.trust) for r in receipt.artifact_refs
        ) if passed else ()
        vev = session.event_log.append(
            "verification",
            self.redactor.redact(
                {
                    "type": "tool_verification",
                    "receipt_hash": receipt.receipt_hash,
                    "receipt_event": receipt_event_id,
                    "tool": receipt.tool,
                    "idempotency_key": receipt.idempotency_key,
                    "results": [r.to_dict() for r in results],
                    "passed": passed,
                    "committed": [{**c.to_dict(), "receipt_hash": receipt.receipt_hash} for c in committed],
                }
            ),
            branch_id=branch_id,
            trust=Trust.SYSTEM,
        )
        committed = tuple(ArtifactRef(c.artifact_id, c.kind, c.hash, c.trust, vev.event_id) for c in committed)
        return CallOutcome(receipt, tuple(results), committed, False, receipt_event_id, vev.event_id)

    # ================================================================ receipts
    def _make_receipt(
        self,
        session: Session,
        call: ActionCall,
        plan_id: str,
        principal: Principal,
        branch_id: str,
        *,
        status: ReceiptStatus,
        error_class: Optional[ErrorClass],
        error_message: Optional[str],
        args_hash: str,
        trust: Trust = Trust.UNTRUSTED,
        artifact_refs: Iterable[Mapping[str, Any]] = (),
        input_hashes: Optional[Mapping[str, str]] = None,
        output_hashes: Optional[Mapping[str, str]] = None,
        output: Optional[Mapping[str, Any]] = None,
        metrics: Optional[Mapping[str, float]] = None,
    ) -> ToolReceipt:
        n = next(self._counter)
        if error_message is not None:
            error_message = self.redactor.redact_text(error_message)[: self.limits.max_error_chars]
        unsigned = ToolReceipt(
            receipt_id="rc_" + short_id(session.tenant_id, session.session_id, call.idempotency_key, args_hash, n),
            tenant_id=session.tenant_id,
            session_id=session.session_id,
            branch_id=branch_id,
            principal_id=principal.principal_id,
            plan_id=plan_id,
            tool=call.tool,
            idempotency_key=call.idempotency_key,
            args_hash=args_hash,
            status=status.value,
            error_class=error_class.value if error_class is not None else None,
            error_message=error_message,
            artifact_refs=tuple(dict(r) for r in artifact_refs),
            input_hashes=dict(input_hashes or {}),
            output_hashes=dict(output_hashes or {}),
            output=self.redactor.redact(dict(output or {})),
            metrics=dict(metrics or {}),
            trust=Trust(trust).value,
            timeout_class=call.timeout_class,
            created_at=float(self._clock()),
            key_id=self._signer.key_id,
        )
        receipt_hash, signature = self._signer.sign(unsigned.body())
        d = unsigned.to_dict()
        d["receipt_hash"], d["signature"] = receipt_hash, signature
        return ToolReceipt.from_dict(d)

    def _reject(
        self,
        session: Session,
        call: ActionCall,
        plan_id: str,
        principal: Principal,
        error_class: ErrorClass,
        message: str,
        *,
        log: bool = True,
    ) -> CallOutcome:
        """Signed ``rejected`` receipt; never executed. Cross-tenant rejections are not logged in the victim's log."""
        try:
            args_hash = call.args_hash
        except (TypeError, ValueError):
            args_hash = content_hash(repr(call.arguments))
        receipt = self._make_receipt(
            session, call, plan_id, principal, session.branch_id,
            status=ReceiptStatus.REJECTED, error_class=error_class, error_message=message, args_hash=args_hash,
            trust=Trust.SYSTEM,
        )
        if not log:
            self.security_events.append({"type": "rejected_unlogged", "receipt": receipt.to_dict()})
            return CallOutcome(receipt)
        ev = session.event_log.append_receipt(receipt.to_dict(), branch_id=session.branch_id)
        return CallOutcome(receipt, (), (), False, ev.event_id, None)

    def verify_receipt(self, receipt: ToolReceipt | Mapping[str, Any]) -> bool:
        """True iff the receipt's hash and HMAC signature are intact."""
        d = receipt.to_dict() if isinstance(receipt, ToolReceipt) else dict(receipt)
        return self._signer.verify(d)

    def _find_prior(self, session: Session, idempotency_key: str, branch_id: str) -> Optional[CallOutcome]:
        """Most recent executed receipt for this key in the branch lineage (with its verification)."""
        found: Optional[tuple[ToolReceipt, str]] = None
        verif: dict[str, Event] = {}
        for ev in session.event_log.replay(branch_id):
            if ev.kind == "receipt" and ev.payload.get("idempotency_key") == idempotency_key:
                if ev.payload.get("status") == ReceiptStatus.REJECTED.value:
                    continue
                if not self._signer.verify(ev.payload):
                    raise _Rejection(ErrorClass.IDEMPOTENCY_CONFLICT, "stored receipt for this key failed signature verification")
                found = (ToolReceipt.from_dict(ev.payload), ev.event_id)
            elif ev.kind == "verification" and ev.payload.get("idempotency_key") == idempotency_key:
                verif[ev.payload.get("receipt_hash", "")] = ev
        if found is None:
            return None
        receipt, rev_id = found
        vev = verif.get(receipt.receipt_hash)
        results: tuple[VerificationResult, ...] = ()
        committed: tuple[ArtifactRef, ...] = ()
        if vev is not None:
            results = tuple(VerificationResult(**r) for r in vev.payload.get("results", []))
            committed = tuple(
                ArtifactRef(c["artifact_id"], c["kind"], c["hash"], c["trust"], vev.event_id)
                for c in vev.payload.get("committed", [])
            )
        return CallOutcome(receipt, results, committed, False, rev_id, vev.event_id if vev else None)

    # ================================================================ branches
    def create_branch(self, session: Session, request: BranchRequest, *, switch: bool = True) -> str:
        """Create a named copy-on-write branch at a committed event (§10 "Branch request")."""
        log = session.event_log
        if request.name is not None and not _BRANCH_NAME_RE.fullmatch(request.name):
            raise BranchError(f"invalid branch name {request.name!r}; use letters, digits, '_', '.', ':' or '-'")
        try:
            log.get(request.parent_event, tenant_id=session.tenant_id)
        except EventLogError as exc:
            raise BranchError(str(exc)) from None
        if request.expected_state is not None:
            actual = {aid: ref.hash for aid, ref in self.artifacts_at(session, request.parent_event).items()}
            mismatched = {
                aid: {"expected": h, "actual": actual.get(aid)}
                for aid, h in request.expected_state.items()
                if actual.get(aid) != h
            }
            if mismatched:
                raise BranchError(f"expected state does not match the log at {request.parent_event}: {mismatched}")
        meta = {
            "expected_state": dict(request.expected_state or {}),
            "memory_snapshot_id": request.memory_snapshot_id,
            "from_branch": session.branch_id,
        }
        try:
            branch_id = log.create_branch(request.parent_event, request.reason, name=request.name, meta=meta)
        except EventLogError as exc:
            raise BranchError(str(exc)) from None
        if switch:
            session.branch_id = branch_id
        return branch_id

    @staticmethod
    def _is_state_event(ev: Event) -> bool:
        p = ev.payload
        return (ev.kind == "ingest" and "artifact_id" in p) or (
            ev.kind == "verification" and bool(p.get("passed")) and bool(p.get("committed"))
        )

    def undo(
        self,
        session: Session,
        *,
        steps: int = 1,
        name: Optional[str] = None,
        reason: str = "user undo",
        memory_snapshot_id: Optional[str] = None,
    ) -> str:
        """Undo the last ``steps`` committed edits by branching from the state before them.

        Nothing is deleted or reversed: the undone history stays on its branch
        (and can be restored with :meth:`restore_branch`); the session switches
        to a new named branch whose head state predates the undone commits.
        """
        if steps < 1:
            raise BranchError("steps must be >= 1")
        lineage = list(session.event_log.replay(session.branch_id))
        commits = [i for i, ev in enumerate(lineage) if ev.kind == "verification" and self._is_state_event(ev)]
        if len(commits) < steps:
            raise BranchError(f"only {len(commits)} committed edit(s) on branch {session.branch_id!r}; cannot undo {steps}")
        first_undone = commits[-steps]
        parents = [ev for ev in lineage[:first_undone] if self._is_state_event(ev)]
        if not parents:
            raise BranchError("no committed state precedes the edits to undo")
        parent = parents[-1]
        expected = {aid: ref.hash for aid, ref in self.artifacts_at(session, parent.event_id).items()}
        if name is None:
            name = f"{session.branch_id}.undo-{short_id(parent.event_id, len(session.event_log), n=8)}"
        return self.create_branch(
            session,
            BranchRequest(parent.event_id, reason, name=name, expected_state=expected, memory_snapshot_id=memory_snapshot_id),
        )

    def restore_branch(self, session: Session, branch_id: str, *, reason: str = "restore") -> str:
        """Switch the session to an existing named branch (e.g. redo after undo)."""
        if branch_id not in session.event_log.branches:
            raise BranchError(f"unknown branch {branch_id!r}")
        session.branch_id = branch_id
        session.event_log.append("note", {"type": "branch_restored", "reason": reason}, branch_id=branch_id)
        return branch_id

    # =================================================================== audit
    def audit(self, session: Session) -> AuditReport:
        """Re-verify the hash chain and every receipt signature in the session log (all branches)."""
        log = session.event_log
        seen: set[str] = set()
        bad: list[str] = []
        foreign: list[str] = []
        n = 0
        for b in log.branches:
            for ev in log.replay(b):
                if ev.event_id in seen or ev.kind != "receipt":
                    continue
                seen.add(ev.event_id)
                n += 1
                if not self._signer.verify(ev.payload):
                    bad.append(ev.event_id)
                elif ev.payload.get("tenant_id") != log.tenant_id or ev.payload.get("session_id") != log.session_id:
                    foreign.append(ev.event_id)
        return AuditReport(log.verify_chain(), n, tuple(bad), tuple(foreign))

    # ========================================================== introspection
    def tool_manifest(self) -> list[dict[str, Any]]:
        return self.registry.manifest()

    def receipts(self, session: Session, branch_id: Optional[str] = None) -> list[ToolReceipt]:
        return [
            ToolReceipt.from_dict(ev.payload)
            for ev in session.event_log.replay(branch_id or session.branch_id)
            if ev.kind == "receipt"
        ]
