"""Versioned orchestrator contracts: the ``m63.actions/v1`` envelope and §10 interfaces.

The model talks to the control plane through a versioned envelope rather than
free-form tool syntax embedded in prose (blueprint §10). This module defines:

* :class:`ActionEnvelope` / :class:`ActionCall` / :class:`Budget` /
  :class:`EscalationFlag` — the exact shape of the §10 JSON example, with
  :func:`parse_envelope` that rejects unknown fields anywhere, unsupported
  versions, malformed ``name@version`` tool references, duplicate idempotency
  keys, and call lists that exceed the envelope's own ``budget.max_calls``;
* the other rows of the §10 interface table: :class:`ToolReceipt`,
  :class:`VerificationRequest` (+ :class:`VerificationResult`),
  :class:`BranchRequest`, :class:`EscalationRequest`, :class:`FinalResponse`.

Everything is a frozen dataclass with ``to_dict``/``from_dict`` so that the
same objects can be hashed (``content_hash``), appended to the event log, and
exported as training trajectories without a second serialisation format.
"""

from __future__ import annotations

import copy
import enum
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from ..common import Trust, canonical_json, content_hash
from .jsonschema_lite import check_schema, json_value_errors, validate

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "TOOL_REF_PATTERN",
    "HASH_PATTERN",
    "ID_PATTERN",
    "ENVELOPE_SCHEMA",
    "CALL_SCHEMA",
    "EnvelopeError",
    "ToolRefError",
    "parse_tool_ref",
    "format_tool_ref",
    "Budget",
    "EscalationFlag",
    "ActionCall",
    "ActionEnvelope",
    "parse_envelope",
    "ReceiptStatus",
    "ErrorClass",
    "ToolReceipt",
    "VerificationRequest",
    "VerificationResult",
    "BranchRequest",
    "EscalationRequest",
    "CompletedArtifact",
    "FinalResponse",
]

SCHEMA_VERSION = "m63.actions/v1"
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (SCHEMA_VERSION,)

#: ``namespace.tool@<positive integer major version>``, e.g. ``timeline.apply_edit@2``.
TOOL_REF_PATTERN = r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*@[1-9][0-9]*$"
HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"
#: Identifiers such as ``session_42/branch_7/plan_19/call_1``.
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:\-]*(?:/[A-Za-z0-9][A-Za-z0-9_.:\-]*)*$"
_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"

MAX_CALLS_PER_ENVELOPE = 64
MAX_REPAIR_ATTEMPTS = 16

_HASH_MAP_SCHEMA = {
    "type": "object",
    "propertyNames": {"type": "string", "pattern": ID_PATTERN, "maxLength": 256},
    "additionalProperties": {"type": "string", "pattern": HASH_PATTERN},
}

CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tool", "arguments", "idempotency_key", "timeout_class", "verify"],
    "properties": {
        "tool": {"type": "string", "maxLength": 128, "pattern": TOOL_REF_PATTERN},
        "arguments": {"type": "object"},
        "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 256, "pattern": ID_PATTERN},
        "timeout_class": {"type": "string", "maxLength": 64, "pattern": _NAME_PATTERN},
        "verify": {
            "type": "array",
            "maxItems": 16,
            "uniqueItems": True,
            "items": {"type": "string", "maxLength": 64, "pattern": _NAME_PATTERN},
        },
        "expected_input_hashes": _HASH_MAP_SCHEMA,
        "expected_output_hashes": _HASH_MAP_SCHEMA,
    },
}

ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "plan_id", "calls", "budget", "escalation"],
    "properties": {
        "schema_version": {"type": "string"},
        "plan_id": {"type": "string", "minLength": 1, "maxLength": 256, "pattern": ID_PATTERN},
        "assistant_message": {"type": ["string", "null"], "maxLength": 8000},
        "calls": {"type": "array", "maxItems": MAX_CALLS_PER_ENVELOPE, "items": CALL_SCHEMA},
        "budget": {
            "type": "object",
            "additionalProperties": False,
            "required": ["max_calls", "max_repair_attempts"],
            "properties": {
                "max_calls": {"type": "integer", "minimum": 0, "maximum": MAX_CALLS_PER_ENVELOPE},
                "max_repair_attempts": {"type": "integer", "minimum": 0, "maximum": MAX_REPAIR_ATTEMPTS},
            },
        },
        "escalation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["required", "reason"],
            "properties": {
                "required": {"type": "boolean"},
                "reason": {"type": ["string", "null"], "maxLength": 2000},
            },
        },
    },
}
check_schema(ENVELOPE_SCHEMA)


class EnvelopeError(ValueError):
    """An action envelope failed parsing or validation; ``errors`` lists every problem."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = list(errors)
        super().__init__("invalid action envelope: " + "; ".join(self.errors))


class ToolRefError(ValueError):
    """A tool reference is not of the form ``name@version``."""


def parse_tool_ref(ref: str) -> tuple[str, int]:
    """Split ``"timeline.apply_edit@2"`` into ``("timeline.apply_edit", 2)``."""
    if not isinstance(ref, str) or not re.fullmatch(TOOL_REF_PATTERN, ref):
        raise ToolRefError(f"malformed tool reference {ref!r}; expected 'name@version' (e.g. 'timeline.apply_edit@2')")
    name, version = ref.rsplit("@", 1)
    return name, int(version)


def format_tool_ref(name: str, version: int) -> str:
    return f"{name}@{int(version)}"


# ------------------------------------------------------------------ envelope
@dataclass(frozen=True)
class Budget:
    """Per-plan budget proposed by the model; the control plane may only tighten it."""

    max_calls: int
    max_repair_attempts: int

    def to_dict(self) -> dict[str, Any]:
        return {"max_calls": self.max_calls, "max_repair_attempts": self.max_repair_attempts}


@dataclass(frozen=True)
class EscalationFlag:
    """The model's own escalation opinion — a *feature* for the router, never a veto (§10)."""

    required: bool
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"required": self.required, "reason": self.reason}


@dataclass(frozen=True)
class ActionCall:
    """One proposed tool call. Proposals carry no authority until the control plane validates them."""

    tool: str
    arguments: Mapping[str, Any]
    idempotency_key: str
    timeout_class: str
    verify: tuple[str, ...] = ()
    expected_input_hashes: Optional[Mapping[str, str]] = None
    expected_output_hashes: Optional[Mapping[str, str]] = None

    @property
    def tool_name(self) -> str:
        return parse_tool_ref(self.tool)[0]

    @property
    def tool_version(self) -> int:
        return parse_tool_ref(self.tool)[1]

    @property
    def args_hash(self) -> str:
        """Hash binding the tool reference and arguments (used for idempotency)."""
        return content_hash({"tool": self.tool, "arguments": self.arguments})

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "tool": self.tool,
            "arguments": copy.deepcopy(dict(self.arguments)),
            "idempotency_key": self.idempotency_key,
            "timeout_class": self.timeout_class,
            "verify": list(self.verify),
        }
        if self.expected_input_hashes is not None:
            d["expected_input_hashes"] = dict(self.expected_input_hashes)
        if self.expected_output_hashes is not None:
            d["expected_output_hashes"] = dict(self.expected_output_hashes)
        return d


@dataclass(frozen=True)
class ActionEnvelope:
    """The ``m63.actions/v1`` envelope (§10 JSON example)."""

    plan_id: str
    calls: tuple[ActionCall, ...]
    budget: Budget
    escalation: EscalationFlag
    assistant_message: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"schema_version": self.schema_version, "plan_id": self.plan_id}
        d["assistant_message"] = self.assistant_message
        d["calls"] = [c.to_dict() for c in self.calls]
        d["budget"] = self.budget.to_dict()
        d["escalation"] = self.escalation.to_dict()
        return d

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def envelope_hash(self) -> str:
        return content_hash(self.to_dict())


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise EnvelopeError([f"duplicate JSON key {k!r}"])
        out[k] = v
    return out


def _reject_constant(name: str) -> Any:
    raise EnvelopeError([f"non-finite JSON constant {name} is not allowed"])


def _load_json(text: str | bytes) -> Any:
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EnvelopeError([f"envelope is not UTF-8: {exc}"]) from None
    try:
        return json.loads(text, object_pairs_hook=_strict_object_pairs, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise EnvelopeError([f"envelope is not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"]) from None


def parse_envelope(data: str | bytes | Mapping[str, Any] | ActionEnvelope) -> ActionEnvelope:
    """Parse and strictly validate an action envelope.

    Accepts JSON text/bytes or an already-decoded mapping. Raises
    :class:`EnvelopeError` (with every problem found) when the envelope:

    * is not JSON, contains duplicate keys, or non-finite numbers;
    * declares a ``schema_version`` other than ``m63.actions/v1``;
    * contains an unknown field at any level of the envelope structure (tool
      ``arguments`` are checked against the tool's own schema by the control
      plane, whose schemas also forbid unknown fields);
    * has a malformed ``name@version`` tool reference;
    * repeats an idempotency key;
    * has more calls than its own ``budget.max_calls``;
    * requests escalation without a reason.
    """
    if isinstance(data, ActionEnvelope):
        data = data.to_dict()
    raw: Any = _load_json(data) if isinstance(data, (str, bytes)) else copy.deepcopy(dict(data)) if isinstance(data, Mapping) else data
    if not isinstance(raw, dict):
        raise EnvelopeError([f"$: expected object, got {type(raw).__name__}"])

    version = raw.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise EnvelopeError(
            [f"$.schema_version: unsupported schema version {version!r}; supported: {list(SUPPORTED_SCHEMA_VERSIONS)}"]
        )

    errors = json_value_errors(raw)
    if errors:
        raise EnvelopeError(errors)
    errors = validate(raw, ENVELOPE_SCHEMA)
    for i, call in enumerate(raw.get("calls", []) if isinstance(raw.get("calls"), list) else []):
        if isinstance(call, dict) and isinstance(call.get("tool"), str):
            try:
                parse_tool_ref(call["tool"])
            except ToolRefError as exc:
                msg = f"$.calls[{i}].tool: {exc}"
                if not any(e.startswith(f"$.calls[{i}].tool:") for e in errors):
                    errors.append(msg)
    if errors:
        raise EnvelopeError(errors)

    seen: dict[str, int] = {}
    for i, call in enumerate(raw["calls"]):
        key = call["idempotency_key"]
        if key in seen:
            errors.append(f"$.calls[{i}].idempotency_key: duplicate idempotency key {key!r} (also used by calls[{seen[key]}])")
        seen.setdefault(key, i)
    if len(raw["calls"]) > raw["budget"]["max_calls"]:
        errors.append(
            f"$.calls: {len(raw['calls'])} calls exceed budget.max_calls={raw['budget']['max_calls']}"
        )
    if raw["escalation"]["required"] and not (raw["escalation"]["reason"] or "").strip():
        errors.append("$.escalation.reason: a reason is required when escalation.required is true")
    if errors:
        raise EnvelopeError(errors)

    calls = tuple(
        ActionCall(
            tool=c["tool"],
            arguments=c["arguments"],
            idempotency_key=c["idempotency_key"],
            timeout_class=c["timeout_class"],
            verify=tuple(c["verify"]),
            expected_input_hashes=c.get("expected_input_hashes"),
            expected_output_hashes=c.get("expected_output_hashes"),
        )
        for c in raw["calls"]
    )
    return ActionEnvelope(
        plan_id=raw["plan_id"],
        calls=calls,
        budget=Budget(raw["budget"]["max_calls"], raw["budget"]["max_repair_attempts"]),
        escalation=EscalationFlag(raw["escalation"]["required"], raw["escalation"]["reason"]),
        assistant_message=raw.get("assistant_message"),
        schema_version=version,
    )


# ------------------------------------------------------------------ receipts
class ReceiptStatus(str, enum.Enum):
    """``ok``: executed and returned; ``failed``: executed but errored/timed out;
    ``rejected``: refused by static validation and never executed."""

    OK = "ok"
    FAILED = "failed"
    REJECTED = "rejected"


class ErrorClass(str, enum.Enum):
    """Stable error classes carried in receipts (training labels depend on them)."""

    SCHEMA_INVALID = "schema_invalid"
    UNKNOWN_TOOL = "unknown_tool"
    UNKNOWN_VERSION = "unknown_version"
    PERMISSION_DENIED = "permission_denied"
    TENANT_ISOLATION = "tenant_isolation"
    CONFIRMATION_REQUIRED = "confirmation_required"
    BUDGET_EXCEEDED = "budget_exceeded"
    RESOURCE_LIMIT = "resource_limit"
    INVALID_TIMEOUT_CLASS = "invalid_timeout_class"
    UNKNOWN_VERIFIER = "unknown_verifier"
    UNKNOWN_ARTIFACT = "unknown_artifact"
    STALE_INPUT = "stale_input"
    PRECONDITION_FAILED = "precondition_failed"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    TIMEOUT = "timeout"
    TOOL_EXCEPTION = "tool_exception"
    INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True)
class ToolReceipt:
    """Signed, hashed record of one call (§10 "Tool receipt" row).

    ``receipt_hash`` is ``content_hash(body())`` and ``signature`` is an
    HMAC-SHA256 of that hash under the control plane's key ``key_id``. Secrets
    are redacted *before* hashing, so the signed record never contains them.
    """

    receipt_id: str
    tenant_id: str
    session_id: str
    branch_id: str
    principal_id: str
    plan_id: str
    tool: str
    idempotency_key: str
    args_hash: str
    status: str
    error_class: Optional[str]
    error_message: Optional[str]
    artifact_refs: tuple[Mapping[str, Any], ...]
    input_hashes: Mapping[str, str]
    output_hashes: Mapping[str, str]
    output: Mapping[str, Any]
    metrics: Mapping[str, float]
    trust: str
    timeout_class: str
    created_at: float
    key_id: str
    receipt_hash: str = ""
    signature: str = ""

    _UNSIGNED = ("receipt_hash", "signature")

    @property
    def ok(self) -> bool:
        return self.status == ReceiptStatus.OK.value

    @property
    def trust_label(self) -> Trust:
        return Trust(self.trust)

    def body(self) -> dict[str, Any]:
        """Everything covered by ``receipt_hash`` (i.e. all fields but hash and signature)."""
        d = self.to_dict()
        for k in self._UNSIGNED:
            d.pop(k)
        return d

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            canonical_json(
                {
                    "receipt_id": self.receipt_id,
                    "tenant_id": self.tenant_id,
                    "session_id": self.session_id,
                    "branch_id": self.branch_id,
                    "principal_id": self.principal_id,
                    "plan_id": self.plan_id,
                    "tool": self.tool,
                    "idempotency_key": self.idempotency_key,
                    "args_hash": self.args_hash,
                    "status": self.status,
                    "error_class": self.error_class,
                    "error_message": self.error_message,
                    "artifact_refs": [dict(r) for r in self.artifact_refs],
                    "input_hashes": dict(self.input_hashes),
                    "output_hashes": dict(self.output_hashes),
                    "output": dict(self.output),
                    "metrics": dict(self.metrics),
                    "trust": self.trust,
                    "timeout_class": self.timeout_class,
                    "created_at": self.created_at,
                    "key_id": self.key_id,
                    "receipt_hash": self.receipt_hash,
                    "signature": self.signature,
                }
            )
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ToolReceipt":
        names = {
            "receipt_id", "tenant_id", "session_id", "branch_id", "principal_id", "plan_id", "tool",
            "idempotency_key", "args_hash", "status", "error_class", "error_message", "artifact_refs",
            "input_hashes", "output_hashes", "output", "metrics", "trust", "timeout_class", "created_at",
            "key_id", "receipt_hash", "signature",
        }
        unknown = set(d) - names
        missing = names - set(d)
        if unknown or missing:
            raise ValueError(f"malformed receipt: unknown={sorted(unknown)} missing={sorted(missing)}")
        kw = dict(d)
        kw["artifact_refs"] = tuple(dict(r) for r in d["artifact_refs"])
        return cls(**kw)

    def compact(self) -> dict[str, Any]:
        """Compact structured summary returned to the model (§10)."""
        return {
            "tool": self.tool,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "error_class": self.error_class,
            "error_message": self.error_message,
            "artifact_refs": [dict(r) for r in self.artifact_refs],
            "trust": self.trust,
            "receipt_hash": self.receipt_hash,
        }


# -------------------------------------------------------------- verification
@dataclass(frozen=True)
class VerificationRequest:
    """§10 "Verification request": probe type, target artifact, acceptance predicate, evidence location."""

    probe: str
    target_artifact: str
    acceptance_predicate: str
    evidence_location: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "target_artifact": self.target_artifact,
            "acceptance_predicate": self.acceptance_predicate,
            "evidence_location": self.evidence_location,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one probe. ``deterministic`` distinguishes checks from model review."""

    probe: str
    passed: bool
    evidence: Mapping[str, Any]
    deterministic: bool = True
    skipped: bool = False
    target_artifact: str = ""
    receipt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            canonical_json(
                {
                    "probe": self.probe,
                    "passed": self.passed,
                    "evidence": dict(self.evidence),
                    "deterministic": self.deterministic,
                    "skipped": self.skipped,
                    "target_artifact": self.target_artifact,
                    "receipt_hash": self.receipt_hash,
                }
            )
        )


# ------------------------------------------------------------------- branches
@dataclass(frozen=True)
class BranchRequest:
    """§10 "Branch request": parent event, reason, expected state, memory snapshot ID.

    ``expected_state`` maps artifact IDs to the hashes the caller believes are
    current at ``parent_event``; the control plane refuses the branch if the
    log disagrees. ``memory_snapshot_id`` references the copy-on-write neural
    memory snapshot owned by the runtime (§6); it is recorded, not interpreted.
    """

    parent_event: str
    reason: str
    name: Optional[str] = None
    expected_state: Optional[Mapping[str, str]] = None
    memory_snapshot_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_event": self.parent_event,
            "reason": self.reason,
            "name": self.name,
            "expected_state": dict(self.expected_state) if self.expected_state is not None else None,
            "memory_snapshot_id": self.memory_snapshot_id,
        }


# ----------------------------------------------------------------- escalation
@dataclass(frozen=True)
class EscalationRequest:
    """§10 "Escalation request": route to the flagship with minimal sufficient context."""

    failed_capability: str
    evidence: tuple[Mapping[str, Any], ...]
    attempted_repairs: int
    remaining_budget: Mapping[str, Any]
    constraints: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    p_fail: Optional[float] = None
    policy_version: Optional[str] = None
    context_refs: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            canonical_json(
                {
                    "failed_capability": self.failed_capability,
                    "evidence": [dict(e) for e in self.evidence],
                    "attempted_repairs": self.attempted_repairs,
                    "remaining_budget": dict(self.remaining_budget),
                    "constraints": list(self.constraints),
                    "reasons": list(self.reasons),
                    "p_fail": self.p_fail,
                    "policy_version": self.policy_version,
                    "context_refs": [dict(c) for c in self.context_refs],
                }
            )
        )


# ------------------------------------------------------------------ final
@dataclass(frozen=True)
class CompletedArtifact:
    """An artifact version backed by a successful, verified receipt."""

    artifact_id: str
    kind: str
    hash: str
    receipt_hash: str
    tool: str
    idempotency_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "hash": self.hash,
            "receipt_hash": self.receipt_hash,
            "tool": self.tool,
            "idempotency_key": self.idempotency_key,
        }


FINAL_STATUSES = ("succeeded", "partial", "failed", "escalated")


@dataclass(frozen=True)
class FinalResponse:
    """§10 "Final response": never claims successful execution without a matching receipt.

    ``completed_artifacts`` and the ``succeeded`` status are derived by the
    control plane from receipts and verification results, not from model text.
    ``receipt_hashes`` lists the successful, verified receipts that back the
    response's claims; failed calls appear in ``verification_summary`` and,
    unless later repaired, in ``unresolved_issues``.
    :meth:`grounding_violations` re-checks that invariant against any set of
    receipts (e.g. the ones replayed from the event log).
    """

    user_message: str
    status: str
    completed_artifacts: tuple[CompletedArtifact, ...] = ()
    verification_summary: tuple[Mapping[str, Any], ...] = ()
    unresolved_issues: tuple[str, ...] = ()
    plan_ids: tuple[str, ...] = ()
    receipt_hashes: tuple[str, ...] = ()
    escalation: Optional[EscalationRequest] = None
    model_message_used: bool = False

    def __post_init__(self) -> None:
        if self.status not in FINAL_STATUSES:
            raise ValueError(f"invalid final status {self.status!r}")
        if self.status == "succeeded" and self.unresolved_issues:
            raise ValueError("a succeeded final response cannot carry unresolved issues")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            canonical_json(
                {
                    "user_message": self.user_message,
                    "status": self.status,
                    "completed_artifacts": [a.to_dict() for a in self.completed_artifacts],
                    "verification_summary": [dict(v) for v in self.verification_summary],
                    "unresolved_issues": list(self.unresolved_issues),
                    "plan_ids": list(self.plan_ids),
                    "receipt_hashes": list(self.receipt_hashes),
                    "escalation": self.escalation.to_dict() if self.escalation else None,
                    "model_message_used": self.model_message_used,
                }
            )
        )

    def grounding_violations(
        self,
        receipts: Mapping[str, ToolReceipt] | Iterable[ToolReceipt],
        verified_receipts: Optional[Iterable[str]] = None,
        verify_signature: Optional[Any] = None,
    ) -> list[str]:
        """List claims in this response that lack a matching successful receipt.

        ``receipts`` maps receipt hash → receipt (or is an iterable of
        receipts). ``verified_receipts`` (optional) is the set of receipt
        hashes whose verification passed. ``verify_signature`` (optional) is a
        callable ``receipt -> bool`` used to reject forged/corrupted receipts.
        """
        by_hash = dict(receipts) if isinstance(receipts, Mapping) else {r.receipt_hash: r for r in receipts}
        verified = set(verified_receipts) if verified_receipts is not None else None
        problems: list[str] = []
        for art in self.completed_artifacts:
            r = by_hash.get(art.receipt_hash)
            if r is None:
                problems.append(f"{art.artifact_id}: no receipt {art.receipt_hash}")
                continue
            if not r.ok:
                problems.append(f"{art.artifact_id}: receipt status is {r.status!r}")
            if r.output_hashes.get(art.artifact_id) != art.hash:
                problems.append(f"{art.artifact_id}: receipt does not produce hash {art.hash}")
            if verify_signature is not None and not verify_signature(r):
                problems.append(f"{art.artifact_id}: receipt signature does not verify")
            if verified is not None and art.receipt_hash not in verified:
                problems.append(f"{art.artifact_id}: receipt {art.receipt_hash} was not verified")
        if self.status == "succeeded":
            for h in self.receipt_hashes:
                r = by_hash.get(h)
                if r is None or not r.ok:
                    problems.append(f"status 'succeeded' but receipt {h} is missing or not ok")
                elif verified is not None and h not in verified:
                    problems.append(f"status 'succeeded' but receipt {h} was not verified")
        return problems
