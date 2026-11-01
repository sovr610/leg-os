"""Agent subsystem: action contracts, deterministic control plane, verifiers, escalation (blueprint §10).

The model proposes; the control plane disposes. The middle model returns a
bounded ``m63.actions/v1`` envelope (:mod:`.schemas`); the deterministic
:class:`ControlPlane` validates it statically, executes tools in a sandbox,
issues signed receipts into the authoritative event log, verifies outputs
(:mod:`.verifiers`), and commits or branches. The :class:`Orchestrator` runs
``intake → context bundle → plan → static validation → execute → verify →
repair/branch → finalize`` and consults the calibrated :class:`EscalationRouter`
(:mod:`.escalation`). :mod:`.timeline` is a small deterministic editing domain
that makes the loop testable end to end.

This package depends only on the standard library and numpy.
"""

from __future__ import annotations

from .artifacts import ArtifactNotFound, ArtifactRef, ArtifactStore
from .control_plane import (
    DEFAULT_TIMEOUT_CLASSES,
    AuditReport,
    BranchError,
    CallOutcome,
    ControlPlane,
    ControlPlaneLimits,
    PreflightReport,
    Principal,
    RunBudget,
    Session,
)
from .escalation import (
    DEFAULT_POLICY,
    FEATURE_NAMES,
    HARD_RULES,
    EscalationContext,
    EscalationDecision,
    EscalationFeatures,
    EscalationPolicy,
    EscalationRouter,
    PolicyRegistry,
    RouterMetrics,
    constraint_complexity,
    evaluate_policy,
    expected_calibration_error,
    fit_logistic,
    fit_platt,
    fit_policy,
    uncertainty_from_text,
)
from .jsonschema_lite import SchemaError, check_schema, is_valid, json_value_errors, validate
from .orchestrator import (
    ContextBundle,
    Orchestrator,
    OrchestratorConfig,
    PlannerResponse,
    RunResult,
    ScriptedPlanner,
    StepRecord,
    UserRequest,
    final_envelope,
)
from .schemas import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ActionCall,
    ActionEnvelope,
    BranchRequest,
    Budget,
    CompletedArtifact,
    EnvelopeError,
    ErrorClass,
    EscalationFlag,
    EscalationRequest,
    FinalResponse,
    ReceiptStatus,
    ToolReceipt,
    ToolRefError,
    VerificationRequest,
    VerificationResult,
    format_tool_ref,
    parse_envelope,
    parse_tool_ref,
)
from .security import REDACTED, ReceiptSigner, Redactor
from .timeline import (
    Clip,
    Timeline,
    TimelineError,
    Transition,
    apply_edit_tool,
    apply_operations,
    find_conflicts,
    render_probe,
    schema_probe,
    summarize_timeline,
    transition_probe,
    validate_operations,
    validate_timeline,
)
from .tools import (
    ArtifactOutput,
    ToolContext,
    ToolInputError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    UnknownToolError,
    UnknownToolVersionError,
)
from .verifiers import (
    VerificationContext,
    Verifier,
    VerifierRegistry,
    UnknownVerifierError,
    default_verifier_registry,
)

__all__ = [
    # schemas / contracts
    "SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS", "ActionCall", "ActionEnvelope", "Budget", "EscalationFlag",
    "EnvelopeError", "ToolRefError", "parse_envelope", "parse_tool_ref", "format_tool_ref", "ReceiptStatus",
    "ErrorClass", "ToolReceipt", "VerificationRequest", "VerificationResult", "BranchRequest",
    "EscalationRequest", "CompletedArtifact", "FinalResponse",
    # json schema
    "SchemaError", "check_schema", "validate", "is_valid", "json_value_errors",
    # tools / artifacts / security
    "ToolSpec", "ToolRegistry", "ToolContext", "ToolResult", "ArtifactOutput", "ToolInputError",
    "UnknownToolError", "UnknownToolVersionError", "ArtifactStore", "ArtifactRef", "ArtifactNotFound",
    "Redactor", "ReceiptSigner", "REDACTED",
    # timeline domain
    "Clip", "Transition", "Timeline", "TimelineError", "validate_timeline", "find_conflicts",
    "validate_operations", "apply_operations", "apply_edit_tool", "summarize_timeline", "schema_probe",
    "render_probe", "transition_probe",
    # verifiers
    "VerificationContext", "Verifier", "VerifierRegistry", "UnknownVerifierError", "default_verifier_registry",
    # control plane
    "DEFAULT_TIMEOUT_CLASSES", "Principal", "ControlPlaneLimits", "RunBudget", "Session", "CallOutcome",
    "PreflightReport", "AuditReport", "BranchError", "ControlPlane",
    # escalation
    "FEATURE_NAMES", "HARD_RULES", "EscalationFeatures", "EscalationContext", "EscalationPolicy",
    "EscalationDecision", "EscalationRouter", "PolicyRegistry", "DEFAULT_POLICY", "uncertainty_from_text",
    "constraint_complexity", "fit_logistic", "fit_platt", "fit_policy", "expected_calibration_error",
    "RouterMetrics", "evaluate_policy",
    # orchestrator
    "UserRequest", "ContextBundle", "PlannerResponse", "ScriptedPlanner", "final_envelope",
    "OrchestratorConfig", "StepRecord", "RunResult", "Orchestrator",
]
