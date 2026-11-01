"""Versioned tool specifications and the tool registry (blueprint §10).

Every call names a tool *and version* (``timeline.apply_edit@2``). A
:class:`ToolSpec` is the control plane's contract for that version: its JSON
argument schema (unknown fields rejected), the permissions a principal needs,
its timeout class, whether it has side effects, which argument fields name the
artifacts it may read or write, the trust label of its output, and the
verifiers that always run after it. Tool implementations evolve without
retraining the backbone because the model only ever sees this manifest (§2).

Tool handlers receive a deep copy of the validated arguments and a read-only
:class:`ToolContext`; they *return* new artifact versions in a
:class:`ToolResult` instead of mutating anything. Only the control plane
stores outputs, and only after verification does a version become current.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from ..common import Trust
from .jsonschema_lite import check_schema
from .schemas import ToolRefError, format_tool_ref, parse_tool_ref

__all__ = [
    "ArtifactOutput",
    "ToolResult",
    "ToolContext",
    "ToolHandler",
    "Precheck",
    "ToolSpec",
    "ToolRegistry",
    "UnknownToolError",
    "UnknownToolVersionError",
    "ToolInputError",
]


class UnknownToolError(LookupError):
    """No tool with this name is registered."""


class UnknownToolVersionError(UnknownToolError):
    """The tool exists but not at the requested version."""


class ToolInputError(ValueError):
    """Raised by tools/prechecks for semantically invalid input (becomes a structured error)."""


@dataclass(frozen=True)
class ArtifactOutput:
    """A new immutable version of an artifact produced by a tool."""

    kind: str
    content: Any


@dataclass(frozen=True)
class ToolResult:
    """What a tool returns: compact structured ``data``, new artifact versions, metrics."""

    data: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, ArtifactOutput] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)


class ToolContext:
    """Read-only, capability-limited view handed to tool handlers and prechecks.

    Exposes the committed artifact versions the call is allowed to read (a
    snapshot taken at validation time, so a tool sees a consistent state) and a
    cooperative cancellation flag set when the call's timeout expires.
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        session_id: str,
        branch_id: str,
        principal_id: str,
        artifacts: Mapping[str, tuple[str, str, Any]],
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.branch_id = branch_id
        self.principal_id = principal_id
        self._artifacts = dict(artifacts)  # artifact_id -> (kind, hash, content)
        self._cancel = cancel_event or threading.Event()

    def read_artifact(self, artifact_id: str) -> Any:
        """Deep copy of the committed content of ``artifact_id`` (``ToolInputError`` if absent)."""
        try:
            return copy.deepcopy(self._artifacts[artifact_id][2])
        except KeyError:
            raise ToolInputError(f"artifact {artifact_id!r} is not readable by this call") from None

    def artifact_hash(self, artifact_id: str) -> Optional[str]:
        entry = self._artifacts.get(artifact_id)
        return entry[1] if entry else None

    def artifact_kind(self, artifact_id: str) -> Optional[str]:
        entry = self._artifacts.get(artifact_id)
        return entry[0] if entry else None

    @property
    def readable_artifacts(self) -> tuple[str, ...]:
        return tuple(sorted(self._artifacts))

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def wait_cancelled(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; returns True early if the call was cancelled."""
        return self._cancel.wait(seconds)


ToolHandler = Callable[[dict, ToolContext], "ToolResult | Mapping[str, Any]"]
#: Deterministic pre-execution check: returns a list of error strings (empty = OK).
Precheck = Callable[[dict, ToolContext], list]


@dataclass(frozen=True)
class ToolSpec:
    """Contract for one tool version.

    Attributes:
        name, version: identity; ``ref`` is ``"name@version"``.
        argument_schema: JSON schema (jsonschema_lite subset) for ``arguments``;
            should set ``additionalProperties: false`` (enforced by default).
        permissions: permissions the calling principal must hold.
        timeout_class: the *maximum* timeout class a call may request.
        side_effect: whether the tool may produce new artifact versions.
        output_trust: trust label on outputs; ``UNTRUSTED`` unless the tool is a
            deterministic internal tool whose output is verified by construction.
        reads / writes: argument fields whose values are artifact IDs the call
            may read / produce new versions of (least privilege).
        default_verifiers: verifiers that always run, whatever the model asks.
        requires_confirmation: destructive tools need a user-issued, single-use
            confirmation bound to the exact arguments (excessive-agency control).
        precheck: deterministic static validation against current state
            (e.g. contradictory timeline edits), run *before* execution.
        output_schema: optional schema for ``ToolResult.data``.
        cost: cost units charged to the run budget per execution.
        handler: the implementation (not part of the manifest or equality).
    """

    name: str
    version: int
    argument_schema: Mapping[str, Any]
    handler: ToolHandler = field(repr=False, compare=False)
    permissions: frozenset[str] = frozenset()
    timeout_class: str = "interactive"
    side_effect: bool = False
    output_trust: Trust = Trust.UNTRUSTED
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    default_verifiers: tuple[str, ...] = ()
    requires_confirmation: bool = False
    precheck: Optional[Precheck] = field(default=None, repr=False, compare=False)
    output_schema: Optional[Mapping[str, Any]] = None
    cost: float = 1.0
    description: str = ""
    allow_additional_arguments: bool = False

    def __post_init__(self) -> None:
        format_tool_ref(self.name, self.version)
        try:
            parse_tool_ref(self.ref)
        except ToolRefError as exc:
            raise ValueError(f"invalid tool identity: {exc}") from None
        object.__setattr__(self, "permissions", frozenset(self.permissions))
        object.__setattr__(self, "output_trust", Trust(self.output_trust))
        check_schema(self.argument_schema)
        if self.output_schema is not None:
            check_schema(self.output_schema)
        if self.argument_schema.get("type") != "object":
            raise ValueError(f"{self.ref}: argument_schema must have type 'object'")
        if self.argument_schema.get("additionalProperties", True) is not False and not self.allow_additional_arguments:
            raise ValueError(f"{self.ref}: argument_schema must set additionalProperties: false")
        props = self.argument_schema.get("properties", {})
        for fld in (*self.reads, *self.writes):
            if fld not in props:
                raise ValueError(f"{self.ref}: artifact field {fld!r} is not an argument")
        if self.writes and not self.side_effect:
            raise ValueError(f"{self.ref}: a tool that writes artifacts must declare side_effect=True")
        if self.cost < 0:
            raise ValueError(f"{self.ref}: cost must be non-negative")

    @property
    def ref(self) -> str:
        return format_tool_ref(self.name, self.version)

    def manifest(self) -> dict[str, Any]:
        """What the model sees: the contract, never the implementation."""
        return {
            "tool": self.ref,
            "description": self.description,
            "arguments": copy.deepcopy(dict(self.argument_schema)),
            "permissions": sorted(self.permissions),
            "timeout_class": self.timeout_class,
            "side_effect": self.side_effect,
            "output_trust": self.output_trust.value,
            "default_verifiers": list(self.default_verifiers),
            "requires_confirmation": self.requires_confirmation,
            "cost": self.cost,
        }


class ToolRegistry:
    """Registry of versioned tools; resolution is always by exact ``name@version``."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[int, ToolSpec]] = {}
        self._lock = threading.RLock()

    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
        with self._lock:
            versions = self._tools.setdefault(spec.name, {})
            if spec.version in versions and not replace:
                raise ValueError(f"tool {spec.ref} is already registered")
            versions[spec.version] = spec
            return spec

    def unregister(self, ref: str) -> None:
        name, version = parse_tool_ref(ref)
        with self._lock:
            self.resolve(ref)
            del self._tools[name][version]
            if not self._tools[name]:
                del self._tools[name]

    def resolve(self, ref: str) -> ToolSpec:
        """Return the spec for ``ref``; raises ``ToolRefError``, ``UnknownToolError`` or ``UnknownToolVersionError``."""
        name, version = parse_tool_ref(ref)
        with self._lock:
            versions = self._tools.get(name)
            if not versions:
                raise UnknownToolError(f"unknown tool {name!r}")
            if version not in versions:
                raise UnknownToolVersionError(
                    f"tool {name!r} has no version {version}; available: {sorted(versions)}"
                )
            return versions[version]

    def __contains__(self, ref: object) -> bool:
        if not isinstance(ref, str):
            return False
        try:
            self.resolve(ref)
        except (UnknownToolError, ToolRefError):
            return False
        return True

    def versions(self, name: str) -> tuple[int, ...]:
        return tuple(sorted(self._tools.get(name, {})))

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(s.ref for vs in self._tools.values() for s in vs.values()))

    def manifest(self) -> list[dict[str, Any]]:
        return [self.resolve(r).manifest() for r in self.refs()]
