"""Verifier registry: deterministic checks first, model review only where necessary (§10).

"Verification is task-specific: a timeline edit may require schema comparison
plus a short render probe" and the runtime must "run deterministic checks
first; use model review only where necessary" (§10 table). A
:class:`Verifier` is a named function returning ``(passed, evidence)``. The
:class:`VerifierRegistry` orders requested verifiers so every deterministic
one runs before any non-deterministic (model-review) one, and skips model
review entirely once a deterministic check has failed. Verifier exceptions
become failed results; they never propagate.

Requested verification is *fail-closed*: a verifier that finds nothing to
check (e.g. ``render_probe`` on a call that produced no timeline) fails rather
than passing vacuously.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

from .jsonschema_lite import validate
from .schemas import ToolReceipt, VerificationRequest, VerificationResult
from .timeline import TIMELINE_JSON_SCHEMA, TIMELINE_KIND, render_probe, schema_probe, transition_probe
from .tools import ToolSpec

__all__ = [
    "VerificationContext",
    "Verifier",
    "VerifierFn",
    "VerifierRegistry",
    "UnknownVerifierError",
    "ARTIFACT_SCHEMAS",
    "default_verifier_registry",
]

#: JSON schemas by artifact kind, used by the generic ``json_schema`` verifier.
ARTIFACT_SCHEMAS: dict[str, Mapping[str, Any]] = {TIMELINE_KIND: TIMELINE_JSON_SCHEMA}
_EDIT_TOOL = "timeline.apply_edit"


class UnknownVerifierError(LookupError):
    pass


@dataclass(frozen=True)
class VerificationContext:
    """Everything a verifier may inspect; all values are detached copies."""

    receipt: ToolReceipt
    spec: ToolSpec
    arguments: Mapping[str, Any]
    inputs: Mapping[str, Any] = field(default_factory=dict)
    input_kinds: Mapping[str, str] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    output_kinds: Mapping[str, str] = field(default_factory=dict)
    output_data: Mapping[str, Any] = field(default_factory=dict)
    request: Optional[VerificationRequest] = None

    def outputs_of_kind(self, kind: str) -> dict[str, Any]:
        return {aid: c for aid, c in self.outputs.items() if self.output_kinds.get(aid) == kind}


VerifierFn = Callable[[VerificationContext], "tuple[bool, Mapping[str, Any]]"]


@dataclass(frozen=True)
class Verifier:
    name: str
    fn: VerifierFn = field(repr=False, compare=False)
    deterministic: bool = True
    acceptance_predicate: str = ""


class VerifierRegistry:
    """Named verifiers with deterministic-first execution."""

    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}

    def register(
        self,
        name: str,
        fn: VerifierFn,
        *,
        deterministic: bool = True,
        acceptance_predicate: str = "",
        replace: bool = False,
    ) -> Verifier:
        if name in self._verifiers and not replace:
            raise ValueError(f"verifier {name!r} already registered")
        v = Verifier(name, fn, deterministic, acceptance_predicate)
        self._verifiers[name] = v
        return v

    def get(self, name: str) -> Verifier:
        try:
            return self._verifiers[name]
        except KeyError:
            raise UnknownVerifierError(f"unknown verifier {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._verifiers

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._verifiers))

    def unknown(self, names: Iterable[str]) -> list[str]:
        return [n for n in names if n not in self._verifiers]

    def order(self, names: Iterable[str]) -> list[Verifier]:
        """De-duplicated verifiers, deterministic ones first (stable within each group)."""
        seen: dict[str, Verifier] = {}
        for n in names:
            if n not in seen:
                seen[n] = self.get(n)
        vs = list(seen.values())
        return [v for v in vs if v.deterministic] + [v for v in vs if not v.deterministic]

    def run(
        self,
        names: Iterable[str],
        ctx: VerificationContext,
        *,
        target_artifact: str = "",
        evidence_location: str = "",
    ) -> list[VerificationResult]:
        """Run verifiers deterministic-first; skip model review after any deterministic failure."""
        results: list[VerificationResult] = []
        failed = False
        for v in self.order(names):
            req = VerificationRequest(
                probe=v.name,
                target_artifact=target_artifact,
                acceptance_predicate=v.acceptance_predicate or f"{v.name} passes",
                evidence_location=evidence_location,
            )
            if failed and not v.deterministic:
                results.append(
                    VerificationResult(
                        v.name, False, {"reason": "skipped: a deterministic check already failed"},
                        deterministic=False, skipped=True, target_artifact=target_artifact,
                        receipt_hash=ctx.receipt.receipt_hash,
                    )
                )
                continue
            try:
                passed, evidence = v.fn(dataclasses.replace(ctx, request=req))
                passed = bool(passed)
                evidence = dict(evidence)
            except Exception as exc:  # verifier bugs are failures, never crashes
                passed, evidence = False, {"error": f"{type(exc).__name__}: {exc}"}
            results.append(
                VerificationResult(
                    v.name, passed, evidence, deterministic=v.deterministic,
                    target_artifact=target_artifact, receipt_hash=ctx.receipt.receipt_hash,
                )
            )
            failed = failed or not passed
        return results


# ----------------------------------------------------------- built-in checks
def _json_schema(ctx: VerificationContext) -> tuple[bool, dict[str, Any]]:
    checked: list[str] = []
    errors: dict[str, list[str]] = {}
    if ctx.spec.output_schema is not None:
        checked.append("data")
        errs = validate(dict(ctx.output_data), ctx.spec.output_schema)
        if errs:
            errors["data"] = errs
    for aid, content in sorted(ctx.outputs.items()):
        schema = ARTIFACT_SCHEMAS.get(ctx.output_kinds.get(aid, ""))
        if schema is None:
            continue
        checked.append(aid)
        errs = validate(content, schema)
        if errs:
            errors[aid] = errs
    if not checked:
        return False, {"reason": "nothing verifiable: no output schema and no output of a known kind"}
    return not errors, {"checked": checked, "errors": errors}


def _per_timeline(
    ctx: VerificationContext, probe: Callable[[str, Any], tuple[bool, dict[str, Any]]]
) -> tuple[bool, dict[str, Any]]:
    timelines = ctx.outputs_of_kind(TIMELINE_KIND)
    if not timelines:
        return False, {"reason": "no timeline output to verify"}
    ok = True
    evidence: dict[str, Any] = {}
    for aid, content in sorted(timelines.items()):
        passed, ev = probe(aid, content)
        ok = ok and passed
        evidence[aid] = ev
    return ok, evidence


def _edit_operations(ctx: VerificationContext) -> Optional[list[Any]]:
    if ctx.spec.name == _EDIT_TOOL and isinstance(ctx.arguments.get("operations"), list):
        return list(ctx.arguments["operations"])
    return None


def _timeline_schema(ctx: VerificationContext) -> tuple[bool, dict[str, Any]]:
    ops = _edit_operations(ctx)

    def probe(aid: str, content: Any) -> tuple[bool, dict[str, Any]]:
        before = ctx.inputs.get(aid) if ctx.input_kinds.get(aid) == TIMELINE_KIND else None
        return schema_probe(content, before if ops is not None else None, ops)

    return _per_timeline(ctx, probe)


def _render_probe(ctx: VerificationContext) -> tuple[bool, dict[str, Any]]:
    return _per_timeline(ctx, lambda aid, content: render_probe(content))


def _transition_probe(ctx: VerificationContext) -> tuple[bool, dict[str, Any]]:
    ops = _edit_operations(ctx)
    return _per_timeline(ctx, lambda aid, content: transition_probe(content, ops))


def default_verifier_registry() -> VerifierRegistry:
    """Registry with ``json_schema``, ``timeline_schema``, ``render_probe``, ``transition_probe``."""
    reg = VerifierRegistry()
    reg.register("json_schema", _json_schema, acceptance_predicate="outputs validate against their declared schemas")
    reg.register(
        "timeline_schema",
        _timeline_schema,
        acceptance_predicate="timeline is schema-valid, satisfies invariants, and equals the replay of the operations",
    )
    reg.register(
        "render_probe",
        _render_probe,
        acceptance_predicate="25 fps probe: duration > 0, no gaps, no overlaps outside transitions, sources in range",
    )
    reg.register(
        "transition_probe",
        _transition_probe,
        acceptance_predicate="every transition fits both adjacent clips and requested transitions are present",
    )
    return reg
