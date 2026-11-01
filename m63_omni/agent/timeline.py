"""A small deterministic editing-timeline domain (blueprint §10 example).

This makes the orchestration loop testable end to end without media: the §10
example envelope calls ``timeline.apply_edit@2`` with ``trim`` and
``transition`` operations and asks for ``timeline_schema``, ``render_probe``
and ``transition_probe`` verification. Everything here is a pure function of
JSON-like data, so edits are replayable and their outputs content-addressed.

Model
-----
A :class:`Timeline` is an ordered single track of :class:`Clip` objects.
``start_ms``/``end_ms`` are *source* in/out points within the clip's asset;
``assets`` maps asset IDs to their source duration (a media constraint). Clips
play back to back; a :class:`Transition` after clip *i* overlaps the tail of
clip *i* and the head of clip *i+1* by ``duration_ms`` (``cut`` has 0 ms), so
the programme duration is ``sum(clip durations) - sum(transition durations)``.

Static validation (run by the control plane *before* execution, §10) rejects
contradictory batches (:func:`find_conflicts`: e.g. two different trims of one
clip, or editing a clip the same batch deletes) and any batch whose sequential
application or final state violates an invariant (:func:`validate_timeline`:
``end_ms <= start_ms``, out-of-range trims, transitions after a nonexistent or
last clip, transitions longer than an adjacent clip, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..common import Trust, content_hash
from .jsonschema_lite import check_schema, validate
from .tools import ArtifactOutput, ToolContext, ToolInputError, ToolResult, ToolSpec

__all__ = [
    "TIMELINE_KIND",
    "TRANSITION_KINDS",
    "TIMELINE_JSON_SCHEMA",
    "OPERATION_SCHEMAS",
    "APPLY_EDIT_ARGS_SCHEMA",
    "TimelineError",
    "Clip",
    "Transition",
    "Timeline",
    "validate_timeline",
    "find_conflicts",
    "validate_operations",
    "apply_operations",
    "apply_edit_tool",
    "summarize_timeline",
    "schema_probe",
    "render_probe",
    "transition_probe",
]

TIMELINE_KIND = "timeline"
TRANSITION_KINDS = ("cut", "dissolve", "fade", "wipe")
MAX_CLIPS = 512
MAX_OPERATIONS = 64
MAX_TRANSITION_MS = 10_000
MAX_SOURCE_MS = 24 * 3600 * 1000
PROBE_STEP_MS = 40  # 25 fps
PROBE_MAX_SAMPLES = 20_000

_ID = {"type": "string", "minLength": 1, "maxLength": 128, "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.:\-]*$"}
_MS = {"type": "integer", "minimum": 0, "maximum": MAX_SOURCE_MS}

_CLIP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["clip_id", "asset_id", "start_ms", "end_ms"],
    "properties": {"clip_id": _ID, "asset_id": _ID, "start_ms": _MS, "end_ms": _MS},
}
_TRANSITION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["after_clip_id", "kind", "duration_ms"],
    "properties": {
        "after_clip_id": _ID,
        "kind": {"enum": list(TRANSITION_KINDS)},
        "duration_ms": {"type": "integer", "minimum": 0, "maximum": MAX_TRANSITION_MS},
    },
}
TIMELINE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["timeline_id", "version", "clips", "transitions", "assets"],
    "properties": {
        "timeline_id": _ID,
        "version": {"type": "integer", "minimum": 0},
        "clips": {"type": "array", "maxItems": MAX_CLIPS, "items": _CLIP_SCHEMA},
        "transitions": {"type": "array", "maxItems": MAX_CLIPS, "items": _TRANSITION_SCHEMA},
        "assets": {
            "type": "object",
            "propertyNames": _ID,
            "additionalProperties": {"type": "integer", "minimum": 1, "maximum": MAX_SOURCE_MS},
        },
    },
}

OPERATION_SCHEMAS: dict[str, dict[str, Any]] = {
    "trim": {
        "type": "object",
        "additionalProperties": False,
        "required": ["op", "clip_id", "start_ms", "end_ms"],
        "properties": {"op": {"const": "trim"}, "clip_id": _ID, "start_ms": _MS, "end_ms": _MS},
    },
    "transition": {
        "type": "object",
        "additionalProperties": False,
        "required": ["op", "after_clip_id", "kind", "duration_ms"],
        "properties": {
            "op": {"const": "transition"},
            "after_clip_id": _ID,
            "kind": {"enum": list(TRANSITION_KINDS)},
            "duration_ms": {"type": "integer", "minimum": 0, "maximum": MAX_TRANSITION_MS},
        },
    },
    "insert": {
        "type": "object",
        "additionalProperties": False,
        "required": ["op", "clip_id", "asset_id", "start_ms", "end_ms", "after_clip_id"],
        "properties": {
            "op": {"const": "insert"},
            "clip_id": _ID,
            "asset_id": _ID,
            "start_ms": _MS,
            "end_ms": _MS,
            "after_clip_id": {"type": ["string", "null"], "minLength": 1, "maxLength": 128},
        },
    },
    "delete": {
        "type": "object",
        "additionalProperties": False,
        "required": ["op", "clip_id"],
        "properties": {"op": {"const": "delete"}, "clip_id": _ID},
    },
}

APPLY_EDIT_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["timeline_id", "operations"],
    "properties": {
        "timeline_id": _ID,
        "operations": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_OPERATIONS,
            "items": {"oneOf": [OPERATION_SCHEMAS[k] for k in ("trim", "transition", "insert", "delete")]},
        },
    },
}
for _s in (TIMELINE_JSON_SCHEMA, APPLY_EDIT_ARGS_SCHEMA):
    check_schema(_s)


class TimelineError(ValueError):
    """Invalid timeline or edit batch; ``errors`` lists every problem found."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors) or "invalid timeline")


# ----------------------------------------------------------------- data model
@dataclass(frozen=True)
class Clip:
    clip_id: str
    asset_id: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {"clip_id": self.clip_id, "asset_id": self.asset_id, "start_ms": self.start_ms, "end_ms": self.end_ms}


@dataclass(frozen=True)
class Transition:
    after_clip_id: str
    kind: str
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {"after_clip_id": self.after_clip_id, "kind": self.kind, "duration_ms": self.duration_ms}


@dataclass(frozen=True)
class Timeline:
    """Immutable timeline value. Use :meth:`to_dict`/:meth:`from_dict` at boundaries."""

    timeline_id: str
    clips: tuple[Clip, ...] = ()
    transitions: tuple[Transition, ...] = ()
    assets: Mapping[str, int] = field(default_factory=dict)
    version: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "clips", tuple(self.clips))
        object.__setattr__(self, "assets", dict(sorted(dict(self.assets).items())))
        order = {c.clip_id: i for i, c in enumerate(self.clips)}
        trs = sorted(self.transitions, key=lambda t: (order.get(t.after_clip_id, len(order)), t.after_clip_id))
        object.__setattr__(self, "transitions", tuple(trs))

    # -------------------------------------------------------------- access
    def index(self, clip_id: str) -> Optional[int]:
        for i, c in enumerate(self.clips):
            if c.clip_id == clip_id:
                return i
        return None

    def clip(self, clip_id: str) -> Optional[Clip]:
        i = self.index(clip_id)
        return self.clips[i] if i is not None else None

    def transition_after(self, clip_id: str) -> Optional[Transition]:
        for t in self.transitions:
            if t.after_clip_id == clip_id:
                return t
        return None

    def layout(self) -> list[dict[str, int | str]]:
        """Record-in/out positions of each clip on the programme timeline."""
        out: list[dict[str, int | str]] = []
        t = 0
        for i, c in enumerate(self.clips):
            if i > 0:
                prev = self.transition_after(self.clips[i - 1].clip_id)
                t -= prev.duration_ms if prev else 0
            out.append({"clip_id": c.clip_id, "record_in_ms": t, "record_out_ms": t + c.duration_ms})
            t += c.duration_ms
        return out

    @property
    def duration_ms(self) -> int:
        lay = self.layout()
        return int(lay[-1]["record_out_ms"]) if lay else 0

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline_id": self.timeline_id,
            "version": self.version,
            "clips": [c.to_dict() for c in self.clips],
            "transitions": [t.to_dict() for t in self.transitions],
            "assets": dict(self.assets),
        }

    @property
    def content_hash(self) -> str:
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Timeline":
        """Strictly parse a timeline dict (schema-checked; unknown fields rejected)."""
        errors = validate(d, TIMELINE_JSON_SCHEMA)
        if errors:
            raise TimelineError(errors)
        return cls(
            timeline_id=d["timeline_id"],
            clips=tuple(Clip(**c) for c in d["clips"]),
            transitions=tuple(Transition(**t) for t in d["transitions"]),
            assets=dict(d["assets"]),
            version=d["version"],
        )


# -------------------------------------------------------------- invariants
def validate_timeline(tl: Timeline) -> list[str]:
    """Every invariant a committed timeline must satisfy (empty list = valid)."""
    errors = validate(tl.to_dict(), TIMELINE_JSON_SCHEMA)
    if errors:
        return errors
    seen: set[str] = set()
    for c in tl.clips:
        if c.clip_id in seen:
            errors.append(f"clip {c.clip_id}: duplicate clip_id")
        seen.add(c.clip_id)
        if c.asset_id not in tl.assets:
            errors.append(f"clip {c.clip_id}: unknown asset {c.asset_id!r}")
        if c.end_ms <= c.start_ms:
            errors.append(f"clip {c.clip_id}: end_ms ({c.end_ms}) must be greater than start_ms ({c.start_ms})")
        elif c.asset_id in tl.assets and c.end_ms > tl.assets[c.asset_id]:
            errors.append(
                f"clip {c.clip_id}: end_ms ({c.end_ms}) exceeds asset {c.asset_id} duration ({tl.assets[c.asset_id]})"
            )
    seen_tr: set[str] = set()
    incoming: dict[str, int] = {}
    outgoing: dict[str, int] = {}
    for t in tl.transitions:
        if t.after_clip_id in seen_tr:
            errors.append(f"transition after {t.after_clip_id}: more than one transition after the same clip")
        seen_tr.add(t.after_clip_id)
        i = tl.index(t.after_clip_id)
        if i is None:
            errors.append(f"transition after {t.after_clip_id}: clip does not exist")
            continue
        if i == len(tl.clips) - 1:
            errors.append(f"transition after {t.after_clip_id}: clip is the last clip; nothing to transition into")
            continue
        if t.kind not in TRANSITION_KINDS:
            errors.append(f"transition after {t.after_clip_id}: unknown kind {t.kind!r}")
        if t.kind == "cut" and t.duration_ms != 0:
            errors.append(f"transition after {t.after_clip_id}: a cut must have duration_ms 0")
        if t.kind != "cut" and t.duration_ms <= 0:
            errors.append(f"transition after {t.after_clip_id}: {t.kind} needs duration_ms > 0")
        a, b = tl.clips[i], tl.clips[i + 1]
        for c in (a, b):
            if c.duration_ms > 0 and t.duration_ms > c.duration_ms:
                errors.append(
                    f"transition after {t.after_clip_id}: duration {t.duration_ms} ms is longer than adjacent clip "
                    f"{c.clip_id} ({c.duration_ms} ms)"
                )
        outgoing[a.clip_id] = t.duration_ms
        incoming[b.clip_id] = t.duration_ms
    for c in tl.clips:
        both = incoming.get(c.clip_id, 0) + outgoing.get(c.clip_id, 0)
        if c.duration_ms > 0 and incoming.get(c.clip_id) and outgoing.get(c.clip_id) and both > c.duration_ms:
            errors.append(
                f"clip {c.clip_id}: incoming + outgoing transitions ({both} ms) exceed its duration ({c.duration_ms} ms)"
            )
    return errors


# --------------------------------------------------------------- operations
def _op_targets(op: Mapping[str, Any]) -> list[str]:
    kind = op.get("op")
    if kind == "transition":
        return [op.get("after_clip_id")]  # type: ignore[list-item]
    if kind == "insert":
        return [op.get("clip_id"), op.get("after_clip_id")]  # type: ignore[list-item]
    return [op.get("clip_id")]  # type: ignore[list-item]


def find_conflicts(operations: Sequence[Mapping[str, Any]]) -> list[str]:
    """Contradictory operations within one batch (independent of timeline state).

    * a clip deleted in the batch may not be referenced by any other operation;
    * at most one ``trim`` per clip and one ``transition`` after a clip
      (identical duplicates are reported as redundant, differing ones as
      contradictory);
    * an ``insert`` may not reuse a clip_id inserted by another operation.
    """
    errors: list[str] = []
    deletes: dict[str, int] = {}
    trims: dict[str, int] = {}
    transitions: dict[str, int] = {}
    inserts: dict[str, int] = {}

    def record(kind: str, idx: dict[str, int], key: Any, i: int) -> None:
        if not isinstance(key, str):
            return
        if key not in idx:
            idx[key] = i
            return
        j = idx[key]
        verb = "duplicates" if dict(operations[j]) == dict(operations[i]) else "contradicts"
        errors.append(f"operations[{i}]: {kind} on {key} {verb} operations[{j}]")

    by_kind = {"delete": deletes, "trim": trims, "insert": inserts}
    for i, op in enumerate(operations):
        kind = op.get("op")
        if kind in by_kind:
            record(kind, by_kind[kind], op.get("clip_id"), i)
        elif kind == "transition":
            record("transition", transitions, op.get("after_clip_id"), i)
    for i, op in enumerate(operations):
        if op.get("op") == "delete":
            continue
        for target in _op_targets(op):
            if isinstance(target, str) and target in deletes:
                errors.append(
                    f"operations[{i}]: {op.get('op')} references clip {target} which operations[{deletes[target]}] deletes"
                )
    return errors


class _State:
    """Mutable scratch state used while applying a batch sequentially."""

    def __init__(self, tl: Timeline) -> None:
        self.timeline_id = tl.timeline_id
        self.version = tl.version
        self.assets = dict(tl.assets)
        self.clips: list[Clip] = list(tl.clips)
        self.transitions: dict[str, Transition] = {t.after_clip_id: t for t in tl.transitions}

    def index(self, clip_id: Any) -> Optional[int]:
        for i, c in enumerate(self.clips):
            if c.clip_id == clip_id:
                return i
        return None

    def freeze(self, version: int) -> Timeline:
        return Timeline(self.timeline_id, tuple(self.clips), tuple(self.transitions.values()), self.assets, version)


def _check_range(where: str, asset_id: str, start: int, end: int, assets: Mapping[str, int]) -> list[str]:
    errs: list[str] = []
    if end <= start:
        errs.append(f"{where}: end_ms ({end}) must be greater than start_ms ({start})")
    if asset_id not in assets:
        errs.append(f"{where}: unknown asset {asset_id!r}")
    elif end > assets[asset_id]:
        errs.append(f"{where}: end_ms ({end}) exceeds asset {asset_id} duration ({assets[asset_id]})")
    return errs


def _apply_one(state: _State, op: Mapping[str, Any], i: int) -> list[str]:
    where = f"operations[{i}] ({op.get('op')})"
    kind = op["op"]
    if kind == "trim":
        idx = state.index(op["clip_id"])
        if idx is None:
            return [f"{where}: clip {op['clip_id']} does not exist"]
        clip = state.clips[idx]
        errs = _check_range(where, clip.asset_id, op["start_ms"], op["end_ms"], state.assets)
        if not errs:
            state.clips[idx] = Clip(clip.clip_id, clip.asset_id, op["start_ms"], op["end_ms"])
        return errs
    if kind == "transition":
        idx = state.index(op["after_clip_id"])
        if idx is None:
            return [f"{where}: clip {op['after_clip_id']} does not exist"]
        if idx == len(state.clips) - 1:
            return [f"{where}: clip {op['after_clip_id']} is the last clip; nothing to transition into"]
        state.transitions[op["after_clip_id"]] = Transition(op["after_clip_id"], op["kind"], op["duration_ms"])
        return []
    if kind == "insert":
        if state.index(op["clip_id"]) is not None:
            return [f"{where}: clip {op['clip_id']} already exists"]
        errs = _check_range(where, op["asset_id"], op["start_ms"], op["end_ms"], state.assets)
        after = op["after_clip_id"]
        pos = 0
        if after is not None:
            j = state.index(after)
            if j is None:
                errs.append(f"{where}: insert position clip {after} does not exist")
            else:
                pos = j + 1
        if not errs:
            state.clips.insert(pos, Clip(op["clip_id"], op["asset_id"], op["start_ms"], op["end_ms"]))
        return errs
    if kind == "delete":
        idx = state.index(op["clip_id"])
        if idx is None:
            return [f"{where}: clip {op['clip_id']} does not exist"]
        removed = state.clips.pop(idx)
        state.transitions.pop(removed.clip_id, None)
        if idx == len(state.clips) and idx > 0:  # removed the last clip: predecessor's transition dangles
            state.transitions.pop(state.clips[idx - 1].clip_id, None)
        return []
    return [f"{where}: unknown operation {kind!r}"]


def _apply(tl: Timeline, operations: Sequence[Mapping[str, Any]]) -> tuple[Optional[Timeline], list[str]]:
    errors = validate({"timeline_id": tl.timeline_id, "operations": list(operations)}, APPLY_EDIT_ARGS_SCHEMA)
    if errors:
        return None, errors
    errors = find_conflicts(operations)
    state = _State(tl)
    for i, op in enumerate(operations):
        errors.extend(_apply_one(state, op, i))
    if errors:
        return None, errors
    new = state.freeze(tl.version + 1)
    final_errors = validate_timeline(new)
    if final_errors:
        return None, [f"resulting timeline: {e}" for e in final_errors]
    return new, []


def validate_operations(tl: Timeline, operations: Sequence[Mapping[str, Any]]) -> list[str]:
    """Static validation of a batch against ``tl`` (empty list = the batch may execute)."""
    return _apply(tl, operations)[1]


def apply_operations(tl: Timeline, operations: Sequence[Mapping[str, Any]]) -> tuple[Timeline, str]:
    """Pure edit: return ``(new_timeline, content_hash)`` or raise :class:`TimelineError`."""
    new, errors = _apply(tl, operations)
    if new is None:
        raise TimelineError(errors)
    return new, new.content_hash


def summarize_timeline(content: Mapping[str, Any]) -> dict[str, Any]:
    """Compact representation for the model's context bundle (§10)."""
    try:
        tl = Timeline.from_dict(content)
    except TimelineError as exc:
        return {"invalid": True, "errors": exc.errors[:5]}
    return {
        "version": tl.version,
        "duration_ms": tl.duration_ms,
        "clips": [c.to_dict() for c in tl.clips],
        "transitions": [t.to_dict() for t in tl.transitions],
        "assets": dict(tl.assets),
    }


# --------------------------------------------------------------------- tool
def _read_timeline(args: Mapping[str, Any], ctx: ToolContext) -> Timeline:
    content = ctx.read_artifact(args["timeline_id"])
    if ctx.artifact_kind(args["timeline_id"]) not in (None, TIMELINE_KIND):
        raise ToolInputError(f"artifact {args['timeline_id']} is not a timeline")
    tl = Timeline.from_dict(content)
    if tl.timeline_id != args["timeline_id"]:
        raise ToolInputError(f"artifact {args['timeline_id']} contains timeline {tl.timeline_id}")
    return tl


def _apply_edit_precheck(args: dict, ctx: ToolContext) -> list[str]:
    try:
        tl = _read_timeline(args, ctx)
    except (ToolInputError, TimelineError) as exc:
        return [str(exc)]
    return validate_operations(tl, args["operations"])


def _apply_edit_handler(args: dict, ctx: ToolContext) -> ToolResult:
    tl = _read_timeline(args, ctx)
    new, h = apply_operations(tl, args["operations"])
    return ToolResult(
        data={
            "timeline_id": new.timeline_id,
            "version": new.version,
            "hash": h,
            "duration_ms": new.duration_ms,
            "clip_count": len(new.clips),
            "operations_applied": len(args["operations"]),
        },
        outputs={new.timeline_id: ArtifactOutput(TIMELINE_KIND, new.to_dict())},
        metrics={"operations": float(len(args["operations"])), "duration_ms": float(new.duration_ms)},
    )


def apply_edit_tool(*, cost: float = 1.0) -> ToolSpec:
    """The ``timeline.apply_edit@2`` tool spec (deterministic, verified-output tool)."""
    return ToolSpec(
        name="timeline.apply_edit",
        version=2,
        argument_schema=APPLY_EDIT_ARGS_SCHEMA,
        handler=_apply_edit_handler,
        permissions=frozenset({"timeline:write"}),
        timeout_class="standard_media_edit",
        side_effect=True,
        output_trust=Trust.TOOL_VERIFIED,
        reads=("timeline_id",),
        writes=("timeline_id",),
        default_verifiers=("timeline_schema",),
        precheck=_apply_edit_precheck,
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["timeline_id", "version", "hash", "duration_ms", "clip_count", "operations_applied"],
            "properties": {
                "timeline_id": _ID,
                "version": {"type": "integer", "minimum": 0},
                "hash": {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"},
                "duration_ms": {"type": "integer", "minimum": 0},
                "clip_count": {"type": "integer", "minimum": 0},
                "operations_applied": {"type": "integer", "minimum": 0},
            },
        },
        cost=cost,
        description="Apply trim/transition/insert/delete operations to a timeline, producing a new version.",
    )


# ------------------------------------------------------------------- probes
def schema_probe(
    after: Any, before: Optional[Mapping[str, Any]] = None, operations: Optional[Sequence[Mapping[str, Any]]] = None
) -> tuple[bool, dict[str, Any]]:
    """Schema validity plus *schema comparison* with the input (§10).

    When ``before`` and ``operations`` are given, the output must equal the
    deterministic re-application of ``operations`` to ``before`` — so a tool
    that silently changed an unreferenced clip fails here.
    """
    evidence: dict[str, Any] = {"schema_errors": validate(after, TIMELINE_JSON_SCHEMA)}
    if evidence["schema_errors"]:
        return False, evidence
    tl = Timeline.from_dict(after)
    evidence["invariant_errors"] = validate_timeline(tl)
    evidence["actual_hash"] = tl.content_hash
    ok = not evidence["invariant_errors"]
    if before is not None and operations is not None:
        try:
            expected, expected_hash = apply_operations(Timeline.from_dict(before), operations)
            evidence["expected_hash"] = expected_hash
            evidence["matches_replay"] = expected_hash == tl.content_hash
            before_ids = {c["clip_id"] for c in before["clips"]}
            after_ids = {c.clip_id for c in tl.clips}
            evidence["clips_added"] = sorted(after_ids - before_ids)
            evidence["clips_removed"] = sorted(before_ids - after_ids)
            ok = ok and evidence["matches_replay"]
        except TimelineError as exc:
            evidence["replay_errors"] = exc.errors
            ok = False
    return ok, evidence


def render_probe(after: Any) -> tuple[bool, dict[str, Any]]:
    """Short deterministic "render": sample the programme at 25 fps.

    Every sample must be covered by at least one clip (no gaps / black frames),
    by at most two clips, and by two only inside a declared transition window
    (no unintended overlaps); every sampled source time must lie inside the
    clip's in/out points and the asset's duration; total duration must be > 0.
    """
    errs = validate(after, TIMELINE_JSON_SCHEMA)
    if errs:
        return False, {"schema_errors": errs}
    tl = Timeline.from_dict(after)
    lay = tl.layout()
    duration = tl.duration_ms
    evidence: dict[str, Any] = {"duration_ms": duration, "clip_count": len(tl.clips)}
    problems: list[str] = []
    if duration <= 0:
        problems.append("total duration must be > 0")
    for i, (c, pos) in enumerate(zip(tl.clips, lay)):
        if c.duration_ms <= 0:
            problems.append(f"clip {c.clip_id} has non-positive duration")
        if i > 0 and pos["record_in_ms"] < lay[i - 1]["record_in_ms"]:  # type: ignore[operator]
            problems.append(f"clip {c.clip_id} starts before its predecessor")
    windows = []
    for i in range(len(tl.clips) - 1):
        t = tl.transition_after(tl.clips[i].clip_id)
        if t and t.duration_ms:
            end = int(lay[i]["record_out_ms"])
            windows.append((end - t.duration_ms, end))
    step = max(PROBE_STEP_MS, -(-duration // PROBE_MAX_SAMPLES)) if duration > 0 else PROBE_STEP_MS
    gap_frames = overlap_frames = out_of_range = samples = 0
    trace: list[Any] = []
    for t_ms in range(0, max(duration, 0), step):
        samples += 1
        active = [i for i, p in enumerate(lay) if p["record_in_ms"] <= t_ms < p["record_out_ms"]]  # type: ignore[operator]
        if not active:
            gap_frames += 1
        elif len(active) > 2 or (len(active) == 2 and not any(a <= t_ms < b for a, b in windows)):
            overlap_frames += 1
        for i in active:
            c = tl.clips[i]
            src = c.start_ms + (t_ms - int(lay[i]["record_in_ms"]))
            if not (c.start_ms <= src < c.end_ms) or src >= tl.assets.get(c.asset_id, 0):
                out_of_range += 1
        trace.append([t_ms, [tl.clips[i].clip_id for i in active]])
    if gap_frames:
        problems.append(f"{gap_frames} sampled frame(s) fall in a gap")
    if overlap_frames:
        problems.append(f"{overlap_frames} sampled frame(s) have overlapping clips outside a transition")
    if out_of_range:
        problems.append(f"{out_of_range} sampled source time(s) out of range")
    evidence.update(
        {
            "samples": samples,
            "step_ms": step,
            "gap_frames": gap_frames,
            "overlap_frames": overlap_frames,
            "out_of_range_samples": out_of_range,
            "probe_hash": content_hash(trace),
            "problems": problems,
        }
    )
    return not problems, evidence


def transition_probe(
    after: Any, operations: Optional[Sequence[Mapping[str, Any]]] = None
) -> tuple[bool, dict[str, Any]]:
    """Check every transition fits its adjacent clips and requested transitions exist exactly."""
    errs = validate(after, TIMELINE_JSON_SCHEMA)
    if errs:
        return False, {"schema_errors": errs}
    tl = Timeline.from_dict(after)
    lay = {p["clip_id"]: p for p in tl.layout()}
    problems = [e for e in validate_timeline(tl) if "transition" in e]
    windows = []
    for t in tl.transitions:
        p = lay.get(t.after_clip_id)
        if p is not None:
            end = int(p["record_out_ms"])
            windows.append({**t.to_dict(), "window_ms": [end - t.duration_ms, end]})
    for i, op in enumerate(operations or ()):
        if op.get("op") != "transition":
            continue
        got = tl.transition_after(op["after_clip_id"])
        if got is None or (got.kind, got.duration_ms) != (op["kind"], op["duration_ms"]):
            problems.append(f"requested transition operations[{i}] after {op['after_clip_id']} is not present as specified")
    return not problems, {"transitions": windows, "problems": problems}
