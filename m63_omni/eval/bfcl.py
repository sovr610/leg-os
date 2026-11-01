"""BFCL-style function-calling accuracy by AST matching (Berkeley Function-Calling Leaderboard, Apache-2.0).

Each item gives the model one or more function definitions and a request; the
model must answer with the right call(s). Scoring parses the output into calls
and matches them against the ground truth without executing anything:

* the function name must match;
* every required parameter must be present and no unknown parameter may appear;
* each value must have the declared type (an int is accepted for a float) and be
  one of the allowed values — strings are compared after BFCL's normalisation
  (case, spaces, and ``,./-_*^`` ignored); an allowed value of ``""`` marks an
  optional parameter that may be omitted.

Categories (inferred from the item ID): ``simple`` (one function, one call),
``multiple`` (pick the right function), ``parallel`` (several calls, any
order), ``parallel_multiple``, and ``irrelevance`` (no function fits: the model
must not call one; prose or ``[]`` passes). ``live_*`` IDs map to the same
categories. Multi-turn and executable categories are out of scope.

Files use the BFCL formats: questions ``{"id", "question", "function"}``,
answers ``{"id", "ground_truth": [{"func": {"param": [allowed, ...]}}]}`` and
model results ``{"id", "result"}``. Outputs may be Python call syntax
(``[f(a=1), g(b="x")]``) or JSON (``[{"name": "f", "arguments": {"a": 1}}]``).

Like :mod:`~m63_omni.eval.ifeval`, this re-implements the checker closely
enough for paired candidate-vs-control comparison; quote absolute BFCL numbers
from the official evaluator.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")


class BFCLError(ValueError):
    pass


@dataclass(frozen=True)
class Call:
    name: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class BFCLItem:
    id: str
    category: str
    functions: tuple[Mapping[str, Any], ...]
    ground_truth: tuple[Mapping[str, Mapping[str, list]], ...] = ()  # empty for irrelevance

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise BFCLError(f"item {self.id}: unknown category {self.category!r}")
        if self.category != "irrelevance" and not self.ground_truth:
            raise BFCLError(f"item {self.id}: no ground truth")


@dataclass(frozen=True)
class BFCLResult:
    id: str
    category: str
    correct: bool
    reason: str = ""
    missing_output: bool = False


def category_of(item_id: str) -> str:
    base = re.sub(r"[_-][\d-]+$", "", item_id).removeprefix("live_").replace("_python", "")
    for c in ("parallel_multiple", "parallel", "multiple", "simple", "irrelevance"):
        if base == c:
            return c
    raise BFCLError(f"cannot infer a supported category from id {item_id!r}")


# ------------------------------------------------------------------ parsing
def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except ValueError as e:
        raise BFCLError(f"argument is not a literal: {ast.dump(node)[:60]}") from e


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _dotted(node.value) + "." + node.attr
    raise BFCLError("unsupported callee")


def _python_calls(text: str) -> list[Call]:
    tree = ast.parse(text, mode="eval").body
    nodes = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
    calls = []
    for n in nodes:
        if not isinstance(n, ast.Call):
            raise BFCLError("expected a function call")
        if n.args:
            raise BFCLError("positional arguments are not allowed")
        calls.append(Call(_dotted(n.func), {kw.arg: _literal(kw.value) for kw in n.keywords if kw.arg is not None}))
    return calls


def _json_calls(obj: Any) -> list[Call]:
    if isinstance(obj, dict) and "tool_calls" in obj:
        obj = obj["tool_calls"]
    items = obj if isinstance(obj, list) else [obj]
    calls = []
    for it in items:
        if not isinstance(it, dict):
            raise BFCLError("JSON call must be an object")
        if "function" in it and isinstance(it["function"], dict):  # OpenAI-style tool call
            it = it["function"]
        if "name" in it:
            args = it.get("arguments", it.get("parameters", {}))
            if isinstance(args, str):
                args = json.loads(args)
            calls.append(Call(it["name"], dict(args)))
        elif len(it) == 1:  # BFCL ground-truth style {"func": {...}}
            (name, args), = it.items()
            calls.append(Call(name, dict(args)))
        else:
            raise BFCLError("JSON call needs a name")
    return calls


def parse_calls(text: str) -> list[Call]:
    """Parse model output into calls; raises :class:`BFCLError` when it is not a call list."""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    if not t or t == "[]":
        return []
    try:
        return _json_calls(json.loads(t))
    except (ValueError, TypeError):
        pass
    try:
        return _python_calls(t)
    except SyntaxError as e:
        raise BFCLError(f"not a function call: {t[:60]!r}") from e


# ----------------------------------------------------------------- matching
def _norm(s: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def _type_ok(value: Any, typ: Optional[str]) -> bool:
    if typ in (None, "any"):
        return True
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ in ("float", "number"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if typ == "string":
        return isinstance(value, str)
    if typ == "boolean":
        return isinstance(value, bool)
    if typ in ("array", "tuple"):
        return isinstance(value, (list, tuple))
    if typ in ("dict", "object"):
        return isinstance(value, dict)
    return True


def _match(value: Any, allowed: Any) -> bool:
    if isinstance(allowed, str):
        return isinstance(value, str) and _norm(value) == _norm(allowed)
    if isinstance(allowed, bool) or isinstance(value, bool):
        return type(value) is type(allowed) and value == allowed
    if isinstance(allowed, (int, float)):
        return isinstance(value, (int, float)) and float(value) == float(allowed)
    if isinstance(allowed, (list, tuple)):
        return isinstance(value, (list, tuple)) and len(value) == len(allowed) and all(map(_match, value, allowed))
    if isinstance(allowed, dict):  # nested ground truth: {"key": [options]}
        if not isinstance(value, dict):
            return False
        for k, opts in allowed.items():
            opts = opts if isinstance(opts, list) else [opts]
            if k not in value:
                if "" not in opts:
                    return False
            elif not any(_match(value[k], o) for o in opts if o != ""):
                return False
        return set(value) <= set(allowed)
    return value == allowed


def check_call(call: Call, expected: Mapping[str, Mapping[str, list]], functions: Sequence[Mapping[str, Any]]) -> str:
    """Empty string when ``call`` matches the single-entry ground truth ``expected``; else the reason."""
    (name, params), = expected.items()
    if call.name != name:
        return f"called {call.name!r}, expected {name!r}"
    doc = next((f for f in functions if f.get("name") == name), None)
    if doc is None:
        return f"no definition for {name!r}"
    schema = doc.get("parameters", {})
    props, required = schema.get("properties", {}), schema.get("required", [])
    for p in required:
        if p not in call.args:
            return f"missing required parameter {p!r}"
    for k, v in call.args.items():
        if k not in props or k not in params:
            return f"unexpected parameter {k!r}"
        if not _type_ok(v, props[k].get("type")):
            return f"wrong type for {k!r}"
        options = params[k] if isinstance(params[k], list) else [params[k]]
        if not any(_match(v, o) for o in options if o != ""):
            return f"wrong value for {k!r}: {v!r}"
    for k, options in params.items():
        if k not in call.args and "" not in (options if isinstance(options, list) else [options]):
            return f"missing parameter {k!r}"
    return ""


def _assign(calls: list[Call], expected: list, functions, used: list[bool], i: int = 0) -> bool:
    """Match expected[i:] to distinct unused calls (order-insensitive; backtracking)."""
    if i == len(expected):
        return True
    for j, c in enumerate(calls):
        if not used[j] and not check_call(c, expected[i], functions):
            used[j] = True
            if _assign(calls, expected, functions, used, i + 1):
                return True
            used[j] = False
    return False


def score_item(item: BFCLItem, output: Optional[str]) -> BFCLResult:
    if output is None:
        return BFCLResult(item.id, item.category, False, "no output", missing_output=True)
    try:
        calls = parse_calls(output)
    except (BFCLError, ValueError) as e:
        calls = None
        why = str(e)
    if item.category == "irrelevance":
        ok = not calls  # prose, a parse failure, or [] all decline to call
        return BFCLResult(item.id, item.category, ok, "" if ok else f"called {[c.name for c in calls]}")
    if calls is None:
        return BFCLResult(item.id, item.category, False, f"unparseable: {why}")
    expected = list(item.ground_truth)
    if len(calls) != len(expected):
        return BFCLResult(item.id, item.category, False, f"{len(calls)} call(s), expected {len(expected)}")
    if len(expected) == 1:
        reason = check_call(calls[0], expected[0], item.functions)
        return BFCLResult(item.id, item.category, not reason, reason)
    ok = _assign(calls, expected, item.functions, [False] * len(calls))
    return BFCLResult(item.id, item.category, ok, "" if ok else "calls do not match the expected set")


def score_bfcl(items: Iterable[BFCLItem], outputs: Mapping[str, str]) -> list[BFCLResult]:
    """Score every item; a missing output is a failure, never a skip."""
    return [score_item(it, outputs.get(it.id)) for it in items]


def summarize(results: Sequence[BFCLResult]) -> dict[str, float]:
    """Per-category accuracy and ``overall`` = unweighted mean over the categories present."""
    if not results:
        raise BFCLError("no results")
    per: dict[str, list[bool]] = {}
    for r in results:
        per.setdefault(r.category, []).append(r.correct)
    out = {c: sum(v) / len(v) for c, v in per.items()}
    out["overall"] = sum(out[c] for c in per) / len(per)
    out["items"] = float(len(results))
    out["missing_outputs"] = float(sum(r.missing_output for r in results))
    return out


# -------------------------------------------------------------------- files
def _jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_bfcl(question_path: str | Path, answer_path: Optional[str | Path] = None) -> list[BFCLItem]:
    """Load a BFCL question file (and its answer file, absent for irrelevance)."""
    answers = {} if answer_path is None else {str(a["id"]): a["ground_truth"] for a in _jsonl(answer_path)}
    items = []
    for q in _jsonl(question_path):
        qid = str(q["id"])
        cat = category_of(qid)
        gt = answers.get(qid, [])
        if cat != "irrelevance" and qid not in answers:
            raise BFCLError(f"no ground truth for {qid}")
        fns = q["function"] if isinstance(q["function"], list) else [q["function"]]
        items.append(BFCLItem(qid, cat, tuple(fns), tuple(gt)))
    return items


def load_outputs(path: str | Path) -> dict[str, str]:
    out = {}
    for obj in _jsonl(path):
        r = obj["result"]
        out[str(obj["id"])] = r if isinstance(r, str) else json.dumps(r)
    return out


# --------------------------------------------------------------- self-test
_WEATHER = {"name": "get_weather", "parameters": {"type": "dict", "required": ["city"], "properties": {
    "city": {"type": "string"}, "unit": {"type": "string"}, "days": {"type": "integer"}}}}
_TRIM = {"name": "timeline.trim", "parameters": {"type": "dict", "required": ["clip_id", "end_ms"], "properties": {
    "clip_id": {"type": "string"}, "end_ms": {"type": "integer"}, "ripple": {"type": "boolean"}}}}
_W_GT = {"get_weather": {"city": ["San Francisco", "SF"], "unit": ["celsius", ""], "days": [3]}}


@dataclass(frozen=True)
class _Fixture:
    item: BFCLItem
    good: tuple[str, ...]
    bad: tuple[str, ...] = field(default=())


SELF_TEST: tuple[_Fixture, ...] = (
    _Fixture(BFCLItem("simple_0", "simple", (_WEATHER,), (_W_GT,)),
             ('get_weather(city="san francisco", days=3)', '[{"name": "get_weather", "arguments": {"city": "SF", "days": 3, "unit": "Celsius"}}]'),
             ('get_weather(city="Paris", days=3)', 'get_weather(city="SF")', 'get_weather(city="SF", days="3")',
              'get_weather(city="SF", days=3, country="US")', "It is sunny.")),
    _Fixture(BFCLItem("multiple_0", "multiple", (_WEATHER, _TRIM),
                      ({"timeline.trim": {"clip_id": ["c_17"], "end_ms": [4000], "ripple": [True, ""]}},)),
             ('timeline.trim(clip_id="c_17", end_ms=4000)',), ('get_weather(city="SF", days=3)',
                                                             'timeline.trim(clip_id="c_17", end_ms=4000, ripple=False)')),
    _Fixture(BFCLItem("parallel_0", "parallel", (_WEATHER,),
                      ({"get_weather": {"city": ["Paris"], "unit": [""], "days": [1]}}, {"get_weather": {"city": ["Rome"], "unit": [""], "days": [2]}})),
             ('[get_weather(city="Rome", days=2), get_weather(city="Paris", days=1)]',),
             ('[get_weather(city="Rome", days=1), get_weather(city="Paris", days=2)]', 'get_weather(city="Rome", days=2)')),
    _Fixture(BFCLItem("parallel_multiple_0", "parallel_multiple", (_WEATHER, _TRIM),
                      (_W_GT, {"timeline.trim": {"clip_id": ["c_2"], "end_ms": [900], "ripple": [""]}})),
             ('[timeline.trim(clip_id="c_2", end_ms=900), get_weather(city="SF", days=3)]',),
             ('[timeline.trim(clip_id="c_2", end_ms=900), get_weather(city="SF", days=3), get_weather(city="SF", days=3)]',
              '[timeline.trim(clip_id="c_2", end_ms=900)]')),
    _Fixture(BFCLItem("irrelevance_0", "irrelevance", (_WEATHER,)), ("I can't help with that using these tools.", "[]"),
             ('get_weather(city="SF", days=1)',)),
)


def self_test() -> None:
    """Known-good outputs must pass and known-bad must fail in every category."""
    from .harness import ScorerSelfTestError

    for fx in SELF_TEST:
        for out in fx.good:
            r = score_item(fx.item, out)
            if not r.correct:
                raise ScorerSelfTestError(f"BFCL scorer rejected known-good output for {fx.item.id}: {r.reason}")
        for out in fx.bad:
            if score_item(fx.item, out).correct:
                raise ScorerSelfTestError(f"BFCL scorer accepted known-bad output for {fx.item.id}: {out!r}")
    missing = set(CATEGORIES) - {fx.item.category for fx in SELF_TEST}
    if missing:
        raise ScorerSelfTestError(f"BFCL categories without self-test fixtures: {sorted(missing)}")
