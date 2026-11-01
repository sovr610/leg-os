"""Schema-guided structured decoding (blueprint §10: valid plans and tool calls).

The model never free-writes JSON. A schema walker emits every structural token
(braces, keys, quotes, separators, constants) itself and lets the model choose
only *values*, each under a constraint that guarantees the finished document
validates against the JSON schema:

* ``enum`` / ``const`` / ``boolean`` / nullable unions / ``oneOf`` with a
  discriminator — chosen by scoring each alternative's token sequence;
* ``string`` — token-by-token under a :class:`RegexConstraint` (``pattern``) or
  :class:`TextConstraint`, with ``minLength``/``maxLength``; generated strings
  never contain ``"``, ``\\`` or control characters, so no escaping is needed;
* ``integer`` / ``number`` — digit tokens under :class:`IntegerConstraint`
  (``minimum``/``maximum``, no leading zeros);
* ``object`` — required properties (plus requested optional ones) in schema
  order; ``array`` — ``minItems``/``maxItems`` with model-chosen length.

Every constraint can always be completed (each prefix it admits has a valid
extension), and when the token budget runs out the shortest valid completion is
forced — so decoding always terminates with a schema-valid value.

The regex engine is a small Thompson-NFA compiler for the subset used by the
action schemas (classes, ranges, groups, alternation, ``* + ? {m,n}``,
``\\d \\w \\s``) with *prefix viability*: whether a partial string can still be
extended to a full match. Production serving would precompile per-state token
masks (as Outlines/XGrammar do); this reference caches masks per NFA state set.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch

# ============================================================== regex → NFA
_PRINTABLE = [chr(c) for c in range(32, 127)]


class RegexUnsupported(ValueError):
    pass


@dataclass
class _CharSet:
    ranges: list[tuple[str, str]]
    negate: bool = False
    any_char: bool = False

    def matches(self, ch: str) -> bool:
        if self.any_char:
            return ch != "\n"
        hit = any(lo <= ch <= hi for lo, hi in self.ranges)
        return hit != self.negate


_SHORTHAND = {
    "d": [("0", "9")],
    "w": [("a", "z"), ("A", "Z"), ("0", "9"), ("_", "_")],
    "s": [(" ", " "), ("\t", "\t"), ("\n", "\n"), ("\r", "\r")],
}


class _Parser:
    """Recursive-descent parser producing a small AST."""

    def __init__(self, pattern: str) -> None:
        self.p = pattern
        self.i = 0

    def parse(self):
        node = self._alt()
        if self.i != len(self.p):
            raise RegexUnsupported(f"unexpected {self.p[self.i]!r} at {self.i} in {self.p!r}")
        return node

    def _peek(self) -> Optional[str]:
        return self.p[self.i] if self.i < len(self.p) else None

    def _alt(self):
        branches = [self._seq()]
        while self._peek() == "|":
            self.i += 1
            branches.append(self._seq())
        return ("alt", branches) if len(branches) > 1 else branches[0]

    def _seq(self):
        items = []
        while self._peek() not in (None, "|", ")"):
            items.append(self._quant())
        return ("seq", items)

    def _quant(self):
        atom = self._atom()
        c = self._peek()
        if c == "*":
            self.i += 1
            return ("rep", atom, 0, None)
        if c == "+":
            self.i += 1
            return ("rep", atom, 1, None)
        if c == "?":
            self.i += 1
            return ("rep", atom, 0, 1)
        if c == "{":
            j = self.p.index("}", self.i)
            body = self.p[self.i + 1 : j]
            self.i = j + 1
            if "," in body:
                lo, hi = body.split(",", 1)
                return ("rep", atom, int(lo or 0), int(hi) if hi else None)
            return ("rep", atom, int(body), int(body))
        return atom

    def _atom(self):
        c = self._peek()
        if c == "(":
            self.i += 1
            if self.p.startswith("?:", self.i):
                self.i += 2
            node = self._alt()
            if self._peek() != ")":
                raise RegexUnsupported("unbalanced parenthesis")
            self.i += 1
            return node
        if c == "[":
            return ("set", self._class())
        if c == ".":
            self.i += 1
            return ("set", _CharSet([], any_char=True))
        if c == "\\":
            self.i += 1
            e = self.p[self.i]
            self.i += 1
            if e in _SHORTHAND:
                return ("set", _CharSet(list(_SHORTHAND[e])))
            if e.upper() in _SHORTHAND and e.isupper():
                return ("set", _CharSet(list(_SHORTHAND[e.lower()]), negate=True))
            return ("set", _CharSet([(e, e)]))
        if c in ("^", "$"):
            raise RegexUnsupported("anchors are only supported at the ends of the pattern")
        self.i += 1
        return ("set", _CharSet([(c, c)]))

    def _class(self) -> _CharSet:
        self.i += 1  # [
        negate = self._peek() == "^"
        if negate:
            self.i += 1
        ranges: list[tuple[str, str]] = []
        first = True
        while True:
            c = self._peek()
            if c is None:
                raise RegexUnsupported("unterminated character class")
            if c == "]" and not first:
                self.i += 1
                break
            first = False
            if c == "\\":
                e = self.p[self.i + 1]
                self.i += 2
                if e in _SHORTHAND:
                    ranges.extend(_SHORTHAND[e])
                    continue
                lo = e
            else:
                lo = c
                self.i += 1
            if self._peek() == "-" and self.i + 1 < len(self.p) and self.p[self.i + 1] != "]":
                hi = self.p[self.i + 1]
                if hi == "\\":
                    hi = self.p[self.i + 2]
                    self.i += 1
                self.i += 2
                ranges.append((lo, hi))
            else:
                ranges.append((lo, lo))
        return _CharSet(ranges, negate)


class Regex:
    """Thompson NFA with prefix viability (``fullmatch`` semantics, anchors optional)."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        body = pattern[1:] if pattern.startswith("^") else pattern
        body = body[:-1] if body.endswith("$") and not body.endswith("\\$") else body
        self.eps: list[list[int]] = []
        self.edges: list[list[tuple[_CharSet, int]]] = []
        start, end = self._compile(_Parser(body).parse())
        self.start, self.accept = start, end
        self._live = self._co_reachable()
        self.dist = self._distance_to_accept()
        self.initial = self._closure({self.start})

    # ---- construction
    def _state(self) -> int:
        self.eps.append([])
        self.edges.append([])
        return len(self.eps) - 1

    def _compile(self, node) -> tuple[int, int]:
        kind = node[0]
        if kind == "set":
            s, e = self._state(), self._state()
            self.edges[s].append((node[1], e))
            return s, e
        if kind == "seq":
            s = e = self._state()
            for item in node[1]:
                a, b = self._compile(item)
                self.eps[e].append(a)
                e = b
            return s, e
        if kind == "alt":
            s, e = self._state(), self._state()
            for br in node[1]:
                a, b = self._compile(br)
                self.eps[s].append(a)
                self.eps[b].append(e)
            return s, e
        if kind == "rep":
            _, atom, lo, hi = node
            s = e = self._state()
            for _ in range(lo):
                a, b = self._compile(atom)
                self.eps[e].append(a)
                e = b
            if hi is None:
                a, b = self._compile(atom)
                self.eps[e].append(a)
                self.eps[b].append(a)
                end = self._state()
                self.eps[e].append(end)
                self.eps[b].append(end)
                return s, end
            end = self._state()
            self.eps[e].append(end)
            for _ in range(hi - lo):
                a, b = self._compile(atom)
                self.eps[e].append(a)
                self.eps[b].append(end)
                e = b
            return s, end
        raise RegexUnsupported(kind)

    def _co_reachable(self) -> frozenset[int]:
        rev: dict[int, list[int]] = {}
        for s in range(len(self.eps)):
            for t in self.eps[s]:
                rev.setdefault(t, []).append(s)
            for _, t in self.edges[s]:
                rev.setdefault(t, []).append(s)
        seen = {self.accept}
        stack = [self.accept]
        while stack:
            for p in rev.get(stack.pop(), []):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return frozenset(seen)

    def _distance_to_accept(self) -> list[float]:
        """Fewest characters needed to reach ``accept`` from each state (0-1 BFS on the reverse graph)."""
        rev: dict[int, list[tuple[int, int]]] = {}
        for s in range(len(self.eps)):
            for t in self.eps[s]:
                rev.setdefault(t, []).append((s, 0))
            for _, t in self.edges[s]:
                rev.setdefault(t, []).append((s, 1))
        dist = [math.inf] * len(self.eps)
        dist[self.accept] = 0
        dq = deque([self.accept])
        while dq:
            u = dq.popleft()
            for p, w in rev.get(u, []):
                if dist[u] + w < dist[p]:
                    dist[p] = dist[u] + w
                    (dq.appendleft if w == 0 else dq.append)(p)
        return dist

    def min_remaining(self, states: frozenset[int]) -> float:
        return min((self.dist[s] for s in states), default=math.inf)

    def _closure(self, states) -> frozenset[int]:
        out = set(states)
        stack = list(states)
        while stack:
            for t in self.eps[stack.pop()]:
                if t not in out:
                    out.add(t)
                    stack.append(t)
        return frozenset(s for s in out if s in self._live)

    # ---- matching
    def step(self, states: frozenset[int], ch: str) -> frozenset[int]:
        nxt = {t for s in states for cs, t in self.edges[s] if cs.matches(ch)}
        return self._closure(nxt) if nxt else frozenset()

    def run(self, text: str, states: Optional[frozenset[int]] = None) -> frozenset[int]:
        st = self.initial if states is None else states
        for ch in text:
            st = self.step(st, ch)
            if not st:
                break
        return st

    def viable(self, text: str) -> bool:
        return bool(self.run(text))

    def fullmatch(self, text: str) -> bool:
        return self.accept in self.run(text)

    def shortest_completion(self, text: str, alphabet: Sequence[str] = _PRINTABLE, min_extra: int = 0) -> Optional[str]:
        """Shortest suffix (from ``alphabet``) making ``text + suffix`` a full match."""
        start = self.run(text)
        if not start:
            return None
        q = deque([(start, "")])
        seen = {(start, 0)}
        while q:
            st, suf = q.popleft()
            if self.accept in st and len(suf) >= min_extra:
                return suf
            for ch in alphabet:
                nxt = self.step(st, ch)
                key = (nxt, min(len(suf) + 1, min_extra))
                if nxt and key not in seen:
                    seen.add(key)
                    q.append((nxt, suf + ch))
        return None


# ============================================================== constraints
_FORBIDDEN = set('"\\') | {chr(c) for c in range(32)} | {"\x7f"}


class Constraint:
    """Character-level constraint for one generated value."""

    min_len: int = 0
    max_len: int = 64

    def viable(self, text: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def complete(self, text: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def completion(self, text: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class TextConstraint(Constraint):
    min_len: int = 0
    max_len: int = 200

    def viable(self, text: str) -> bool:
        return len(text) <= self.max_len and not (_FORBIDDEN & set(text))

    def complete(self, text: str) -> bool:
        return self.viable(text) and len(text) >= self.min_len

    def completion(self, text: str) -> str:
        return "x" * max(0, self.min_len - len(text))


@dataclass
class RegexConstraint(Constraint):
    pattern: str = ".*"
    min_len: int = 0
    max_len: int = 128
    regex: Regex = field(init=False)

    def __post_init__(self) -> None:
        self.regex = Regex(self.pattern)

    def viable_states(self, states: frozenset[int], n: int) -> bool:
        """Whether a prefix of length ``n`` that reached ``states`` can still complete within ``max_len``."""
        return bool(states) and n + self.regex.min_remaining(states) <= self.max_len and self.min_len <= self.max_len

    def viable(self, text: str) -> bool:
        if len(text) > self.max_len or (_FORBIDDEN & set(text)):
            return False
        return self.viable_states(self.regex.run(text), len(text))

    def complete(self, text: str) -> bool:
        return self.min_len <= len(text) <= self.max_len and not (_FORBIDDEN & set(text)) and self.regex.fullmatch(text)

    def completion(self, text: str) -> str:
        suf = self.regex.shortest_completion(text, alphabet=[c for c in _PRINTABLE if c not in _FORBIDDEN],
                                             min_extra=max(0, self.min_len - len(text)))
        if suf is None:
            raise ValueError(f"no completion for {text!r} under {self.pattern!r}")
        return suf


@dataclass
class IntegerConstraint(Constraint):
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    max_len: int = 12

    def __post_init__(self) -> None:
        lo = 0 if self.minimum is None else self.minimum
        if lo < 0:
            raise ValueError("negative integer ranges are not supported by the reference decoder")
        self.lo = lo
        self.hi = self.maximum if self.maximum is not None else 10 ** (self.max_len - 1) - 1
        self.max_len = len(str(self.hi))

    def viable(self, text: str) -> bool:
        if not text:
            return True
        if not text.isdigit() or len(text) > self.max_len or (len(text) > 1 and text[0] == "0"):
            return False
        v = int(text)
        if v > self.hi:
            return False
        if self.lo <= v:
            return True
        if text == "0":
            return False
        k = 1
        while len(text) + k <= self.max_len:  # some extension lands in [lo, hi]
            low, high = v * 10**k, v * 10**k + 10**k - 1
            if high >= self.lo and low <= self.hi:
                return True
            k += 1
        return False

    def complete(self, text: str) -> bool:
        return bool(text) and self.viable(text) and self.lo <= int(text) <= self.hi

    def completion(self, text: str) -> str:
        if not text:
            return str(self.lo)
        v = int(text)
        k = 1
        while True:
            low = max(v * 10**k, self.lo)
            if low <= v * 10**k + 10**k - 1 and low <= self.hi:
                return str(low)[len(text):]
            k += 1


# ============================================================ the decoder
class StructuredDecodeError(RuntimeError):
    pass


class SchemaDecoder:
    """Walks a JSON schema, forcing structure and letting a runtime session choose values.

    ``session`` must provide ``next_logits()``, ``feed(ids)`` and ``score(ids)``
    (:class:`m63_omni.runtime.session.Session` does). ``tokenizer`` must provide
    ``encode_ordinary``, ``token_bytes`` and ``vocab_size`` (``ByteBPETokenizer``).
    """

    def __init__(self, session, tokenizer, *, temperature: float = 0.0, generator: Optional[torch.Generator] = None,
                 max_string_tokens: int = 24, max_array_items: int = 4, max_string_chars: int = 64) -> None:
        self.session, self.tok = session, tokenizer
        self.temperature, self.generator = temperature, generator
        self.max_string_tokens = max_string_tokens
        self.max_array_items = max_array_items
        self.max_string_chars = max_string_chars
        self.text = ""
        self.token_text: list[Optional[str]] = []
        for i in range(tokenizer.vocab_size):
            if tokenizer.is_special(i):
                self.token_text.append(None)
                continue
            try:
                t = tokenizer.token_bytes(i).decode("utf-8")
            except (UnicodeDecodeError, ValueError):  # partial UTF-8, or an ID the tokenizer never produces
                t = None
            self.token_text.append(t if t else None)
        self._clean = torch.tensor([t is not None and not (_FORBIDDEN & set(t)) for t in self.token_text])
        self._lens = torch.tensor([len(t) if t else 0 for t in self.token_text])
        self._mask_cache: dict[Any, torch.Tensor] = {}

    # ---------------------------------------------------------- primitives
    def force(self, text: str) -> None:
        if text:
            self.session.feed(self.tok.encode_ordinary(text))
            self.text += text

    def pick(self, options: Sequence[str]) -> int:
        """Score complete alternatives by log-probability (no state change); sample or argmax."""
        if len(options) == 1:
            return 0
        scores = torch.tensor([self.session.score(self.tok.encode_ordinary(o)) for o in options])
        if self.temperature > 0:
            return int(torch.multinomial(torch.softmax(scores / self.temperature, -1), 1, generator=self.generator))
        return int(scores.argmax())

    def choose(self, options: Sequence[str]) -> int:
        i = self.pick(options)
        self.force(options[i])
        return i

    def _allowed(self, c: Constraint, cur: str) -> torch.Tensor:
        """Token mask for extending ``cur`` under ``c`` (cached per regex state set / remaining length)."""
        remaining = c.max_len - len(cur)
        if isinstance(c, TextConstraint):
            key = ("txt", remaining)
            if key not in self._mask_cache:
                self._mask_cache[key] = self._clean & (self._lens <= remaining)
            return self._mask_cache[key]
        if isinstance(c, RegexConstraint):
            states = c.regex.run(cur)
            key = ("re", c.pattern, states, remaining)
            if key not in self._mask_cache:
                mask = torch.zeros(len(self.token_text), dtype=torch.bool)
                for i in self._clean.nonzero().flatten().tolist():
                    t = self.token_text[i]
                    if len(t) > remaining:
                        continue
                    st = states
                    for ch in t:
                        st = c.regex.step(st, ch)
                        if not st:
                            break
                    mask[i] = c.viable_states(st, len(cur) + len(t))
                self._mask_cache[key] = mask
            return self._mask_cache[key]
        mask = torch.zeros(len(self.token_text), dtype=torch.bool)
        for i in self._clean.nonzero().flatten().tolist():
            mask[i] = c.viable(cur + self.token_text[i])
        return mask

    def gen(self, c: Constraint, terminator: str) -> str:
        """Generate a constrained value; the end decision is scored with ``terminator``'s first token."""
        cur = ""
        end_tok = self.tok.encode_ordinary(terminator)[0]
        for _ in range(self.max_string_tokens):
            logits = self.session.next_logits()[0].float()
            allowed = self._allowed(c, cur)
            can_end = c.complete(cur)
            if not allowed.any() and not can_end:
                break
            masked = torch.where(allowed, logits, torch.full_like(logits, -math.inf))
            end_score = logits[end_tok] if can_end else torch.tensor(-math.inf)
            cand = torch.cat([masked, end_score.view(1)])
            if self.temperature > 0:
                idx = int(torch.multinomial(torch.softmax(cand / self.temperature, -1), 1, generator=self.generator))
            else:
                idx = int(cand.argmax())
            if idx == len(self.token_text):
                return cur
            t = self.token_text[idx]
            self.session.feed([idx])
            self.text += t
            cur += t
        if not c.complete(cur):
            suf = c.completion(cur)
            self.force(suf)
            cur += suf
        return cur

    # ------------------------------------------------------------- walker
    def value(self, schema: Mapping[str, Any], after: str = ",", include_optional: Sequence[str] = ()) -> Any:
        if "const" in schema:
            self.force(json.dumps(schema["const"]))
            return schema["const"]
        if "enum" in schema:
            opts = list(schema["enum"])
            return opts[self.choose([json.dumps(o) for o in opts])]
        if "oneOf" in schema or "anyOf" in schema:
            return self._union(schema.get("oneOf") or schema.get("anyOf"), after)
        types = schema.get("type")
        if isinstance(types, list):
            non_null = [t for t in types if t != "null"]
            if "null" in types and non_null:
                if non_null[0] == "string":
                    if self.pick(["null", '"']) == 0:
                        self.force("null")
                        return None
                    return self.value({**schema, "type": "string"}, after)
                return self.value({**schema, "type": non_null[0]}, after)
            types = types[0]
        if types == "string":
            self.force('"')
            s = self.gen(self._string_constraint(schema), '"')
            self.force('"')
            return s
        if types in ("integer", "number"):
            return int(self.gen(IntegerConstraint(schema.get("minimum"), schema.get("maximum")), after))
        if types == "boolean":
            return self.choose(["false", "true"]) == 1
        if types == "null":
            self.force("null")
            return None
        if types == "array":
            return self._array(schema)
        if types == "object" or "properties" in schema:
            return self._object(schema, include_optional=include_optional)
        raise StructuredDecodeError(f"unsupported schema {schema!r}")

    def _string_constraint(self, schema: Mapping[str, Any]) -> Constraint:
        max_len = min(int(schema.get("maxLength", self.max_string_chars)), self.max_string_chars)
        min_len = int(schema.get("minLength", 0))
        if "pattern" in schema:
            return RegexConstraint(schema["pattern"], min_len, max_len)
        return TextConstraint(min_len, max_len)

    def _object(self, schema: Mapping[str, Any], *, prefilled: Optional[dict] = None, opened: bool = False,
                include_optional: Sequence[str] = ()) -> dict:
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        keys = [k for k in props if k in required or k in include_optional]
        out = dict(prefilled or {})
        if not opened:
            self.force("{")
        first = not out
        for i, k in enumerate(k for k in keys if k not in out):
            self.force(("" if first else ", ") + json.dumps(k) + ": ")
            first = False
            remaining = [x for x in keys if x not in out and x != k]
            out[k] = self.value(props[k], after="," if remaining else "}")
        self.force("}")
        return out

    def _array(self, schema: Mapping[str, Any]) -> list:
        items = schema.get("items", {})
        lo = int(schema.get("minItems", 0))
        hi = min(int(schema.get("maxItems", self.max_array_items)), max(self.max_array_items, lo))
        self.force("[")
        out: list = []
        while len(out) < hi:
            if len(out) >= lo:
                opts = ["]", self._opener(items)] if not out else [", ", "]"]
                stop = self.pick(opts) == (0 if not out else 1)
                if stop:
                    break
            if out:
                self.force(", ")
            out.append(self.value(items, after=","))
        self.force("]")
        return out

    @staticmethod
    def _opener(schema: Mapping[str, Any]) -> str:
        if "oneOf" in schema or "anyOf" in schema or schema.get("type") == "object":
            return "{"
        t = schema.get("type")
        return {"string": '"', "array": "[", "boolean": "true", "null": "null"}.get(t if isinstance(t, str) else "", "1")

    def _union(self, branches: Sequence[Mapping[str, Any]], after: str) -> Any:
        disc = _discriminator(branches)
        if disc is None:
            return self.value(branches[0], after)
        opts = ["{" + json.dumps(disc) + ": " + json.dumps(b["properties"][disc]["const"]) for b in branches]
        i = self.choose(opts)
        b = branches[i]
        return self._object(b, prefilled={disc: b["properties"][disc]["const"]}, opened=True)


def _discriminator(branches: Sequence[Mapping[str, Any]]) -> Optional[str]:
    if not branches or not all(b.get("type") == "object" for b in branches):
        return None
    first = branches[0].get("properties", {})
    for k in first:
        if all("const" in b.get("properties", {}).get(k, {}) for b in branches) and \
                len({json.dumps(b["properties"][k]["const"]) for b in branches}) == len(branches) and \
                all(next(iter(b.get("properties", {}))) == k for b in branches):
            return k
    return None
