"""Quality, language-ID, and safety/privacy filters (blueprint §8 transformations 2, 4, 5).

* :func:`quality_check` — Gopher-style heuristics: length, symbol-to-word
  ratio, alphabetic-word fraction, mean word length, duplicate lines, most
  frequent word-bigram coverage, and duplicated word-5-gram coverage. Word
  rules apply only to whitespace-delimited scripts so CJK/Thai are not
  penalised by construction.
* :func:`identify_language` — a deliberately simple language-ID *stub* based on
  Unicode script detection plus a few stopword checks for Latin script. It is a
  placeholder interface for a trained classifier, not a production LID model.
* :func:`redact_pii` / :func:`safety_privacy_filter` — regex redaction of
  emails, credit-card-like numbers (Luhn-validated), IPv4/IPv6 addresses, and
  phone numbers. The result carries the filtered text *and* a record of what
  fired (kind + span, never the raw value, so the audit log itself does not
  leak PII).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "QualityConfig",
    "QualityResult",
    "quality_check",
    "detect_script",
    "LanguageGuess",
    "identify_language",
    "whitespace_delimited",
    "luhn_valid",
    "PIIFinding",
    "RedactionResult",
    "redact_pii",
    "PII_KINDS",
    "SafetyConfig",
    "FilterOutcome",
    "safety_privacy_filter",
]

# ---------------------------------------------------------------- script / LID
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("Latin", 0x0041, 0x005A),
    ("Latin", 0x0061, 0x007A),
    ("Latin", 0x00C0, 0x024F),
    ("Latin", 0x1E00, 0x1EFF),
    ("Greek", 0x0370, 0x03FF),
    ("Cyrillic", 0x0400, 0x04FF),
    ("Hebrew", 0x0590, 0x05FF),
    ("Arabic", 0x0600, 0x06FF),
    ("Arabic", 0x0750, 0x077F),
    ("Devanagari", 0x0900, 0x097F),
    ("Thai", 0x0E00, 0x0E7F),
    ("Hangul", 0x1100, 0x11FF),
    ("Hangul", 0x3130, 0x318F),
    ("Kana", 0x3040, 0x30FF),
    ("Han", 0x3400, 0x4DBF),
    ("Han", 0x4E00, 0x9FFF),
    ("Han", 0xF900, 0xFAFF),
    ("Hangul", 0xAC00, 0xD7AF),
)

_NON_SPACED_SCRIPTS = frozenset({"Han", "Kana", "Thai"})

_SCRIPT_LANG = {
    "Greek": "el",
    "Cyrillic": "ru",
    "Hebrew": "he",
    "Arabic": "ar",
    "Devanagari": "hi",
    "Thai": "th",
    "Hangul": "ko",
}

_LATIN_STOPWORDS: Mapping[str, frozenset[str]] = {
    "en": frozenset("the and of to is in that it for with as was on are this be by".split()),
    "fr": frozenset("le la les et des est une un du que dans pour pas sur au".split()),
    "de": frozenset("der die das und ist nicht mit ein eine zu den von auf für".split()),
    "es": frozenset("el la los las y es que de en un una por para con no".split()),
}


def _script_of(ch: str) -> str | None:
    cp = ord(ch)
    for name, lo, hi in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    return "Other" if ch.isalpha() else None


def detect_script(text: str) -> tuple[str, float, dict[str, int]]:
    """Dominant script of the letters in ``text``: ``(script, fraction, counts)``.

    Kana and Han are reported together as ``"Japanese"`` when kana make up at
    least 10% of CJK letters. Text without letters returns ``("Zyyy", 0.0, {})``.
    """
    counts: Counter[str] = Counter()
    for ch in text:
        s = _script_of(ch)
        if s is not None:
            counts[s] += 1
    total = sum(counts.values())
    if total == 0:
        return "Zyyy", 0.0, {}
    kana, han = counts.get("Kana", 0), counts.get("Han", 0)
    merged = Counter(counts)
    if kana and kana >= 0.1 * (kana + han):
        merged.pop("Kana")
        merged.pop("Han", None)
        merged["Japanese"] = kana + han
    script, n = max(merged.items(), key=lambda kv: (kv[1], kv[0]))
    return script, n / total, dict(counts)


def whitespace_delimited(script: str) -> bool:
    """Whether word-based quality rules are meaningful for ``script``."""
    return script not in _NON_SPACED_SCRIPTS and script != "Japanese"


@dataclass(frozen=True)
class LanguageGuess:
    """Result of the script-based language-ID stub."""

    language: str  # BCP-47-like code, "und-<Script>" when unknown, "und" for no letters
    script: str
    confidence: float


def identify_language(text: str) -> LanguageGuess:
    """Language-ID stub: script detection, plus stopword voting for Latin text."""
    script, frac, _ = detect_script(text)
    if script == "Zyyy":
        return LanguageGuess("und", script, 0.0)
    if script == "Japanese":
        return LanguageGuess("ja", script, frac)
    if script == "Han":
        return LanguageGuess("zh", script, frac)
    if script in _SCRIPT_LANG:
        return LanguageGuess(_SCRIPT_LANG[script], script, frac)
    if script == "Latin":
        words = re.findall(r"[^\W\d_]+", text.casefold())
        if words:
            votes = {lang: sum(w in sw for w in words) for lang, sw in _LATIN_STOPWORDS.items()}
            lang, hits = max(votes.items(), key=lambda kv: (kv[1], kv[0] == "en"))
            if hits / len(words) >= 0.05:
                return LanguageGuess(lang, script, frac * min(1.0, 4 * hits / len(words)))
        return LanguageGuess("und-Latn", script, frac)
    return LanguageGuess(f"und-{script}", script, frac)


# -------------------------------------------------------------------- quality
@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for :func:`quality_check` (Gopher-style defaults, relaxed for short docs)."""

    min_chars: int = 64
    max_chars: int = 1_000_000
    min_words: int = 10
    max_words: int = 100_000
    min_mean_word_len: float = 2.5
    max_mean_word_len: float = 12.0
    max_symbol_ratio: float = 0.1  # ('#' + ellipsis) per word
    min_alpha_word_frac: float = 0.7  # words containing at least one letter
    max_dup_line_frac: float = 0.3
    max_top_bigram_frac: float = 0.2  # chars covered by the most frequent repeated word bigram
    max_dup_5gram_frac: float = 0.15  # chars covered by repeated word 5-grams
    max_non_printable_frac: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)


def _top_bigram_frac(words: list[str], total_chars: int) -> float:
    if len(words) < 2 or total_chars == 0:
        return 0.0
    grams = Counter(zip(words, words[1:]))
    (a, b), count = grams.most_common(1)[0]
    if count < 2:
        return 0.0
    return min(1.0, count * (len(a) + len(b)) / total_chars)


def _dup_5gram_frac(words: list[str], total_chars: int) -> float:
    n = 5
    if len(words) < 2 * n or total_chars == 0:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(words)
    for i, g in enumerate(grams):
        if counts[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    return min(1.0, sum(len(w) for w, c in zip(words, covered) if c) / total_chars)


def quality_check(text: str, config: QualityConfig | None = None, *, kind: str = "prose") -> QualityResult:
    """Heuristic quality gate (§8 transformation 4).

    ``kind="code"`` skips the prose-only rules (symbol ratio, alphabetic-word
    fraction, mean word length). Word rules are skipped for scripts that do not
    delimit words with whitespace.
    """
    cfg = config or QualityConfig()
    reasons: list[str] = []
    n_chars = len(text)
    words = text.split()
    script, _, _ = detect_script(text)
    spaced = whitespace_delimited(script)
    word_chars = sum(len(w) for w in words)
    non_printable = sum(1 for ch in text if not ch.isprintable() and ch not in "\n\t\r")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    dup_lines = len(lines) - len(set(lines)) if lines else 0
    metrics: dict[str, float] = {
        "chars": float(n_chars),
        "words": float(len(words)),
        "mean_word_len": word_chars / len(words) if words else 0.0,
        "symbol_ratio": (text.count("#") + text.count("...") + text.count("…")) / max(1, len(words)),
        "alpha_word_frac": sum(any(c.isalpha() for c in w) for w in words) / max(1, len(words)),
        "dup_line_frac": dup_lines / len(lines) if lines else 0.0,
        "top_bigram_frac": _top_bigram_frac(words, word_chars),
        "dup_5gram_frac": _dup_5gram_frac(words, word_chars),
        "non_printable_frac": non_printable / max(1, n_chars),
    }
    if n_chars < cfg.min_chars:
        reasons.append("too_short")
    if n_chars > cfg.max_chars:
        reasons.append("too_long")
    if metrics["non_printable_frac"] > cfg.max_non_printable_frac:
        reasons.append("non_printable")
    if metrics["dup_line_frac"] > cfg.max_dup_line_frac:
        reasons.append("duplicate_lines")
    if spaced:
        if len(words) < cfg.min_words:
            reasons.append("too_few_words")
        if len(words) > cfg.max_words:
            reasons.append("too_many_words")
        if metrics["top_bigram_frac"] > cfg.max_top_bigram_frac:
            reasons.append("repetitive_bigram")
        if metrics["dup_5gram_frac"] > cfg.max_dup_5gram_frac:
            reasons.append("repetitive_ngrams")
        if kind != "code" and words:
            if not cfg.min_mean_word_len <= metrics["mean_word_len"] <= cfg.max_mean_word_len:
                reasons.append("mean_word_length")
            if metrics["symbol_ratio"] > cfg.max_symbol_ratio:
                reasons.append("symbol_ratio")
            if metrics["alpha_word_frac"] < cfg.min_alpha_word_frac:
                reasons.append("non_alphabetic")
    else:
        # character-level repetition proxy for unspaced scripts
        if n_chars >= 20:
            grams = Counter(text[i : i + 4] for i in range(n_chars - 3))
            g, c = grams.most_common(1)[0]
            frac = c * 4 / n_chars if c > 1 else 0.0
            metrics["top_char4gram_frac"] = min(1.0, frac)
            if frac > 0.3:
                reasons.append("repetitive_chars")
    return QualityResult(not reasons, tuple(reasons), metrics)


# ----------------------------------------------------------------------- PII
PII_KINDS: tuple[str, ...] = ("email", "credit_card", "ip_address", "phone")

_PLACEHOLDER = {
    "email": "[EMAIL]",
    "credit_card": "[CREDIT_CARD]",
    "ip_address": "[IP_ADDRESS]",
    "phone": "[PHONE]",
}

_EMAIL_RE = re.compile(r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d])")
_IPV4_RE = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.]*\d)")
_IPV6_RE = re.compile(
    r"(?<![\w:])(?:"
    r"(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}"
    r"|(?:[0-9A-Fa-f]{1,4}:){1,6}:(?:[0-9A-Fa-f]{1,4}:){0,5}[0-9A-Fa-f]{1,4}"
    r"|::(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4}"
    r")(?![\w:])"
)
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:"
    r"\+\d{7,15}"
    r"|(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){1,4}"
    r")(?![\w])"
)
_DATE_RE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")


def luhn_valid(digits: str) -> bool:
    """Luhn checksum over a string of digits (separators must be removed first)."""
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_phone(s: str) -> bool:
    digits = sum(c.isdigit() for c in s)
    if not 7 <= digits <= 15:
        return False
    if _DATE_RE.fullmatch(s.strip()):
        return False
    groups = len(re.findall(r"\d+", s))
    return s.startswith("+") or "(" in s or groups >= 3


@dataclass(frozen=True)
class PIIFinding:
    """One redaction: kind and span in the ORIGINAL text (the value is not stored)."""

    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class RedactionResult:
    text: str
    findings: tuple[PIIFinding, ...]
    rejected_candidates: Mapping[str, int] = field(default_factory=dict)  # e.g. Luhn failures

    @property
    def modified(self) -> bool:
        return bool(self.findings)

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(f.kind for f in self.findings))


def redact_pii(text: str, kinds: Iterable[str] = PII_KINDS) -> RedactionResult:
    """Redact PII in priority order email > credit card > IP > phone.

    Spans are claimed against the original text; a later (lower-priority)
    pattern cannot match inside an already-claimed span. Credit-card-like digit
    runs are redacted only when they pass the Luhn check; failures are counted
    in ``rejected_candidates`` (they may still be caught by the phone rule).
    """
    kinds = tuple(kinds)
    unknown = set(kinds) - set(PII_KINDS)
    if unknown:
        raise ValueError(f"unknown PII kinds {sorted(unknown)}")
    claimed: list[tuple[int, int, str]] = []
    rejected: Counter[str] = Counter()

    def free(s: int, e: int) -> bool:
        return all(e <= cs or s >= ce for cs, ce, _ in claimed)

    for kind in PII_KINDS:
        if kind not in kinds:
            continue
        if kind == "email":
            for m in _EMAIL_RE.finditer(text):
                if free(*m.span()):
                    claimed.append((m.start(), m.end(), kind))
        elif kind == "credit_card":
            for m in _CARD_RE.finditer(text):
                digits = re.sub(r"\D", "", m.group())
                if 13 <= len(digits) <= 19 and luhn_valid(digits):
                    if free(*m.span()):
                        claimed.append((m.start(), m.end(), kind))
                else:
                    rejected["credit_card_luhn_fail"] += 1
        elif kind == "ip_address":
            for rx in (_IPV4_RE, _IPV6_RE):
                for m in rx.finditer(text):
                    if free(*m.span()):
                        claimed.append((m.start(), m.end(), kind))
        elif kind == "phone":
            for m in _PHONE_RE.finditer(text):
                if _is_phone(m.group()) and free(*m.span()):
                    claimed.append((m.start(), m.end(), kind))
    claimed.sort()
    out: list[str] = []
    pos = 0
    for s, e, kind in claimed:
        out.append(text[pos:s])
        out.append(_PLACEHOLDER[kind])
        pos = e
    out.append(text[pos:])
    findings = tuple(PIIFinding(k, s, e) for s, e, k in claimed)
    return RedactionResult("".join(out), findings, dict(rejected))


# -------------------------------------------------------------------- safety
_SPECIAL_LITERAL_RE = re.compile(r"<\|[A-Za-z_/][A-Za-z0-9_/]*\|>")


@dataclass(frozen=True)
class SafetyConfig:
    """Safety/privacy filter configuration (§8 transformation 5).

    ``blocked_terms`` is a placeholder for a real safety classifier: any
    case-insensitive whole-word hit drops the document.
    """

    redact_kinds: tuple[str, ...] = PII_KINDS
    blocked_terms: tuple[str, ...] = ()
    max_pii_findings: int | None = None  # drop docs that are mostly PII dumps

    def to_dict(self) -> dict[str, Any]:
        return {"redact_kinds": list(self.redact_kinds), "blocked_terms": len(self.blocked_terms), "max_pii_findings": self.max_pii_findings}


@dataclass(frozen=True)
class FilterOutcome:
    """Filtered text plus a record of every rule that fired."""

    text: str
    keep: bool
    fired: dict[str, int]
    reasons: tuple[str, ...] = ()
    findings: tuple[PIIFinding, ...] = ()

    @property
    def modified(self) -> bool:
        return bool(self.findings)


def safety_privacy_filter(text: str, config: SafetyConfig | None = None) -> FilterOutcome:
    """Redact PII, flag special-token literals, and drop blocked content.

    Special-token look-alikes (``<|tool_call|>`` in raw text) are only *counted*
    (``special_token_literal``): the tokenizer already encodes them as plain
    bytes, so no rewriting is needed for injection safety.
    """
    cfg = config or SafetyConfig()
    red = redact_pii(text, cfg.redact_kinds)
    fired: Counter[str] = Counter(red.counts)
    fired.update(red.rejected_candidates)
    n_special = len(_SPECIAL_LITERAL_RE.findall(text))
    if n_special:
        fired["special_token_literal"] = n_special
    reasons: list[str] = []
    if cfg.blocked_terms:
        low = red.text.casefold()
        for term in cfg.blocked_terms:
            if re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", low):
                fired["blocked_term"] += 1
        if fired.get("blocked_term"):
            reasons.append("unsafe_content")
    if cfg.max_pii_findings is not None and len(red.findings) > cfg.max_pii_findings:
        reasons.append("pii_dump")
    return FilterOutcome(red.text, not reasons, dict(fired), tuple(reasons), red.findings)
