"""IFEval-style verifiable instruction following (Zhou et al., 2023; the benchmark is Apache-2.0).

Each prompt carries one or more *verifiable* instructions ("answer in all
lowercase", "at least 300 words", "wrap the answer in JSON", ...). A response
follows an instruction when a deterministic check passes, so scoring needs no
judge model. Metrics, as in the official scorer:

* **prompt-level strict** — the share of prompts whose every instruction is followed;
* **instruction-level strict** — the share of all instructions followed;
* **loose** variants — an instruction also counts if it passes after removing the
  first and/or last line or the ``*`` markdown characters (8 response variants).

Input files use the official format, one JSON object per line::

    {"key": 1000, "prompt": "...", "instruction_id_list": ["punctuation:no_comma", ...],
     "kwargs": [{}, {"num_words": 300, "relation": "at least"}, ...]}

and responses ``{"prompt": "...", "response": "..."}`` (or keyed by ``"key"``).

This is a re-implementation of the official checks, not the official scorer:
word counts use the same ``\\w+`` tokenizer, but sentence splitting and
capital-word counting use simple regexes instead of NLTK, and
``language:response_language`` (which needs a language detector) is not
supported — load with ``skip_unsupported=True`` to drop those prompts, and the
count of dropped prompts is reported. That is adequate for the
:mod:`~m63_omni.eval.parity` gate, which scores the candidate and the control
with the *same* scorer on the same prompts; quote absolute IFEval numbers from
the official scorer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

RELATIONS = ("less than", "at least")
CONSTRAINED_OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")


class IFEvalError(ValueError):
    pass


def _compare(count: int, threshold: int, relation: str) -> bool:
    if relation == "less than":
        return count < threshold
    if relation == "at least":
        return count >= threshold
    raise IFEvalError(f"relation must be one of {RELATIONS}, got {relation!r}")


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text)  # the official count_words tokenizer (RegexpTokenizer(r"\w+"))


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", text.strip())
    return [p for p in parts if p.strip()]


# ------------------------------------------------------------------ checks
def _keywords_existence(r: str, keywords: Sequence[str]) -> bool:
    return all(re.search(re.escape(k), r, flags=re.IGNORECASE) for k in keywords)


def _keywords_frequency(r: str, keyword: str, frequency: int, relation: str) -> bool:
    return _compare(len(re.findall(re.escape(keyword), r, flags=re.IGNORECASE)), frequency, relation)


def _forbidden_words(r: str, forbidden_words: Sequence[str]) -> bool:
    return not any(re.search(r"\b" + re.escape(w) + r"\b", r, flags=re.IGNORECASE) for w in forbidden_words)


def _letter_frequency(r: str, letter: str, let_frequency: int, let_relation: str) -> bool:
    return _compare(r.lower().count(letter.lower()), let_frequency, let_relation)


def _number_sentences(r: str, num_sentences: int, relation: str) -> bool:
    return _compare(len(_sentences(r)), num_sentences, relation)


def _number_paragraphs(r: str, num_paragraphs: int) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", r)
    n = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            if i in (0, len(paragraphs) - 1):
                n -= 1
            else:
                return False
    return n == num_paragraphs


def _number_words(r: str, num_words: int, relation: str) -> bool:
    return _compare(len(_words(r)), num_words, relation)


def _nth_paragraph_first_word(r: str, num_paragraphs: int, nth_paragraph: int, first_word: str) -> bool:
    paragraphs = re.split(r"\n\n", r)
    n = sum(1 for p in paragraphs if p.strip())
    if nth_paragraph > n:
        return False
    paragraph = paragraphs[nth_paragraph - 1].strip()
    if not paragraph:
        return False
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    first = ""
    for ch in word:
        if ch in {".", ",", "?", "!", "'", '"'}:
            break
        first += ch.lower()
    return n == num_paragraphs and first == first_word.lower()


def _number_placeholders(r: str, num_placeholders: int) -> bool:
    return len(re.findall(r"\[.*?\]", r)) >= num_placeholders


def _postscript(r: str, postscript_marker: str) -> bool:
    v = r.lower()
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(postscript_marker.lower()) + r".*$"
    return bool(re.findall(pattern, v, flags=re.MULTILINE))


def _bullet_lists(r: str, num_bullets: int) -> bool:
    star = re.findall(r"^\s*\*[^\*].*$", r, flags=re.MULTILINE)
    dash = re.findall(r"^\s*-.*$", r, flags=re.MULTILINE)
    return len(star) + len(dash) == num_bullets


def _constrained_response(r: str) -> bool:
    v = r.strip()
    return any(option in v for option in CONSTRAINED_OPTIONS)


def _highlighted_sections(r: str, num_highlights: int) -> bool:
    n = sum(1 for h in re.findall(r"\*[^\n\*]*\*", r) if h.strip("*").strip())
    n += sum(1 for h in re.findall(r"\*\*[^\n\*]*\*\*", r) if h.removeprefix("**").removesuffix("**").strip())
    return n >= num_highlights


def _multiple_sections(r: str, section_spliter: str, num_sections: int) -> bool:
    sections = re.split(r"\s?" + re.escape(section_spliter) + r"\s?\d+\s?", r)
    return len(sections) - 1 >= num_sections


def _json_format(r: str) -> bool:
    v = r.strip()
    for prefix in ("```json", "```Json", "```JSON", "```"):
        v = v.removeprefix(prefix)
    v = v.removesuffix("```").strip()
    try:
        json.loads(v)
    except ValueError:
        return False
    return True


def _title(r: str) -> bool:
    return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", r))


def _two_responses(r: str) -> bool:
    valid = []
    parts = r.split("******")
    for i, p in enumerate(parts):
        if not p.strip():
            if i not in (0, len(parts) - 1):
                return False
        else:
            valid.append(p)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _repeat_prompt(r: str, prompt_to_repeat: str) -> bool:
    return r.strip().lower().startswith(prompt_to_repeat.strip().lower())


def _end_checker(r: str, end_phrase: str) -> bool:
    return r.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def _capital_word_frequency(r: str, capital_frequency: int, capital_relation: str) -> bool:
    return _compare(sum(1 for w in re.findall(r"\b\w+\b", r) if w.isupper()), capital_frequency, capital_relation)


def _english_capital(r: str) -> bool:
    return r.isupper()  # the official check also requires langdetect == "en"


def _english_lowercase(r: str) -> bool:
    return r.islower()  # the official check also requires langdetect == "en"


def _no_comma(r: str) -> bool:
    return "," not in r


def _quotation(r: str) -> bool:
    v = r.strip()
    return len(v) > 1 and v[0] == '"' and v[-1] == '"'


#: Instruction ID -> check(response, **kwargs). IDs and kwarg names follow the official benchmark.
CHECKS: dict[str, Callable[..., bool]] = {
    "keywords:existence": _keywords_existence,
    "keywords:frequency": _keywords_frequency,
    "keywords:forbidden_words": _forbidden_words,
    "keywords:letter_frequency": _letter_frequency,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "length_constraints:number_words": _number_words,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:number_bullet_lists": _bullet_lists,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_format:number_highlighted_sections": _highlighted_sections,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:json_format": _json_format,
    "detectable_format:title": _title,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "change_case:english_capital": _english_capital,
    "change_case:english_lowercase": _english_lowercase,
    "punctuation:no_comma": _no_comma,
}
UNSUPPORTED = ("language:response_language",)


# ------------------------------------------------------------------ items
@dataclass(frozen=True)
class IFEvalItem:
    key: str
    prompt: str
    instruction_ids: tuple[str, ...]
    kwargs: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if len(self.instruction_ids) != len(self.kwargs):
            raise IFEvalError(f"item {self.key}: instruction_id_list and kwargs differ in length")
        unknown = [i for i in self.instruction_ids if i not in CHECKS]
        if unknown:
            raise IFEvalError(f"item {self.key}: unsupported instruction(s) {unknown}")


@dataclass(frozen=True)
class IFEvalResult:
    key: str
    instruction_ids: tuple[str, ...]
    strict: tuple[bool, ...]
    loose: tuple[bool, ...]
    missing_response: bool = False

    @property
    def prompt_strict(self) -> bool:
        return all(self.strict)

    @property
    def prompt_loose(self) -> bool:
        return all(self.loose)


def _item(obj: Mapping[str, Any]) -> IFEvalItem:
    kwargs = tuple({k: v for k, v in (kw or {}).items() if v is not None} for kw in obj["kwargs"])
    return IFEvalItem(str(obj["key"]), obj["prompt"], tuple(obj["instruction_id_list"]), kwargs)


def _jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_ifeval(path: str | Path, *, skip_unsupported: bool = False) -> tuple[list[IFEvalItem], list[str]]:
    """Load official-format prompts. Returns ``(items, skipped keys)``.

    Prompts using an unsupported instruction raise unless ``skip_unsupported``,
    in which case they are dropped from *both* arms of any comparison.
    """
    items, skipped = [], []
    for obj in _jsonl(path):
        if any(i not in CHECKS for i in obj["instruction_id_list"]):
            if not skip_unsupported:
                bad = sorted({i for i in obj["instruction_id_list"] if i not in CHECKS})
                raise IFEvalError(f"prompt {obj['key']} uses unsupported instruction(s) {bad}; pass skip_unsupported=True")
            skipped.append(str(obj["key"]))
            continue
        items.append(_item(obj))
    return items, skipped


def load_responses(path: str | Path) -> dict[str, str]:
    """``{prompt or key: response}`` from an official-format response file."""
    out: dict[str, str] = {}
    for obj in _jsonl(path):
        k = str(obj["key"]) if "key" in obj else obj["prompt"]
        out[k] = obj["response"]
    return out


def _loose_variants(response: str) -> list[str]:
    lines = response.split("\n")
    rm_first = "\n".join(lines[1:]).strip()
    rm_last = "\n".join(lines[:-1]).strip()
    rm_both = "\n".join(lines[1:-1]).strip()
    base = [response, rm_first, rm_last, rm_both]
    return base + [v.replace("*", "") for v in base]


def check(instruction_id: str, response: str, kwargs: Mapping[str, Any] | None = None, *, loose: bool = False) -> bool:
    fn = CHECKS[instruction_id]
    kw = dict(kwargs or {})
    variants = _loose_variants(response) if loose else [response]
    return any(v.strip() and fn(v, **kw) for v in variants)


def score_ifeval(items: Iterable[IFEvalItem], responses: Mapping[str, str]) -> list[IFEvalResult]:
    """Score every item. A missing response fails all of its instructions (it is never skipped)."""
    out = []
    for it in items:
        r = responses.get(it.key, responses.get(it.prompt))
        if r is None:
            fails = tuple(False for _ in it.instruction_ids)
            out.append(IFEvalResult(it.key, it.instruction_ids, fails, fails, missing_response=True))
            continue
        out.append(IFEvalResult(
            it.key, it.instruction_ids,
            tuple(check(i, r, kw) for i, kw in zip(it.instruction_ids, it.kwargs)),
            tuple(check(i, r, kw, loose=True) for i, kw in zip(it.instruction_ids, it.kwargs)),
        ))
    return out


def summarize(results: Sequence[IFEvalResult]) -> dict[str, float]:
    if not results:
        raise IFEvalError("no results")
    n_instr = sum(len(r.strict) for r in results)
    return {
        "prompt_strict": sum(r.prompt_strict for r in results) / len(results),
        "instruction_strict": sum(sum(r.strict) for r in results) / n_instr,
        "prompt_loose": sum(r.prompt_loose for r in results) / len(results),
        "instruction_loose": sum(sum(r.loose) for r in results) / n_instr,
        "prompts": float(len(results)),
        "instructions": float(n_instr),
        "missing_responses": float(sum(r.missing_response for r in results)),
    }


# --------------------------------------------------------------- self-test
#: (instruction id, kwargs, response that follows it, response that does not)
SELF_TEST: tuple[tuple[str, dict, str, str], ...] = (
    ("punctuation:no_comma", {}, "No commas here at all.", "Yes, a comma."),
    ("change_case:english_lowercase", {}, "all lowercase words.", "Not all lowercase."),
    ("change_case:english_capital", {}, "ALL CAPS HERE.", "ALL CAPS but one."),
    ("length_constraints:number_words", {"num_words": 5, "relation": "at least"}, "one two three four five", "one two"),
    ("length_constraints:number_words", {"num_words": 3, "relation": "less than"}, "one two", "one two three"),
    ("length_constraints:number_sentences", {"num_sentences": 2, "relation": "at least"}, "First. Second one.", "Only one."),
    ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "Para one.\n***\nPara two.", "Para one only."),
    ("length_constraints:nth_paragraph_first_word", {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "then"},
     "First part.\n\nThen the second.", "First part.\n\nAnd the second."),
    ("keywords:existence", {"keywords": ["timeline", "render"]}, "Render the timeline.", "Render it."),
    ("keywords:forbidden_words", {"forbidden_words": ["cut"]}, "Trim the clip.", "Cut the clip."),
    ("keywords:frequency", {"keyword": "clip", "frequency": 2, "relation": "at least"}, "clip and clip", "one clip"),
    ("keywords:letter_frequency", {"letter": "z", "let_frequency": 2, "let_relation": "less than"}, "zoom", "zizz"),
    ("detectable_format:json_format", {}, '```json\n{"ok": true}\n```', "{not json}"),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* a\n* b", "* a\n* b\n* c"),
    ("detectable_format:title", {}, "<<A Title>>\nBody.", "No title."),
    ("detectable_format:constrained_response", {}, "My answer is yes.", "Probably yes."),
    ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "*one* and *two*", "*one* only"),
    ("detectable_format:multiple_sections", {"section_spliter": "Section", "num_sections": 2},
     "Section 1\nA.\nSection 2\nB.", "Section 1\nA."),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body.\nP.S. More.", "Body only."),
    ("detectable_content:number_placeholders", {"num_placeholders": 2}, "[name] at [place]", "[name] only"),
    ("combination:two_responses", {}, "Answer one.\n******\nAnswer two.", "Only one answer."),
    ("combination:repeat_prompt", {"prompt_to_repeat": "Say hi."}, "Say hi. Hi!", "Hi!"),
    ("startend:end_checker", {"end_phrase": "Is there anything else?"}, "Done. Is there anything else?", "Done."),
    ("startend:quotation", {}, '"Quoted."', "Not quoted."),
    ("change_case:capital_word_frequency", {"capital_frequency": 2, "capital_relation": "at least"},
     "USE BOTH words", "Only ONE word"),
)


def self_test() -> None:
    """Known-good responses must pass and known-bad must fail (a broken scorer cannot certify a model)."""
    from .harness import ScorerSelfTestError

    for iid, kw, good, bad in SELF_TEST:
        if not check(iid, good, kw):
            raise ScorerSelfTestError(f"IFEval scorer rejected a known-good response for {iid}")
        if check(iid, bad, kw):
            raise ScorerSelfTestError(f"IFEval scorer accepted a known-bad response for {iid}")
    missing = set(CHECKS) - {s[0] for s in SELF_TEST}
    if missing:
        raise ScorerSelfTestError(f"IFEval checks without self-test fixtures: {sorted(missing)}")


def by_key(results: Sequence[IFEvalResult]) -> dict[str, IFEvalResult]:
    out: dict[str, IFEvalResult] = {}
    for r in results:
        if r.key in out:
            raise IFEvalError(f"duplicate IFEval key {r.key}")
        out[r.key] = r
    return out
