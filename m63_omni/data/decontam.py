"""Benchmark decontamination and source-event holdout splitting (blueprint §8, §12 step 0, §13).

§8: "Evaluation decontamination covers captions, transcripts, frames, and
near-duplicates, with holdouts split by source event rather than random clip
so that adjacent segments cannot leak across train and test." §12 step 0
requires holdouts to be isolated *before* data processing, and §15 prescribes
"Rebuild holdouts by source event and near-duplicate media" when contamination
is suspected.

* :class:`NGramIndex` — word n-gram (default 13) overlap index built from
  holdout captions, transcripts, and text. Scripts without spaces (CJK, kana,
  Hangul) are tokenised per character so they are not one giant "word".
  Holdout items shorter than ``n`` tokens (but at least ``min_tokens``) are
  indexed whole.
* :class:`Decontaminator` — combines the n-gram index, a frame/clip
  embedding index (re-using :class:`~m63_omni.data.dedup.MediaNearDupIndex`),
  and the set of held-out source events.
* :func:`split_by_source_event` — deterministic, hash-based group split: an
  event is in exactly one split, and an event's assignment does not change
  when other events are added.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterable, Iterator, Sequence, TypeVar

import numpy as np

from .dedup import MediaNearDupIndex

__all__ = [
    "ngram_tokens",
    "iter_ngrams",
    "NGramIndex",
    "ContaminationReport",
    "Decontaminator",
    "HoldoutSplit",
    "split_by_source_event",
    "HoldoutError",
]

T = TypeVar("T")

_CJK = r"぀-ヿ㐀-䶿一-鿿豈-﫿가-힯฀-๿"
_TOKEN_RE = re.compile(rf"[{_CJK}]|(?:(?![{_CJK}])[^\W_])+")


class HoldoutError(ValueError):
    """Raised when a holdout split cannot be made safely."""


def ngram_tokens(text: str) -> list[str]:
    """Normalised tokens for overlap matching: NFKC + casefold, alphanumeric
    runs, with CJK/kana/Hangul/Thai characters as individual tokens."""
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def iter_ngrams(tokens: Sequence[str], n: int) -> Iterator[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


class NGramIndex:
    """Exact word-n-gram overlap index over holdout text (§8)."""

    def __init__(self, n: int = 13, min_tokens: int = 8) -> None:
        if n < 1 or min_tokens < 1 or min_tokens > n:
            raise ValueError("need 1 <= min_tokens <= n")
        self.n = int(n)
        self.min_tokens = int(min_tokens)
        self._grams: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.skipped_short: list[str] = []
        self.n_items = 0

    def add(self, holdout_id: str, text: str) -> int:
        """Index ``text``; returns the number of grams added."""
        toks = ngram_tokens(text)
        self.n_items += 1
        if len(toks) >= self.n:
            table = self._grams[self.n]
            count = 0
            for g in iter_ngrams(toks, self.n):
                table[g].add(holdout_id)
                count += 1
            return count
        if len(toks) >= self.min_tokens:
            self._grams[len(toks)][" ".join(toks)].add(holdout_id)
            return 1
        self.skipped_short.append(holdout_id)
        return 0

    def matches(self, text: str) -> tuple[dict[str, int], int]:
        """``({holdout_id: n_matching_grams}, total_matching_grams)`` for ``text``."""
        toks = ngram_tokens(text)
        per_id: dict[str, int] = defaultdict(int)
        total = 0
        for length, table in self._grams.items():
            if len(toks) < length or not table:
                continue
            for g in iter_ngrams(toks, length):
                ids = table.get(g)
                if ids:
                    total += 1
                    for hid in ids:
                        per_id[hid] += 1
        return dict(per_id), total


@dataclass(frozen=True)
class ContaminationReport:
    """Outcome of a decontamination check."""

    contaminated: bool
    kind: str  # "text" | "media" | "event"
    holdout_ids: tuple[str, ...] = ()
    n_overlapping: int = 0
    max_similarity: float = 0.0


class Decontaminator:
    """Checks training items against isolated evaluation holdouts (§8, §13).

    Text: any shared ``n``-gram (``min_ngram_hits`` configurable) with a holdout
    caption, transcript, or text. Media: any frame/clip/audio embedding at
    cosine >= ``media_threshold`` to a holdout embedding. Events: training items
    whose source event was held out are always contaminated.
    """

    def __init__(
        self,
        n: int = 13,
        *,
        min_tokens: int = 8,
        min_ngram_hits: int = 1,
        media_threshold: float = 0.95,
        seed: int = 0,
    ) -> None:
        self.text_index = NGramIndex(n=n, min_tokens=min_tokens)
        self.media_index = MediaNearDupIndex(threshold=media_threshold, seed=seed)
        self.min_ngram_hits = int(min_ngram_hits)
        self.holdout_events: set[str] = set()
        self._holdout_kinds: dict[str, str] = {}

    @property
    def n(self) -> int:
        return self.text_index.n

    def params(self) -> dict[str, Any]:
        return {
            "n": self.text_index.n,
            "min_tokens": self.text_index.min_tokens,
            "min_ngram_hits": self.min_ngram_hits,
            "media": self.media_index.params(),
            "n_holdout_text_items": self.text_index.n_items,
            "n_holdout_media_items": len(self.media_index),
            "n_holdout_events": len(self.holdout_events),
        }

    # ------------------------------------------------------------- building
    def add_holdout_text(self, holdout_id: str, text: str, kind: str = "text") -> None:
        """Index a holdout caption / transcript / text item."""
        if kind not in ("text", "caption", "transcript"):
            raise ValueError(f"unknown holdout text kind {kind!r}")
        self._holdout_kinds[holdout_id] = kind
        self.text_index.add(holdout_id, text)

    def add_holdout_media(self, holdout_id: str, embeddings: Sequence[float] | np.ndarray) -> None:
        """Index holdout frame / clip / audio embeddings (one vector per row)."""
        self.media_index.add(holdout_id, embeddings)

    def add_holdout_events(self, events: Iterable[str]) -> None:
        self.holdout_events.update(e for e in events if e)

    def add_holdout_document(self, doc: Any) -> None:
        """Index everything a holdout document carries: text, timed segments,
        embeddings, and its source event (duck-typed on the pipeline ``Document``)."""
        doc_id = str(doc.doc_id)
        if getattr(doc, "text", None):
            self.add_holdout_text(doc_id, doc.text, kind="caption" if getattr(doc, "media", None) else "text")
        for i, seg in enumerate(getattr(doc, "segments", ()) or ()):
            self.add_holdout_text(f"{doc_id}#seg{i}", seg.text, kind="transcript")
        emb = getattr(doc, "embeddings", None)
        if emb is not None:
            self.add_holdout_media(doc_id, emb)
        event = getattr(doc, "source_event", None)
        if event:
            self.holdout_events.add(event)

    @classmethod
    def from_holdout(cls, docs: Iterable[Any], **kwargs: Any) -> "Decontaminator":
        dec = cls(**kwargs)
        for d in docs:
            dec.add_holdout_document(d)
        return dec

    # --------------------------------------------------------------- checks
    def check_text(self, text: str) -> ContaminationReport:
        per_id, total = self.text_index.matches(text)
        hit = total >= self.min_ngram_hits and total > 0
        return ContaminationReport(hit, "text", tuple(sorted(per_id)), total)

    def is_contaminated(self, text: str) -> bool:
        """True when ``text`` shares an n-gram with any holdout text item."""
        return self.check_text(text).contaminated

    def check_media(self, embeddings: Sequence[float] | np.ndarray) -> ContaminationReport:
        """Any query frame/clip near-duplicating any holdout frame/clip contaminates."""
        if len(self.media_index) == 0:
            return ContaminationReport(False, "media")
        matches = self.media_index.query(embeddings)
        if not matches:
            return ContaminationReport(False, "media")
        return ContaminationReport(
            True,
            "media",
            tuple(sorted(m.item_id for m in matches)),
            len(matches),
            max(m.max_similarity for m in matches),
        )

    def check_event(self, source_event: str | None) -> ContaminationReport:
        hit = bool(source_event) and source_event in self.holdout_events
        return ContaminationReport(hit, "event", (source_event,) if hit else ())


# ------------------------------------------------------------------- holdouts
@dataclass(frozen=True)
class HoldoutSplit(Generic[T]):
    """Train/test split where every source event lives in exactly one side."""

    train: list[T]
    test: list[T]
    train_events: frozenset[str]
    test_events: frozenset[str]
    seed: int = 0
    test_fraction: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)


def _event_unit(seed: int, event: str) -> float:
    h = hashlib.sha256(f"m63-holdout:{seed}:{event}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def split_by_source_event(
    items: Iterable[T],
    test_fraction: float = 0.1,
    *,
    seed: int = 0,
    key: Callable[[T], str | None] | None = None,
) -> HoldoutSplit[T]:
    """Split ``items`` into train/test BY SOURCE EVENT, never by individual clip (§8).

    Each event is assigned by ``sha256(seed, event)`` mapped to ``[0, 1)``:
    ``< test_fraction`` goes to test. This is deterministic and stable — adding
    new events never moves existing ones. If the hash rule leaves a side empty
    while there are >= 2 events, the extreme event is moved so both sides are
    populated. Items without a group key raise :class:`HoldoutError` rather than
    being split as independent clips.
    """
    if not 0.0 < test_fraction < 1.0:
        raise HoldoutError("test_fraction must be in (0, 1)")
    keyf = key or (lambda it: getattr(it, "source_event"))
    items = list(items)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        ev = keyf(it)
        if ev is None or ev == "":
            raise HoldoutError(f"item {i} has no source event; refusing to split by clip")
        groups[str(ev)].append(i)
    units = {ev: _event_unit(seed, ev) for ev in groups}
    test_events = {ev for ev, u in units.items() if u < test_fraction}
    if len(groups) >= 2:
        if not test_events:
            test_events = {min(units, key=lambda e: (units[e], e))}
        elif len(test_events) == len(groups):
            test_events.discard(max(units, key=lambda e: (units[e], e)))
    train_events = set(groups) - test_events
    train = [it for it in items if str(keyf(it)) in train_events]
    test = [it for it in items if str(keyf(it)) in test_events]
    return HoldoutSplit(
        train=train,
        test=test,
        train_events=frozenset(train_events),
        test_events=frozenset(test_events),
        seed=seed,
        test_fraction=test_fraction,
        params={"method": "sha256_event_threshold", "n_events": len(groups)},
    )
