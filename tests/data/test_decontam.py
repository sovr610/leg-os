"""Benchmark decontamination and source-event holdout splitting (§8, §12 step 0, §13)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from m63_omni.common import Modality
from m63_omni.data.decontam import (
    Decontaminator,
    HoldoutError,
    NGramIndex,
    ngram_tokens,
    split_by_source_event,
)
from m63_omni.data.pipeline import Document, MediaMeta, TimedSegment

HOLDOUT_CAPTION = (
    "A red kite circles twice above the harbour before diving toward the fishing boat "
    "while gulls scatter across the grey morning sky"
)


def test_ngram_tokens_normalise_and_split_cjk():
    assert ngram_tokens("Hello, WORLD!  it's") == ["hello", "world", "it", "s"]
    assert ngram_tokens("abc数据x") == ["abc", "数", "据", "x"]


def test_13gram_hit_and_near_miss():
    dec = Decontaminator(n=13)
    dec.add_holdout_text("eval/cap-1", HOLDOUT_CAPTION, kind="caption")
    words = HOLDOUT_CAPTION.split()
    leak = "Unrelated preamble. " + " ".join(words[3:16]).upper() + "!!! trailing text"
    assert dec.is_contaminated(leak)
    rep = dec.check_text(leak)
    assert rep.contaminated and rep.holdout_ids == ("eval/cap-1",) and rep.n_overlapping == 1
    only_12 = "Unrelated preamble " + " ".join(words[3:15]) + " trailing text"
    assert not dec.is_contaminated(only_12)
    assert not dec.is_contaminated("A completely different sentence about cooking pasta at home tonight.")


def test_ngram_size_is_configurable_and_short_holdouts():
    dec = Decontaminator(n=5, min_tokens=3)
    dec.add_holdout_text("t1", "the answer to the riddle is forty two", kind="text")
    assert dec.is_contaminated("I heard that the riddle is forty two today")
    idx = NGramIndex(n=13, min_tokens=8)
    idx.add("short", "what is the capital city of france please")  # 8 tokens < 13: indexed whole
    idx.add("tiny", "yes")
    assert idx.skipped_short == ["tiny"]
    hits, total = idx.matches("Q: What is the capital city of France, please? A: Paris")
    assert hits == {"short": 1} and total == 1


def test_transcripts_and_cjk_holdouts():
    dec = Decontaminator(n=8)
    dec.add_holdout_text("asr-1", "音频和视频按时间戳交错排列而不是拼接", kind="transcript")
    assert dec.is_contaminated("训练数据：音频和视频按时间戳交错排列。")
    with pytest.raises(ValueError):
        dec.add_holdout_text("x", "text", kind="bogus")


def test_media_decontamination_and_events():
    rng = np.random.default_rng(5)
    frames = rng.standard_normal((4, 32))
    dec = Decontaminator()
    dec.add_holdout_media("eval/clip-9", frames)
    dec.add_holdout_events(["event-9"])
    near = frames[2] + 0.01 * rng.standard_normal(32)
    rep = dec.check_media(np.stack([rng.standard_normal(32), near]))
    assert rep.contaminated and rep.holdout_ids == ("eval/clip-9",) and rep.max_similarity > 0.95
    assert not dec.check_media(rng.standard_normal((3, 32))).contaminated
    assert dec.check_event("event-9").contaminated and not dec.check_event("event-1").contaminated
    assert not dec.check_event(None).contaminated


def test_from_holdout_documents():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((3, 16))
    doc = Document(
        "eval-v1", "eval_src", Modality.VIDEO, text=HOLDOUT_CAPTION,
        media=MediaMeta(duration_s=8, has_video=True, fps=30),
        source_event="match-final",
        segments=[TimedSegment(0, 4000, "the commentator says the striker scores in the final minute of extra time")],
        embeddings=emb,
    )
    dec = Decontaminator.from_holdout([doc], n=8)
    assert dec.check_event("match-final").contaminated
    assert dec.is_contaminated("so the striker scores in the final minute of extra time again")
    assert dec.is_contaminated(HOLDOUT_CAPTION)
    assert dec.check_media(emb[1]).contaminated
    assert dec.params()["n_holdout_events"] == 1


@dataclass
class Clip:
    clip_id: str
    source_event: str | None


def _clips(n_events: int, per_event: int = 5) -> list[Clip]:
    return [Clip(f"e{e}-seg{s}", f"event-{e}") for e in range(n_events) for s in range(per_event)]


def test_split_by_source_event_never_splits_an_event():
    clips = _clips(40)
    split = split_by_source_event(clips, test_fraction=0.25, seed=11)
    assert split.train_events.isdisjoint(split.test_events)
    assert split.train_events | split.test_events == {c.source_event for c in clips}
    assert {c.source_event for c in split.train}.isdisjoint({c.source_event for c in split.test})
    # adjacent segments of an event stay together
    for ev in split.test_events:
        assert sum(c.source_event == ev for c in split.test) == 5
    assert len(split.train) + len(split.test) == len(clips)
    assert 0 < len(split.test_events) < 40


def test_split_is_deterministic_and_stable():
    clips = _clips(30)
    a = split_by_source_event(clips, 0.2, seed=3)
    b = split_by_source_event(list(reversed(clips)), 0.2, seed=3)
    assert a.test_events == b.test_events
    grown = split_by_source_event(clips + _clips(60)[150:], 0.2, seed=3)  # add events 30..59
    assert a.test_events <= grown.test_events  # existing assignments never move
    assert split_by_source_event(clips, 0.2, seed=4).test_events != a.test_events


def test_split_guarantees_both_sides_and_refuses_ungrouped():
    two = _clips(2, per_event=3)
    s = split_by_source_event(two, 0.01, seed=0)
    assert len(s.test_events) == 1 and len(s.train_events) == 1
    with pytest.raises(HoldoutError):
        split_by_source_event([Clip("a", "ev"), Clip("b", None)], 0.5)
    with pytest.raises(HoldoutError):
        split_by_source_event(two, 0.0)
