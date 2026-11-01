"""Exact and near-duplicate detection for text and media (§8)."""

from __future__ import annotations

import numpy as np
import pytest

from m63_omni.data.dedup import (
    ExactDeduplicator,
    MediaNearDupIndex,
    MinHasher,
    NearDuplicateIndex,
    bytes_hash,
    cosine_similarity,
    jaccard,
    normalize_text,
    optimal_lsh_params,
    shingles,
    text_hash,
)

BASE = (
    "Data quality is the largest program risk because the architecture cannot compensate for poor "
    "alignment, contaminated evaluations, or unusable rights. Candidate pools can include web, science, "
    "and code mixes, but every source requires an independent license, consent, privacy, and takedown review."
)
NEAR = BASE.replace("largest", "biggest")
OTHER = (
    "Audio and video chunks are interleaved in timestamp order rather than concatenating an entire video "
    "followed by its soundtrack, which makes temporal alignment learnable for the orchestrator."
)


def test_exact_dedup_normalises_text():
    assert normalize_text("  Hello\tWORLD \n") == "hello world"
    assert text_hash("Hello  World") == text_hash("hello world")
    assert text_hash("Hello World", normalize=False) != text_hash("hello world", normalize=False)
    assert text_hash("ﬁle") == text_hash("file")  # NFKC
    ex = ExactDeduplicator()
    assert ex.add_text("a", BASE) is None
    assert ex.add_text("b", BASE.upper().replace(" ", "  ")) == "a"
    assert ex.add_text("c", NEAR) is None
    assert ex.add_bytes("d", b"\x00\x01") is None
    assert ex.add_bytes("e", b"\x00\x01") == "d"
    assert bytes_hash(b"x").startswith("sha256:") and len(ex) == 3


def test_minhash_estimates_jaccard():
    mh = MinHasher(num_perm=256, seed=7)
    a, b = shingles(BASE), shingles(NEAR)
    true = jaccard(a, b)
    est = MinHasher.estimate_jaccard(mh.signature(BASE), mh.signature(NEAR))
    assert 0.8 < true < 1.0
    assert abs(est - true) < 0.08
    assert MinHasher.estimate_jaccard(mh.signature(BASE), mh.signature(OTHER)) < 0.2
    # seeded and deterministic
    assert np.array_equal(MinHasher(seed=7).signature(BASE), MinHasher(seed=7).signature(BASE))
    assert not np.array_equal(MinHasher(seed=7).signature(BASE), MinHasher(seed=8).signature(BASE))
    assert jaccard(set(), set()) == 1.0


def test_lsh_params_respect_budget():
    for t in (0.5, 0.8, 0.9):
        b, r = optimal_lsh_params(t, 128)
        assert b * r <= 128 and b >= 1 and r >= 1
    # higher threshold -> more rows per band (stricter bands)
    assert optimal_lsh_params(0.9, 128)[1] >= optimal_lsh_params(0.5, 128)[1]


def test_near_duplicate_index():
    idx = NearDuplicateIndex(threshold=0.8, seed=1)
    assert idx.check_and_add("base", BASE) is None
    hit = idx.check_and_add("near", NEAR)
    assert hit is not None and hit[0] == "base" and hit[1] >= 0.8
    assert idx.check_and_add("other", OTHER) is None
    assert len(idx) == 2  # the near-duplicate was not indexed
    assert [k for k, _ in idx.query(BASE)] == ["base"]


def test_near_duplicate_threshold_is_configurable():
    words = BASE.split()
    edited = " ".join(w if i % 4 else w.upper() + "x" for i, w in enumerate(words))
    j = jaccard(shingles(BASE), shingles(edited))
    assert 0.3 < j < 0.75
    loose, strict = NearDuplicateIndex(threshold=0.2, seed=3), NearDuplicateIndex(threshold=0.9, seed=3)
    loose.add("base", BASE)
    strict.add("base", BASE)
    assert loose.query(edited) and not strict.query(edited)


def _unit(rng, d):
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def test_media_near_dup_frames_and_clips():
    rng = np.random.default_rng(0)
    d = 64
    clip = np.stack([_unit(rng, d) for _ in range(6)])
    reencoded = clip + 0.02 * rng.standard_normal(clip.shape)  # same clip, re-encoded
    unrelated = np.stack([_unit(rng, d) for _ in range(6)])
    idx = MediaNearDupIndex(threshold=0.95, seed=0)
    assert idx.check_and_add("clip", clip) is None
    m = idx.check_and_add("copy", reencoded)
    assert m is not None and m.item_id == "clip" and m.max_similarity > 0.95 and m.matched_fraction == 1.0
    assert idx.check_and_add("other", unrelated) is None
    # one shared frame: reported by query, but not enough to dedupe a whole clip
    shares_one = np.stack([_unit(rng, d) for _ in range(5)] + [clip[2]])
    q = idx.query(shares_one)
    assert q and q[0].item_id == "clip" and q[0].matched_fraction == pytest.approx(1 / 6)
    assert idx.check_and_add("partial", shares_one, min_match_fraction=0.5) is None
    # bucketed search agrees with exhaustive search
    bucketed = {x.item_id for x in idx.query(reencoded)}
    assert bucketed == {x.item_id for x in idx.query(reencoded, exhaustive=True)} == {"clip", "partial"}


def test_media_index_validation():
    idx = MediaNearDupIndex(dim=4)
    with pytest.raises(ValueError):
        idx.add("a", np.zeros(4))
    with pytest.raises(ValueError):
        idx.add("b", np.ones(5))
    idx.add("c", np.ones(4))
    with pytest.raises(KeyError):
        idx.add("c", np.ones(4))
    assert cosine_similarity([1, 0], [[1, 0], [0, 1]]).tolist() == [[1.0, 0.0]]
