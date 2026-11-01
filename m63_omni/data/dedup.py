"""Exact and near-duplicate detection for text and media (blueprint §8, Appendix B).

§8: "Text uses document-level deduplication plus approximate matching, while
media uses hashes and embedding-based near-duplicate detection across frames,
clips, and audio." This module provides:

* exact dedup — SHA-256 of normalised text (NFKC, casefold, whitespace
  collapsed) or of raw bytes (:class:`ExactDeduplicator`);
* near-dup text — seeded MinHash signatures (numpy) with LSH banding tuned to a
  configurable Jaccard threshold; candidates are verified against the estimated
  Jaccard before being reported (:class:`NearDuplicateIndex`);
* near-dup media — cosine similarity over caller-supplied embedding vectors
  (one vector per frame / clip / audio window) with a random-hyperplane
  bucketing index (:class:`MediaNearDupIndex`). The same index backs
  frame/clip decontamination in :mod:`m63_omni.data.decontam`.
"""

from __future__ import annotations

import functools
import hashlib
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..common import content_hash

__all__ = [
    "normalize_text",
    "text_hash",
    "bytes_hash",
    "ExactDeduplicator",
    "shingles",
    "jaccard",
    "MinHasher",
    "optimal_lsh_params",
    "MinHashLSH",
    "NearDuplicateIndex",
    "cosine_similarity",
    "MediaMatch",
    "MediaNearDupIndex",
]

# --------------------------------------------------------------------- exact
def normalize_text(text: str) -> str:
    """Normalisation used for exact text dedup: NFKC, casefold, collapse whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_hash(text: str, *, normalize: bool = True) -> str:
    """``sha256:<hex>`` of (normalised) text."""
    return content_hash(normalize_text(text) if normalize else text)


def bytes_hash(data: bytes) -> str:
    """``sha256:<hex>`` of raw media bytes."""
    return content_hash(bytes(data))


class ExactDeduplicator:
    """First-seen-wins exact dedup over content hashes."""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._seen)

    def lookup(self, digest: str) -> str | None:
        """ID of the item first registered with ``digest``, if any."""
        return self._seen.get(digest)

    def add(self, item_id: str, digest: str) -> str | None:
        """Register ``digest`` for ``item_id``; returns the original's ID if it is a duplicate."""
        first = self._seen.get(digest)
        if first is not None:
            return first
        self._seen[digest] = item_id
        return None

    def add_text(self, item_id: str, text: str) -> str | None:
        return self.add(item_id, text_hash(text))

    def add_bytes(self, item_id: str, data: bytes) -> str | None:
        return self.add(item_id, bytes_hash(data))


# ------------------------------------------------------------------- minhash
_EMPTY_HASH = 1 << 32  # unreachable by the 32-bit hash family; marks "no shingles"


def shingles(text: str, n: int = 5, mode: str = "char") -> set[str]:
    """Shingle set of normalised text: character ``n``-grams or word ``n``-grams.

    Texts shorter than one shingle yield a single shingle (the whole text);
    empty text yields the empty set.
    """
    norm = normalize_text(text)
    if not norm:
        return set()
    if mode == "char":
        if len(norm) <= n:
            return {norm}
        return {norm[i : i + n] for i in range(len(norm) - n + 1)}
    if mode == "word":
        words = norm.split()
        if len(words) <= n:
            return {" ".join(words)}
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
    raise ValueError(f"unknown shingle mode {mode!r}")


def jaccard(a: set[str], b: set[str]) -> float:
    """Exact Jaccard similarity of two sets (1.0 for two empty sets)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _hash32(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little")


class MinHasher:
    """Seeded MinHash with the multiply-add-shift family over 32-bit shingle hashes:
    ``h_i(x) = ((a_i * x + b_i) mod 2^64) >> 32`` with ``a_i, b_i`` uniform 64-bit.

    This family is strongly universal for 32-bit keys (Dietzfelbinger), and the
    ``mod 2^64`` is exactly numpy's ``uint64`` wraparound, so no big-integer
    arithmetic is needed.
    """

    def __init__(self, num_perm: int = 128, ngram: int = 5, mode: str = "char", seed: int = 0) -> None:
        if num_perm < 1:
            raise ValueError("num_perm must be >= 1")
        self.num_perm = int(num_perm)
        self.ngram = int(ngram)
        self.mode = mode
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        top = np.iinfo(np.uint64).max
        self._a = rng.integers(0, top, size=num_perm, dtype=np.uint64, endpoint=True)
        self._b = rng.integers(0, top, size=num_perm, dtype=np.uint64, endpoint=True)

    def signature_from_shingles(self, items: Iterable[str]) -> np.ndarray:
        hashes = np.fromiter((_hash32(s) for s in items), dtype=np.uint64)
        if hashes.size == 0:
            return np.full(self.num_perm, _EMPTY_HASH, dtype=np.uint64)
        with np.errstate(over="ignore"):
            phv = (self._a[:, None] * hashes[None, :] + self._b[:, None]) >> np.uint64(32)
        return phv.min(axis=1)

    def signature(self, text: str) -> np.ndarray:
        return self.signature_from_shingles(shingles(text, self.ngram, self.mode))

    @staticmethod
    def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        if sig_a.shape != sig_b.shape:
            raise ValueError("signatures have different lengths")
        return float(np.mean(sig_a == sig_b))


@functools.lru_cache(maxsize=256)
def optimal_lsh_params(
    threshold: float, num_perm: int, fp_weight: float = 0.5, fn_weight: float = 0.5
) -> tuple[int, int]:
    """Bands ``b`` and rows ``r`` (``b*r <= num_perm``) minimising the weighted
    false-positive/false-negative area of the S-curve ``1-(1-s^r)^b`` around ``threshold``."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    best: tuple[float, int, int] | None = None
    lo = np.linspace(0.0, threshold, 200)
    hi = np.linspace(threshold, 1.0, 200)
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            fp = np.trapezoid(1.0 - (1.0 - lo**r) ** b, lo)
            fn = np.trapezoid((1.0 - hi**r) ** b, hi)
            err = fp_weight * fp + fn_weight * fn
            if best is None or err < best[0]:
                best = (float(err), b, r)
    assert best is not None
    return best[1], best[2]


class MinHashLSH:
    """LSH banding index over MinHash signatures."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, bands: int | None = None, rows: int | None = None) -> None:
        if bands is None or rows is None:
            bands, rows = optimal_lsh_params(threshold, num_perm)
        if bands * rows > num_perm:
            raise ValueError("bands * rows must not exceed num_perm")
        self.threshold = float(threshold)
        self.num_perm = int(num_perm)
        self.bands = int(bands)
        self.rows = int(rows)
        self._tables: list[dict[bytes, list[str]]] = [defaultdict(list) for _ in range(self.bands)]
        self._keys: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def _band_keys(self, sig: np.ndarray) -> list[bytes]:
        if sig.shape != (self.num_perm,):
            raise ValueError(f"signature must have shape ({self.num_perm},)")
        return [sig[i * self.rows : (i + 1) * self.rows].tobytes() for i in range(self.bands)]

    def insert(self, key: str, sig: np.ndarray) -> None:
        if key in self._keys:
            raise KeyError(f"duplicate LSH key {key!r}")
        self._keys[key] = sig
        for table, bk in zip(self._tables, self._band_keys(sig)):
            table[bk].append(key)

    def signature(self, key: str) -> np.ndarray:
        return self._keys[key]

    def candidates(self, sig: np.ndarray) -> list[str]:
        """Keys sharing at least one band with ``sig`` (insertion order, no duplicates)."""
        seen: dict[str, None] = {}
        for table, bk in zip(self._tables, self._band_keys(sig)):
            for key in table.get(bk, ()):
                seen.setdefault(key, None)
        return list(seen)


class NearDuplicateIndex:
    """MinHash + LSH near-duplicate detector for text (§8 "approximate matching")."""

    def __init__(
        self, threshold: float = 0.8, num_perm: int = 128, ngram: int = 5, mode: str = "char", seed: int = 0
    ) -> None:
        self.threshold = float(threshold)
        self.hasher = MinHasher(num_perm=num_perm, ngram=ngram, mode=mode, seed=seed)
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    def __len__(self) -> int:
        return len(self.lsh)

    def params(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "num_perm": self.hasher.num_perm,
            "ngram": self.hasher.ngram,
            "mode": self.hasher.mode,
            "seed": self.hasher.seed,
            "bands": self.lsh.bands,
            "rows": self.lsh.rows,
        }

    def _query_sig(self, sig: np.ndarray) -> list[tuple[str, float]]:
        hits = []
        for key in self.lsh.candidates(sig):
            sim = MinHasher.estimate_jaccard(sig, self.lsh.signature(key))
            if sim >= self.threshold:
                hits.append((key, sim))
        hits.sort(key=lambda kv: -kv[1])
        return hits

    def query(self, text: str) -> list[tuple[str, float]]:
        """Indexed documents whose estimated Jaccard with ``text`` is >= threshold, best first."""
        return self._query_sig(self.hasher.signature(text))

    def add(self, doc_id: str, text: str) -> None:
        self.lsh.insert(doc_id, self.hasher.signature(text))

    def check_and_add(self, doc_id: str, text: str) -> tuple[str, float] | None:
        """Return ``(original_id, similarity)`` if ``text`` near-duplicates an indexed
        document; otherwise index it and return ``None``."""
        sig = self.hasher.signature(text)
        hits = self._query_sig(sig)
        if hits:
            return hits[0]
        self.lsh.insert(doc_id, sig)
        return None


# --------------------------------------------------------------------- media
def _as_matrix(embeddings: Sequence[float] | np.ndarray) -> np.ndarray:
    m = np.asarray(embeddings, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty vector or (n, d) matrix")
    if not np.all(np.isfinite(m)):
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embeddings contain a zero vector")
    return m / norms


def cosine_similarity(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix between rows of ``a`` and rows of ``b``."""
    return _as_matrix(a) @ _as_matrix(b).T


@dataclass(frozen=True)
class MediaMatch:
    """A stored media item matched by a query."""

    item_id: str
    max_similarity: float
    matched_fraction: float  # fraction of query vectors with a match in this item


class MediaNearDupIndex:
    """Cosine near-duplicate index over frame / clip / audio embeddings.

    Each item contributes one or more unit vectors. Vectors are bucketed by
    ``n_tables`` independent random-hyperplane signatures of ``n_bits`` bits;
    a query compares exactly (cosine) against every vector sharing a bucket in
    any table (or against all vectors with ``exhaustive=True``). With the
    defaults (8 tables x 6 bits) a pair at cosine 0.95 is missed with
    probability < 0.5%.
    """

    def __init__(
        self,
        dim: int | None = None,
        threshold: float = 0.95,
        n_tables: int = 8,
        n_bits: int = 6,
        seed: int = 0,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a cosine similarity in [-1, 1]")
        self.threshold = float(threshold)
        self.n_tables = int(n_tables)
        self.n_bits = int(n_bits)
        self.seed = int(seed)
        self.dim: int | None = None
        self._planes: np.ndarray | None = None
        self._vectors: list[np.ndarray] = []
        self._owner: list[str] = []
        self._items: dict[str, list[int]] = {}
        self._tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(self.n_tables)]
        if dim is not None:
            self._init_dim(int(dim))

    def _init_dim(self, dim: int) -> None:
        self.dim = dim
        rng = np.random.default_rng(self.seed)
        self._planes = rng.standard_normal((self.n_tables, self.n_bits, dim))
        self._weights = (1 << np.arange(self.n_bits)).astype(np.int64)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items

    def params(self) -> dict[str, object]:
        return {"threshold": self.threshold, "n_tables": self.n_tables, "n_bits": self.n_bits, "seed": self.seed, "dim": self.dim}

    def _prepare(self, embeddings: Sequence[float] | np.ndarray) -> np.ndarray:
        m = _as_matrix(embeddings)
        if self.dim is None:
            self._init_dim(m.shape[1])
        if m.shape[1] != self.dim:
            raise ValueError(f"embedding dim {m.shape[1]} != index dim {self.dim}")
        return m

    def _bucket_keys(self, m: np.ndarray) -> np.ndarray:
        assert self._planes is not None
        bits = np.einsum("tbd,nd->ntb", self._planes, m) > 0  # (n, tables, bits)
        return (bits.astype(np.int64) * self._weights).sum(axis=2)  # (n, tables)

    def add(self, item_id: str, embeddings: Sequence[float] | np.ndarray) -> None:
        if item_id in self._items:
            raise KeyError(f"duplicate media item {item_id!r}")
        m = self._prepare(embeddings)
        keys = self._bucket_keys(m)
        idxs = []
        for row, vec in enumerate(m):
            vid = len(self._vectors)
            self._vectors.append(vec)
            self._owner.append(item_id)
            idxs.append(vid)
            for t in range(self.n_tables):
                self._tables[t][int(keys[row, t])].append(vid)
        self._items[item_id] = idxs

    def query(self, embeddings: Sequence[float] | np.ndarray, *, exhaustive: bool = False) -> list[MediaMatch]:
        """Items with at least one vector at cosine >= threshold to a query vector, best first."""
        if not self._vectors:
            return []
        m = self._prepare(embeddings)
        if exhaustive:
            cand_sets = [range(len(self._vectors))] * m.shape[0]
        else:
            keys = self._bucket_keys(m)
            cand_sets = []
            for row in range(m.shape[0]):
                cands: set[int] = set()
                for t in range(self.n_tables):
                    cands.update(self._tables[t].get(int(keys[row, t]), ()))
                cand_sets.append(sorted(cands))
        best: dict[str, float] = {}
        matched_rows: dict[str, set[int]] = defaultdict(set)
        for row, cands in enumerate(cand_sets):
            cands = list(cands)
            if not cands:
                continue
            sims = np.stack([self._vectors[i] for i in cands]) @ m[row]
            for vid, sim in zip(cands, sims):
                if sim >= self.threshold:
                    owner = self._owner[vid]
                    matched_rows[owner].add(row)
                    if sim > best.get(owner, -2.0):
                        best[owner] = float(sim)
        n = m.shape[0]
        out = [MediaMatch(k, best[k], len(matched_rows[k]) / n) for k in best]
        out.sort(key=lambda x: (-x.max_similarity, x.item_id))
        return out

    def check_and_add(
        self, item_id: str, embeddings: Sequence[float] | np.ndarray, *, min_match_fraction: float = 0.5
    ) -> MediaMatch | None:
        """Return the best match covering >= ``min_match_fraction`` of the query
        vectors (so a shared black frame alone does not make two clips duplicates);
        otherwise index the item and return ``None``."""
        for match in self.query(embeddings):
            if match.matched_fraction >= min_match_fraction:
                return match
        self.add(item_id, embeddings)
        return None
