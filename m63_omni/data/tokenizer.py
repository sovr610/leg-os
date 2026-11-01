"""Byte-level BPE tokenizer and the fixed special-token registry (blueprint §5, §8, §12 step 4).

§5 specifies text/code as "byte-level BPE; tied embeddings; special tool and
timestamp tokens" with a "131,072-token vocabulary; reserve 256 special IDs
initially" and "multilingual fertility checks". This module implements all of
that from scratch with the standard library only:

* GPT-style pre-tokenisation using stdlib :mod:`re` (``[^\\W\\d_]`` stands in for
  ``\\p{L}``), with a catch-all branch so *every* character is covered and
  ``"".join(pretokenize(s)) == s`` holds for any string;
* a deterministic trainer (:func:`train_bpe`) with incremental pair counts;
* encode/decode that round-trips arbitrary well-formed Unicode (emoji, CJK,
  combining marks, zero-width and exotic whitespace) because the base alphabet
  is the 256 byte values;
* JSON save/load with a content fingerprint (the tokenizer version that runtime
  prefix-cache keys include, §14).

Special tokens occupy the TOP ``special_tokens`` IDs of the vocabulary:
``id = vocab_size - special_tokens + index`` where ``index`` is the position in
:data:`SPECIAL_TOKENS`. :func:`special_id` computes this from configuration
alone, so the model package never needs a trained tokenizer to find, e.g., the
``<|media|>`` placeholder ID.

Injection safety (§10): special-token strings that appear in raw text are
encoded as ordinary bytes unless the caller passes ``allow_special``. Untrusted
text can therefore never forge a ``<|tool_call|>`` or role token.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..common import Modality, Trust, content_hash

__all__ = [
    "NAMED_SPECIAL_TOKENS",
    "SPECIAL_TOKENS",
    "REFERENCE_SPECIAL_COUNT",
    "PRETOKENIZE_PATTERN",
    "ROLE_TOKENS",
    "MODALITY_SPAN_TOKENS",
    "TRUST_TOKENS",
    "special_token_names",
    "special_index",
    "special_id",
    "special_ids",
    "special_ids_for_config",
    "special_name",
    "is_special_id",
    "pretokenize",
    "ByteBPETokenizer",
    "train_bpe",
    "FertilityStats",
    "fertility_report",
    "TokenizerError",
]


class TokenizerError(ValueError):
    """Raised for invalid tokenizer layouts, IDs, or serialised files."""


# ----------------------------------------------------------------- special tokens
#: The named special tokens, in their fixed order. Their index in this tuple is
#: their offset from the bottom of the special-ID block. Never reorder: IDs are
#: part of the checkpoint contract.
NAMED_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|developer|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_result|>",
    "<|/tool_result|>",
    "<|image|>",
    "<|/image|>",
    "<|video|>",
    "<|/video|>",
    "<|audio|>",
    "<|/audio|>",
    "<|av_chunk|>",
    "<|ts|>",
    "<|media|>",  # placeholder position filled with an encoder output embedding
    "<|memory|>",
    "<|asset_ref|>",
    "<|trust_untrusted|>",
    "<|trust_verified|>",
    "<|escalate|>",
)

#: §5: "reserve 256 special IDs initially".
REFERENCE_SPECIAL_COUNT = 256

_RESERVED_RE = re.compile(r"<\|reserved_(\d+)\|>\Z")


def _reserved_name(i: int) -> str:
    return f"<|reserved_{i}|>"


def special_token_names(count: int) -> tuple[str, ...]:
    """The ordered special-token names for a block of ``count`` special IDs.

    The first ``len(NAMED_SPECIAL_TOKENS)`` entries are the named tokens; the
    rest are ``<|reserved_N|>`` with ``N`` counting reserved slots from 0.
    """
    n_named = len(NAMED_SPECIAL_TOKENS)
    if count < n_named:
        raise TokenizerError(f"special_tokens={count} cannot hold the {n_named} named special tokens")
    return NAMED_SPECIAL_TOKENS + tuple(_reserved_name(i) for i in range(count - n_named))


#: The registry padded to the reference reserve of 256 (§5). Smaller configs
#: (e.g. the ``tiny`` preset's 64) use a prefix; larger ones extend it.
SPECIAL_TOKENS: tuple[str, ...] = special_token_names(REFERENCE_SPECIAL_COUNT)

_NAMED_INDEX = {name: i for i, name in enumerate(NAMED_SPECIAL_TOKENS)}

#: Chat-role tokens (§10 role separation).
ROLE_TOKENS: Mapping[str, str] = {
    "system": "<|system|>",
    "developer": "<|developer|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
}

#: Open/close span tokens per encoder modality (§5).
MODALITY_SPAN_TOKENS: Mapping[Modality, tuple[str, str]] = {
    Modality.IMAGE: ("<|image|>", "<|/image|>"),
    Modality.VIDEO: ("<|video|>", "<|/video|>"),
    Modality.AUDIO: ("<|audio|>", "<|/audio|>"),
}

#: Trust-label tokens wrapped around content of the given provenance (§6, §10).
TRUST_TOKENS: Mapping[Trust, str] = {
    Trust.UNTRUSTED: "<|trust_untrusted|>",
    Trust.TOOL_VERIFIED: "<|trust_verified|>",
}


def special_index(name: str) -> int:
    """Offset of ``name`` within the special-ID block (independent of vocab size)."""
    idx = _NAMED_INDEX.get(name)
    if idx is not None:
        return idx
    m = _RESERVED_RE.match(name)
    if m:
        return len(NAMED_SPECIAL_TOKENS) + int(m.group(1))
    raise KeyError(f"unknown special token {name!r}")


def _check_layout(vocab_size: int, special_tokens: int) -> None:
    if not (0 < special_tokens < vocab_size):
        raise TokenizerError(f"need 0 < special_tokens ({special_tokens}) < vocab_size ({vocab_size})")


def special_id(name: str, vocab_size: int, special_tokens: int) -> int:
    """ID of special token ``name`` from configuration alone (no trained tokenizer needed).

    ``id = vocab_size - special_tokens + special_index(name)``. For the reference
    config (131072, 256): ``<|pad|>`` = 130816 and ``<|media|>`` = 130836.
    Raises ``KeyError`` if the name is unknown or does not fit in the block.
    """
    _check_layout(vocab_size, special_tokens)
    idx = special_index(name)
    if idx >= special_tokens:
        raise KeyError(f"special token {name!r} (index {idx}) does not fit in {special_tokens} special IDs")
    return vocab_size - special_tokens + idx


def special_ids(vocab_size: int, special_tokens: int) -> dict[str, int]:
    """Mapping of every special-token name in the block to its ID."""
    _check_layout(vocab_size, special_tokens)
    base = vocab_size - special_tokens
    names = special_token_names(special_tokens) if special_tokens >= len(NAMED_SPECIAL_TOKENS) else NAMED_SPECIAL_TOKENS[:special_tokens]
    return {name: base + i for i, name in enumerate(names)}


def special_ids_for_config(cfg: Any) -> dict[str, int]:
    """:func:`special_ids` for an :class:`~m63_omni.config.M63Config` (uses ``cfg.model``)."""
    return special_ids(cfg.model.vocab_size, cfg.model.special_tokens)


def is_special_id(token_id: int, vocab_size: int, special_tokens: int) -> bool:
    """True when ``token_id`` lies in the special-ID block."""
    return vocab_size - special_tokens <= token_id < vocab_size


def special_name(token_id: int, vocab_size: int, special_tokens: int) -> str | None:
    """Name of the special token with ``token_id``, or ``None`` for a regular ID."""
    if not is_special_id(token_id, vocab_size, special_tokens):
        return None
    idx = token_id - (vocab_size - special_tokens)
    if idx < len(NAMED_SPECIAL_TOKENS):
        return NAMED_SPECIAL_TOKENS[idx]
    return _reserved_name(idx - len(NAMED_SPECIAL_TOKENS))


# ------------------------------------------------------------- pre-tokenisation
#: GPT-2/GPT-4-style split expressed with stdlib ``re``:
#: contractions | optional-space + letters | optional-space + 1-3 digits |
#: optional-space + symbols (incl. underscore) | whitespace not followed by
#: non-space | whitespace | any remaining character (catch-all, guarantees
#: lossless coverage).
PRETOKENIZE_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"| ?[^\W\d_]+"
    r"| ?\d{1,3}"
    r"| ?(?:[^\s\w]|_)+"
    r"|\s+(?!\S)"
    r"|\s+"
    r"|[\s\S]"
)
_PRETOKENIZE_RE = re.compile(PRETOKENIZE_PATTERN)


def pretokenize(text: str, pattern: re.Pattern[str] = _PRETOKENIZE_RE) -> list[str]:
    """Split ``text`` into pre-tokens; concatenating them always reproduces ``text``."""
    return pattern.findall(text)


# --------------------------------------------------------------------- tokenizer
_N_BYTES = 256
_CACHE_LIMIT = 100_000


class ByteBPETokenizer:
    """Byte-level BPE with special tokens at the top of the vocabulary (§5).

    Regular IDs: ``0..255`` are raw bytes; ``256 + rank`` is the merge of rank
    ``rank``. IDs between the last merge and the special block are valid
    embedding rows that the tokenizer never emits (a corpus may not support the
    full merge budget). Special IDs follow :func:`special_id`.
    """

    FORMAT = "m63-byte-bpe"
    FORMAT_VERSION = 1

    def __init__(
        self,
        merges: Sequence[tuple[int, int]],
        vocab_size: int,
        special_tokens: int,
        *,
        pattern: str = PRETOKENIZE_PATTERN,
    ) -> None:
        _check_layout(vocab_size, special_tokens)
        n_regular = vocab_size - special_tokens
        if n_regular < _N_BYTES:
            raise TokenizerError(f"byte-level BPE needs >= 256 regular IDs; got {n_regular}")
        if len(merges) > n_regular - _N_BYTES:
            raise TokenizerError(f"{len(merges)} merges exceed the {n_regular - _N_BYTES} available regular IDs")
        self._vocab_size = int(vocab_size)
        self._special_count = int(special_tokens)
        self._n_regular = n_regular
        self._merges: list[tuple[int, int]] = []
        self._ranks: dict[tuple[int, int], int] = {}
        self._vocab: list[bytes] = [bytes([i]) for i in range(_N_BYTES)]
        for rank, pair in enumerate(merges):
            a, b = int(pair[0]), int(pair[1])
            limit = _N_BYTES + rank
            if not (0 <= a < limit and 0 <= b < limit):
                raise TokenizerError(f"merge {rank} {pair} references an undefined token")
            if (a, b) in self._ranks:
                raise TokenizerError(f"duplicate merge {pair}")
            self._merges.append((a, b))
            self._ranks[(a, b)] = rank
            self._vocab.append(self._vocab[a] + self._vocab[b])
        self._special_names = special_token_names(special_tokens)
        self._special_base = n_regular
        self._special_to_id = {name: n_regular + i for i, name in enumerate(self._special_names)}
        self._pattern_src = pattern
        self._pat = re.compile(pattern)
        self._cache: dict[str, tuple[int, ...]] = {}
        self._split_cache: dict[frozenset[str], re.Pattern[str]] = {}
        self._fingerprint = content_hash(self._fingerprint_payload())

    # ------------------------------------------------------------ properties
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def special_tokens(self) -> int:
        """Number of special IDs reserved at the top of the vocabulary."""
        return self._special_count

    @property
    def n_regular(self) -> int:
        return self._n_regular

    @property
    def n_merges(self) -> int:
        return len(self._merges)

    @property
    def merges(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._merges)

    @property
    def special_names(self) -> tuple[str, ...]:
        return self._special_names

    @property
    def pad_id(self) -> int:
        return self._special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self._special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self._special_to_id["<|eos|>"]

    @property
    def fingerprint(self) -> str:
        """Content hash identifying this exact tokenizer version."""
        return self._fingerprint

    def special_id(self, name: str) -> int:
        try:
            return self._special_to_id[name]
        except KeyError:
            raise KeyError(f"unknown special token {name!r}") from None

    def is_special(self, token_id: int) -> bool:
        return self._special_base <= token_id < self._vocab_size

    def token_bytes(self, token_id: int) -> bytes:
        """Raw bytes of a regular token (specials return their UTF-8 name)."""
        if self.is_special(token_id):
            return self._special_names[token_id - self._special_base].encode("utf-8")
        if 0 <= token_id < len(self._vocab):
            return self._vocab[token_id]
        raise TokenizerError(f"token id {token_id} is not produced by this tokenizer")

    # --------------------------------------------------------------- encoding
    def pretokenize(self, text: str) -> list[str]:
        return pretokenize(text, self._pat)

    def _bpe(self, piece: str) -> tuple[int, ...]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = list(piece.encode("utf-8"))
        ranks = self._ranks
        while len(ids) >= 2:
            best = None
            for pair in zip(ids, ids[1:]):
                r = ranks.get(pair)
                if r is not None and (best is None or r < best):
                    best = r
            if best is None:
                break
            a, b = self._merges[best]
            new_id = _N_BYTES + best
            out: list[int] = []
            i, n = 0, len(ids)
            while i < n:
                if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        result = tuple(ids)
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[piece] = result
        return result

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode treating every character (including ``<|...|>`` strings) as plain text."""
        out: list[int] = []
        for piece in self._pat.findall(text):
            out.extend(self._bpe(piece))
        return out

    def _special_splitter(self, allowed: frozenset[str]) -> re.Pattern[str]:
        pat = self._split_cache.get(allowed)
        if pat is None:
            alts = "|".join(re.escape(s) for s in sorted(allowed, key=lambda s: (-len(s), s)))
            pat = re.compile(f"({alts})")
            self._split_cache[allowed] = pat
        return pat

    def encode(
        self,
        text: str,
        *,
        allow_special: bool | Iterable[str] = False,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode ``text`` to token IDs.

        ``allow_special``: ``False`` (default) encodes special-token strings as
        ordinary bytes — the injection-safe behaviour for any untrusted text.
        ``True`` parses every registered special token; an iterable of names
        parses only those.
        """
        ids: list[int] = [self.bos_id] if add_bos else []
        if allow_special is False or not text:
            ids.extend(self.encode_ordinary(text))
        else:
            if allow_special is True:
                allowed = frozenset(self._special_names)
            else:
                allowed = frozenset(allow_special)
                unknown = allowed - set(self._special_names)
                if unknown:
                    raise KeyError(f"unknown special tokens {sorted(unknown)}")
            if allowed:
                for i, part in enumerate(self._special_splitter(allowed).split(text)):
                    if i % 2 == 1:
                        ids.append(self._special_to_id[part])
                    elif part:
                        ids.extend(self.encode_ordinary(part))
            else:
                ids.extend(self.encode_ordinary(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    # --------------------------------------------------------------- decoding
    def decode_bytes(self, ids: Iterable[int], *, skip_special: bool = False) -> bytes:
        parts: list[bytes] = []
        for t in ids:
            t = int(t)
            if self.is_special(t):
                if not skip_special:
                    parts.append(self._special_names[t - self._special_base].encode("utf-8"))
            elif 0 <= t < len(self._vocab):
                parts.append(self._vocab[t])
            else:
                raise TokenizerError(f"token id {t} is not produced by this tokenizer")
        return b"".join(parts)

    def decode(self, ids: Iterable[int], *, skip_special: bool = False, errors: str = "replace") -> str:
        """Decode IDs to text. Valid encodings round-trip exactly; arbitrary ID
        sequences may split UTF-8 sequences, hence ``errors="replace"`` by default."""
        return self.decode_bytes(ids, skip_special=skip_special).decode("utf-8", errors=errors)

    # ---------------------------------------------------------- serialisation
    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "format_version": self.FORMAT_VERSION,
            "vocab_size": self._vocab_size,
            "special_tokens": self._special_count,
            "pattern": self._pattern_src,
            "merges": [list(m) for m in self._merges],
            "special_names": list(self._special_names),
        }

    def to_dict(self) -> dict[str, Any]:
        d = self._fingerprint_payload()
        d["fingerprint"] = self._fingerprint
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ByteBPETokenizer":
        if data.get("format") != cls.FORMAT or data.get("format_version") != cls.FORMAT_VERSION:
            raise TokenizerError("not an m63 byte-BPE tokenizer file (or unsupported version)")
        tok = cls(
            [tuple(m) for m in data["merges"]],  # type: ignore[misc]
            int(data["vocab_size"]),
            int(data["special_tokens"]),
            pattern=str(data["pattern"]),
        )
        if list(data.get("special_names", tok.special_names)) != list(tok.special_names):
            raise TokenizerError("special-token registry in file does not match this code's registry")
        expected = data.get("fingerprint")
        if expected is not None and expected != tok.fingerprint:
            raise TokenizerError("tokenizer fingerprint mismatch: file is corrupted or was edited")
        return tok

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __repr__(self) -> str:
        return (
            f"ByteBPETokenizer(vocab_size={self._vocab_size}, special_tokens={self._special_count}, "
            f"merges={len(self._merges)}, fingerprint={self._fingerprint[:19]}...)"
        )


# ----------------------------------------------------------------------- trainer
def train_bpe(
    texts: Iterable[str],
    vocab_size: int,
    special_tokens: int,
    *,
    min_frequency: int = 2,
    pattern: str = PRETOKENIZE_PATTERN,
) -> ByteBPETokenizer:
    """Train a byte-level BPE tokenizer (§5).

    Learns up to ``vocab_size - special_tokens - 256`` merges. Pairs are chosen
    by frequency with a deterministic tie-break (lowest ``(left, right)`` IDs),
    so identical inputs always produce identical tokenizers. Training stops
    early when no pair occurs at least ``min_frequency`` times.
    """
    _check_layout(vocab_size, special_tokens)
    n_target = vocab_size - special_tokens - _N_BYTES
    if n_target < 0:
        raise TokenizerError(f"byte-level BPE needs >= 256 regular IDs; got {vocab_size - special_tokens}")
    pat = re.compile(pattern)
    word_freq: dict[bytes, int] = defaultdict(int)
    for text in texts:
        for piece in pat.findall(text):
            word_freq[piece.encode("utf-8")] += 1

    words: list[list[int]] = [list(w) for w in word_freq]
    freqs: list[int] = list(word_freq.values())
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for i, w in enumerate(words):
        f = freqs[i]
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            pair_where[p].add(i)

    merges: list[tuple[int, int]] = []
    for rank in range(n_target):
        if not pair_counts:
            break
        pair, count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if count < min_frequency:
            break
        new_id = _N_BYTES + rank
        merges.append(pair)
        a, b = pair
        for i in sorted(pair_where.pop(pair, ())):
            w, f = words[i], freqs[i]
            for p in zip(w, w[1:]):
                c = pair_counts[p] - f
                if c > 0:
                    pair_counts[p] = c
                else:
                    pair_counts.pop(p, None)
                where = pair_where.get(p)
                if where is not None:
                    where.discard(i)
            out: list[int] = []
            j, n = 0, len(w)
            while j < n:
                if j + 1 < n and w[j] == a and w[j + 1] == b:
                    out.append(new_id)
                    j += 2
                else:
                    out.append(w[j])
                    j += 1
            words[i] = out
            for p in zip(out, out[1:]):
                pair_counts[p] += f
                pair_where[p].add(i)
        pair_counts.pop(pair, None)
    return ByteBPETokenizer(merges, vocab_size, special_tokens, pattern=pattern)


# ---------------------------------------------------------------------- fertility
@dataclass(frozen=True)
class FertilityStats:
    """Tokenizer fertility for one language sample (§5 "multilingual fertility checks")."""

    sample: str
    n_chars: int
    n_bytes: int
    n_words: int
    n_tokens: int
    tokens_per_char: float
    tokens_per_word: float
    bytes_per_token: float


def fertility_report(
    tokenizer: ByteBPETokenizer, samples: Mapping[str, str | Sequence[str]]
) -> dict[str, FertilityStats]:
    """Tokens per character and per whitespace-delimited word for each named sample.

    ``samples`` maps a label (usually a language code) to a text or list of texts.
    Scripts without whitespace word boundaries (CJK, Thai) naturally show a high
    tokens-per-word value; tokens-per-character is the comparable metric there.
    """
    report: dict[str, FertilityStats] = {}
    for label, value in samples.items():
        texts = [value] if isinstance(value, str) else list(value)
        n_chars = n_bytes = n_words = n_tokens = 0
        for t in texts:
            n_chars += len(t)
            n_bytes += len(t.encode("utf-8"))
            n_words += len(t.split())
            n_tokens += len(tokenizer.encode(t))
        report[label] = FertilityStats(
            sample=label,
            n_chars=n_chars,
            n_bytes=n_bytes,
            n_words=n_words,
            n_tokens=n_tokens,
            tokens_per_char=n_tokens / n_chars if n_chars else 0.0,
            tokens_per_word=n_tokens / n_words if n_words else 0.0,
            bytes_per_token=n_bytes / n_tokens if n_tokens else 0.0,
        )
    return report
