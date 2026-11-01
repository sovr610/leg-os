"""Sequence packing with document-boundary masks and recurrent-state reset flags (blueprint §8, App. B).

§8: "Packed sequences carry document-boundary masks and recurrent-state reset
flags." Appendix B requires causality tests over "document resets, packed
sequences" and boundary lengths around the chunk size. Every tensor in a
:class:`PackedBatch` has shape ``(B, T)``:

* ``input_ids``   long — tokens, ``pad_id`` in padding;
* ``labels``      long — ``input_ids[t+1]`` when ``t+1`` is in the SAME packed
  segment, else ``-100`` (so labels never cross a document boundary); padding
  ``-100``;
* ``segment_ids`` long — segment index within the row starting at 0, padding ``-1``;
* ``position_ids`` long — restart at 0 at every segment start;
* ``reset_mask``  bool — True at the first token of every segment (GDN/recurrent
  state must be reset there);
* ``loss_mask``   bool — ``labels != -100``;
* ``modality_ids`` long — per-position :class:`~m63_omni.common.Modality`
  (default TEXT).

Documents longer than ``T`` are split into explicit continuation chunks of at
most ``T`` tokens, recorded in :attr:`PackedBatch.continuations` — never
silently dropped. Each chunk is its own segment (positions restart, state
resets) because rows are processed independently. Empty documents are
recorded in :attr:`PackedBatch.skipped_empty`. Packing is deterministic given
``seed``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from ..common import Modality

__all__ = [
    "IGNORE_INDEX",
    "PAD_SEGMENT",
    "TokenizedDoc",
    "SegmentInfo",
    "PackedBatch",
    "pack_documents",
    "single_document_batch",
    "PACKING_STRATEGIES",
]

IGNORE_INDEX = -100
PAD_SEGMENT = -1
PACKING_STRATEGIES = ("first_fit", "first_fit_decreasing", "sequential")


@dataclass(frozen=True)
class TokenizedDoc:
    """A tokenised document ready for packing."""

    doc_id: str
    tokens: tuple[int, ...]
    modality_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(int(t) for t in self.tokens))
        if self.modality_ids is not None:
            mods = tuple(int(m) for m in self.modality_ids)
            if len(mods) != len(self.tokens):
                raise ValueError(f"doc {self.doc_id}: modality_ids length != tokens length")
            object.__setattr__(self, "modality_ids", mods)


@dataclass(frozen=True)
class SegmentInfo:
    """Where one (piece of a) document landed."""

    doc_id: str
    row: int
    segment: int  # index within the row (== segment_ids value)
    offset: int  # start column in the row
    length: int
    chunk_index: int  # 0 for unsplit documents
    n_chunks: int  # 1 for unsplit documents
    doc_token_start: int  # offset of this chunk within the original document

    @property
    def is_continuation(self) -> bool:
        return self.n_chunks > 1


@dataclass
class PackedBatch:
    """Packed rows; see module docstring for exact tensor semantics."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    segment_ids: torch.Tensor
    position_ids: torch.Tensor
    reset_mask: torch.Tensor
    loss_mask: torch.Tensor
    modality_ids: torch.Tensor
    doc_ids: list[list[str]]
    segments: list[list[SegmentInfo]] = field(default_factory=list)
    continuations: list[SegmentInfo] = field(default_factory=list)
    skipped_empty: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.input_ids.shape)  # type: ignore[return-value]

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def n_tokens(self) -> int:
        """Non-padding positions."""
        return int((self.segment_ids >= 0).sum())

    @property
    def n_targets(self) -> int:
        return int(self.loss_mask.sum())

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids,
            "labels": self.labels,
            "segment_ids": self.segment_ids,
            "position_ids": self.position_ids,
            "reset_mask": self.reset_mask,
            "loss_mask": self.loss_mask,
            "modality_ids": self.modality_ids,
        }

    def to(self, device: torch.device | str) -> "PackedBatch":
        moved = {k: v.to(device) for k, v in self.tensors().items()}
        return PackedBatch(**moved, doc_ids=self.doc_ids, segments=self.segments,
                           continuations=self.continuations, skipped_empty=self.skipped_empty)

    def document_mask(self) -> torch.Tensor:
        """(B, T, T) bool: query ``i`` may attend key ``j`` iff ``j <= i`` and same segment."""
        seg = self.segment_ids
        same = (seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] >= 0)
        T = seg.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool, device=seg.device).tril()
        return same & causal

    def validate(self) -> None:
        """Assert the packing invariants (used by tests and as a loader self-check)."""
        B, T = self.shape
        for name, t in self.tensors().items():
            if tuple(t.shape) != (B, T):
                raise AssertionError(f"{name} has shape {tuple(t.shape)}, expected {(B, T)}")
        for name in ("input_ids", "labels", "segment_ids", "position_ids", "modality_ids"):
            if self.tensors()[name].dtype != torch.long:
                raise AssertionError(f"{name} must be torch.long")
        if self.reset_mask.dtype != torch.bool or self.loss_mask.dtype != torch.bool:
            raise AssertionError("reset_mask and loss_mask must be torch.bool")
        if not torch.equal(self.loss_mask, self.labels != IGNORE_INDEX):
            raise AssertionError("loss_mask != (labels != -100)")
        seg, pos = self.segment_ids, self.position_ids
        same_next = (seg[:, 1:] == seg[:, :-1]) & (seg[:, :-1] >= 0)
        lab = self.labels[:, :-1]
        if not torch.all(torch.where(same_next, True, lab == IGNORE_INDEX)):
            raise AssertionError("a label crosses a segment boundary")
        if not torch.all(lab[same_next & (lab != IGNORE_INDEX)] == self.input_ids[:, 1:][same_next & (lab != IGNORE_INDEX)]):
            raise AssertionError("labels are not the next input token")
        if not torch.all(self.labels[:, -1] == IGNORE_INDEX):
            raise AssertionError("last column cannot have a label")
        starts = torch.zeros_like(self.reset_mask)
        starts[:, 0] = seg[:, 0] >= 0
        starts[:, 1:] = (seg[:, 1:] != seg[:, :-1]) & (seg[:, 1:] >= 0)
        if not torch.equal(starts, self.reset_mask):
            raise AssertionError("reset_mask must mark exactly the segment starts")
        if not torch.all(pos[self.reset_mask] == 0):
            raise AssertionError("positions must restart at 0 at every segment start")
        if not torch.all(torch.where(same_next, pos[:, 1:] == pos[:, :-1] + 1, True)):
            raise AssertionError("positions must increase by 1 within a segment")
        if not torch.all(self.labels[seg < 0] == IGNORE_INDEX):
            raise AssertionError("padding must have label -100")
        for r, ids in enumerate(self.doc_ids):
            n = int(seg[r].max()) + 1 if bool((seg[r] >= 0).any()) else 0
            if len(ids) != n:
                raise AssertionError(f"row {r}: {len(ids)} doc ids for {n} segments")


@dataclass
class _Piece:
    doc_id: str
    tokens: tuple[int, ...]
    modality: tuple[int, ...]
    chunk_index: int
    n_chunks: int
    doc_token_start: int
    order: int


def _coerce(docs: Iterable[TokenizedDoc | Sequence[int]]) -> list[TokenizedDoc]:
    out = []
    for i, d in enumerate(docs):
        if isinstance(d, TokenizedDoc):
            out.append(d)
        else:
            out.append(TokenizedDoc(f"doc-{i}", tuple(d)))
    return out


def _build(
    rows: list[list[_Piece]],
    seq_len: int,
    pad_id: int,
    ignore_target_modalities: frozenset[int],
) -> tuple[dict[str, torch.Tensor], list[list[str]], list[list[SegmentInfo]]]:
    B = len(rows)
    input_ids = torch.full((B, seq_len), pad_id, dtype=torch.long)
    labels = torch.full((B, seq_len), IGNORE_INDEX, dtype=torch.long)
    segment_ids = torch.full((B, seq_len), PAD_SEGMENT, dtype=torch.long)
    position_ids = torch.zeros((B, seq_len), dtype=torch.long)
    reset_mask = torch.zeros((B, seq_len), dtype=torch.bool)
    modality_ids = torch.full((B, seq_len), int(Modality.TEXT), dtype=torch.long)
    doc_ids: list[list[str]] = []
    infos: list[list[SegmentInfo]] = []
    for r, pieces in enumerate(rows):
        col = 0
        row_ids: list[str] = []
        row_infos: list[SegmentInfo] = []
        for s, p in enumerate(pieces):
            L = len(p.tokens)
            toks = torch.tensor(p.tokens, dtype=torch.long)
            input_ids[r, col : col + L] = toks
            segment_ids[r, col : col + L] = s
            position_ids[r, col : col + L] = torch.arange(L)
            reset_mask[r, col] = True
            mods = torch.tensor(p.modality, dtype=torch.long)
            modality_ids[r, col : col + L] = mods
            if L > 1:
                tgt = toks[1:].clone()
                if ignore_target_modalities:
                    drop = torch.tensor([m in ignore_target_modalities for m in p.modality[1:]], dtype=torch.bool)
                    tgt[drop] = IGNORE_INDEX
                labels[r, col : col + L - 1] = tgt
            row_ids.append(p.doc_id)
            row_infos.append(SegmentInfo(p.doc_id, r, s, col, L, p.chunk_index, p.n_chunks, p.doc_token_start))
            col += L
        doc_ids.append(row_ids)
        infos.append(row_infos)
    tensors = {
        "input_ids": input_ids,
        "labels": labels,
        "segment_ids": segment_ids,
        "position_ids": position_ids,
        "reset_mask": reset_mask,
        "loss_mask": labels != IGNORE_INDEX,
        "modality_ids": modality_ids,
    }
    return tensors, doc_ids, infos


def pack_documents(
    docs: Iterable[TokenizedDoc | Sequence[int]],
    seq_len: int,
    *,
    pad_id: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    strategy: str = "first_fit",
    min_rows: int = 0,
    ignore_target_modalities: Iterable[int] = (),
) -> PackedBatch:
    """Pack variable-length documents into ``(B, seq_len)`` rows.

    1. Each document longer than ``seq_len`` is split into ``ceil(len/seq_len)``
       continuation chunks (all recorded).
    2. Pieces are shuffled with ``random.Random(seed)`` when ``shuffle``.
    3. Pieces are placed by ``strategy``: ``first_fit`` (first row with room),
       ``first_fit_decreasing`` (longest first, stable), or ``sequential``
       (next-fit; preserves order).

    ``ignore_target_modalities`` (e.g. ``{IMAGE, VIDEO, AUDIO}``) additionally
    sets ``labels = -100`` where the *target* is an encoder placeholder
    position, since encoder outputs are inputs, not next-token targets (§9).
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if strategy not in PACKING_STRATEGIES:
        raise ValueError(f"strategy must be one of {PACKING_STRATEGIES}")
    pieces: list[_Piece] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for d in _coerce(docs):
        if d.doc_id in seen:
            raise ValueError(f"duplicate doc_id {d.doc_id!r}")
        seen.add(d.doc_id)
        n = len(d.tokens)
        if n == 0:
            skipped.append(d.doc_id)
            continue
        mods = d.modality_ids if d.modality_ids is not None else (int(Modality.TEXT),) * n
        n_chunks = -(-n // seq_len)
        for c in range(n_chunks):
            a, b = c * seq_len, min(n, (c + 1) * seq_len)
            pieces.append(_Piece(d.doc_id, d.tokens[a:b], mods[a:b], c, n_chunks, a, len(pieces)))
    if shuffle:
        random.Random(seed).shuffle(pieces)
    if strategy == "first_fit_decreasing":
        pieces.sort(key=lambda p: -len(p.tokens))  # stable: ties keep (shuffled) order

    rows: list[list[_Piece]] = []
    free: list[int] = []
    for p in pieces:
        L = len(p.tokens)
        target = -1
        if strategy == "sequential":
            if rows and free[-1] >= L:
                target = len(rows) - 1
        else:
            for i, f in enumerate(free):
                if f >= L:
                    target = i
                    break
        if target < 0:
            rows.append([])
            free.append(seq_len)
            target = len(rows) - 1
        rows[target].append(p)
        free[target] -= L
    while len(rows) < min_rows:
        rows.append([])
        free.append(seq_len)

    tensors, doc_ids, infos = _build(rows, seq_len, pad_id, frozenset(int(m) for m in ignore_target_modalities))
    continuations = sorted(
        (si for row in infos for si in row if si.n_chunks > 1),
        key=lambda si: (si.doc_id, si.chunk_index),
    )
    return PackedBatch(**tensors, doc_ids=doc_ids, segments=infos, continuations=continuations, skipped_empty=skipped)


def single_document_batch(
    tokens: Sequence[int] | torch.Tensor,
    *,
    modality_ids: Sequence[int] | None = None,
    doc_id: str = "doc-0",
    seq_len: int | None = None,
    pad_id: int = 0,
) -> PackedBatch:
    """A ``(1, T)`` PackedBatch for one document (``T = len(tokens)`` unless ``seq_len`` pads it).

    Raises if the document does not fit ``seq_len`` — use :func:`pack_documents`
    for splitting.
    """
    if isinstance(tokens, torch.Tensor):
        if tokens.dim() != 1:
            raise ValueError("tokens must be 1-D")
        tokens = tokens.tolist()
    toks = tuple(int(t) for t in tokens)
    if not toks:
        raise ValueError("empty document")
    T = len(toks) if seq_len is None else int(seq_len)
    if len(toks) > T:
        raise ValueError(f"document of {len(toks)} tokens does not fit seq_len={T}")
    doc = TokenizedDoc(doc_id, toks, tuple(modality_ids) if modality_ids is not None else None)
    mods = doc.modality_ids if doc.modality_ids is not None else (int(Modality.TEXT),) * len(toks)
    piece = _Piece(doc_id, toks, mods, 0, 1, 0, 0)
    tensors, doc_ids, infos = _build([[piece]], T, pad_id, frozenset())
    return PackedBatch(**tensors, doc_ids=doc_ids, segments=infos)
