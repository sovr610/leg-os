"""Packing semantics: boundary masks, reset flags, position restarts, splits (§8, Appendix B)."""

from __future__ import annotations

import pytest
import torch

from m63_omni.common import Modality
from m63_omni.data.packing import (
    IGNORE_INDEX,
    PACKING_STRATEGIES,
    TokenizedDoc,
    pack_documents,
    single_document_batch,
)


def _doc(i: int, n: int) -> TokenizedDoc:
    # token values encode (doc, offset) so we can check provenance of every position
    return TokenizedDoc(f"d{i}", tuple(1000 * (i + 1) + k for k in range(n)))


def _reconstruct(batch) -> dict[str, list[int]]:
    """Reassemble every document from its packed segments (in chunk order)."""
    pieces: dict[str, list[tuple[int, list[int]]]] = {}
    for row in batch.segments:
        for si in row:
            toks = batch.input_ids[si.row, si.offset : si.offset + si.length].tolist()
            pieces.setdefault(si.doc_id, []).append((si.chunk_index, toks))
    return {d: [t for _, ts in sorted(p) for t in ts] for d, p in pieces.items()}


def test_packed_semantics_small_example():
    docs = [TokenizedDoc("a", (1, 2, 3)), TokenizedDoc("b", (4, 5)), TokenizedDoc("c", (6, 7, 8, 9))]
    b = pack_documents(docs, 10, pad_id=0, shuffle=False, strategy="sequential")
    assert b.shape == (1, 10)
    assert b.input_ids.tolist() == [[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    assert b.labels.tolist() == [[2, 3, -100, 5, -100, 7, 8, 9, -100, -100]]
    assert b.segment_ids.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 2, -1]]
    assert b.position_ids.tolist() == [[0, 1, 2, 0, 1, 0, 1, 2, 3, 0]]
    assert b.reset_mask.tolist() == [[True, False, False, True, False, True, False, False, False, False]]
    assert b.loss_mask.tolist() == (b.labels != -100).tolist()
    assert b.modality_ids.tolist() == [[int(Modality.TEXT)] * 10]
    assert b.doc_ids == [["a", "b", "c"]]
    for name, t in b.tensors().items():
        assert t.dtype == (torch.bool if name in ("reset_mask", "loss_mask") else torch.long), name
    b.validate()


def test_labels_never_cross_document_boundaries():
    docs = [_doc(i, n) for i, n in enumerate([5, 17, 1, 30, 9, 64, 2, 40, 63, 11])]
    b = pack_documents(docs, 64, seed=3)
    b.validate()
    seg, lab, ids = b.segment_ids, b.labels, b.input_ids
    for r in range(b.batch_size):
        for t in range(b.seq_len - 1):
            if lab[r, t] != IGNORE_INDEX:
                assert seg[r, t] == seg[r, t + 1] >= 0
                assert lab[r, t] == ids[r, t + 1]
                assert int(lab[r, t]) // 1000 == int(ids[r, t]) // 1000  # same document
    # every document start resets state and restarts positions
    assert torch.equal(b.reset_mask, (b.position_ids == 0) & (b.segment_ids >= 0))
    n_segments = sum(len(r) for r in b.doc_ids)
    assert int(b.reset_mask.sum()) == n_segments
    # targets per doc = len - 1 (no doc here exceeds T)
    assert b.n_targets == sum(len(d.tokens) - 1 for d in docs)
    # attention mask never crosses documents
    mask = b.document_mask()
    same = b.segment_ids[:, :, None] == b.segment_ids[:, None, :]
    assert not torch.any(mask & ~same)


@pytest.mark.parametrize("T", [64, 2048])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_boundary_lengths(T, delta):
    n = T + delta
    b = pack_documents([_doc(0, n)], T, pad_id=0)
    b.validate()
    assert _reconstruct(b) == {"d0": list(_doc(0, n).tokens)}
    if n <= T:
        assert b.shape == (1, T) and b.continuations == []
        assert b.n_targets == n - 1
        assert int(b.reset_mask.sum()) == 1
        assert (b.segment_ids == -1).sum() == T - n
    else:
        assert b.shape == (2, T)
        assert [(c.chunk_index, c.n_chunks, c.length, c.doc_token_start) for c in b.continuations] == [
            (0, 2, T, 0), (1, 2, 1, T),
        ]
        # the cross-row prediction is lost, everything else is supervised
        assert b.n_targets == (T - 1) + 0
        assert int(b.reset_mask.sum()) == 2
        assert b.doc_ids == [["d0"], ["d0"]]


def test_single_token_and_long_split_docs():
    T = 64
    docs = [_doc(0, 1), _doc(1, 3 * T + 5), _doc(2, 2 * T)]
    b = pack_documents(docs, T, seed=1)
    b.validate()
    assert _reconstruct(b) == {d.doc_id: list(d.tokens) for d in docs}
    cont = {(c.doc_id, c.chunk_index): c for c in b.continuations}
    assert sorted(cont) == [("d1", 0), ("d1", 1), ("d1", 2), ("d1", 3), ("d2", 0), ("d2", 1)]
    assert [cont[("d1", i)].length for i in range(4)] == [T, T, T, 5]
    single = next(si for row in b.segments for si in row if si.doc_id == "d0")
    assert single.length == 1 and b.labels[single.row, single.offset] == IGNORE_INDEX
    assert b.reset_mask[single.row, single.offset]


def test_deterministic_given_seed_and_strategies():
    docs = [_doc(i, 3 + (7 * i) % 50) for i in range(25)]
    a, b = pack_documents(docs, 64, seed=42), pack_documents(docs, 64, seed=42)
    for k, v in a.tensors().items():
        assert torch.equal(v, b.tensors()[k]), k
    assert a.doc_ids == b.doc_ids
    c = pack_documents(docs, 64, seed=43)
    assert c.doc_ids != a.doc_ids
    for strategy in PACKING_STRATEGIES:
        p = pack_documents(docs, 64, seed=0, strategy=strategy)
        p.validate()
        assert _reconstruct(p) == {d.doc_id: list(d.tokens) for d in docs}
    ffd = pack_documents(docs, 64, strategy="first_fit_decreasing")
    seq = pack_documents(docs, 64, strategy="sequential")
    assert ffd.batch_size <= seq.batch_size


def test_modalities_and_media_target_masking():
    mods = (0, 0, 2, 2, 2, 3, 3, 0)
    doc = TokenizedDoc("av", (1, 2, 9, 9, 9, 9, 9, 3), mods)
    b = pack_documents([doc], 8)
    assert b.modality_ids[0].tolist() == list(mods)
    assert b.labels[0].tolist() == [2, 9, 9, 9, 9, 9, 3, -100]
    masked = pack_documents([doc], 8, ignore_target_modalities=[Modality.VIDEO, Modality.AUDIO])
    masked.validate()
    assert masked.labels[0].tolist() == [2, -100, -100, -100, -100, -100, 3, -100]
    with pytest.raises(ValueError):
        TokenizedDoc("bad", (1, 2), (0,))


def test_empty_docs_recorded_and_raw_lists_accepted():
    b = pack_documents([[5, 6, 7], [], TokenizedDoc("x", (8,))], 4, shuffle=False, strategy="sequential")
    assert b.skipped_empty == ["doc-1"]
    assert b.doc_ids == [["doc-0", "x"]]
    with pytest.raises(ValueError):
        pack_documents([TokenizedDoc("x", (1,)), TokenizedDoc("x", (2,))], 4)
    assert pack_documents([], 8, min_rows=2).shape == (2, 8)


def test_single_document_batch():
    b = single_document_batch(torch.tensor([10, 11, 12, 13]))
    b.validate()
    assert b.shape == (1, 4)
    assert b.labels.tolist() == [[11, 12, 13, -100]]
    assert b.position_ids.tolist() == [[0, 1, 2, 3]]
    assert b.reset_mask.tolist() == [[True, False, False, False]]
    padded = single_document_batch([1, 2], seq_len=4, pad_id=7, modality_ids=[0, 1], doc_id="img")
    padded.validate()
    assert padded.input_ids.tolist() == [[1, 2, 7, 7]] and padded.segment_ids.tolist() == [[0, 0, -1, -1]]
    assert padded.modality_ids.tolist() == [[0, 1, 0, 0]] and padded.doc_ids == [["img"]]
    with pytest.raises(ValueError):
        single_document_batch([1, 2, 3], seq_len=2)
