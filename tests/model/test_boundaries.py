"""Appendix B boundary lengths: 1, 63, 64, 65, 2,047, 2,048, 2,049, 32K, and the maximum context."""

import pytest
import torch
import torch.nn.functional as F

from m63_omni.common import Trust
from m63_omni.config import tiny
from m63_omni.kernels.gdn import gdn_chunked, gdn_recurrent_reference
from m63_omni.model.m63 import ContextLengthError, M63Omni
from m63_omni.runtime.session import Session


def _gdn(T, seed=0, H=2, D=8):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(1, T, H, D, generator=g), dim=-1)
    k = F.normalize(torch.randn(1, T, H, D, generator=g), dim=-1)
    v = torch.randn(1, T, H, D, generator=g)
    gate = -torch.rand(1, T, H, generator=g) * 0.05
    beta = torch.rand(1, T, H, generator=g)
    reset = torch.zeros(1, T, dtype=torch.bool)
    reset[0, T // 2] = True
    return q, k, v, gate, beta, reset


@pytest.mark.parametrize("T", [1, 63, 64, 65, 2047, 2048, 2049])
def test_gdn_chunk_and_segment_boundaries(T):
    q, k, v, g, b, r = _gdn(T, seed=T)
    o1, s1 = gdn_recurrent_reference(q, k, v, g, b, reset=r)
    o2, s2 = gdn_chunked(q, k, v, g, b, chunk_size=64, reset=r)
    torch.testing.assert_close(o2, o1, atol=5e-5, rtol=1e-4)
    torch.testing.assert_close(s2, s1, atol=5e-5, rtol=1e-4)


@pytest.mark.slow
def test_gdn_32k_chunked_matches_recurrent():
    q, k, v, g, b, r = _gdn(32768, seed=1, H=1, D=8)
    o1, s1 = gdn_recurrent_reference(q, k, v, g, b, reset=r)
    o2, s2 = gdn_chunked(q, k, v, g, b, chunk_size=64, reset=r)
    torch.testing.assert_close(o2, o1, atol=2e-4, rtol=1e-3)
    torch.testing.assert_close(s2, s1, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("T", [63, 64, 65])
def test_model_prefill_decode_at_chunk_boundaries(T):
    torch.manual_seed(0)
    m = M63Omni(tiny(gdn={"train_chunk": 64}, multimodal={"enabled": False})).eval()
    ids = torch.randint(0, 400, (1, T + 2))
    with torch.no_grad():
        full = m(ids).logits
        out = m(ids[:, :T], use_cache=True)
        c, pieces = out.cache, [out.logits]
        for t in (T, T + 1):
            o = m.step(ids[:, t : t + 1], c)
            c = o.cache
            pieces.append(o.logits)
    torch.testing.assert_close(torch.cat(pieces, 1), full, atol=1e-5, rtol=1e-5)


@pytest.fixture(scope="module")
def seg2048():
    torch.manual_seed(0)
    return M63Omni(tiny(memory={"update_segment_tokens": 2048}, multimodal={"enabled": False},
                        model={"context_targets": [4096, 8192]})).eval()


@pytest.mark.parametrize("n", [2047, 2048, 2049])
def test_memory_update_boundaries_in_sessions(seg2048, n):
    s = Session(seg2048, "t", "s")
    h0 = s.memory_hash()
    s.ingest(torch.randint(0, 400, (n,)).tolist(), trust=Trust.USER)
    snaps = [e for e in s.log.replay() if e.kind == "memory_snapshot"]
    assert len(snaps) == n // 2048 and len(s._buffer_trust) == n % 2048
    assert (s.memory_hash() != h0) == (n >= 2048)
    if n == 2047:
        s.ingest([1])  # the 2,048th token completes the first segment
        assert len([e for e in s.log.replay() if e.kind == "memory_snapshot"]) == 1


@pytest.mark.parametrize("n", [2047, 2048, 2049])
def test_segmented_forward_boundaries(seg2048, n):
    ids = torch.randint(0, 400, (1, n + 1))
    with torch.no_grad():
        out = seg2048.forward_segmented(ids[:, :n])
        alt = ids.clone()
        alt[:, n - 1] = (alt[:, n - 1] + 1) % 400
        out2 = seg2048.forward_segmented(alt[:, :n])
    assert len(out.memory_info) == (n // 2048) * len(seg2048.anchor_indices)
    torch.testing.assert_close(out.logits[:, : n - 1], out2.logits[:, : n - 1], atol=1e-6, rtol=0)


def test_maximum_context_is_enforced_not_truncated():
    torch.manual_seed(0)
    cfg = tiny(multimodal={"enabled": False}, model={"context_targets": [96, 128]})
    m = M63Omni(cfg).eval()
    ids = torch.randint(0, 400, (1, 128))
    with torch.no_grad():
        full = m(ids).logits  # exactly the maximum supported context
        out = m(ids[:, :126], use_cache=True)
        c = out.cache
        for t in (126, 127):
            o = m.step(ids[:, t : t + 1], c)
            c = o.cache
        torch.testing.assert_close(o.logits[:, -1], full[:, -1], atol=1e-5, rtol=1e-5)
        with pytest.raises(ContextLengthError):
            m.step(ids[:, :1], c)  # position 129 > 128
        with pytest.raises(ContextLengthError):
            m(torch.randint(0, 400, (1, 129)))
