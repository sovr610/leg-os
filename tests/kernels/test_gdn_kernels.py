"""GDN equivalence: chunked prefill vs recurrent decode (Appendix B "GDN equivalence")."""

import pytest
import torch
import torch.nn.functional as F

from m63_omni.kernels.gdn import causal_depthwise_conv, gdn_chunked, gdn_recurrent_reference


def _inputs(B, T, H, D, seed=0, resets=0.0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, D, generator=g), dim=-1)
    k = F.normalize(torch.randn(B, T, H, D, generator=g), dim=-1)
    v = torch.randn(B, T, H, D, generator=g)
    gate = -torch.rand(B, T, H, generator=g) * 0.7
    beta = torch.rand(B, T, H, generator=g)
    reset = torch.rand(B, T, generator=g) < resets
    return q, k, v, gate, beta, reset


@pytest.mark.parametrize("T", [1, 63, 64, 65, 127, 129])
def test_chunk_boundaries_match_recurrent(T):
    q, k, v, g, b, r = _inputs(2, T, 2, 8, seed=T, resets=0.05)
    s0 = torch.randn(2, 2, 8, 8)
    o1, s1 = gdn_recurrent_reference(q, k, v, g, b, initial_state=s0, reset=r)
    o2, s2 = gdn_chunked(q, k, v, g, b, chunk_size=64, initial_state=s0, reset=r)
    torch.testing.assert_close(o2, o1, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(s2, s1, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("chunk", [1, 3, 7, 8, 16])
def test_odd_and_even_chunk_sizes(chunk):
    q, k, v, g, b, r = _inputs(1, 37, 3, 4, seed=chunk, resets=0.1)
    o1, s1 = gdn_recurrent_reference(q, k, v, g, b, reset=r)
    o2, s2 = gdn_chunked(q, k, v, g, b, chunk_size=chunk, reset=r)
    torch.testing.assert_close(o2, o1, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(s2, s1, atol=2e-5, rtol=1e-4)


def test_gradients_match():
    q, k, v, g, b, r = _inputs(1, 21, 2, 4, seed=3, resets=0.1)
    grads = []
    for fn in (gdn_recurrent_reference, lambda *a, **kw: gdn_chunked(*a, chunk_size=8, **kw)):
        ts = [t.clone().requires_grad_(True) for t in (q, k, v, g, b)]
        o, s = fn(*ts, reset=r)
        (o.pow(2).sum() + s.sum()).backward()
        grads.append([t.grad for t in ts])
    for a, c in zip(*grads):
        assert torch.isfinite(c).all()
        torch.testing.assert_close(c, a, atol=1e-4, rtol=1e-3)


def test_split_state_continuation():
    q, k, v, g, b, r = _inputs(2, 50, 2, 4, seed=5, resets=0.05)
    o, s = gdn_chunked(q, k, v, g, b, chunk_size=8, reset=r)
    oa, sa = gdn_chunked(q[:, :23], k[:, :23], v[:, :23], g[:, :23], b[:, :23], chunk_size=8, reset=r[:, :23])
    ob, sb = gdn_recurrent_reference(q[:, 23:], k[:, 23:], v[:, 23:], g[:, 23:], b[:, 23:], initial_state=sa, reset=r[:, 23:])
    torch.testing.assert_close(torch.cat([oa, ob], 1), o, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(sb, s, atol=2e-5, rtol=1e-4)


def test_reset_isolates_documents():
    q, k, v, g, b, _ = _inputs(1, 30, 2, 4, seed=7)
    r = torch.zeros(1, 30, dtype=torch.bool)
    r[0, 12] = True
    o_packed, _ = gdn_chunked(q, k, v, g, b, chunk_size=8, reset=r, initial_state=torch.randn(1, 2, 4, 4))
    o_alone, _ = gdn_chunked(q[:, 12:], k[:, 12:], v[:, 12:], g[:, 12:], b[:, 12:], chunk_size=8)
    torch.testing.assert_close(o_packed[:, 12:], o_alone, atol=2e-5, rtol=1e-4)


def test_bf16_inputs_within_documented_tolerance():
    q, k, v, g, b, r = _inputs(1, 40, 2, 8, seed=9, resets=0.05)
    o32, _ = gdn_recurrent_reference(q, k, v, g, b, reset=r)
    o16, _ = gdn_chunked(q.bfloat16(), k.bfloat16(), v.bfloat16(), g, b, chunk_size=16, reset=r)
    assert o16.dtype == torch.bfloat16
    torch.testing.assert_close(o16.float(), o32, atol=5e-2, rtol=5e-2)


def test_conv_split_and_document_isolation():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 20, 6, generator=g)
    w = torch.randn(6, 4, generator=g)
    r = torch.zeros(2, 20, dtype=torch.bool)
    r[:, 9] = True
    y, st = causal_depthwise_conv(x, w, reset=r)
    y1, s1 = causal_depthwise_conv(x[:, :7], w, reset=r[:, :7])
    y2, s2 = causal_depthwise_conv(x[:, 7:], w, conv_state=s1, reset=r[:, 7:])
    torch.testing.assert_close(torch.cat([y1, y2], 1), y)
    torch.testing.assert_close(s2, st)
    alone, _ = causal_depthwise_conv(x[:, 9:], w)
    torch.testing.assert_close(y[:, 9:], alone)
