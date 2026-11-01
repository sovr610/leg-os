"""Kimi Delta Attention: chunked prefill vs recurrent decode, GDN special case, gradients (tournament mixer)."""

import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from m63_omni.config import GDNConfig
from m63_omni.kernels.gdn import gdn_chunked
from m63_omni.kernels.kda import kda_chunked, kda_recurrent_reference
from m63_omni.model.kda import KimiDeltaAttention

D64 = torch.float64


def _inputs(B=2, T=37, H=2, Dk=8, Dv=6, seed=0, dtype=torch.float32, decay=2.0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, Dk, generator=g, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(B, T, H, Dk, generator=g, dtype=dtype), dim=-1)
    v = torch.randn(B, T, H, Dv, generator=g, dtype=dtype)
    gate = -torch.rand(B, T, H, Dk, generator=g, dtype=dtype) * decay
    beta = torch.rand(B, T, H, generator=g, dtype=dtype)
    s0 = torch.randn(B, H, Dk, Dv, generator=g, dtype=dtype)
    return q, k, v, gate, beta, s0


@pytest.mark.parametrize("T", [1, 63, 64, 65, 127, 129])
@pytest.mark.parametrize("chunk", [16, 64])
def test_chunked_matches_recurrent(T, chunk):
    q, k, v, g, b, s0 = _inputs(T=T)
    reset = torch.zeros(2, T, dtype=torch.bool)
    if T > 20:
        reset[0, 17] = reset[1, T // 2] = reset[1, T // 2 + 1] = True  # in-chunk resets, back to back
    o1, s1 = kda_recurrent_reference(q, k, v, g, b, initial_state=s0, reset=reset)
    o2, s2 = kda_chunked(q, k, v, g, b, chunk_size=chunk, initial_state=s0, reset=reset)
    torch.testing.assert_close(o2, o1, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(s2, s1, atol=2e-5, rtol=2e-5)


def test_strong_decay_is_exact():
    """Per-channel decays down to exp(-40) per token: pairwise decays are masked before exp, never divided."""
    q, k, v, g, b, s0 = _inputs(T=70, dtype=D64, decay=40.0)
    o1, s1 = kda_recurrent_reference(q, k, v, g, b, initial_state=s0)
    o2, s2 = kda_chunked(q, k, v, g, b, chunk_size=32, initial_state=s0)
    assert torch.isfinite(o2).all()
    torch.testing.assert_close(o2, o1, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(s2, s1, atol=1e-12, rtol=1e-10)


def test_uniform_channel_decay_is_exactly_gdn():
    q, k, v, _, b, s0 = _inputs(T=45, dtype=D64)
    gs = -torch.rand(2, 45, 2, dtype=D64)
    reset = torch.zeros(2, 45, dtype=torch.bool)
    reset[1, 20] = True
    o1, s1 = kda_chunked(q, k, v, gs.unsqueeze(-1).expand_as(q), b, chunk_size=16, initial_state=s0, reset=reset)
    o2, s2 = gdn_chunked(q, k, v, gs, b, chunk_size=16, initial_state=s0, reset=reset)
    torch.testing.assert_close(o1, o2, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(s1, s2, atol=1e-12, rtol=1e-12)


def test_state_carries_across_calls():
    q, k, v, g, b, _ = _inputs(T=50)
    o_full, s_full = kda_chunked(q, k, v, g, b, chunk_size=16)
    o_a, s_a = kda_chunked(q[:, :23], k[:, :23], v[:, :23], g[:, :23], b[:, :23], chunk_size=16)
    o_b, s_b = kda_chunked(q[:, 23:], k[:, 23:], v[:, 23:], g[:, 23:], b[:, 23:], chunk_size=16, initial_state=s_a)
    torch.testing.assert_close(torch.cat([o_a, o_b], 1), o_full, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(s_b, s_full, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("chunk", [2, 3, 8])
def test_kda_chunked_gradcheck(chunk):
    q, k, v, g, b, s0 = (x[:1].clone().requires_grad_() for x in _inputs(T=7, H=1, Dk=3, Dv=3, dtype=D64, decay=0.5))
    reset = torch.zeros(1, 7, dtype=torch.bool)
    reset[0, 4] = True
    assert gradcheck(lambda *a: kda_chunked(*a[:5], chunk_size=chunk, initial_state=a[5], reset=reset), (q, k, v, g, b, s0))


def test_kda_layer_gradcheck_and_modes_agree():
    torch.manual_seed(0)
    layer = KimiDeltaAttention(8, GDNConfig(heads=1, key_dim=4, value_dim=4, inner_width=4, conv_kernel=2, train_chunk=3)).double()
    x = torch.randn(1, 7, 8, dtype=D64, requires_grad=True)
    reset = torch.zeros(1, 7, dtype=torch.bool)
    reset[0, 3] = True
    assert gradcheck(lambda t: layer(t, reset=reset, mode="chunked")[0], (x,))
    oc, sc = layer(x, reset=reset, mode="chunked")
    orr, sr = layer(x, reset=reset, mode="recurrent")
    torch.testing.assert_close(oc, orr)
    torch.testing.assert_close(sc.recurrent, sr.recurrent)
    g = layer.decay(x)
    assert g.shape == (1, 7, 1, 4) and (g < 0).all()
    assert not torch.allclose(g[..., 0], g[..., 1])  # channel-wise, not per head
