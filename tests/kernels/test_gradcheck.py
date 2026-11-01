"""Float64 finite-difference gradient checks (Appendix B: "every optimized kernel has a naive reference").

Kernels compute in max(FP32, input dtype), so double inputs exercise the real code paths.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from m63_omni.config import AttentionConfig, GDNConfig
from m63_omni.kernels.attention import attention_reference, build_attention_mask
from m63_omni.kernels.gdn import causal_depthwise_conv, gdn_chunked, gdn_recurrent_reference
from m63_omni.kernels.memory import assoc_grad_closed_form
from m63_omni.kernels.moe import IMPLEMENTATIONS
from m63_omni.model.attention import GatedGQA
from m63_omni.model.cache import resolve_stream
from m63_omni.model.gdn import GatedDeltaNet

D64 = torch.float64


def _gdn_inputs(T=7, H=1, D=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(1, T, H, D, generator=g, dtype=D64), dim=-1).requires_grad_()
    k = F.normalize(torch.randn(1, T, H, D, generator=g, dtype=D64), dim=-1).requires_grad_()
    v = torch.randn(1, T, H, D, generator=g, dtype=D64, requires_grad=True)
    gate = (-torch.rand(1, T, H, generator=g, dtype=D64) * 0.5 - 0.05).requires_grad_()
    beta = (torch.rand(1, T, H, generator=g, dtype=D64) * 0.8 + 0.1).requires_grad_()
    s0 = torch.randn(1, H, D, D, generator=g, dtype=D64, requires_grad=True)
    reset = torch.zeros(1, T, dtype=torch.bool)
    reset[0, 4] = True
    return q, k, v, gate, beta, s0, reset


@pytest.mark.parametrize("chunk", [2, 3, 8])
def test_gdn_chunked_gradcheck(chunk):
    q, k, v, g, b, s0, reset = _gdn_inputs()
    assert gradcheck(lambda *a: gdn_chunked(*a[:5], chunk_size=chunk, initial_state=a[5], reset=reset), (q, k, v, g, b, s0))


def test_gdn_recurrent_gradcheck():
    q, k, v, g, b, s0, reset = _gdn_inputs(T=5, seed=1)
    assert gradcheck(lambda *a: gdn_recurrent_reference(*a[:5], initial_state=a[5], reset=reset), (q, k, v, g, b, s0))


def test_causal_conv_gradcheck():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 6, 3, generator=g, dtype=D64, requires_grad=True)
    w = torch.randn(3, 4, generator=g, dtype=D64, requires_grad=True)
    st = torch.randn(2, 3, 3, generator=g, dtype=D64, requires_grad=True)
    reset = torch.tensor([[False, False, True, False, False, False], [False] * 6])
    assert gradcheck(lambda x, w, st: causal_depthwise_conv(x, w, conv_state=st, reset=reset), (x, w, st))


@pytest.mark.parametrize("impl", sorted(IMPLEMENTATIONS))
def test_moe_kernels_gradcheck(impl):
    g = torch.Generator().manual_seed(3)
    N, d, h, E, k = 6, 3, 4, 3, 2
    x = torch.randn(N, d, generator=g, dtype=D64, requires_grad=True)
    idx = torch.tensor([[0, 1], [1, 2], [0, 2], [0, 1], [2, 0], [1, 0]])
    w = torch.rand(N, k, generator=g, dtype=D64, requires_grad=True)
    ws = [torch.randn(E, d, h, generator=g, dtype=D64, requires_grad=True) * 1 for _ in range(2)]
    ws.append(torch.randn(E, h, d, generator=g, dtype=D64, requires_grad=True))
    ws = [t.detach().requires_grad_() for t in ws]
    try:
        IMPLEMENTATIONS[impl](x, idx, w, *ws)
    except RuntimeError as e:  # e.g. grouped_mm without float64 support on this platform
        pytest.skip(f"{impl} does not support float64 here: {e}")
    assert gradcheck(lambda x, w, a, b, c: IMPLEMENTATIONS[impl](x, idx, w, a, b, c), (x, w, *ws))


def test_memory_closed_form_is_differentiable_for_outer_training():
    g = torch.Generator().manual_seed(4)
    W1 = torch.randn(1, 4, 3, generator=g, dtype=D64, requires_grad=True)
    W2 = torch.randn(1, 3, 4, generator=g, dtype=D64, requires_grad=True)
    k = torch.randn(1, 5, 3, generator=g, dtype=D64, requires_grad=True)
    v = torch.randn(1, 5, 3, generator=g, dtype=D64, requires_grad=True)
    assert gradcheck(assoc_grad_closed_form, (W1, W2, k, v))


def test_attention_reference_gradcheck():
    g = torch.Generator().manual_seed(5)
    q = torch.randn(1, 2, 4, 3, generator=g, dtype=D64, requires_grad=True)
    k = torch.randn(1, 1, 6, 3, generator=g, dtype=D64, requires_grad=True)
    v = torch.randn(1, 1, 6, 3, generator=g, dtype=D64, requires_grad=True)
    pos = torch.arange(4).view(1, 4)
    seg = torch.tensor([[0, 0, 1, 1]])
    mask = build_attention_mask(pos, seg, pos, seg, n_mem=2)
    assert gradcheck(lambda q, k, v: attention_reference(q, k, v, mask), (q, k, v))


def test_gdn_layer_gradcheck():
    torch.manual_seed(6)
    layer = GatedDeltaNet(8, GDNConfig(heads=2, key_dim=3, value_dim=3, inner_width=6, train_chunk=2)).double()
    x = torch.randn(1, 5, 8, dtype=D64, requires_grad=True)
    reset = torch.tensor([[True, False, False, True, False]])
    assert gradcheck(lambda x: layer(x, reset=reset)[0], (x,))


def test_gated_gqa_layer_gradcheck():
    torch.manual_seed(7)
    layer = GatedGQA(8, AttentionConfig(query_heads=2, kv_heads=1, head_dim=4, rotary_dims=2, rope_theta=100.0)).double()
    layer.backend = "reference"
    x = torch.randn(1, 5, 8, dtype=D64, requires_grad=True)
    mem = torch.randn(1, 2, 8, dtype=D64, requires_grad=True)
    idx = resolve_stream(1, 5, x.device, segment_ids=torch.tensor([[0, 0, 0, 1, 1]]))
    assert gradcheck(lambda x, mem: layer(x, idx, extra_kv=mem)[0], (x, mem))
