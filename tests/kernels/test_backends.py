"""Optional fast backends must match the naive references (§11: "optimized code cannot become
authoritative until it matches that reference")."""

import pytest
import torch
import torch.nn.functional as F

from m63_omni.kernels.attention import attention_flex, attention_reference, build_attention_mask, flex_available
from m63_omni.kernels.gdn import gdn_chunked
from m63_omni.kernels.gdn_fla import fla_available, gdn_fla, varlen_reference
from m63_omni.kernels.moe import IMPLEMENTATIONS, grouped_mm_available, moe_reference

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")  # eager FlexAttention warns that it is uncompiled


# ------------------------------------------------------------------ FlexAttention
@pytest.mark.skipif(not flex_available(), reason="FlexAttention unavailable")
@pytest.mark.parametrize("Tq,Tk,n_mem", [(13, 13, 0), (13, 13, 3), (1, 21, 4), (5, 37, 2)])
def test_flex_matches_reference_mask(Tq, Tk, n_mem):
    g = torch.Generator().manual_seed(Tq + Tk)
    B, Hq, Hkv, D = 2, 4, 2, 8
    q = torch.randn(B, Hq, Tq, D, generator=g)
    k = torch.randn(B, Hkv, n_mem + Tk, D, generator=g)
    v = torch.randn(B, Hkv, n_mem + Tk, D, generator=g)
    k_pos = torch.arange(Tk).expand(B, Tk)
    q_pos = k_pos[:, Tk - Tq :]
    k_seg = torch.stack([torch.zeros(Tk, dtype=torch.long), (torch.arange(Tk) >= Tk // 2).long()])
    q_seg = k_seg[:, Tk - Tq :]
    ref = attention_reference(q, k, v, build_attention_mask(q_pos, q_seg, k_pos, k_seg, n_mem))
    out = attention_flex(q, k, v, q_pos, q_seg, k_pos, k_seg, n_mem)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not flex_available(), reason="FlexAttention unavailable")
def test_model_with_flex_backend_matches_sdpa_including_decode(model, cfg, rand_ids):
    ids = rand_ids(cfg, 2, 21)
    seg = torch.tensor([[0] * 9 + [1] * 12, [0] * 21])
    with torch.no_grad():
        sdpa = model(ids, segment_ids=seg).logits
        model.set_kernel_mode(attention_backend="flex")
        flex = model(ids, segment_ids=seg).logits
        out = model(ids[:, :15], use_cache=True)
        steps = [out.logits]
        cache = out.cache
        for t in range(15, 21):
            o = model.step(ids[:, t : t + 1], cache)
            cache = o.cache
            steps.append(o.logits)
        model.set_kernel_mode(attention_backend="sdpa")
        full = model(ids).logits
    torch.testing.assert_close(flex, sdpa, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.cat(steps, 1), full, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------- grouped-mm MoE
@pytest.mark.skipif(not grouped_mm_available(), reason="torch._grouped_mm unavailable")
@pytest.mark.parametrize("skew", [False, True])
def test_grouped_mm_moe_matches_reference(skew):
    g = torch.Generator().manual_seed(0)
    N, d, h, E, k = 29, 8, 12, 6, 2
    x = torch.randn(N, d, generator=g, requires_grad=True)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(N)])
    if skew:
        idx[:, 0], idx[:, 1] = 0, 1  # experts 2..5 receive nothing (empty groups)
    w = torch.rand(N, k, generator=g)
    ws = [(torch.randn(E, d, h, generator=g) * 0.3).requires_grad_(), (torch.randn(E, d, h, generator=g) * 0.3).requires_grad_(),
          (torch.randn(E, h, d, generator=g) * 0.3).requires_grad_()]
    out = IMPLEMENTATIONS["grouped_mm"](x, idx, w, *ws)
    grads = torch.autograd.grad(out.pow(2).sum(), [x, *ws])
    x2 = x.detach().clone().requires_grad_()
    ws2 = [t.detach().clone().requires_grad_() for t in ws]
    ref = moe_reference(x2, idx, w, *ws2)
    ref_grads = torch.autograd.grad(ref.pow(2).sum(), [x2, *ws2])
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not grouped_mm_available(), reason="torch._grouped_mm unavailable")
def test_model_with_grouped_mm_moe(model, cfg, rand_ids):
    ids = rand_ids(cfg, 2, 17)
    with torch.no_grad():
        ref = model(ids).logits
        model.set_kernel_mode(moe_impl="grouped_mm")
        out = model(ids).logits
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_unknown_backends_are_rejected(model):
    with pytest.raises(ValueError):
        model.set_kernel_mode(moe_impl="magic")
    with pytest.raises(ValueError):
        model.set_kernel_mode(attention_backend="magic")
    with pytest.raises(ValueError):
        model.set_kernel_mode(gdn_backend="magic")


# ------------------------------------------------------------- fla GDN adapter
def _gdn_inputs(B=3, T=31, H=2, D=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, D, generator=g), dim=-1)
    k = F.normalize(torch.randn(B, T, H, D, generator=g), dim=-1)
    v = torch.randn(B, T, H, D, generator=g)
    gate = -torch.rand(B, T, H, generator=g) * 0.6
    beta = torch.rand(B, T, H, generator=g)
    reset = torch.rand(B, T, generator=g) < 0.12
    reset[1, 0] = True  # a row that ignores its initial state
    reset[B - 1, 5] = reset[B - 1, 6] = True  # consecutive resets (a length-1 document)
    return q, k, v, gate, beta, reset, torch.randn(B, H, D, D, generator=g)


@pytest.mark.parametrize("seed", range(3))
def test_fla_varlen_adapter_matches_chunked_reference(seed):
    q, k, v, g, b, r, s0 = _gdn_inputs(seed=seed)
    o_ref, s_ref = gdn_chunked(q, k, v, g, b, chunk_size=8, initial_state=s0, reset=r)
    o, s = gdn_fla(q, k, v, g, b, initial_state=s0, reset=r, kernel=varlen_reference)
    torch.testing.assert_close(o, o_ref, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(s, s_ref, atol=2e-5, rtol=1e-4)


def test_fla_adapter_gradients_flow():
    q, k, v, g, b, r, s0 = _gdn_inputs(B=2, T=17)
    ts = [t.clone().requires_grad_(True) for t in (q, k, v, g, b)]
    o, s = gdn_fla(*ts, initial_state=s0, reset=r, kernel=varlen_reference)
    (o.sum() + s.sum()).backward()
    ref = [t.clone().requires_grad_(True) for t in (q, k, v, g, b)]
    o2, s2 = gdn_chunked(*ref, chunk_size=4, initial_state=s0, reset=r)
    (o2.sum() + s2.sum()).backward()
    for a, c in zip(ts, ref):
        torch.testing.assert_close(a.grad, c.grad, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not fla_available(), reason="needs CUDA and flash-linear-attention")
def test_real_fla_kernel_matches_reference():  # pragma: no cover - GPU only
    q, k, v, g, b, r, s0 = [t.cuda() for t in _gdn_inputs(T=200)]
    o_ref, s_ref = gdn_chunked(q, k, v, g, b, chunk_size=64, initial_state=s0, reset=r)
    o, s = gdn_fla(q, k, v, g, b, initial_state=s0, reset=r)
    torch.testing.assert_close(o, o_ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(s, s_ref, atol=2e-3, rtol=2e-3)
