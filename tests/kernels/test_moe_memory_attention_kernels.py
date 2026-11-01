"""MoE dispatch, memory-gradient, and attention kernels vs their naive references (Appendix B)."""

import pytest
import torch

from m63_omni.kernels.attention import attention, attention_reference, build_attention_mask
from m63_omni.kernels.memory import assoc_grad_autograd, assoc_grad_closed_form, clip_by_norm
from m63_omni.kernels.moe import expert_counts, moe_grouped, moe_padded_bmm, moe_reference


def _moe_inputs(N=37, d=8, h=12, E=6, k=2, skew=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, d, generator=g)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(N)])
    if skew:
        idx[:, 0] = 0  # every token hits expert 0
        idx[:, 1] = torch.where(idx[:, 1] == 0, torch.ones_like(idx[:, 1]), idx[:, 1])
    w = torch.rand(N, k, generator=g)
    w = w / w.sum(-1, keepdim=True)
    ws = [torch.randn(E, d, h, generator=g) * 0.3, torch.randn(E, d, h, generator=g) * 0.3, torch.randn(E, h, d, generator=g) * 0.3]
    return x, idx, w, ws


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("impl", [moe_grouped, moe_padded_bmm])
def test_moe_implementations_match_reference(impl, skew):
    x, idx, w, ws = _moe_inputs(skew=skew)
    torch.testing.assert_close(impl(x, idx, w, *ws), moe_reference(x, idx, w, *ws), atol=1e-5, rtol=1e-5)


def test_moe_gradients_match_and_are_dropless():
    x, idx, w, ws = _moe_inputs(seed=3)
    grads = []
    for fn in (moe_reference, moe_grouped, moe_padded_bmm):
        xs = x.clone().requires_grad_(True)
        wt = [t.clone().requires_grad_(True) for t in ws]
        fn(xs, idx, w, *wt).pow(2).sum().backward()
        grads.append([xs.grad] + [t.grad for t in wt])
    for other in grads[1:]:
        for a, b in zip(grads[0], other):
            torch.testing.assert_close(b, a, atol=1e-4, rtol=1e-4)
    assert int(expert_counts(idx, 6).sum()) == idx.numel()


@pytest.mark.parametrize("masked", [False, True])
def test_memory_closed_form_gradient_matches_autograd(masked):
    g = torch.Generator().manual_seed(1)
    W1, W2 = torch.randn(3, 10, 6, generator=g) * 0.3, torch.randn(3, 6, 10, generator=g) * 0.3
    k, v = torch.randn(3, 9, 6, generator=g), torch.randn(3, 9, 6, generator=g)
    valid = (torch.rand(3, 9, generator=g) > 0.3) if masked else None
    l1, a1, b1 = assoc_grad_closed_form(W1, W2, k, v, valid)
    l2, a2, b2 = assoc_grad_autograd(W1, W2, k, v, valid)
    torch.testing.assert_close(l1, l2)
    torch.testing.assert_close(a1, a2, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(b1, b2, atol=1e-6, rtol=1e-5)


def test_clip_by_norm_bounds_each_session():
    x = torch.randn(4, 5, 5) * torch.tensor([0.01, 1, 10, 100]).view(4, 1, 1)
    y = clip_by_norm(x, 1.0)
    assert (y.flatten(1).norm(dim=-1) <= 1.0 + 1e-5).all()
    torch.testing.assert_close(y[0], x[0])  # small tensors untouched


def test_attention_fast_path_matches_reference_with_masks_and_memory():
    g = torch.Generator().manual_seed(0)
    B, Hq, Hkv, T, D, M = 2, 4, 2, 11, 8, 3
    q = torch.randn(B, Hq, T, D, generator=g)
    k = torch.randn(B, Hkv, M + T, D, generator=g)
    v = torch.randn(B, Hkv, M + T, D, generator=g)
    pos = torch.arange(T).expand(B, T)
    seg = torch.tensor([[0] * 5 + [1] * 6, [0] * 11])
    mask = build_attention_mask(pos, seg, pos, seg, n_mem=M)
    torch.testing.assert_close(attention(q, k, v, mask), attention_reference(q, k, v, mask), atol=1e-5, rtol=1e-5)


def test_mask_semantics():
    pos = torch.arange(4).view(1, 4)
    seg = torch.tensor([[0, 0, 1, 1]])
    m = build_attention_mask(pos, seg, pos, seg, n_mem=2)[0, 0]
    assert m[:, :2].all()  # memory visible to all queries
    body = m[:, 2:]
    assert body.tolist() == [[True, False, False, False], [True, True, False, False],
                             [False, False, True, False], [False, False, True, True]]
