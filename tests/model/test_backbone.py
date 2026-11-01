"""Backbone correctness: causality, document isolation, decode equivalence (Appendix B)."""

import pytest
import torch

from m63_omni.model.cache import HybridCache


def test_future_suffix_replacement_does_not_move_prefix(model, cfg, rand_ids):
    ids = rand_ids(cfg, 2, 40)
    alt = ids.clone()
    alt[:, 23:] = rand_ids(cfg, 2, 17, seed=9)
    with torch.no_grad():
        a = model(ids).logits
        b = model(alt).logits
    torch.testing.assert_close(a[:, :23], b[:, :23], atol=1e-6, rtol=0)
    assert not torch.allclose(a[:, 23:], b[:, 23:])


def test_packed_documents_are_isolated(model, cfg, rand_ids):
    ids = rand_ids(cfg, 1, 30)
    seg = torch.tensor([[0] * 13 + [1] * 17])
    pos = torch.tensor([list(range(13)) + list(range(17))])
    with torch.no_grad():
        packed = model(ids, segment_ids=seg, position_ids=pos).logits
        alone = model(ids[:, 13:]).logits
    torch.testing.assert_close(packed[:, 13:], alone, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("memory_enabled", [True, False])
def test_chunked_prefill_matches_recurrent_decode(model, cfg, rand_ids, memory_enabled):
    ids = rand_ids(cfg, 2, 33)
    with torch.no_grad():
        full = model(ids, memory_enabled=memory_enabled).logits
        out = model(ids[:, :17], use_cache=True, memory_enabled=memory_enabled)
        cache, pieces = out.cache, [out.logits]
        for t in range(17, 33):
            o = model.step(ids[:, t : t + 1], cache, memory_enabled=memory_enabled)
            cache = o.cache
            pieces.append(o.logits)
    torch.testing.assert_close(torch.cat(pieces, 1), full, atol=1e-5, rtol=1e-5)


def test_irregular_prefill_chunks(model, cfg, rand_ids):
    ids = rand_ids(cfg, 1, 25)
    with torch.no_grad():
        full = model(ids).logits
        cache, pieces, t = None, [], 0
        for n in (1, 7, 8, 9):
            o = model(ids[:, t : t + n], cache=cache, use_cache=True)
            cache, t = o.cache, t + n
            pieces.append(o.logits)
    torch.testing.assert_close(torch.cat(pieces, 1), full, atol=1e-5, rtol=1e-5)
    assert cache.next_pos == 25


def test_recurrent_mode_for_whole_sequence_matches_chunked(model, cfg, rand_ids):
    ids = rand_ids(cfg, 1, 19)
    with torch.no_grad():
        a = model(ids, gdn_mode="chunked").logits
        b = model(ids, gdn_mode="recurrent").logits
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_reference_kernels_match_fast_kernels(model, cfg, rand_ids):
    ids = rand_ids(cfg, 2, 21)
    with torch.no_grad():
        fast = model(ids).logits
        model.set_kernel_mode(moe_impl="reference", reference_attention=True)
        ref = model(ids).logits
        model.set_kernel_mode(moe_impl="padded", reference_attention=False)
        padded = model(ids).logits
    torch.testing.assert_close(ref, fast, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(padded, fast, atol=1e-5, rtol=1e-5)


def test_cache_state_sizes(model, cfg, rand_ids):
    with torch.no_grad():
        cache = model(rand_ids(cfg, 1, 10), use_cache=True).cache
    assert isinstance(cache, HybridCache) and cache.next_pos == 10
    a = cfg.attention
    assert cache.kv_bytes() == 2 * 10 * a.kv_heads * a.head_dim * 4 * 2  # 2 attention layers, fp32
    clone = cache.clone()
    clone.layers[0].recurrent.add_(1.0)
    assert not torch.equal(clone.layers[0].recurrent, cache.layers[0].recurrent)


def test_loss_and_gradients_finite(cfg, rand_ids):
    from m63_omni.model.m63 import M63Omni

    torch.manual_seed(0)
    m = M63Omni(cfg)
    ids = rand_ids(cfg, 2, 24)
    labels = torch.cat([ids[:, 1:], torch.full((2, 1), -100)], 1)
    out = m(ids, labels=labels, compute_mtp=True)
    (out.lm_loss + out.mtp_loss + out.router_aux_loss()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.blocks[0].moe.w_gate.grad.abs().sum() > 0


# --------------------------------------------- KDA in the hybrid pattern (tournament alternative)
@pytest.fixture
def kda_model():
    from m63_omni.config import tiny
    from m63_omni.model.m63 import M63Omni

    torch.manual_seed(0)
    cfg = tiny(multimodal={"enabled": False}, model={"hybrid_pattern": ["kda", "gdn", "kda", "gated_gqa"]})
    return cfg, M63Omni(cfg).eval()


def test_kda_backbone_prefill_decode_and_isolation(kda_model, rand_ids):
    from m63_omni.model.kda import KimiDeltaAttention

    cfg, model = kda_model
    assert sum(isinstance(b.mixer, KimiDeltaAttention) for b in model.blocks) == 4
    ids = rand_ids(cfg, 2, 29)
    with torch.no_grad():
        full = model(ids).logits
        out = model(ids[:, :13], use_cache=True)
        cache, pieces = out.cache, [out.logits]
        for t in range(13, 29):
            o = model.step(ids[:, t : t + 1], cache)
            cache = o.cache
            pieces.append(o.logits)
        torch.testing.assert_close(torch.cat(pieces, 1), full, atol=1e-5, rtol=1e-5)
        seg = torch.tensor([[0] * 11 + [1] * 18])
        pos = torch.tensor([list(range(11)) + list(range(18))])
        packed = model(ids[:1], segment_ids=seg, position_ids=pos).logits
        alone = model(ids[:1, 11:]).logits
    torch.testing.assert_close(packed[:, 11:], alone, atol=1e-5, rtol=1e-5)
    assert cache.recurrent_bytes() > 0
