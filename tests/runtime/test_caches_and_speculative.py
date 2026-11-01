"""Paged KV, prefix cache, MTP speculative decoding with state rollback (§14, Appendix B)."""

import pytest
import torch

from m63_omni.config import ConfigError, tiny
from m63_omni.model.m63 import M63Omni
from m63_omni.runtime import speculative
from m63_omni.runtime.paged_kv import PageAllocatorError, PagedKVStore
from m63_omni.runtime.prefix_cache import PrefixCache, PrefixKey


def test_paged_kv_matches_contiguous_and_copy_on_write():
    st = PagedKVStore(num_pages=16, block_tokens=4, kv_heads=2, head_dim=3)
    st.create("a", "t")
    k, v = torch.randn(2, 10, 3), torch.randn(2, 10, 3)
    st.append("a", k[:, :6], v[:, :6])
    st.append("a", k[:, 6:], v[:, 6:])
    gk, gv = st.gather("a", "t")
    torch.testing.assert_close(gk, k)
    torch.testing.assert_close(gv, v)
    used = st.pages_in_use
    st.fork("a", "b", "t")
    assert st.pages_in_use == used  # shared, not copied
    st.append("b", torch.ones(2, 1, 3), torch.ones(2, 1, 3))
    assert st.copies == 1  # tail page copied on first write
    torch.testing.assert_close(st.gather("a", "t")[0], k)  # parent untouched
    assert st.gather("b", "t")[0].shape[1] == 11
    st.truncate("b", 3)
    assert st.gather("b", "t")[0].shape[1] == 3
    with pytest.raises(PageAllocatorError):
        st.truncate("b", 4)  # cannot roll forward
    st.free_seq("b")
    st.free_seq("a")
    assert st.pages_in_use == 0


def test_paged_kv_cross_tenant_and_exhaustion():
    st = PagedKVStore(num_pages=2, block_tokens=2, kv_heads=1, head_dim=1)
    st.create("a", "t1")
    with pytest.raises(PermissionError):
        st.fork("a", "b", "t2")
    with pytest.raises(PageAllocatorError):
        st.append("a", torch.ones(1, 5, 1), torch.ones(1, 5, 1))


def test_prefix_cache_keys_never_cross_tenants_or_assets():
    clock = [0.0]
    pc = PrefixCache(capacity=2, ttl_seconds=10, clock=lambda: clock[0])
    k = PrefixKey("t1", "m1", "tok1", "policy", "sha256:asset", 256)
    pc.put(k, "S")
    assert pc.get(k) == "S"
    for variant in (PrefixKey("t2", "m1", "tok1", "policy", "sha256:asset", 256),
                    PrefixKey("t1", "m2", "tok1", "policy", "sha256:asset", 256),
                    PrefixKey("t1", "m1", "tok2", "policy", "sha256:asset", 256),
                    PrefixKey("t1", "m1", "tok1", "policy", "sha256:other", 256),
                    PrefixKey("t1", "m1", "tok1", "policy", "sha256:asset", 255)):
        assert pc.get(variant) is None
    clock[0] = 11.0
    assert pc.get(k) is None  # expired
    pc.put(PrefixKey("t1", "a", "b", "c", "d", 1), 1)
    pc.put(PrefixKey("t1", "a", "b", "c", "e", 1), 2)
    pc.put(PrefixKey("t2", "a", "b", "c", "f", 1), 3)
    assert len(pc) == 2 and pc.evict_tenant("t1") == 1


@pytest.fixture(params=["separate", "shared"])
def deep(request):
    """A tiny model with a trained depth-3 MTP (separate or shared weights per depth)."""
    torch.manual_seed(0)
    c = tiny(multimodal={"enabled": False}, mtp={"layers": 3, "draft_tokens": 3, "share_weights": request.param == "shared"})
    return c, M63Omni(c).eval()


@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("draft", [1, 2, 3])
def test_greedy_speculative_equals_target_only(deep, rand_ids, draft, use_cache):
    cfg, model = deep
    p = rand_ids(cfg, 1, 14)
    a, ca, _ = speculative.generate(model, p, 11, draft_tokens=0)
    b, cb, st = speculative.generate(model, p, 11, draft_tokens=draft, use_mtp_cache=use_cache)
    assert a.tolist() == b.tolist()
    assert ca.next_pos == cb.next_pos == 14 + 11
    nxt = torch.tensor([[a[-1]]])
    with torch.no_grad():
        torch.testing.assert_close(model.step(nxt, cb).logits, model.step(nxt, ca).logits, atol=1e-5, rtol=1e-5)


def test_rejected_drafts_roll_back_every_state(deep, rand_ids, monkeypatch):
    cfg, model = deep
    real = speculative._draft_cached

    def bad_draft(m, mc, h, nxt, k, t, g):
        d, q, n = real(m, mc, h, nxt, k, t, g)
        return [(x + 1) % 50 for x in d], q, n  # always wrong

    monkeypatch.setattr(speculative, "_draft_cached", bad_draft)
    p = rand_ids(cfg, 1, 9)
    ref, cref, _ = speculative.generate(model, p, 8, draft_tokens=0)
    out, cout, st = speculative.generate(model, p, 8, draft_tokens=2)
    assert out.tolist() == ref.tolist() and st.rollbacks > 0
    nxt = torch.tensor([[out[-1]]])
    with torch.no_grad():
        torch.testing.assert_close(model.step(nxt, cout).logits, model.step(nxt, cref).logits, atol=1e-5, rtol=1e-5)


def test_stochastic_speculative_is_seeded(deep, rand_ids):
    cfg, model = deep
    p = rand_ids(cfg, 1, 8)
    runs = [speculative.generate(model, p, 6, draft_tokens=2, temperature=1.0, generator=torch.Generator().manual_seed(3))[0]
            for _ in range(2)]
    assert runs[0].tolist() == runs[1].tolist()
    assert all(0 <= t < cfg.model.vocab_size for t in runs[0].tolist())


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_mtp_cache_drafts_equal_full_recompute(deep, rand_ids, temperature):
    cfg, model = deep
    p = rand_ids(cfg, 1, 30)
    runs = [speculative.generate(model, p, 20, draft_tokens=3, temperature=temperature, use_mtp_cache=c,
                                 generator=torch.Generator().manual_seed(7)) for c in (False, True)]
    (a, _, sa), (b, _, sb) = runs
    assert a.tolist() == b.tolist() and sa.accept_history == sb.accept_history
    assert sb.mtp_positions < sa.mtp_positions / 3


def test_drafting_beyond_the_trained_mtp_depth_is_rejected(model, cfg, rand_ids):
    assert model.mtp.depth == 1 and cfg.mtp.draft_tokens == 1
    with pytest.raises(ValueError, match="trained MTP depth"):
        speculative.generate(model, rand_ids(cfg, 1, 6), 4, draft_tokens=2)
    with pytest.raises(ConfigError, match="draft_tokens"):
        tiny(mtp={"draft_tokens": 2})


def test_each_draft_uses_its_own_depth(rand_ids):
    """Depth j drafts token j: perturbing depth 2 changes the second draft, never the first."""
    torch.manual_seed(0)
    c = tiny(multimodal={"enabled": False}, mtp={"layers": 2, "draft_tokens": 2})
    m = M63Omni(c).eval()
    h = torch.randn(9, c.model.hidden_size)
    ids = rand_ids(c, 1, 10)[0].tolist()
    with torch.no_grad():
        a, _ = speculative._draft(m, h, ids, 2, 0.0, None)
        b, _, _ = speculative._draft_cached(m, speculative.MTPCache(), h, ids, 2, 0.0, None)
        m.mtp.at(2).ffn.down.weight.mul_(-5.0)
        c2, _ = speculative._draft(m, h, ids, 2, 0.0, None)
    assert a == b and a[0] == c2[0]
    assert len(m.mtp.depths) == 2 and m.mtp.at(1) is not m.mtp.at(2)


def test_multi_depth_mtp_loss_respects_document_boundaries():
    torch.manual_seed(0)
    c = tiny(multimodal={"enabled": False}, mtp={"layers": 2, "draft_tokens": 2})
    m = M63Omni(c)
    ids = torch.randint(0, 400, (1, 12))
    labels = torch.cat([ids[:, 1:], torch.full((1, 1), -100)], 1)
    labels[0, 5] = -100  # document boundary after position 5
    out = m(ids, labels=labels, compute_mtp=True)
    # depth 1 needs labels t and t+1 valid: 12 positions - 2 invalid labels - 2 positions preceding them = 8
    assert out.mtp_tokens == 8 and torch.isfinite(out.mtp_loss)
    out.mtp_loss.backward()
    assert all(p.grad is not None for p in m.mtp.at(2).parameters())
