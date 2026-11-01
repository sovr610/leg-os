"""Optional second-pass loop: schedule, per-pass caches, training, sessions, serving arithmetic."""

import pytest
import torch

from m63_omni.common import Trust
from m63_omni.config import ConfigError, reference, tiny
from m63_omni.model.m63 import M63Omni
from m63_omni.model.manifest import build_manifest
from m63_omni.runtime import capacity as cap
from m63_omni.runtime import speculative
from m63_omni.runtime.prefix_cache import PrefixKey
from m63_omni.runtime.session import IncompatibleSnapshotError, Session

LOOP = {"enabled": True, "first_layer": 5, "last_layer": 8}


def _cfg(memory=True, **loop):
    """12 layers (3 hybrid groups); the middle group loops; memory anchors at 4 and 12 sit outside it."""
    return tiny(multimodal={"enabled": False}, model={"layers": 12}, memory={"enabled": memory, "anchors": [4, 12]},
                loop={**LOOP, **loop})


@pytest.fixture
def looped():
    torch.manual_seed(0)
    c = _cfg()
    return c, M63Omni(c).eval()


def test_schedule_and_validation():
    c = _cfg()
    assert c.loop_schedule(1) == list(range(12))
    assert c.loop_schedule(2) == list(range(8)) + [4, 5, 6, 7] + list(range(8, 12))
    assert c.execution_kinds(2).count("gated_gqa") == 4 and c.expected_train_passes() == 1.5
    with pytest.raises(ConfigError, match="whole hybrid groups"):
        _cfg(first_layer=6)
    with pytest.raises(ConfigError, match="outside the looped layers"):
        tiny(multimodal={"enabled": False}, model={"layers": 12}, memory={"anchors": [8, 12]}, loop=LOOP)
    with pytest.raises(ConfigError, match="both 1 pass and max_passes"):
        _cfg(train_pass_probabilities=[0.0, 1.0])
    with pytest.raises(ConfigError, match="default_passes must be 1"):
        tiny(loop={"default_passes": 2})
    with pytest.raises(ConfigError, match="loop passes"):
        tiny().loop_schedule(2)  # the loop is off
    assert reference().replace(loop={"enabled": True}, memory={"enabled": False}).loop_schedule(2)[36:40] == [12, 13, 14, 15]


@pytest.mark.parametrize("passes", [1, 2])
def test_prefill_and_decode_match_the_full_forward_at_every_pass_count(looped, rand_ids, passes):
    cfg, model = looped
    ids = rand_ids(cfg, 2, 23)
    with torch.no_grad():
        full = model(ids, loop_passes=passes).logits
        out = model(ids[:, :10], use_cache=True, loop_passes=passes)
        cache, pieces = out.cache, [out.logits]
        for t in range(10, 23):
            o = model.step(ids[:, t : t + 1], cache)
            cache = o.cache
            pieces.append(o.logits)
    torch.testing.assert_close(torch.cat(pieces, 1), full, atol=1e-5, rtol=1e-5)
    assert cache.loop_passes == passes and len(cache.layers) == len(cfg.loop_schedule(passes))


def test_one_pass_is_the_unlooped_model_and_two_passes_differ(looped, rand_ids):
    cfg, model = looped
    plain = M63Omni(cfg.replace(loop={"enabled": False})).eval()
    plain.load_state_dict(model.state_dict())
    ids = rand_ids(cfg, 1, 17)
    with torch.no_grad():
        one, two, ref = model(ids, loop_passes=1), model(ids, loop_passes=2), plain(ids)
    torch.testing.assert_close(one.logits, ref.logits, atol=0, rtol=0)
    assert not torch.allclose(one.logits, two.logits)
    k = cfg.moe.top_k
    for i, s in enumerate(two.router_stats):  # stats stay per layer; looped layers count both passes
        assert int(s.counts.sum()) == (2 if 4 <= i < 8 else 1) * 17 * k
    model.update_router_biases(two.router_stats)


def test_a_cache_fixes_the_pass_count(looped, rand_ids):
    cfg, model = looped
    ids = rand_ids(cfg, 1, 8)
    with torch.no_grad():
        cache = model.prefill(ids, loop_passes=1).cache
        with pytest.raises(ValueError, match="re-prefill"):
            model(ids[:, :1], cache=cache, use_cache=True, loop_passes=2)
        two = model.prefill(ids, loop_passes=2).cache
    assert two.kv_bytes() == pytest.approx(cache.kv_bytes() * 4 / 3)  # 3 attention executions -> 4
    assert two.clone().loop_passes == 2


def test_training_samples_pass_counts_reproducibly_and_resumes_exactly(tmp_path):
    from m63_omni.train.checkpoint import CheckpointManager
    from m63_omni.train.trainer import Trainer, synthetic_batch

    cfg = _cfg(memory=False)

    def trainer(seed, **kw):
        torch.manual_seed(seed)
        return Trainer(M63Omni(cfg), lambda c: synthetic_batch(cfg, c, 2, 32), total_positions=10**6, **kw)

    a = trainer(1)
    hist = a.fit(6)
    assert {h["loop_passes"] for h in hist} == {1, 2}
    assert [h["loop_passes"] for h in hist] == [a.loop_passes(s) for s in range(6)]
    b = trainer(1, checkpoints=CheckpointManager(tmp_path))
    b.fit(3)
    path = b.save()[0]
    c = trainer(7)
    c.resume(path)
    c.fit(3)
    assert all(torch.equal(x, y) for x, y in zip(a.model.state_dict().values(), c.model.state_dict().values()))


def test_segmented_memory_training_runs_with_the_loop(looped, rand_ids):
    cfg, model = looped
    ids = rand_ids(cfg, 1, 48)
    labels = torch.cat([ids[:, 1:], torch.full((1, 1), -100)], 1)
    out = model.forward_segmented(ids, labels, loop_passes=2)
    assert out.loop_passes == 2 and torch.isfinite(out.lm_loss) and out.memory_loss is not None


def test_session_think_harder_is_logged_replayed_and_forked(looped, rand_ids):
    cfg, model = looped
    s = Session(model, "t", "s")
    first = s.ingest(rand_ids(cfg, 1, 12)[0].tolist(), trust=Trust.USER)
    s.generate(3)
    assert s.loop_passes == 1 and s.set_loop_passes(1) is None
    ev = s.think_harder()
    assert ev.payload["loop_passes"] == 2 and s.loop_passes == 2 and s.cache.loop_passes == 2
    s.ingest(rand_ids(cfg, 1, 5, seed=3)[0].tolist(), trust=Trust.USER)
    after = s.next_logits().clone()
    fresh = Session(model, "t", "s2", loop_passes=2)  # the same transcript prefilled at 2 passes from the start
    for e in s.log.replay(s.branch_id):
        if e.kind == "ingest":
            fresh.ingest(e.payload["tokens"], trust=Trust.USER)
        elif e.kind == "note" and e.payload.get("role") == "assistant":
            fresh.feed(e.payload["tokens"])
    torch.testing.assert_close(after, fresh.next_logits(), atol=1e-5, rtol=1e-5)
    s.replay()
    torch.testing.assert_close(s.next_logits(), after, atol=1e-5, rtol=1e-5)
    assert s.loop_passes == 2
    early = s.fork(first.event_id, "before the switch")
    assert early.loop_passes == 1 and early.cache.loop_passes == 1
    snap = s.snapshot_memory()
    with pytest.raises(IncompatibleSnapshotError, match="loop pass"):
        early.load_memory_snapshot(snap)
    with pytest.raises(ConfigError):
        s.set_loop_passes(3)


def test_speculative_decoding_at_two_passes(rand_ids):
    torch.manual_seed(0)
    cfg = _cfg(memory=False, max_passes=2)
    cfg = cfg.replace(mtp={"layers": 2, "draft_tokens": 2})
    model = M63Omni(cfg).eval()
    p = rand_ids(cfg, 1, 12)
    a, ca, _ = speculative.generate(model, p, 8, draft_tokens=0, loop_passes=2)
    b, cb, _ = speculative.generate(model, p, 8, draft_tokens=2, loop_passes=2)
    assert a.tolist() == b.tolist() and ca.loop_passes == cb.loop_passes == 2


def test_serving_and_training_arithmetic():
    c = reference().replace(loop={"enabled": True}, memory={"enabled": False})  # middle 24 of 48 layers
    m = build_manifest(c)
    assert cap.kv_bytes_per_token(c, loop_passes=2) == 1.5 * cap.kv_bytes_per_token(c) == 72 * 1024
    assert cap.gdn_state_bytes(c, loop_passes=2) == 1.5 * cap.gdn_state_bytes(c)
    extra = m.loop_extra_active(c)
    assert extra == pytest.approx(sum(m.layer_active[12:36])) and 2.4e9 < extra < 2.6e9
    assert m.active_with_passes(c, 2, include_head=False) == pytest.approx(m.active_excl_head + extra)
    base = cap.training_compute(reference().replace(memory={"enabled": False}))
    looped = cap.training_compute(c, manifest=m)
    assert looped.backbone == pytest.approx(base.backbone + 6 * 0.5 * extra * 4.75e12, rel=1e-6)
    assert 1.15 < looped.total / base.total < 1.25  # 50/50 training mix: +0.5 pass over half the layers
    f1, f2 = cap.serving_footprint(c, 131072), cap.serving_footprint(c, 131072, loop_passes=2)
    assert f2.kv == 1.5 * f1.kv and f2.mtp_kv == f1.mtp_kv
    from m63_omni.model.card import model_card_markdown
    assert "Second-pass loop: layers 13-36" in model_card_markdown(c)
    k1, k2 = PrefixKey("t", "m", "tok", "p", "h", 8), PrefixKey("t", "m", "tok", "p", "h", 8, loop_passes=2)
    assert k1.digest() != k2.digest()
