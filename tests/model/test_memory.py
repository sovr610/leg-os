"""Neural memory: read timing, bounded updates, freezing, segmented training (§6, Appendix B)."""

import pytest
import torch

from m63_omni.model.memory import MemoryFrozenError, MemoryState


def test_read_depends_only_on_completed_state(model):
    anchor = model.memory[str(model.anchor_indices[0])]
    st = anchor.init_state(2)
    a = anchor.read(st)
    assert a.shape == (2, model.cfg.memory.read_tokens_per_anchor, model.cfg.model.hidden_size)
    torch.testing.assert_close(anchor.read(st), a)  # no hidden input -> no current-segment leakage


def test_update_is_bounded_and_changes_state(model):
    cfg = model.cfg.memory
    anchor = model.memory[str(model.anchor_indices[0])]
    st = anchor.init_state(2)
    h = torch.randn(2, cfg.update_segment_tokens, model.cfg.model.hidden_size)
    new, info = anchor.update(st, h)
    assert new.updates == 1
    assert not torch.equal(new.W1, st.W1)
    assert (info["eta"] <= cfg.max_eta).all() and (info["theta"] <= cfg.max_theta).all() and (info["alpha"] <= cfg.max_alpha).all()
    assert (new.S1.flatten(1).norm(dim=-1) <= cfg.grad_clip + 1e-5).all()


def test_repeated_updates_memorise_a_segment(model):
    anchor = model.memory[str(model.anchor_indices[0])]
    with torch.no_grad():
        anchor.gate.bias.copy_(torch.tensor([0.0, 3.0, -6.0]))  # large inner step, little forgetting
    st = anchor.init_state(1)
    h = torch.randn(1, 16, model.cfg.model.hidden_size)
    before = float(anchor.retrieval_loss(st, h).detach())
    for _ in range(20):
        st, _ = anchor.update(st, h)
    assert float(anchor.retrieval_loss(st, h).detach()) < before


def test_frozen_memory_rejects_updates(model):
    anchor = model.memory[str(model.anchor_indices[0])]
    st = anchor.init_state(1)
    frozen = MemoryState(**{**st.__dict__, "frozen": True})
    with pytest.raises(MemoryFrozenError):
        anchor.update(frozen, torch.randn(1, 4, model.cfg.model.hidden_size))
    anchor.read(frozen)  # reads are still allowed


def test_memory_off_path_differs_but_works(model, cfg, rand_ids):
    ids = rand_ids(cfg, 1, 12)
    with torch.no_grad():
        on = model(ids).logits
        off = model(ids, memory_enabled=False).logits
    assert torch.isfinite(off).all() and not torch.allclose(on, off)


def test_segmented_forward_is_causal_across_segments(model, cfg, rand_ids):
    S = cfg.memory.update_segment_tokens
    ids = rand_ids(cfg, 1, 3 * S)
    alt = ids.clone()
    alt[:, 2 * S + 3 :] = rand_ids(cfg, 1, S - 3, seed=4)
    with torch.no_grad():
        a = model.forward_segmented(ids).logits
        b = model.forward_segmented(alt).logits
    torch.testing.assert_close(a[:, : 2 * S + 3], b[:, : 2 * S + 3], atol=1e-6, rtol=0)


def test_segmented_memory_changes_later_segments(model, cfg, rand_ids):
    S = cfg.memory.update_segment_tokens
    ids = rand_ids(cfg, 1, 2 * S)
    with torch.no_grad():
        seg = model.forward_segmented(ids)
        plain = model(ids)
    torch.testing.assert_close(seg.logits[:, :S], plain.logits[:, :S], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(seg.logits[:, S:], plain.logits[:, S:])  # segment 2 reads updated memory
    assert len(seg.memory_info) == 2 * len(model.anchor_indices)


def test_segmented_training_reaches_memory_parameters(cfg, rand_ids):
    from m63_omni.model.m63 import M63Omni

    torch.manual_seed(0)
    m = M63Omni(cfg)
    S = cfg.memory.update_segment_tokens
    ids = rand_ids(cfg, 2, 3 * S)
    labels = torch.cat([ids[:, 1:], torch.full((2, 1), -100)], 1)
    out = m.forward_segmented(ids, labels, bptt_segments=2)
    (out.lm_loss + out.memory_loss).backward()
    a = m.memory[str(m.anchor_indices[0])]
    for p in (a.W1_0, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight, a.gate.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


@pytest.mark.parametrize("target", ["zero", "init"])
def test_memory_decay_target(target):
    from m63_omni.config import tiny
    from m63_omni.model.m63 import M63Omni

    torch.manual_seed(0)
    m = M63Omni(tiny(memory={"decay_target": target, "max_alpha": 0.5}, multimodal={"enabled": False}))
    anchor = m.memory[str(m.anchor_indices[0])]
    with torch.no_grad():
        anchor.gate.bias.copy_(torch.tensor([0.0, -30.0, 30.0]))  # no inner learning step, maximal forgetting
    st = anchor.init_state(1)
    st.W1 = st.W1 + torch.randn_like(st.W1)  # start away from M0
    h = torch.randn(1, 16, m.cfg.model.hidden_size)
    for _ in range(40):
        st, _ = anchor.update(st, h)
    to_m0 = float((st.W1[0] - anchor.W1_0).detach().norm())
    if target == "init":
        assert to_m0 < 1e-3  # forgetting returns to the learned initial memory
    else:
        assert float(st.W1.detach().norm()) < 1e-3 and to_m0 > 0.5  # forgetting erodes M0 itself


def test_memory_decay_target_is_validated():
    from m63_omni.config import ConfigError, tiny

    with pytest.raises(ConfigError, match="decay_target"):
        tiny(memory={"decay_target": "half"})
