"""Training loop, checkpoint/resume, schedules, monitors, objectives, parallel layouts (§7, §9, §11, §12)."""

import pytest
import torch

from m63_omni.config import reference
from m63_omni.model.m63 import M63Omni
from m63_omni.model.moe import RouterStats
from m63_omni.train.checkpoint import CheckpointManager, CheckpointMismatchError, load_checkpoint
from m63_omni.train.monitor import NonFiniteError, RouterMonitor, SpikeDetector, check_finite
from m63_omni.train.objectives import contrastive_alignment, ctc_loss
from m63_omni.train.parallel import LONG_CONTEXT_256, SHORT_CONTEXT_256, ParallelLayout, validate_layout
from m63_omni.train.schedule import batch_tokens, wsd_lr
from m63_omni.train.trainer import Trainer, synthetic_batch


def _trainer(cfg, seed=0, **kw):
    torch.manual_seed(seed)
    m = M63Omni(cfg)
    return Trainer(m, lambda c: synthetic_batch(cfg, c, 4, 64), total_positions=kw.pop("total", 10**6), **kw)


def test_synthetic_batches_respect_document_contract(cfg):
    b = synthetic_batch(cfg, 3, 4, 64)
    assert torch.equal(b.input_ids, synthetic_batch(cfg, 3, 4, 64).input_ids)  # pure function of the cursor
    starts = b.reset_mask
    assert starts[:, 0].all()
    assert (b.position_ids[starts] == 0).all()
    ends = torch.cat([b.segment_ids[:, 1:] != b.segment_ids[:, :-1], torch.ones(4, 1, dtype=torch.bool)], 1)
    assert (b.labels[ends] == -100).all()


def test_loss_decreases(cfg):
    tr = _trainer(cfg, total=4 * 64 * 40)
    hist = tr.fit(40)
    assert hist[-1]["lm"] < hist[0]["lm"] - 1.0
    assert all(h["grad_norm"] > 0 for h in hist)


def test_resume_reproduces_uninterrupted_run(cfg, tmp_path):
    a = _trainer(cfg, seed=1)
    a.fit(6)
    b = _trainer(cfg, seed=1, checkpoints=CheckpointManager(tmp_path))
    b.fit(3)
    path = b.save()[0]
    c = _trainer(cfg, seed=123)
    c.resume(path)
    c.fit(3)
    for (k, x), y in zip(a.model.state_dict().items(), c.model.state_dict().values()):
        assert torch.equal(x, y), k
    assert c.state.cursor == 6 and c.state.positions == a.state.positions


def test_checkpoint_contents_and_retention(cfg, tmp_path):
    tr = _trainer(cfg, checkpoints=CheckpointManager(tmp_path, keep_rolling=2, milestone_every_positions=512),
                  checkpoint_every=1)
    tr.fit(5)
    mgr = tr.ckpt
    assert len(list((tmp_path / "rolling").glob("*.pt"))) == 2
    assert len(mgr.milestones()) >= 2
    payload = torch.load(mgr.latest(), weights_only=False)
    assert any(k.endswith("router.bias") for k in payload["model"])
    assert payload["data_cursor"] == 5 and "rng" in payload
    assert payload["model"]["blocks.0.moe.router.bias"].abs().sum() > 0  # bias controller ran
    other = M63Omni(cfg.replace(moe={"top_k": 1}))
    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(mgr.latest(), model=other, config_fingerprint=other.cfg.fingerprint())


def test_nonfinite_guard(cfg):
    m = M63Omni(cfg)
    loss = (m.embed.weight.sum() * float("nan"))
    with pytest.raises(NonFiniteError):
        check_finite(loss, m)


def test_segmented_memory_training_step(cfg):
    torch.manual_seed(0)
    m = M63Omni(cfg)
    S = cfg.memory.update_segment_tokens
    tr = Trainer(m, lambda c: synthetic_batch(cfg, c, 2, 3 * S), total_positions=10**5, segmented_memory=True)
    met = tr.train_step()
    assert "memory" in met and met["memory"] > 0


def test_wsd_schedule_and_batch_ramp():
    t = reference().training
    total = 100e9
    assert wsd_lr(0, total, t) < t.peak_lr / 100
    assert wsd_lr(5e9, total, t) == t.peak_lr
    assert wsd_lr(total, total, t) == pytest.approx(t.peak_lr * t.min_lr_ratio)
    assert wsd_lr(95e9, total, t) < t.peak_lr
    assert batch_tokens(0, t) == 2_097_152
    assert batch_tokens(1e12, t) == 8_388_608
    assert batch_tokens(50e9, t) % t.seq_len == 0


def test_router_monitor_alarms():
    mon = RouterMonitor(1, 4, window_tokens=10, overload_ratio=2.0, overload_patience=2, min_modality_tokens=1)

    def stats(counts):
        c = torch.tensor(counts)
        mc = torch.zeros(6, 4, dtype=torch.long)
        mc[0] = c
        return [RouterStats(c, mc, 10, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))]

    first = mon.update(stats([10, 10, 0, 0]))
    assert {a.kind for a in first} >= {"dead_expert"}
    mon.update(stats([17, 1, 1, 1]))
    kinds = [a.kind for a in mon.update(stats([17, 1, 1, 1]))]
    assert "overload" in kinds and "modality_skew" in kinds


def test_spike_detector():
    d = SpikeDetector(warmup_steps=5)
    for i in range(30):
        assert not d.update(i, 3.0 + 0.01 * (i % 3))
    assert d.update(31, 9.0)


def test_auxiliary_objectives():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(8, 16, generator=g)
    assert contrastive_alignment(a, a + 0.01 * torch.randn(8, 16, generator=g)) < contrastive_alignment(a, torch.randn(8, 16, generator=g))
    logits = torch.randn(2, 20, 30, generator=g)
    loss = ctc_loss(logits, torch.randint(1, 30, (2, 5), generator=g), torch.tensor([20, 18]), torch.tensor([5, 4]))
    assert torch.isfinite(loss)


def test_parallel_layouts():
    c = reference()
    assert validate_layout(c, SHORT_CONTEXT_256, 256) == []
    assert validate_layout(c, LONG_CONTEXT_256, 256) == []
    assert SHORT_CONTEXT_256.expert_dp == 2  # "two expert-data replicas" (§11)
    bad = validate_layout(c, ParallelLayout(tp=4, pp=5, cp=1, dp=16, ep=8), 320)
    assert any("PP" in e for e in bad)
    bad = validate_layout(c, ParallelLayout(tp=4, pp=8, cp=1, dp=8, ep=24), 256)
    assert any("routed_experts" in e for e in bad) and any("hybrid groups" in e for e in bad)
