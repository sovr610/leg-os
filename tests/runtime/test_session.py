"""Sessions: replay, branching, memory governance, deletion, reclaim (§6, §14, Appendix A/B)."""

import pytest
import torch

from m63_omni.common import Trust
from m63_omni.runtime.event_log import TenantIsolationError
from m63_omni.runtime.session import IncompatibleSnapshotError, Session, SessionDeletedError, SessionManager


def _toks(cfg, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.model.vocab_size - cfg.model.special_tokens, (n,), generator=g).tolist()


def test_replay_reproduces_live_state(model, cfg):
    s = Session(model, "t", "s")
    S = cfg.memory.update_segment_tokens
    s.ingest(_toks(cfg, S + 5, 0))
    s.ingest(_toks(cfg, S, 1), trust=Trust.UNTRUSTED)
    s.generate(4)
    live_logits, live_mem = s._last_logits.clone(), s.memory_hash()
    r = Session(model, "t", "s", event_log=s.log)
    r.replay()
    torch.testing.assert_close(r._last_logits, live_logits, atol=1e-5, rtol=1e-5)
    assert r.memory_hash() == live_mem and r.tokens == s.tokens


def test_fork_restores_committed_state_and_branches_are_independent(model, cfg):
    s = Session(model, "t", "s")
    e1 = s.ingest(_toks(cfg, 10, 0))
    s.ingest(_toks(cfg, 7, 1))
    child = s.fork(e1.event_id, "undo")
    assert child.tokens == 10 and s.tokens == 17
    parent_logits = s._last_logits.clone()
    child.ingest(_toks(cfg, 5, 2))
    torch.testing.assert_close(s._last_logits, parent_logits)  # no cross-branch writes
    assert [e.kind for e in child.log.replay(child.branch_id)][-2:] == ["branch", "ingest"]


def test_fork_without_checkpoint_replays_identically(model, cfg):
    s = Session(model, "t", "s")
    e1 = s.ingest(_toks(cfg, 9, 0))
    s.ingest(_toks(cfg, 4, 1))
    via_cp = s.fork(e1.event_id, "a")
    s.checkpoints.pop(e1.event_id)
    via_replay = s.fork(e1.event_id, "b")
    torch.testing.assert_close(via_replay._last_logits, via_cp._last_logits, atol=1e-5, rtol=1e-5)


def test_untrusted_segments_never_write_persistent_memory(model, cfg):
    S = cfg.memory.update_segment_tokens
    s = Session(model, "t", "s")
    h0 = s.memory_hash()
    s.ingest(_toks(cfg, S, 0), trust=Trust.UNTRUSTED)
    assert s.memory_hash() == h0
    notes = [e.payload for e in s.log.replay() if e.kind == "note"]
    assert notes and notes[-1]["reason"] == "untrusted_content_quarantined"
    s.ingest(_toks(cfg, S, 1), trust=Trust.USER)
    assert s.memory_hash() != h0
    assert any(e.kind == "memory_snapshot" for e in s.log.replay())


def test_mixed_segment_with_untrusted_tail_is_quarantined(model, cfg):
    S = cfg.memory.update_segment_tokens
    s = Session(model, "t", "s")
    h0 = s.memory_hash()
    s.ingest(_toks(cfg, S - 3, 0), trust=Trust.USER)
    s.ingest(_toks(cfg, 3, 1), trust=Trust.UNTRUSTED)  # completes the segment
    assert s.memory_hash() == h0


def test_generation_keeps_memory_frozen(model, cfg):
    s = Session(model, "t", "s")
    s.ingest(_toks(cfg, cfg.memory.update_segment_tokens, 0))
    h = s.memory_hash()
    s.generate(cfg.memory.update_segment_tokens + 3)
    assert s.memory_hash() == h


def test_delete_removes_all_three_forms(model, cfg):
    s = Session(model, "t", "s")
    e = s.ingest(_toks(cfg, 8, 0))
    s.generate(2)
    s.delete()
    assert s.verify_deleted() and s.log.verify_deleted(e.event_id)
    with pytest.raises(SessionDeletedError):
        s.ingest([1, 2])
    with pytest.raises(SessionDeletedError):
        s.fork(e.event_id, "x")


def test_reclaim_then_transparent_replay(model, cfg):
    clock = [0.0]
    s = Session(model, "t", "s", clock=lambda: clock[0], ttl_seconds=10)
    s.ingest(_toks(cfg, 12, 0))
    ref = Session(model, "t", "ref")
    ref.ingest(_toks(cfg, 12, 0))
    clock[0] = 100.0
    assert s.expired()
    s.reclaim()
    assert s.cache is None
    s.ingest(_toks(cfg, 5, 1))
    ref.ingest(_toks(cfg, 5, 1))
    torch.testing.assert_close(s._last_logits, ref._last_logits, atol=1e-5, rtol=1e-5)


def test_snapshot_compatibility_and_tenancy(model, cfg):
    s = Session(model, "t", "s")
    s.ingest(_toks(cfg, cfg.memory.update_segment_tokens, 0))
    snap = s.snapshot_memory()
    other = Session(model, "t", "s2", tokenizer_version="bpe-v1")
    with pytest.raises(IncompatibleSnapshotError):
        other.load_memory_snapshot(snap)
    with pytest.raises(TenantIsolationError):
        Session(model, "u", "s3").load_memory_snapshot(snap)
    same = Session(model, "t", "s4")
    same.load_memory_snapshot(snap)
    assert same.memory_hash() == s.memory_hash()


def test_manager_isolation_and_reclaimer(model, cfg):
    clock = [0.0]
    mgr = SessionManager(model, clock=lambda: clock[0])
    a = mgr.open("ta", "s", ttl_seconds=5)
    a.ingest([1, 2, 3])
    with pytest.raises(TenantIsolationError):
        mgr.get("tb", "s")
    clock[0] = 60.0
    assert mgr.reclaim_idle() == [("ta", "s")]


def test_memory_off_session(model, cfg):
    s = Session(model, "t", "s", memory_enabled=False)
    s.ingest(_toks(cfg, 2 * cfg.memory.update_segment_tokens, 0))
    assert s.memory is None and len(s.generate(3)) == 3
    assert not any(e.kind == "memory_snapshot" for e in s.log.replay())


def test_session_uses_a_provided_empty_log(model, tmp_path):
    from m63_omni.runtime.event_log import EventLog

    log = EventLog("t", "s", path=tmp_path / "log.jsonl")
    assert len(log) == 0  # an empty log is falsy; the session must still adopt it
    s = Session(model, "t", "s", event_log=log)
    s.ingest([1, 2, 3])
    assert s.log is log and (tmp_path / "log.jsonl").exists()
