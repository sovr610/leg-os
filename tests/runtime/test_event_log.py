"""Authoritative event log: hash chain, branches, tenant isolation, verified deletion (§6, §10)."""

import json

import pytest

from m63_omni.common import Trust
from m63_omni.runtime.event_log import EventLog, EventLogError, TenantIsolationError


def _log(tmp_path=None):
    t = iter(range(10**6))
    return EventLog("tenant", "sess", path=None if tmp_path is None else tmp_path / "log.jsonl", clock=lambda: float(next(t)))


def test_chain_and_tamper_detection():
    log = _log()
    e1 = log.append_ingest(segment_hash="sha256:a", asset_refs=["x"], trust=Trust.USER)
    log.append("plan", {"steps": [1, 2]})
    assert log.verify_chain()
    object.__setattr__(log.get(e1.event_id), "payload", {"segment_hash": "sha256:evil"})
    assert not log.verify_chain()


def test_branches_lineage_and_replay():
    log = _log()
    a = log.append("plan", {"v": 1})
    b = log.append("plan", {"v": 2})
    br = log.create_branch(a.event_id, "undo", name="undo_1")
    c = log.append("plan", {"v": 3}, branch_id=br)
    assert [e.payload.get("v") for e in log.replay("main")] == [1, 2]
    assert [e.kind for e in log.replay(br)] == ["plan", "branch", "plan"]
    assert log.is_ancestor(a.event_id, c.event_id) and not log.is_ancestor(b.event_id, c.event_id)
    with pytest.raises(EventLogError):
        log.create_branch(a.event_id, "dup", name="undo_1")


def test_tenant_isolation_and_kind_validation():
    log = _log()
    e = log.append("plan", {})
    with pytest.raises(TenantIsolationError):
        log.get(e.event_id, tenant_id="other")
    with pytest.raises(TenantIsolationError):
        log.append("plan", {}, tenant_id="other")
    with pytest.raises(EventLogError):
        log.append("rm_rf", {})
    for read in (lambda: list(log.replay(tenant_id="other")), lambda: log.lineage(e.event_id, tenant_id="other"),
                 lambda: list(log.events(tenant_id="other")), lambda: log.create_branch(e.event_id, "x", tenant_id="other")):
        with pytest.raises(TenantIsolationError):
            read()
    assert [x.event_id for x in log.events(tenant_id="tenant")] == [e.event_id]


def test_persistence_roundtrip_and_deletion_purges_disk(tmp_path):
    log = _log(tmp_path)
    secret = log.append("ingest", {"text": "my secret address"}, trust=Trust.USER)
    log.append("plan", {"v": 1})
    br = log.create_branch(secret.event_id, "alt")
    log.append("note", {"x": 1}, branch_id=br)
    reloaded = EventLog("tenant", "sess", path=tmp_path / "log.jsonl")
    assert len(reloaded) == len(log) and reloaded.verify_chain()
    assert reloaded.branches[br].reason == "alt"
    log.delete([secret.event_id], "gdpr")
    assert log.verify_deleted(secret.event_id) and log.verify_chain()
    assert "my secret address" not in (tmp_path / "log.jsonl").read_text()
    with pytest.raises(EventLogError):
        log.create_branch(secret.event_id, "from deleted")
    with pytest.raises(TenantIsolationError):
        EventLog("other", "sess", path=tmp_path / "log.jsonl")


def test_receipts_carry_trust():
    log = _log()
    ev = log.append_receipt({"status": "ok", "trust": "untrusted"})
    assert ev.trust == "untrusted" and json.loads(json.dumps(ev.to_dict()))["kind"] == "receipt"
