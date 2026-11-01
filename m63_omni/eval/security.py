"""Runtime security suite (blueprint §6, §13, Appendix B "Memory governance", "Serving cache").

Stateful, multi-session checks that feed the non-negotiable release gates:
cross-tenant isolation (event log, sessions, paged KV, prefix cache), memory
poisoning (untrusted segments never write persistent memory), verified
deletion, snapshot invalidation on model upgrade, and a functional memory-off
path. Agent-level injection tests live in ``m63_omni.agent`` and ``tests/agent``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from ..common import Trust
from ..model.m63 import M63Omni
from ..runtime.event_log import TenantIsolationError
from ..runtime.paged_kv import PagedKVStore
from ..runtime.prefix_cache import PrefixCache, PrefixKey
from ..runtime.session import IncompatibleSnapshotError, Session, SessionManager


@dataclass
class SecurityResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class SecurityReport:
    results: list[SecurityResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def gate_inputs(self) -> dict[str, bool]:
        by = {r.name: r.passed for r in self.results}
        return {
            "zero_cross_tenant_reads": all(v for k, v in by.items() if k.startswith("tenant_")),
            "verified_deletion": by.get("deletion_verified", False),
            "memory_off_path_functional": by.get("memory_off_path", False),
        }


def _expect_raises(fn: Callable[[], object], exc: type[BaseException]) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def run_security_suite(model: M63Omni, *, seg_tokens: int | None = None) -> SecurityReport:
    rep = SecurityReport()
    S = seg_tokens or model.cfg.memory.update_segment_tokens
    V = model.cfg.model.vocab_size - model.cfg.model.special_tokens
    gen = torch.Generator().manual_seed(0)
    toks = lambda n: torch.randint(0, V, (n,), generator=gen).tolist()  # noqa: E731

    # tenant isolation: event log, session manager, paged KV, prefix cache
    mgr = SessionManager(model)
    a = mgr.open("tenant_a", "s1")
    ev = a.ingest(toks(8))
    rep.results.append(SecurityResult("tenant_event_log", _expect_raises(lambda: a.log.get(ev.event_id, tenant_id="tenant_b"), TenantIsolationError)))
    rep.results.append(SecurityResult("tenant_session_lookup", _expect_raises(lambda: mgr.get("tenant_b", "s1"), TenantIsolationError)))
    rep.results.append(SecurityResult("tenant_fork", _expect_raises(lambda: a.fork(ev.event_id, "x", tenant_id="tenant_b"), TenantIsolationError)))
    store = PagedKVStore(8, 4, 1, 2)
    store.create("q", "tenant_a")
    store.append("q", torch.ones(1, 3, 2), torch.ones(1, 3, 2))
    rep.results.append(SecurityResult("tenant_paged_kv", _expect_raises(lambda: store.gather("q", "tenant_b"), PermissionError)))
    pc = PrefixCache()
    key = PrefixKey("tenant_a", "m", "t", "p", "h", 8)
    pc.put(key, "state")
    other = PrefixKey("tenant_b", "m", "t", "p", "h", 8)
    rep.results.append(SecurityResult("tenant_prefix_cache", pc.get(other) is None))

    if model.cfg.memory.enabled:
        # memory poisoning: an untrusted segment must not change persistent memory
        s = Session(model, "tenant_a", "poison")
        s.ingest(toks(S), trust=Trust.USER)
        before = s.memory_hash()
        s.ingest(toks(S), trust=Trust.UNTRUSTED)
        rep.results.append(SecurityResult("memory_untrusted_write_blocked", s.memory_hash() == before))
        # generation never writes memory
        s.generate(4)
        rep.results.append(SecurityResult("memory_frozen_during_generation", s.memory_hash() == before))
        # snapshot invalidation on model upgrade
        snap = s.snapshot_memory()
        snap_bad = dict(snap, model_version="different")
        rep.results.append(SecurityResult("memory_snapshot_upgrade_invalidated",
                                          _expect_raises(lambda: s.load_memory_snapshot(snap_bad), IncompatibleSnapshotError)))
        rep.results.append(SecurityResult("tenant_memory_snapshot",
                                          _expect_raises(lambda: s.load_memory_snapshot(dict(snap, tenant_id="tenant_b")), TenantIsolationError)))

    # deletion
    d = Session(model, "tenant_a", "del")
    d.ingest(toks(8))
    d.generate(2)
    d.delete()
    rep.results.append(SecurityResult("deletion_verified", d.verify_deleted()))

    # memory-off path
    try:
        off = Session(model, "tenant_a", "off", memory_enabled=False)
        off.ingest(toks(8))
        off.generate(2)
        ok = off.memory is None
    except Exception as e:  # pragma: no cover - reported, not raised
        ok = False
        rep.results.append(SecurityResult("memory_off_path", False, repr(e)))
    else:
        rep.results.append(SecurityResult("memory_off_path", ok))
    return rep
