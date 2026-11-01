"""Authoritative, append-only, branchable event log (blueprint §6, §10, §14).

The event log is the source of truth for assets, timeline versions, plans, tool
receipts, permissions, and provenance. Recurrent state, attention KV, and neural
memory are *accelerators* that can always be rebuilt by replaying this log.

Properties enforced here:

* append-only with a hash chain (``chain_hash`` covers the parent chain hash,
  the kind, branch, trust label, and payload hash), so tampering is detectable;
* branches: every event has a parent; a branch is a named head pointer. Undo
  creates or restores a named branch instead of destructively reversing state;
* tenant isolation: every read checks the caller's tenant;
* verified deletion: payloads are purged, a tombstone event is appended, and
  :meth:`EventLog.verify_deleted` proves the content is gone.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

from ..common import Trust, canonical_json, content_hash, short_id

MAIN_BRANCH = "main"
DELETED_PAYLOAD = {"__deleted__": True}

EVENT_KINDS = frozenset(
    {
        "session_start",
        "ingest",
        "plan",
        "receipt",
        "verification",
        "branch",
        "memory_snapshot",
        "escalation",
        "final",
        "delete",
        "note",
    }
)


class TenantIsolationError(PermissionError):
    pass


class EventLogError(RuntimeError):
    pass


@dataclass(frozen=True)
class Event:
    event_id: str
    seq: int
    tenant_id: str
    session_id: str
    branch_id: str
    parent_id: Optional[str]
    kind: str
    trust: str
    payload: dict
    payload_hash: str
    chain_hash: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Branch:
    branch_id: str
    head: Optional[str]
    parent_event: Optional[str]
    reason: str = ""
    created_seq: int = 0
    meta: dict = field(default_factory=dict)


def _chain(parent_chain: str, kind: str, branch_id: str, trust: str, payload_hash: str) -> str:
    return content_hash([parent_chain, kind, branch_id, trust, payload_hash])


class EventLog:
    """Per-(tenant, session) append-only event log with branches.

    ``path`` enables JSONL persistence; the file is append-only except for
    deletion, which rewrites the file with purged payloads (verified deletion
    must not leave the content on disk).
    """

    def __init__(
        self,
        tenant_id: str,
        session_id: str,
        path: str | Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.path = Path(path) if path is not None else None
        self._clock = clock
        self._lock = threading.RLock()
        self._events: dict[str, Event] = {}
        self._order: list[str] = []
        self._branches: dict[str, Branch] = {MAIN_BRANCH: Branch(MAIN_BRANCH, None, None, "root")}
        self._deleted: set[str] = set()
        if self.path is not None and self.path.exists():
            self._load()

    # ---------------------------------------------------------------- access
    def _check_tenant(self, tenant_id: Optional[str]) -> None:
        if tenant_id is not None and tenant_id != self.tenant_id:
            raise TenantIsolationError(f"tenant {tenant_id!r} may not access log of tenant {self.tenant_id!r}")

    def get(self, event_id: str, tenant_id: Optional[str] = None) -> Event:
        self._check_tenant(tenant_id)
        try:
            return self._events[event_id]
        except KeyError:
            raise EventLogError(f"unknown event {event_id}") from None

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._events

    @property
    def branches(self) -> dict[str, Branch]:
        return dict(self._branches)

    def head(self, branch_id: str = MAIN_BRANCH) -> Optional[Event]:
        b = self._branch(branch_id)
        return self._events[b.head] if b.head else None

    def _branch(self, branch_id: str) -> Branch:
        try:
            return self._branches[branch_id]
        except KeyError:
            raise EventLogError(f"unknown branch {branch_id}") from None

    # ---------------------------------------------------------------- append
    def append(
        self,
        kind: str,
        payload: dict,
        *,
        branch_id: str = MAIN_BRANCH,
        trust: Trust | str = Trust.SYSTEM,
        tenant_id: Optional[str] = None,
    ) -> Event:
        self._check_tenant(tenant_id)
        if kind not in EVENT_KINDS:
            raise EventLogError(f"unknown event kind {kind!r}")
        if not isinstance(payload, dict):
            raise EventLogError("payload must be a dict")
        trust_v = Trust(trust).value
        with self._lock:
            branch = self._branch(branch_id)
            parent_id = branch.head
            parent_chain = self._events[parent_id].chain_hash if parent_id else content_hash([self.tenant_id, self.session_id])
            payload = json.loads(canonical_json(payload))  # detach + normalise
            payload_hash = content_hash(payload)
            seq = len(self._order)
            chain_hash = _chain(parent_chain, kind, branch_id, trust_v, payload_hash)
            event_id = "ev_" + short_id(self.tenant_id, self.session_id, seq, chain_hash)
            ev = Event(
                event_id=event_id,
                seq=seq,
                tenant_id=self.tenant_id,
                session_id=self.session_id,
                branch_id=branch_id,
                parent_id=parent_id,
                kind=kind,
                trust=trust_v,
                payload=payload,
                payload_hash=payload_hash,
                chain_hash=chain_hash,
                created_at=float(self._clock()),
            )
            self._events[event_id] = ev
            self._order.append(event_id)
            branch.head = event_id
            self._persist(ev)
            return ev

    def append_ingest(
        self,
        *,
        segment_hash: str,
        asset_refs: Iterable[str],
        trust: Trust | str,
        preprocessing: dict | None = None,
        branch_id: str = MAIN_BRANCH,
        extra: dict | None = None,
    ) -> Event:
        payload = {
            "segment_hash": segment_hash,
            "asset_refs": list(asset_refs),
            "preprocessing": preprocessing or {},
        }
        if extra:
            payload.update(extra)
        return self.append("ingest", payload, branch_id=branch_id, trust=trust)

    def append_receipt(self, receipt: dict, *, branch_id: str = MAIN_BRANCH) -> Event:
        trust = receipt.get("trust", Trust.UNTRUSTED.value)
        return self.append("receipt", receipt, branch_id=branch_id, trust=trust)

    # -------------------------------------------------------------- branches
    def create_branch(self, parent_event_id: str, reason: str, name: Optional[str] = None, meta: dict | None = None,
                      *, tenant_id: Optional[str] = None) -> str:
        """Create a branch whose history is ``parent_event_id``'s lineage.

        Branching is only valid at a committed event (Appendix A).
        """
        self._check_tenant(tenant_id)
        with self._lock:
            if parent_event_id not in self._events:
                raise EventLogError(f"cannot branch from unknown event {parent_event_id}")
            if parent_event_id in self._deleted:
                raise EventLogError("cannot branch from a deleted event")
            branch_id = name or "br_" + short_id(parent_event_id, reason, len(self._branches), n=10)
            if branch_id in self._branches:
                raise EventLogError(f"branch {branch_id} already exists")
            self._branches[branch_id] = Branch(branch_id, parent_event_id, parent_event_id, reason, len(self._order), dict(meta or {}))
            self.append(
                "branch",
                {"parent_event": parent_event_id, "reason": reason, "meta": dict(meta or {})},
                branch_id=branch_id,
            )
            return branch_id

    def lineage(self, event_id: str, *, tenant_id: Optional[str] = None) -> list[Event]:
        """Events from the root to ``event_id`` inclusive (following parent links)."""
        self._check_tenant(tenant_id)
        out: list[Event] = []
        cur: Optional[str] = event_id
        while cur is not None:
            ev = self._events[cur]
            out.append(ev)
            cur = ev.parent_id
        return out[::-1]

    def replay(self, branch_id: str = MAIN_BRANCH, *, include_deleted: bool = False,
               tenant_id: Optional[str] = None) -> Iterator[Event]:
        self._check_tenant(tenant_id)
        head = self._branch(branch_id).head
        if head is None:
            return iter(())
        events = self.lineage(head)
        if not include_deleted:
            events = [e for e in events if e.event_id not in self._deleted]
        return iter(events)

    def events(self, *, include_deleted: bool = False, tenant_id: Optional[str] = None) -> Iterator[Event]:
        """Every event across all branches in append order (for audit)."""
        self._check_tenant(tenant_id)
        return iter([self._events[e] for e in self._order if include_deleted or e not in self._deleted])

    def is_ancestor(self, ancestor_id: str, event_id: str) -> bool:
        return any(e.event_id == ancestor_id for e in self.lineage(event_id))

    # ----------------------------------------------------------- integrity
    def verify_chain(self) -> bool:
        """Recompute every chain hash. Deleted events keep their original payload hash."""
        for eid in self._order:
            ev = self._events[eid]
            parent_chain = (
                self._events[ev.parent_id].chain_hash if ev.parent_id else content_hash([self.tenant_id, self.session_id])
            )
            if eid not in self._deleted and content_hash(ev.payload) != ev.payload_hash:
                return False
            if _chain(parent_chain, ev.kind, ev.branch_id, ev.trust, ev.payload_hash) != ev.chain_hash:
                return False
        return True

    # ------------------------------------------------------------- deletion
    def delete(self, event_ids: Iterable[str], reason: str, *, tenant_id: Optional[str] = None) -> Event:
        """Purge payloads of ``event_ids`` and append a tombstone on ``main``.

        The tombstone records *which* events were deleted and why, never their content.
        """
        self._check_tenant(tenant_id)
        ids = list(event_ids)
        with self._lock:
            for eid in ids:
                ev = self.get(eid)
                self._events[eid] = Event(**{**ev.to_dict(), "payload": dict(DELETED_PAYLOAD)})
                self._deleted.add(eid)
            tomb = self.append("delete", {"deleted_events": ids, "reason": reason})
            self._rewrite()
            return tomb

    def verify_deleted(self, event_id: str) -> bool:
        ev = self._events.get(event_id)
        if ev is None:
            return True
        if event_id not in self._deleted or ev.payload != DELETED_PAYLOAD:
            return False
        if self.path is not None and self.path.exists():
            for line in self.path.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("event_id") == event_id and rec.get("payload") != DELETED_PAYLOAD:
                    return False
        return True

    def is_deleted(self, event_id: str) -> bool:
        return event_id in self._deleted

    # ---------------------------------------------------------- persistence
    def _persist(self, ev: Event) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(canonical_json({"type": "event", **ev.to_dict()}) + "\n")
        if ev.kind == "branch":
            b = self._branches[ev.branch_id]
            with self.path.open("a") as fh:
                fh.write(canonical_json({"type": "branch_meta", "branch_id": b.branch_id, "parent_event": b.parent_event,
                                         "reason": b.reason, "meta": b.meta}) + "\n")

    def _rewrite(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as fh:
            for eid in self._order:
                ev = self._events[eid]
                fh.write(canonical_json({"type": "event", **ev.to_dict(), "deleted": eid in self._deleted}) + "\n")
                if ev.kind == "branch" and ev.parent_id == self._branches[ev.branch_id].parent_event:
                    b = self._branches[ev.branch_id]
                    fh.write(canonical_json({"type": "branch_meta", "branch_id": b.branch_id, "parent_event": b.parent_event,
                                             "reason": b.reason, "meta": b.meta}) + "\n")
        tmp.replace(self.path)

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text().splitlines():
            rec = json.loads(line)
            typ = rec.pop("type")
            if typ == "branch_meta":
                b = self._branches.setdefault(rec["branch_id"], Branch(rec["branch_id"], None, rec["parent_event"]))
                b.parent_event, b.reason, b.meta = rec["parent_event"], rec["reason"], rec["meta"]
                continue
            deleted = rec.pop("deleted", False)
            ev = Event(**rec)
            if ev.tenant_id != self.tenant_id or ev.session_id != self.session_id:
                raise TenantIsolationError("event log file belongs to a different tenant/session")
            self._events[ev.event_id] = ev
            self._order.append(ev.event_id)
            if deleted or ev.payload == DELETED_PAYLOAD:
                self._deleted.add(ev.event_id)
            self._branches.setdefault(ev.branch_id, Branch(ev.branch_id, None, ev.parent_id, ""))
            self._branches[ev.branch_id].head = ev.event_id
        if not self.verify_chain():
            raise EventLogError(f"hash chain verification failed for {self.path}")
