"""Sessions: working context + bounded neural memory + authoritative event log (§6, §14, Appendix A).

State layers (blueprint §6 table):

* working context — :class:`HybridCache` (GDN state + GQA KV), exact for one
  model version, recomputable by replaying the event log;
* session neural memory — per-anchor :class:`MemoryState`, a disposable
  accelerator snapshotted with model/tokenizer/event IDs;
* authoritative event log — :class:`EventLog`, the source of truth.

Governance rules implemented here:

* memory updates happen only at ``update_segment_tokens`` boundaries of
  *ingested* content, never during generation (memory is frozen while decoding);
* a segment containing untrusted content never writes persistent memory (it may
  still inform the current task through the working context);
* branches are valid only at committed events; a fork copies (copy-on-write in
  production, clone in this reference) the checkpoint at that event or replays
  from the log when no checkpoint exists — it never invents a partial state;
* delete removes all three forms and is verifiable;
* a model or tokenizer upgrade invalidates memory snapshots.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from ..common import Trust, content_hash
from ..model.cache import HybridCache
from ..model.m63 import M63Omni
from ..model.memory import MemoryState
from .event_log import MAIN_BRANCH, Event, EventLog, TenantIsolationError


class IncompatibleSnapshotError(RuntimeError):
    pass


class SessionDeletedError(RuntimeError):
    pass


@dataclass
class Checkpoint:
    event_id: str
    cache: Optional[HybridCache]
    memory: Optional[dict[int, MemoryState]]
    buffer: dict[int, list[torch.Tensor]]
    buffer_trust: list[str]
    last_logits: Optional[torch.Tensor]


@dataclass
class _MemoryUpdate:
    anchor_ids: list[int]
    written: bool
    reason: str
    info: dict = field(default_factory=dict)


def _clone_mem(mem: Optional[dict[int, MemoryState]]) -> Optional[dict[int, MemoryState]]:
    return None if mem is None else {i: s.clone() for i, s in mem.items()}


class Session:
    def __init__(
        self,
        model: M63Omni,
        tenant_id: str,
        session_id: str,
        *,
        event_log: Optional[EventLog] = None,
        branch_id: str = MAIN_BRANCH,
        tokenizer_version: str = "bpe-v0",
        memory_enabled: bool = True,
        clock: Callable[[], float] = time.time,
        ttl_seconds: Optional[float] = None,
        _checkpoints: Optional[dict[str, Checkpoint]] = None,
    ) -> None:
        self.model = model
        self.tenant_id, self.session_id, self.branch_id = tenant_id, session_id, branch_id
        self.model_version = model.cfg.fingerprint()
        self.tokenizer_version = tokenizer_version
        self.memory_enabled = memory_enabled and model.cfg.memory.enabled
        self.clock = clock
        self.ttl = ttl_seconds if ttl_seconds is not None else model.cfg.runtime.session_ttl_seconds
        self.last_active = clock()
        self.deleted = False
        new_log = event_log is None
        self.log = event_log or EventLog(tenant_id, session_id, clock=clock)
        if self.log.tenant_id != tenant_id:
            raise TenantIsolationError("event log belongs to another tenant")
        self.checkpoints: dict[str, Checkpoint] = _checkpoints if _checkpoints is not None else {}
        self._reset_state()
        if new_log:
            self.log.append("session_start", {"model_version": self.model_version, "tokenizer_version": tokenizer_version,
                                              "memory_enabled": self.memory_enabled}, branch_id=branch_id)

    # ------------------------------------------------------------------ state
    def _reset_state(self) -> None:
        self.cache: Optional[HybridCache] = None
        self.memory = self._fresh_memory() if self.memory_enabled else None
        self._buffer: dict[int, list[torch.Tensor]] = {i: [] for i in self.model.anchor_indices}
        self._buffer_trust: list[str] = []
        self._last_logits: Optional[torch.Tensor] = None

    def _fresh_memory(self) -> dict[int, MemoryState]:
        mem = self.model.init_memory(1)
        for s in mem.values():
            s.meta.update({"model_version": self.model_version, "tokenizer_version": self.tokenizer_version})
        return mem

    def _alive(self) -> None:
        if self.deleted:
            raise SessionDeletedError(f"session {self.session_id} was deleted")
        self.last_active = self.clock()
        if self.cache is None and self.log.head(self.branch_id) is not None and self._has_content():
            self.replay()  # state was reclaimed; rebuild from the authoritative log

    def _has_content(self) -> bool:
        return any(e.kind == "ingest" or (e.kind == "note" and e.payload.get("role") == "assistant")
                   for e in self.log.replay(self.branch_id))

    @property
    def tokens(self) -> int:
        return 0 if self.cache is None else self.cache.next_pos

    # ----------------------------------------------------------------- ingest
    @torch.no_grad()
    def _process(self, ids: list[int], trust: Trust, allow_memory_write: bool) -> list[_MemoryUpdate]:
        x = torch.tensor([ids], dtype=torch.long, device=self.model.embed.weight.device)
        out = self.model(x, cache=self.cache, use_cache=True, memory=self.memory, memory_enabled=self.memory_enabled)
        self.cache = out.cache
        self._last_logits = out.logits[:, -1]
        updates: list[_MemoryUpdate] = []
        if not self.memory_enabled:
            return updates
        for i in self.model.anchor_indices:
            self._buffer[i].append(out.anchor_hidden[i][0])
        self._buffer_trust.extend([trust.value] * len(ids))
        S = self.model.cfg.memory.update_segment_tokens
        while len(self._buffer_trust) >= S:
            seg_trust = self._buffer_trust[:S]
            self._buffer_trust = self._buffer_trust[S:]
            segs = {}
            for i in self.model.anchor_indices:
                h = torch.cat(self._buffer[i], dim=0)
                segs[i], rest = h[:S], h[S:]
                self._buffer[i] = [rest] if rest.shape[0] else []
            untrusted = any(not Trust(t).may_write_persistent_memory for t in seg_trust)
            if not allow_memory_write or untrusted:
                reason = "untrusted_content_quarantined" if untrusted else "write_not_allowed"
                updates.append(_MemoryUpdate(list(segs), False, reason))
                continue
            info = {}
            for i, h in segs.items():
                self.memory[i], inf = self.model.memory[str(i)].update(self.memory[i], h.unsqueeze(0).detach())
                info[i] = {k: float(v.mean()) for k, v in inf.items()}
            updates.append(_MemoryUpdate(list(segs), True, "segment_boundary", info))
        return updates

    def ingest(
        self,
        token_ids: Sequence[int],
        *,
        trust: Trust = Trust.USER,
        asset_refs: Sequence[str] = (),
        preprocessing: Optional[dict] = None,
        allow_memory_write: bool = True,
    ) -> Event:
        """Append content to the session. Returns the authoritative ingest event."""
        self._alive()
        ids = [int(t) for t in token_ids]
        if not ids:
            raise ValueError("empty ingest")
        updates = self._process(ids, trust, allow_memory_write)
        ev = self.log.append_ingest(
            segment_hash=content_hash(ids), asset_refs=asset_refs, trust=trust, preprocessing=preprocessing,
            branch_id=self.branch_id, extra={"tokens": ids, "allow_memory_write": allow_memory_write},
        )
        for u in updates:
            if u.written:
                self.log.append("memory_snapshot", {"parent_event": ev.event_id, "anchors": u.anchor_ids,
                                                    "state_hash": self.memory_hash(), "info": u.info},
                                branch_id=self.branch_id)
            else:
                self.log.append("note", {"memory_write": "skipped", "reason": u.reason, "parent_event": ev.event_id},
                                branch_id=self.branch_id)
        self._checkpoint()
        return ev

    # --------------------------------------------------------------- generate
    @torch.no_grad()
    def _decode(self, n: int, temperature: float, generator: Optional[torch.Generator], eos_id: Optional[int]) -> list[int]:
        if self._last_logits is None:
            raise RuntimeError("nothing to continue from; ingest a prompt first")
        frozen = None if self.memory is None else {i: MemoryState(**{**s.__dict__, "frozen": True}) for i, s in self.memory.items()}
        produced: list[int] = []
        logits = self._last_logits
        for _ in range(n):
            if temperature > 0:
                tok = int(torch.multinomial(torch.softmax(logits[0].float() / temperature, -1), 1, generator=generator))
            else:
                tok = int(logits[0].argmax())
            produced.append(tok)
            out = self.model.step(torch.tensor([[tok]], device=logits.device), self.cache, memory=frozen,
                                  memory_enabled=self.memory_enabled)
            self.cache = out.cache
            logits = out.logits[:, -1]
            if eos_id is not None and tok == eos_id:
                break
        self._last_logits = logits
        return produced

    def generate(self, max_new_tokens: int, *, temperature: float = 0.0, generator: Optional[torch.Generator] = None,
                 eos_id: Optional[int] = None) -> list[int]:
        """Decode with neural memory frozen. Output is unverified prose: it is logged as
        untrusted and never feeds the memory-write buffer."""
        self._alive()
        produced = self._decode(max_new_tokens, temperature, generator, eos_id)
        self.log.append("note", {"role": "assistant", "tokens": produced}, branch_id=self.branch_id, trust=Trust.UNTRUSTED)
        self._checkpoint()
        return produced

    # -------------------------------------------------------------- branching
    def _checkpoint(self) -> None:
        head = self.log.head(self.branch_id)
        if head is None:
            return
        self.checkpoints[head.event_id] = Checkpoint(
            head.event_id,
            None if self.cache is None else self.cache.clone(),
            _clone_mem(self.memory),
            {i: [t.clone() for t in v] for i, v in self._buffer.items()},
            list(self._buffer_trust),
            None if self._last_logits is None else self._last_logits.clone(),
        )

    def _restore(self, cp: Checkpoint) -> None:
        self.cache = None if cp.cache is None else cp.cache.clone()
        self.memory = _clone_mem(cp.memory)
        self._buffer = {i: [t.clone() for t in v] for i, v in cp.buffer.items()}
        self._buffer_trust = list(cp.buffer_trust)
        self._last_logits = None if cp.last_logits is None else cp.last_logits.clone()

    def fork(self, parent_event_id: str, reason: str, *, name: Optional[str] = None, tenant_id: Optional[str] = None) -> "Session":
        """Create a branch at a committed event with copy-on-write state."""
        if tenant_id is not None and tenant_id != self.tenant_id:
            raise TenantIsolationError("cannot fork another tenant's session")
        if self.deleted:
            raise SessionDeletedError("cannot fork a deleted session")
        self.log.get(parent_event_id, tenant_id=self.tenant_id)
        branch = self.log.create_branch(parent_event_id, reason, name)
        child = Session(self.model, self.tenant_id, self.session_id, event_log=self.log, branch_id=branch,
                        tokenizer_version=self.tokenizer_version, memory_enabled=self.memory_enabled, clock=self.clock,
                        ttl_seconds=self.ttl, _checkpoints=self.checkpoints)
        cp = self.checkpoints.get(parent_event_id)
        if cp is not None:
            child._restore(cp)
        else:
            child.replay()
        child._checkpoint()
        return child

    @torch.no_grad()
    def replay(self) -> None:
        """Rebuild working context and memory from this branch's event lineage."""
        self._reset_state()
        for ev in self.log.replay(self.branch_id):
            if ev.kind == "ingest":
                self._process(ev.payload["tokens"], Trust(ev.trust), ev.payload.get("allow_memory_write", True))
            elif ev.kind == "note" and ev.payload.get("role") == "assistant" and ev.payload.get("tokens"):
                self._replay_generated(ev.payload["tokens"])

    def _replay_generated(self, tokens: list[int]) -> None:
        frozen = None if self.memory is None else {i: MemoryState(**{**s.__dict__, "frozen": True}) for i, s in self.memory.items()}
        x = torch.tensor([tokens], device=self.model.embed.weight.device)
        out = self.model(x, cache=self.cache, use_cache=True, memory=frozen, memory_enabled=self.memory_enabled)
        self.cache = out.cache
        self._last_logits = out.logits[:, -1]

    # ------------------------------------------------------------ governance
    def memory_hash(self) -> Optional[str]:
        if self.memory is None:
            return None
        return content_hash([content_hash(s.W1.detach().float().cpu().numpy().tobytes()) + content_hash(s.W2.detach().float().cpu().numpy().tobytes())
                             for _, s in sorted(self.memory.items())])

    def snapshot_memory(self) -> dict:
        if self.memory is None:
            raise RuntimeError("memory is disabled for this session")
        head = self.log.head(self.branch_id)
        return {
            "model_version": self.model_version,
            "tokenizer_version": self.tokenizer_version,
            "event_id": None if head is None else head.event_id,
            "tenant_id": self.tenant_id,
            "states": _clone_mem(self.memory),
        }

    def load_memory_snapshot(self, snap: dict) -> None:
        if snap.get("tenant_id") != self.tenant_id:
            raise TenantIsolationError("memory snapshot belongs to another tenant")
        if snap.get("model_version") != self.model_version or snap.get("tokenizer_version") != self.tokenizer_version:
            raise IncompatibleSnapshotError("memory snapshot was produced by a different model/tokenizer version")
        self.memory = _clone_mem(snap["states"])

    def reset_memory(self) -> None:
        if self.memory_enabled:
            self.memory = self._fresh_memory()
            self.log.append("note", {"memory_reset": True}, branch_id=self.branch_id)

    def expired(self, now: Optional[float] = None) -> bool:
        return ((now if now is not None else self.clock()) - self.last_active) > self.ttl

    def reclaim(self) -> None:
        """Evict accelerators (cache, memory, checkpoints). The log remains; state replays on next use."""
        self.cache = None
        self.memory = self._fresh_memory() if self.memory_enabled else None
        self._last_logits = None
        for eid in [e for e, cp in self.checkpoints.items() if self.log.is_ancestor(e, self.log.head(self.branch_id).event_id)]:
            del self.checkpoints[eid]

    def delete(self, reason: str = "user_request") -> Event:
        """Remove all three state forms: working context, neural memory, event payloads."""
        ids = [e.event_id for e in self.log.replay(self.branch_id) if not self.log.is_deleted(e.event_id)]
        tomb = self.log.delete(ids, reason, tenant_id=self.tenant_id)
        self.cache = None
        self.memory = None
        self._buffer = {i: [] for i in self.model.anchor_indices}
        self._buffer_trust = []
        self._last_logits = None
        for eid in ids:
            self.checkpoints.pop(eid, None)
        self.deleted = True
        return tomb

    def verify_deleted(self) -> bool:
        if not self.deleted or self.cache is not None or self.memory is not None:
            return False
        evs = [e for e in self.log.replay(self.branch_id, include_deleted=True) if e.kind != "delete"]
        return all(self.log.verify_deleted(e.event_id) for e in evs) and not any(e in self.checkpoints for e in (x.event_id for x in evs))

    def state_bytes(self) -> dict[str, int]:
        mem = 0 if self.memory is None else sum(s.weight_and_momentum_bytes() for s in self.memory.values())
        return {
            "kv": 0 if self.cache is None else self.cache.kv_bytes(),
            "recurrent": 0 if self.cache is None else self.cache.recurrent_bytes(),
            "memory": mem,
        }


class SessionManager:
    """Tenant-scoped registry with a TTL reclaimer."""

    def __init__(self, model: M63Omni, clock: Callable[[], float] = time.time) -> None:
        self.model, self.clock = model, clock
        self._sessions: dict[tuple[str, str], Session] = {}

    def open(self, tenant_id: str, session_id: str, **kw) -> Session:
        key = (tenant_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = Session(self.model, tenant_id, session_id, clock=self.clock, **kw)
        return self._sessions[key]

    def get(self, tenant_id: str, session_id: str) -> Session:
        for (t, s), sess in self._sessions.items():
            if s == session_id and t != tenant_id:
                raise TenantIsolationError("session belongs to another tenant")
        return self._sessions[(tenant_id, session_id)]

    def reclaim_idle(self) -> list[tuple[str, str]]:
        now = self.clock()
        out = []
        for key, s in self._sessions.items():
            if not s.deleted and s.cache is not None and s.expired(now):
                s.reclaim()
                out.append(key)
        return out
