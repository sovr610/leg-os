"""Sessions: working context + bounded neural memory + authoritative event log (§6, §14, Appendix A).

State layers (blueprint §6 table):

* working context — :class:`HybridCache` (GDN state + GQA KV), exact for one
  model version, recomputable by replaying the event log;
* session neural memory — per-anchor :class:`MemoryState`, a disposable
  accelerator snapshotted with model/tokenizer/event IDs;
* authoritative event log — :class:`EventLog`, the source of truth. Raw media
  lives in a tenant-scoped content-addressed :class:`ArtifactStore`; events
  carry only typed references (hashes) and preprocessing parameters (§10).

Governance rules implemented here:

* memory updates happen only at ``update_segment_tokens`` boundaries of
  *ingested* content, never during generation (memory is frozen while decoding);
* a segment containing untrusted content never writes persistent memory (it may
  still inform the current task through the working context);
* branches are valid only at committed events; a fork or branch switch restores
  the nearest checkpoint on the branch lineage and replays the remaining events
  from the log — it never invents a partial state;
* delete removes all three forms (plus this session's media blobs) and is verifiable;
* a model or tokenizer upgrade invalidates memory snapshots.

The session can share its :class:`EventLog` with the agent control plane, so
plans, receipts, ingests, and model output form one authoritative history; see
``runtime.planner``.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from ..agent.artifacts import ArtifactStore
from ..common import Modality, Trust, content_hash
from ..model.cache import HybridCache
from ..model.interleave import AVPart, ImagePart, Part, TextPart, build_sequence
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


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Deterministic ``.npy`` encoding (no pickle) used for media blobs."""
    buf = io.BytesIO()
    np.save(buf, t.detach().cpu().contiguous().numpy(), allow_pickle=False)
    return buf.getvalue()


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    return torch.from_numpy(np.load(io.BytesIO(data), allow_pickle=False).copy())


def _least_trusted(labels: Sequence[str]) -> Trust:
    return max((Trust(t) for t in labels), key=lambda t: t.rank, default=Trust.USER)


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
        artifact_store: Optional[ArtifactStore] = None,
        announce_attach: bool = False,
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
        self.store = artifact_store if artifact_store is not None else ArtifactStore()
        self._released: dict[str, bool] = {}
        new_log = event_log is None
        self.log = event_log if event_log is not None else EventLog(tenant_id, session_id, clock=clock)
        if self.log.tenant_id != tenant_id:
            raise TenantIsolationError("event log belongs to another tenant")
        self.checkpoints: dict[str, Checkpoint] = _checkpoints if _checkpoints is not None else {}
        self._reset_state()
        meta = {"model_version": self.model_version, "tokenizer_version": tokenizer_version,
                "memory_enabled": self.memory_enabled}
        if new_log:
            self.log.append("session_start", meta, branch_id=branch_id)
        elif announce_attach:
            self.log.append("note", {"runtime_attached": meta}, branch_id=branch_id)

    # ------------------------------------------------------------------ state
    def _reset_state(self) -> None:
        self.cache: Optional[HybridCache] = None
        self.memory = self._fresh_memory() if self.memory_enabled else None
        self._buffer: dict[int, list[torch.Tensor]] = {i: [] for i in self.model.anchor_indices}
        self._buffer_trust: list[str] = []
        self._last_logits: Optional[torch.Tensor] = None
        self._pending_assistant: list[int] = []

    def _fresh_memory(self) -> dict[int, MemoryState]:
        mem = self.model.init_memory(1)
        for s in mem.values():
            s.meta.update({"model_version": self.model_version, "tokenizer_version": self.tokenizer_version})
        return mem

    def _frozen(self) -> Optional[dict[int, MemoryState]]:
        return None if self.memory is None else {i: MemoryState(**{**s.__dict__, "frozen": True}) for i, s in self.memory.items()}

    def _alive(self) -> None:
        if self.deleted:
            raise SessionDeletedError(f"session {self.session_id} was deleted")
        self.last_active = self.clock()
        if self.cache is None and self._has_content():
            self._rebuild(self.branch_id, use_checkpoints=True)  # state was reclaimed; rebuild from the log

    def _has_content(self) -> bool:
        return any(e.kind == "ingest" or (e.kind == "note" and e.payload.get("role") == "assistant")
                   for e in self.log.replay(self.branch_id))

    @property
    def tokens(self) -> int:
        return 0 if self.cache is None else self.cache.next_pos

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def _forward(self, ids: Optional[list[int]] = None, embeds: Optional[torch.Tensor] = None,
                 modality_ids: Optional[torch.Tensor] = None, memory=None):
        dev = self.model.embed.weight.device
        x = None if ids is None else torch.tensor([ids], dtype=torch.long, device=dev)
        out = self.model(x, inputs_embeds=embeds, modality_ids=modality_ids, cache=self.cache, use_cache=True,
                         memory=memory, memory_enabled=self.memory_enabled)
        self.cache = out.cache
        self._last_logits = out.logits[:, -1]
        return out

    @torch.no_grad()
    def _process(self, trust: Sequence[str], allow_memory_write: bool, *, ids: Optional[list[int]] = None,
                 embeds: Optional[torch.Tensor] = None, modality_ids: Optional[torch.Tensor] = None) -> list[_MemoryUpdate]:
        out = self._forward(ids, embeds, modality_ids, memory=self.memory)
        updates: list[_MemoryUpdate] = []
        if not self.memory_enabled:
            return updates
        for i in self.model.anchor_indices:
            self._buffer[i].append(out.anchor_hidden[i][0])
        self._buffer_trust.extend(trust)
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

    def _log_memory_updates(self, ev: Event, updates: list[_MemoryUpdate]) -> None:
        for u in updates:
            if u.written:
                self.log.append("memory_snapshot", {"parent_event": ev.event_id, "anchors": u.anchor_ids,
                                                    "state_hash": self.memory_hash(), "info": u.info},
                                branch_id=self.branch_id)
            else:
                self.log.append("note", {"memory_write": "skipped", "reason": u.reason, "parent_event": ev.event_id},
                                branch_id=self.branch_id)

    # ----------------------------------------------------------------- ingest
    def ingest(
        self,
        token_ids: Sequence[int],
        *,
        trust: Trust = Trust.USER,
        asset_refs: Sequence[str] = (),
        preprocessing: Optional[dict] = None,
        allow_memory_write: bool = True,
    ) -> Event:
        """Append text/tool tokens to the session. Returns the authoritative ingest event."""
        self._alive()
        self.commit_assistant()
        ids = [int(t) for t in token_ids]
        if not ids:
            raise ValueError("empty ingest")
        updates = self._process([trust.value] * len(ids), allow_memory_write, ids=ids)
        ev = self.log.append_ingest(
            segment_hash=content_hash(ids), asset_refs=asset_refs, trust=trust, preprocessing=preprocessing,
            branch_id=self.branch_id, extra={"tokens": ids, "allow_memory_write": allow_memory_write},
        )
        self._log_memory_updates(ev, updates)
        self._checkpoint()
        return ev

    def ingest_media(self, parts: Sequence[Part], *, allow_memory_write: bool = True) -> Event:
        """Encode and append interleaved text/image/audio/video parts (§5).

        Raw tensors go to the artifact store; the ingest event records only blob
        hashes and preprocessing parameters, so replay rebuilds identical state.
        """
        self._alive()
        self.commit_assistant()
        if self.model.multimodal is None:
            raise ValueError("model was built without the multimodal front end")
        specs = [self._store_part(p) for p in parts]
        seq = build_sequence(self.model, parts)
        updates = self._process(seq.trust, allow_memory_write, embeds=seq.inputs_embeds, modality_ids=seq.modality_ids)
        mm = self.model.cfg.multimodal
        ev = self.log.append_ingest(
            segment_hash=seq.content_hash, asset_refs=[h for s in specs for h in _spec_blobs(s)],
            trust=_least_trusted(seq.trust),
            preprocessing={"patch": mm.patch_size, "sr": mm.audio_sample_rate, "hop_ms": mm.hop_ms, "mel_bins": mm.mel_bins},
            branch_id=self.branch_id,
            extra={"multimodal": specs, "positions": seq.length, "allow_memory_write": allow_memory_write},
        )
        self._log_memory_updates(ev, updates)
        self._checkpoint()
        return ev

    def _put(self, kind: str, t: torch.Tensor) -> str:
        return self.store.put_bytes(self.tenant_id, kind, tensor_to_bytes(t), owner=self.session_id)

    def _store_part(self, p: Part) -> dict:
        if isinstance(p, TextPart):
            return {"type": "text", "ids": [int(t) for t in p.ids], "trust": p.trust.value}
        if isinstance(p, ImagePart):
            return {"type": "image", "blob": self._put("image", p.image), "n_tokens": p.n_tokens,
                    "asset_id": p.asset_id, "trust": p.trust.value}
        if isinstance(p, AVPart):
            return {
                "type": "av", "asset_id": p.asset_id,
                "frames": None if p.frames is None else self._put("video_frames", p.frames),
                "frame_times_ms": None if p.frame_times_ms is None else [int(x) for x in p.frame_times_ms.tolist()],
                "audio": None if p.audio is None else self._put("audio", p.audio),
                "tokens_per_frame": p.tokens_per_frame, "window_ms": p.window_ms,
                "hifi": p.high_fidelity_audio, "trust": p.trust.value,
            }
        raise TypeError(f"unknown part {type(p).__name__}")

    def _load_part(self, s: dict) -> Part:
        get = lambda h: bytes_to_tensor(self.store.get_bytes(self.tenant_id, h)[1])  # noqa: E731
        trust = Trust(s["trust"])
        if s["type"] == "text":
            return TextPart(s["ids"], trust)
        if s["type"] == "image":
            return ImagePart(get(s["blob"]), s["n_tokens"], s["asset_id"], trust)
        return AVPart(
            s["asset_id"],
            frames=None if s["frames"] is None else get(s["frames"]),
            frame_times_ms=None if s["frame_times_ms"] is None else torch.tensor(s["frame_times_ms"]),
            audio=None if s["audio"] is None else get(s["audio"]),
            tokens_per_frame=s["tokens_per_frame"], window_ms=s["window_ms"],
            high_fidelity_audio=s["hifi"], trust=trust,
        )

    # ------------------------------------------------- assistant-side decoding
    def next_logits(self) -> torch.Tensor:
        """Logits for the next token given everything in the working context. [1, V]"""
        self._alive()
        if self._last_logits is None:
            raise RuntimeError("nothing to continue from; ingest a prompt first")
        return self._last_logits

    def feed(self, token_ids: Sequence[int]) -> None:
        """Append model-output tokens (memory frozen). They are logged on :meth:`commit_assistant`."""
        self._alive()
        ids = [int(t) for t in token_ids]
        if ids:
            self._forward(ids, memory=self._frozen())
            self._pending_assistant.extend(ids)

    @torch.no_grad()
    def score(self, token_ids: Sequence[int]) -> float:
        """Log-probability of ``token_ids`` as the next tokens, without changing any state."""
        ids = [int(t) for t in token_ids]
        if not ids:
            return 0.0
        logp = torch.log_softmax(self.next_logits()[0].float(), -1)[ids[0]].item()
        if len(ids) > 1:
            x = torch.tensor([ids[:-1]], device=self.model.embed.weight.device)
            out = self.model(x, cache=self.cache, use_cache=True, memory=self._frozen(), memory_enabled=self.memory_enabled)
            lp = torch.log_softmax(out.logits[0].float(), -1)
            logp += float(lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].sum())
        return logp

    def commit_assistant(self, meta: Optional[dict] = None) -> Optional[Event]:
        """Log pending model-output tokens as one untrusted assistant note and checkpoint."""
        if not self._pending_assistant:
            return None
        payload = {"role": "assistant", "tokens": list(self._pending_assistant), **(meta or {})}
        self._pending_assistant = []
        ev = self.log.append("note", payload, branch_id=self.branch_id, trust=Trust.UNTRUSTED)
        self._checkpoint()
        return ev

    @torch.no_grad()
    def _decode(self, n: int, temperature: float, generator: Optional[torch.Generator], eos_id: Optional[int]) -> list[int]:
        produced: list[int] = []
        for _ in range(n):
            logits = self.next_logits()[0]
            if temperature > 0:
                tok = int(torch.multinomial(torch.softmax(logits.float() / temperature, -1), 1, generator=generator))
            else:
                tok = int(logits.argmax())
            produced.append(tok)
            self.feed([tok])
            if eos_id is not None and tok == eos_id:
                break
        return produced

    def generate(self, max_new_tokens: int, *, temperature: float = 0.0, generator: Optional[torch.Generator] = None,
                 eos_id: Optional[int] = None) -> list[int]:
        """Decode with neural memory frozen. Output is unverified prose: it is logged as
        untrusted and never feeds the memory-write buffer."""
        self._alive()
        self.commit_assistant()
        produced = self._decode(max_new_tokens, temperature, generator, eos_id)
        self.commit_assistant()
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
        self._pending_assistant = []

    @torch.no_grad()
    def _rebuild(self, branch_id: str, *, use_checkpoints: bool) -> None:
        """Restore the nearest checkpoint on ``branch_id``'s lineage, then replay later events."""
        lineage = list(self.log.replay(branch_id, tenant_id=self.tenant_id))
        start = 0
        self._reset_state()
        if use_checkpoints:
            for i in range(len(lineage) - 1, -1, -1):
                cp = self.checkpoints.get(lineage[i].event_id)
                if cp is not None:
                    self._restore(cp)
                    start = i + 1
                    break
        for ev in lineage[start:]:
            self._apply(ev)

    def _apply(self, ev: Event) -> None:
        if ev.kind == "ingest" and "multimodal" in ev.payload:
            seq = build_sequence(self.model, [self._load_part(s) for s in ev.payload["multimodal"]])
            self._process(seq.trust, ev.payload.get("allow_memory_write", True),
                          embeds=seq.inputs_embeds, modality_ids=seq.modality_ids)
        elif ev.kind == "ingest" and "tokens" in ev.payload:
            ids = ev.payload["tokens"]
            self._process([ev.trust] * len(ids), ev.payload.get("allow_memory_write", True), ids=ids)
        elif ev.kind == "note" and ev.payload.get("role") == "assistant" and ev.payload.get("tokens"):
            self._forward(ev.payload["tokens"], memory=self._frozen())
        elif ev.kind == "note" and ev.payload.get("memory_reset") and self.memory_enabled:
            self.memory = self._fresh_memory()

    def replay(self) -> None:
        """Rebuild working context and memory from this branch's event lineage alone (no checkpoints)."""
        self._rebuild(self.branch_id, use_checkpoints=False)

    def attach_branch(self, branch_id: str) -> None:
        """Adopt an existing branch (e.g. one the control plane created for undo/repair)."""
        if self.deleted:
            raise SessionDeletedError("session was deleted")
        self.commit_assistant()
        self.branch_id = branch_id
        self._rebuild(branch_id, use_checkpoints=True)
        self._checkpoint()

    def fork(self, parent_event_id: str, reason: str, *, name: Optional[str] = None, tenant_id: Optional[str] = None) -> "Session":
        """Create a branch at a committed event with copy-on-write state."""
        if tenant_id is not None and tenant_id != self.tenant_id:
            raise TenantIsolationError("cannot fork another tenant's session")
        if self.deleted:
            raise SessionDeletedError("cannot fork a deleted session")
        self.commit_assistant()
        self.log.get(parent_event_id, tenant_id=self.tenant_id)
        branch = self.log.create_branch(parent_event_id, reason, name, tenant_id=self.tenant_id)
        child = Session(self.model, self.tenant_id, self.session_id, event_log=self.log, branch_id=branch,
                        tokenizer_version=self.tokenizer_version, memory_enabled=self.memory_enabled, clock=self.clock,
                        ttl_seconds=self.ttl, artifact_store=self.store, _checkpoints=self.checkpoints)
        child._rebuild(branch, use_checkpoints=True)
        child._checkpoint()
        return child

    # ------------------------------------------------------------ governance
    def memory_hash(self) -> Optional[str]:
        if self.memory is None:
            return None
        return content_hash([content_hash(s.W1.detach().float().cpu().numpy().tobytes())
                             + content_hash(s.W2.detach().float().cpu().numpy().tobytes())
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
            self._checkpoint()

    def expired(self, now: Optional[float] = None) -> bool:
        return ((now if now is not None else self.clock()) - self.last_active) > self.ttl

    def reclaim(self) -> None:
        """Evict accelerators (cache, memory, checkpoints). The log remains; state replays on next use."""
        self.commit_assistant()
        self.cache = None
        self.memory = self._fresh_memory() if self.memory_enabled else None
        self._last_logits = None
        head = self.log.head(self.branch_id)
        if head is not None:
            for eid in [e for e in self.checkpoints if self.log.is_ancestor(e, head.event_id)]:
                del self.checkpoints[eid]

    def delete(self, reason: str = "user_request") -> Event:
        """Remove all three state forms and this session's media blobs."""
        self._pending_assistant = []
        lineage = [e for e in self.log.replay(self.branch_id) if not self.log.is_deleted(e.event_id)]
        blobs = {h for e in lineage if e.kind == "ingest" for s in e.payload.get("multimodal", []) for h in _spec_blobs(s)}
        tomb = self.log.delete([e.event_id for e in lineage], reason, tenant_id=self.tenant_id)
        for h in blobs:
            self._released[h] = self.store.release(self.tenant_id, h, self.session_id)
        self.cache = None
        self.memory = None
        self._buffer = {i: [] for i in self.model.anchor_indices}
        self._buffer_trust = []
        self._last_logits = None
        for e in lineage:
            self.checkpoints.pop(e.event_id, None)
        self.deleted = True
        return tomb

    def verify_deleted(self) -> bool:
        if not self.deleted or self.cache is not None or self.memory is not None:
            return False
        evs = [e for e in self.log.replay(self.branch_id, include_deleted=True) if e.kind != "delete"]
        logs_ok = all(self.log.verify_deleted(e.event_id) for e in evs) and not any(e.event_id in self.checkpoints for e in evs)
        media_ok = all(not self.store.contains(self.tenant_id, h) for h, purged in self._released.items() if purged)
        return logs_ok and media_ok

    def state_bytes(self) -> dict[str, int]:
        mem = 0 if self.memory is None else sum(s.weight_and_momentum_bytes() for s in self.memory.values())
        return {
            "kv": 0 if self.cache is None else self.cache.kv_bytes(),
            "recurrent": 0 if self.cache is None else self.cache.recurrent_bytes(),
            "memory": mem,
        }


def _spec_blobs(spec: dict) -> list[str]:
    if spec.get("type") == "image":
        return [spec["blob"]]
    if spec.get("type") == "av":
        return [h for h in (spec.get("frames"), spec.get("audio")) if h]
    return []


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


__all__ = ["Session", "SessionManager", "Checkpoint", "IncompatibleSnapshotError", "SessionDeletedError",
           "tensor_to_bytes", "bytes_to_tensor", "Modality"]
