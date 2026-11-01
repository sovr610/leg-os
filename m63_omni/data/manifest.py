"""Provenance ledger and immutable data manifests (blueprint §8, §12 step 4, §15, §16).

§8: "every source requires an independent license, consent, privacy, and
takedown review" and "Every batch can be reproduced from a manifest containing
source versions, transformations, seeds, and ordering." §16 requires that
"Source, rights, transformations, dedup, decontamination, and takedown records
are queryable"; §15's response to a takedown is "Remove affected data/code,
regenerate manifests".

* :class:`SourceRecord` — rights/consent/takedown state of one source version.
* :class:`ProvenanceLedger` — append-only, hash-chained history of source
  registrations, rights changes, takedowns, and per-document pipeline decisions
  (admitted / dropped / modified, with stage and reason).
* :class:`DataManifest` — frozen, content-hashed record of sources+versions,
  the ordered transformation list with parameters, seeds, ordering, and
  document entries. Construction REFUSES any source whose rights are not
  ``cleared``, that has a takedown, or whose consent is missing/withdrawn, so an
  invalid manifest cannot exist.
* :func:`batch_order` — deterministic, version-independent ordering derived
  from the manifest alone (SHA-256 keyed permutation).
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..common import Modality, canonical_json, content_hash

__all__ = [
    "RightsStatus",
    "ConsentStatus",
    "TakedownStatus",
    "SourceRecord",
    "LedgerEntry",
    "ProvenanceLedger",
    "TransformRecord",
    "ManifestEntry",
    "DataManifest",
    "ManifestError",
    "batch_order",
    "iter_epochs",
]


class ManifestError(ValueError):
    """Raised when a manifest would include unusable data or fails verification."""


class RightsStatus(str, enum.Enum):
    CLEARED = "cleared"
    PENDING = "pending"
    REJECTED = "rejected"


class ConsentStatus(str, enum.Enum):
    NOT_REQUIRED = "not_required"  # e.g. public-domain or licensed non-personal media
    OBTAINED = "obtained"
    MISSING = "missing"
    WITHDRAWN = "withdrawn"


class TakedownStatus(str, enum.Enum):
    NONE = "none"
    REQUESTED = "requested"
    ENFORCED = "enforced"


@dataclass(frozen=True)
class SourceRecord:
    """Rights and provenance of one version of a data source."""

    source_id: str
    version: str
    license: str
    rights_status: RightsStatus = RightsStatus.PENDING
    consent: ConsentStatus = ConsentStatus.NOT_REQUIRED
    takedown: TakedownStatus = TakedownStatus.NONE
    modality: Modality = Modality.TEXT
    uri: str = ""
    synthetic: bool = False  # synthetic captions/data must be labelled as such (§8)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rights_status", RightsStatus(self.rights_status))
        object.__setattr__(self, "consent", ConsentStatus(self.consent))
        object.__setattr__(self, "takedown", TakedownStatus(self.takedown))
        object.__setattr__(self, "modality", Modality(self.modality))
        if not self.source_id or not self.version:
            raise ManifestError("source_id and version are required")

    def ineligibility_reasons(self) -> list[str]:
        reasons = []
        if self.rights_status is not RightsStatus.CLEARED:
            reasons.append(f"rights_{self.rights_status.value}")
        if self.takedown is not TakedownStatus.NONE:
            reasons.append(f"takedown_{self.takedown.value}")
        if self.consent in (ConsentStatus.MISSING, ConsentStatus.WITHDRAWN):
            reasons.append(f"consent_{self.consent.value}")
        return reasons

    @property
    def eligible(self) -> bool:
        return not self.ineligibility_reasons()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "version": self.version,
            "license": self.license,
            "rights_status": self.rights_status.value,
            "consent": self.consent.value,
            "takedown": self.takedown.value,
            "modality": int(self.modality),
            "uri": self.uri,
            "synthetic": self.synthetic,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SourceRecord":
        return cls(**dict(d))

    def replace(self, **changes: Any) -> "SourceRecord":
        d = self.to_dict()
        d.update(changes)
        return SourceRecord.from_dict(d)


# --------------------------------------------------------------------- ledger
@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    kind: str  # register | update | takedown | decision
    source_id: str
    payload_json: str
    chain_hash: str
    doc_id: str | None = None
    stage: str | None = None
    action: str | None = None
    reason: str | None = None

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)


class ProvenanceLedger:
    """Append-only, hash-chained provenance and lineage ledger (§8, §16)."""

    GENESIS = "sha256:" + "0" * 64

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._sources: dict[str, SourceRecord] = {}
        self._admitted: dict[str, list[str]] = {}

    # ---------------------------------------------------------------- write
    def _append(self, kind: str, source_id: str, payload: Mapping[str, Any], **extra: Any) -> LedgerEntry:
        prev = self._entries[-1].chain_hash if self._entries else self.GENESIS
        pj = canonical_json(payload)
        chain = content_hash({"prev": prev, "kind": kind, "source_id": source_id, "payload": pj, **extra})
        entry = LedgerEntry(len(self._entries), kind, source_id, pj, chain, **extra)
        self._entries.append(entry)
        return entry

    def register(self, record: SourceRecord) -> SourceRecord:
        """Register a source (or a new version of it)."""
        self._sources[record.source_id] = record
        self._append("register", record.source_id, record.to_dict())
        return record

    def update(self, source_id: str, *, reason: str = "", **changes: Any) -> SourceRecord:
        """Change rights/consent/takedown/version fields; the change is logged."""
        rec = self.get(source_id).replace(**changes)
        self._sources[source_id] = rec
        self._append("update", source_id, {"changes": {k: getattr(rec, k) for k in changes}, "reason": reason})
        return rec

    def request_takedown(self, source_id: str, reason: str = "", *, enforced: bool = False) -> SourceRecord:
        """Record a takedown; the source becomes ineligible for every future manifest."""
        status = TakedownStatus.ENFORCED if enforced else TakedownStatus.REQUESTED
        rec = self.get(source_id).replace(takedown=status.value)
        self._sources[source_id] = rec
        self._append("takedown", source_id, {"status": status.value, "reason": reason,
                                             "affected_documents": self.documents_from(source_id)})
        return rec

    def record_decision(
        self, doc_id: str, source_id: str, stage: str, action: str, reason: str = "", detail: Mapping[str, Any] | None = None
    ) -> LedgerEntry:
        """Record a per-document pipeline decision (``admitted``/``dropped``/``modified``)."""
        if action not in ("admitted", "dropped", "modified", "flagged"):
            raise ValueError(f"unknown decision action {action!r}")
        if action == "admitted":
            self._admitted.setdefault(source_id, []).append(doc_id)
        return self._append("decision", source_id, dict(detail or {}), doc_id=doc_id, stage=stage, action=action, reason=reason)

    # ----------------------------------------------------------------- read
    def __contains__(self, source_id: object) -> bool:
        return source_id in self._sources

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, source_id: str) -> SourceRecord:
        try:
            return self._sources[source_id]
        except KeyError:
            raise KeyError(f"unknown source {source_id!r}") from None

    def sources(self) -> list[SourceRecord]:
        return [self._sources[k] for k in sorted(self._sources)]

    def eligible_sources(self) -> list[SourceRecord]:
        return [s for s in self.sources() if s.eligible]

    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def history(self, source_id: str) -> list[LedgerEntry]:
        return [e for e in self._entries if e.source_id == source_id and e.kind != "decision"]

    def decisions(self, *, doc_id: str | None = None, source_id: str | None = None,
                  stage: str | None = None, action: str | None = None) -> list[LedgerEntry]:
        """Query lineage records (§16 "queryable" data lineage)."""
        out = []
        for e in self._entries:
            if e.kind != "decision":
                continue
            if doc_id is not None and e.doc_id != doc_id:
                continue
            if source_id is not None and e.source_id != source_id:
                continue
            if stage is not None and e.stage != stage:
                continue
            if action is not None and e.action != action:
                continue
            out.append(e)
        return out

    def documents_from(self, source_id: str) -> list[str]:
        """Documents admitted from ``source_id`` (what a takedown must remove, §15)."""
        return list(self._admitted.get(source_id, ()))

    def verify_chain(self) -> bool:
        prev = self.GENESIS
        for e in self._entries:
            extra = {k: getattr(e, k) for k in ("doc_id", "stage", "action", "reason") if getattr(e, k) is not None}
            expect = content_hash({"prev": prev, "kind": e.kind, "source_id": e.source_id, "payload": e.payload_json, **extra})
            if expect != e.chain_hash:
                return False
            prev = e.chain_hash
        return True


# ------------------------------------------------------------------- manifest
@dataclass(frozen=True)
class TransformRecord:
    """One applied transformation and its exact parameters (stored as canonical JSON)."""

    name: str
    params_json: str = "{}"
    version: str = "1"

    @classmethod
    def create(cls, name: str, params: Mapping[str, Any] | None = None, version: str = "1") -> "TransformRecord":
        return cls(name, canonical_json(dict(params or {})), version)

    @property
    def params(self) -> dict[str, Any]:
        return json.loads(self.params_json)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "version": self.version}


@dataclass(frozen=True)
class ManifestEntry:
    """One document admitted to the manifest."""

    doc_id: str
    source_id: str
    source_version: str
    content_hash: str
    modality: int = int(Modality.TEXT)
    n_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _freeze_json(obj: Mapping[str, Any] | None) -> str:
    return canonical_json(dict(obj or {}))


@dataclass(frozen=True)
class DataManifest:
    """Immutable, content-hashed training-data manifest (§8).

    Build with :meth:`build`. ``manifest_hash`` covers every field; loading a
    manifest whose stored hash does not match its content raises.
    """

    sources: tuple[SourceRecord, ...]
    transformations: tuple[TransformRecord, ...]
    seed: int
    entries: tuple[ManifestEntry, ...] = ()
    ordering_json: str = '{"batch_size":1,"drop_last":false,"method":"sha256_permutation"}'
    stats_json: str = "{}"
    schema_version: int = 1
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "transformations", tuple(self.transformations))
        object.__setattr__(self, "entries", tuple(self.entries))
        self._validate()
        digest = self.compute_hash()
        if self.manifest_hash and self.manifest_hash != digest:
            raise ManifestError("manifest hash mismatch: content was modified")
        object.__setattr__(self, "manifest_hash", digest)

    # ------------------------------------------------------------ validation
    def _validate(self) -> None:
        problems: list[str] = []
        by_id: dict[str, SourceRecord] = {}
        for s in self.sources:
            if s.source_id in by_id:
                problems.append(f"duplicate source {s.source_id}")
            by_id[s.source_id] = s
            reasons = s.ineligibility_reasons()
            if reasons:
                problems.append(f"source {s.source_id}@{s.version}: {', '.join(reasons)}")
        seen_docs: set[str] = set()
        for e in self.entries:
            if e.doc_id in seen_docs:
                problems.append(f"duplicate document {e.doc_id}")
            seen_docs.add(e.doc_id)
            src = by_id.get(e.source_id)
            if src is None:
                problems.append(f"document {e.doc_id} references unlisted source {e.source_id}")
            elif src.version != e.source_version:
                problems.append(f"document {e.doc_id} pins {e.source_id}@{e.source_version}, manifest has {src.version}")
        if not self.transformations:
            problems.append("manifest must list the applied transformations")
        ordering = self.ordering
        if int(ordering.get("batch_size", 1)) < 1:
            problems.append("ordering.batch_size must be >= 1")
        if problems:
            raise ManifestError("refusing to build manifest: " + "; ".join(problems))

    # -------------------------------------------------------------- accessors
    @property
    def ordering(self) -> dict[str, Any]:
        return json.loads(self.ordering_json)

    @property
    def stats(self) -> dict[str, Any]:
        return json.loads(self.stats_json)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(e.doc_id for e in self.entries)

    def source(self, source_id: str) -> SourceRecord:
        for s in self.sources:
            if s.source_id == source_id:
                return s
        raise KeyError(source_id)

    def _content(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "sources": [s.to_dict() for s in self.sources],
            "transformations": [t.to_dict() for t in self.transformations],
            "ordering": self.ordering,
            "entries": [e.to_dict() for e in self.entries],
            "stats": self.stats,
        }

    def compute_hash(self) -> str:
        return content_hash(self._content())

    def verify(self) -> bool:
        return self.compute_hash() == self.manifest_hash

    # --------------------------------------------------------- construction
    @classmethod
    def build(
        cls,
        sources: Iterable[SourceRecord],
        transformations: Sequence[TransformRecord | tuple[str, Mapping[str, Any]]],
        seed: int,
        entries: Iterable[ManifestEntry] = (),
        *,
        ordering: Mapping[str, Any] | None = None,
        stats: Mapping[str, Any] | None = None,
    ) -> "DataManifest":
        """Build a manifest; sources are sorted by id for a canonical hash."""
        trs = tuple(t if isinstance(t, TransformRecord) else TransformRecord.create(t[0], t[1]) for t in transformations)
        order = {"method": "sha256_permutation", "batch_size": 1, "drop_last": False}
        order.update(dict(ordering or {}))
        return cls(
            sources=tuple(sorted(sources, key=lambda s: s.source_id)),
            transformations=trs,
            seed=int(seed),
            entries=tuple(entries),
            ordering_json=_freeze_json(order),
            stats_json=_freeze_json(stats),
        )

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        d = self._content()
        d["manifest_hash"] = self.manifest_hash
        return d

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DataManifest":
        return cls(
            sources=tuple(SourceRecord.from_dict(s) for s in d["sources"]),
            transformations=tuple(TransformRecord.create(t["name"], t["params"], t.get("version", "1")) for t in d["transformations"]),
            seed=int(d["seed"]),
            entries=tuple(ManifestEntry(**e) for e in d["entries"]),
            ordering_json=_freeze_json(d.get("ordering")),
            stats_json=_freeze_json(d.get("stats")),
            schema_version=int(d.get("schema_version", 1)),
            manifest_hash=str(d.get("manifest_hash", "")),
        )

    @classmethod
    def from_json(cls, text: str) -> "DataManifest":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DataManifest":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _order_key(manifest: DataManifest, epoch: int, doc_id: str) -> bytes:
    return hashlib.sha256(f"{manifest.manifest_hash}|{manifest.seed}|{epoch}|{doc_id}".encode("utf-8")).digest()


def batch_order(manifest: DataManifest, epoch: int) -> list[tuple[str, ...]]:
    """Deterministic batches of document IDs for ``epoch``, from the manifest alone.

    Documents are ordered by ``sha256(manifest_hash | seed | epoch | doc_id)``
    (independent of Python/numpy RNG versions), then grouped into batches of
    ``ordering["batch_size"]``; a short final batch is kept unless
    ``ordering["drop_last"]``.
    """
    if epoch < 0:
        raise ValueError("epoch must be >= 0")
    ordering = manifest.ordering
    if ordering.get("method", "sha256_permutation") != "sha256_permutation":
        raise ManifestError(f"unsupported ordering method {ordering.get('method')!r}")
    bs = int(ordering.get("batch_size", 1))
    ids = sorted(manifest.doc_ids, key=lambda d: _order_key(manifest, epoch, d))
    batches = [tuple(ids[i : i + bs]) for i in range(0, len(ids), bs)]
    if ordering.get("drop_last") and batches and len(batches[-1]) < bs:
        batches.pop()
    return batches


def iter_epochs(manifest: DataManifest, epochs: int) -> Iterator[tuple[int, list[tuple[str, ...]]]]:
    """``(epoch, batch_order(manifest, epoch))`` for ``epochs`` epochs."""
    for ep in range(epochs):
        yield ep, batch_order(manifest, ep)
