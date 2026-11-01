"""Tenant-scoped, content-addressed artifact storage (blueprint §6, §10).

"Raw media stays in artifact storage; the model receives typed references and
compact representations" (§10). Blobs here are immutable and addressed by
``content_hash``; a tool never mutates a blob, it returns a new version and the
control plane stores it. Which version of a named artifact (``asset_91``) is
*current* on a branch is not stored here at all: it is derived from the
authoritative event log (ingest events and verified commits), so undo by
branching automatically restores earlier versions.

Storage is partitioned per tenant with no cross-tenant deduplication, so a
tenant can neither read nor probe for another tenant's content (a lookup of a
foreign hash is indistinguishable from a missing one).
"""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional

from ..common import canonical_json, content_hash

__all__ = ["ArtifactRef", "ArtifactNotFound", "ArtifactStore"]


class ArtifactNotFound(KeyError):
    """No blob with this hash exists *for this tenant*."""


@dataclass(frozen=True)
class ArtifactRef:
    """Typed reference to one version of a named artifact, as shown to the model."""

    artifact_id: str
    kind: str
    hash: str
    trust: str
    event_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "hash": self.hash,
            "trust": self.trust,
            "event_id": self.event_id,
        }


class ArtifactStore:
    """In-memory content-addressed blob store partitioned by tenant.

    Holds JSON blobs (tool artifacts) and raw binary blobs (media). Binary blobs
    carry an owner set so a session can release its media on deletion without
    breaking another session of the same tenant that ingested identical bytes.
    """

    def __init__(self) -> None:
        self._blobs: dict[tuple[str, str], tuple[str, str]] = {}
        self._bytes: dict[tuple[str, str], tuple[str, bytes]] = {}
        self._owners: dict[tuple[str, str], set[str]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ binary blobs
    def put_bytes(self, tenant_id: str, kind: str, data: bytes, *, owner: Optional[str] = None) -> str:
        """Store raw bytes (idempotent); returns the ``sha256:`` hash of the bytes."""
        h = content_hash(bytes(data))
        with self._lock:
            self._bytes.setdefault((tenant_id, h), (kind, bytes(data)))
            if owner is not None:
                self._owners.setdefault((tenant_id, h), set()).add(owner)
        return h

    def get_bytes(self, tenant_id: str, blob_hash: str) -> tuple[str, bytes]:
        with self._lock:
            entry = self._bytes.get((tenant_id, blob_hash))
        if entry is None:
            raise ArtifactNotFound(f"artifact {blob_hash} not found for tenant {tenant_id!r}")
        return entry

    def release(self, tenant_id: str, blob_hash: str, owner: str) -> bool:
        """Drop ``owner``'s reference; purge the bytes when no owner remains. Returns True if purged."""
        with self._lock:
            owners = self._owners.get((tenant_id, blob_hash))
            if owners is not None:
                owners.discard(owner)
                if owners:
                    return False
                del self._owners[(tenant_id, blob_hash)]
            return self._bytes.pop((tenant_id, blob_hash), None) is not None

    def put(self, tenant_id: str, kind: str, content: Any) -> str:
        """Store a JSON-serialisable blob; returns its ``sha256:`` hash (idempotent)."""
        encoded = canonical_json(content)
        h = content_hash(json.loads(encoded))
        with self._lock:
            self._blobs.setdefault((tenant_id, h), (kind, encoded))
        return h

    def get(self, tenant_id: str, blob_hash: str) -> tuple[str, Any]:
        """Return ``(kind, content)`` for a blob owned by ``tenant_id``."""
        with self._lock:
            entry = self._blobs.get((tenant_id, blob_hash))
        if entry is None:
            raise ArtifactNotFound(f"artifact {blob_hash} not found for tenant {tenant_id!r}")
        kind, encoded = entry
        return kind, json.loads(encoded)

    def contains(self, tenant_id: str, blob_hash: str) -> bool:
        with self._lock:
            return (tenant_id, blob_hash) in self._blobs or (tenant_id, blob_hash) in self._bytes

    def delete_tenant(self, tenant_id: str) -> int:
        """Purge every blob of a tenant (deletion policy, §6); returns the count removed."""
        with self._lock:
            keys = [k for k in self._blobs if k[0] == tenant_id]
            for k in keys:
                del self._blobs[k]
            bkeys = [k for k in self._bytes if k[0] == tenant_id]
            for k in bkeys:
                del self._bytes[k]
                self._owners.pop(k, None)
        return len(keys) + len(bkeys)

    def snapshot(self, tenant_id: str, blob_hash: str) -> Any:
        """Deep copy of the content only."""
        return copy.deepcopy(self.get(tenant_id, blob_hash)[1])
