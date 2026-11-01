"""Shared primitives: modality IDs, trust labels, canonical hashing.

Data and runtime code share these schemas so that training and serving cannot
drift apart (blueprint §11: "Data and runtime code share the same modality and
trust schemas, preventing training/serving skew").
"""

from __future__ import annotations

import enum
import hashlib
import json
from typing import Any


class Modality(enum.IntEnum):
    """Modality IDs carried alongside every position in the residual stream."""

    TEXT = 0
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3
    TOOL = 4  # tool schemas, receipts, artifact references (always text-encoded)
    MEMORY = 5  # neural-memory read tokens (never produced by the data loader)


class Trust(str, enum.Enum):
    """Trust labels, ordered from most to least privileged.

    Untrusted content (web results, unverified tool output, uploaded documents)
    may inform the current task but can never silently become persistent
    instruction memory or be promoted into a higher-trust role (§6).
    """

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    TOOL_VERIFIED = "tool_verified"
    UNTRUSTED = "untrusted"

    @property
    def rank(self) -> int:
        return _TRUST_RANK[self]

    def at_least(self, other: "Trust") -> bool:
        """True when ``self`` is at least as privileged as ``other``."""
        return self.rank <= other.rank

    @property
    def may_write_persistent_memory(self) -> bool:
        return self is not Trust.UNTRUSTED


_TRUST_RANK = {
    Trust.SYSTEM: 0,
    Trust.DEVELOPER: 1,
    Trust.USER: 2,
    Trust.TOOL_VERIFIED: 3,
    Trust.UNTRUSTED: 4,
}


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding used for every hash in the system."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def content_hash(obj: Any) -> str:
    """``sha256:<hex>`` of bytes, str, or a canonical-JSON-encodable object."""
    if isinstance(obj, bytes):
        data = obj
    elif isinstance(obj, str):
        data = obj.encode("utf-8")
    else:
        data = canonical_json(obj).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def short_id(*parts: Any, n: int = 16) -> str:
    """Stable short identifier derived from arbitrary parts."""
    return hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).hexdigest()[:n]
