"""Multimodal prefix cache keyed for safety, not just speed (blueprint §14).

A prefix key covers the model version, tokenizer version, tenant, trust policy,
and the content hash of the multimodal prefix (asset hashes, preprocessing
configuration, timestamps). Reuse is allowed only on an *exact* key match at a
*verified recurrent-state checkpoint boundary*: a matching attention-KV prefix
alone is not enough, because GDN state at an arbitrary token is not
recoverable from KV (§14). A match never crosses tenants or assets.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..common import content_hash


@dataclass(frozen=True)
class PrefixKey:
    tenant_id: str
    model_version: str
    tokenizer_version: str
    trust_policy: str
    prefix_hash: str  # hash over token IDs + asset hashes + preprocessing + timestamps
    boundary_tokens: int  # checkpointed boundary length
    loop_passes: int = 1  # a 1-pass prefix state cannot serve a 2-pass request

    def digest(self) -> str:
        return content_hash([self.tenant_id, self.model_version, self.tokenizer_version, self.trust_policy,
                             self.prefix_hash, self.boundary_tokens, self.loop_passes])


class PrefixCache:
    def __init__(self, capacity: int = 128, ttl_seconds: float = 3600.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.capacity, self.ttl, self.clock = capacity, ttl_seconds, clock
        self._store: "OrderedDict[str, tuple[PrefixKey, Any, float]]" = OrderedDict()
        self.hits = self.misses = 0

    def put(self, key: PrefixKey, state: Any) -> None:
        d = key.digest()
        self._store[d] = (key, state, self.clock())
        self._store.move_to_end(d)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def get(self, key: PrefixKey) -> Optional[Any]:
        d = key.digest()
        item = self._store.get(d)
        if item is None or item[0] != key or self.clock() - item[2] > self.ttl:
            if item is not None and self.clock() - item[2] > self.ttl:
                del self._store[d]
            self.misses += 1
            return None
        self._store.move_to_end(d)
        self.hits += 1
        return item[1]

    def evict_tenant(self, tenant_id: str) -> int:
        doomed = [d for d, (k, _, _) in self._store.items() if k.tenant_id == tenant_id]
        for d in doomed:
            del self._store[d]
        return len(doomed)

    def __len__(self) -> int:
        return len(self._store)
