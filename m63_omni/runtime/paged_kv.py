"""Paged KV storage with reference counting and copy-on-write (blueprint §11, §14).

KV pages hold ``block_tokens`` positions for one attention layer. Sequences own
page tables; forking a sequence shares every page (refcount++) and marks them
read-only; the first append into a shared tail page copies it. Pages are
reclaimed when their refcount reaches zero. Tenant IDs are recorded per
sequence, and a sequence can only be forked within its tenant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


class PageAllocatorError(RuntimeError):
    pass


@dataclass
class _Seq:
    tenant_id: str
    pages: list[int] = field(default_factory=list)
    length: int = 0


class PagedKVStore:
    """One store per attention layer group. Tensors: [num_pages, 2 (K/V), Hkv, block, D]."""

    def __init__(self, num_pages: int, block_tokens: int, kv_heads: int, head_dim: int,
                 dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> None:
        self.block = block_tokens
        self.pages = torch.zeros(num_pages, 2, kv_heads, block_tokens, head_dim, dtype=dtype, device=device)
        self.refcount = [0] * num_pages
        self.free = list(range(num_pages - 1, -1, -1))
        self.seqs: dict[str, _Seq] = {}
        self.copies = 0  # number of copy-on-write page copies performed

    # ---------------------------------------------------------- allocation
    def _alloc(self) -> int:
        if not self.free:
            raise PageAllocatorError("out of KV pages")
        p = self.free.pop()
        self.refcount[p] = 1
        return p

    def _release(self, p: int) -> None:
        self.refcount[p] -= 1
        if self.refcount[p] == 0:
            self.free.append(p)

    @property
    def pages_in_use(self) -> int:
        return sum(1 for r in self.refcount if r > 0)

    # ----------------------------------------------------------- sequences
    def create(self, seq_id: str, tenant_id: str) -> None:
        if seq_id in self.seqs:
            raise PageAllocatorError(f"sequence {seq_id} exists")
        self.seqs[seq_id] = _Seq(tenant_id)

    def fork(self, parent: str, child: str, tenant_id: str) -> None:
        src = self.seqs[parent]
        if src.tenant_id != tenant_id:
            raise PermissionError("cannot fork a sequence across tenants")
        for p in src.pages:
            self.refcount[p] += 1
        self.seqs[child] = _Seq(tenant_id, list(src.pages), src.length)

    def free_seq(self, seq_id: str) -> None:
        s = self.seqs.pop(seq_id)
        for p in s.pages:
            self._release(p)

    def truncate(self, seq_id: str, length: int) -> None:
        """Roll back to ``length`` tokens (MTP rejection / branch rollback)."""
        s = self.seqs[seq_id]
        if length > s.length:
            raise PageAllocatorError("cannot truncate forward")
        keep = -(-length // self.block)
        for p in s.pages[keep:]:
            self._release(p)
        s.pages = s.pages[:keep]
        s.length = length

    def append(self, seq_id: str, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: [Hkv, n, D]."""
        s = self.seqs[seq_id]
        n = k.shape[1]
        i = 0
        while i < n:
            off = s.length % self.block
            if off == 0:
                s.pages.append(self._alloc())
            page = s.pages[-1]
            if self.refcount[page] > 1:  # copy-on-write
                new = self._alloc()
                self.pages[new].copy_(self.pages[page])
                self._release(page)
                s.pages[-1] = new
                page = new
                self.copies += 1
            take = min(self.block - off, n - i)
            self.pages[page, 0, :, off : off + take] = k[:, i : i + take]
            self.pages[page, 1, :, off : off + take] = v[:, i : i + take]
            s.length += take
            i += take

    def gather(self, seq_id: str, tenant_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.seqs[seq_id]
        if s.tenant_id != tenant_id:
            raise PermissionError("cross-tenant KV read")
        if not s.pages:
            shape = (self.pages.shape[2], 0, self.pages.shape[4])
            return self.pages.new_zeros(shape), self.pages.new_zeros(shape)
        kv = self.pages[s.pages]  # [P, 2, H, block, D]
        kv = kv.permute(1, 2, 0, 3, 4).reshape(2, kv.shape[2], -1, kv.shape[4])[:, :, : s.length]
        return kv[0], kv[1]
