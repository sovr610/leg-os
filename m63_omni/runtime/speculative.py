"""Target-verified MTP speculative decoding with complete state rollback (§14, Appendix A).

The MTP module only *proposes* tokens. The target model verifies every draft;
rejected drafts roll back every recurrent, attention, and neural-memory state by
restoring the pre-verification snapshot and replaying the accepted tokens, so
the cache always reflects exactly the visible transcript. A target-only path is
always available (``draft_tokens=0``) and must produce identical greedy output.

Speed must be reported as completed-response latency under mixed traffic, not
accepted tokens per step (§14); :class:`SpecStats` records both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from ..model.cache import HybridCache, KVLayerState, resolve_stream
from ..model.m63 import M63Omni
from ..model.memory import MemoryState


@dataclass
class SpecStats:
    steps: int = 0
    drafted: int = 0
    accepted: int = 0
    rollbacks: int = 0
    target_forwards: int = 0
    mtp_positions: int = 0  # MTP positions computed (the work the MTP cache removes)
    accept_history: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0


def _sample(logits: torch.Tensor, temperature: float, gen: Optional[torch.Generator]) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=gen).squeeze(-1)


@dataclass
class MTPCache:
    """MTP attention KV for *committed* positions only (both their hidden state and next token are final).

    Speculative positions are computed functionally on top of it and never stored,
    so a rejected draft needs no MTP rollback.
    """

    state: Optional[KVLayerState] = None
    n: int = 0


def _stream_at(n: int, length: int, device: torch.device):
    if n == 0:
        return resolve_stream(1, length, device)
    pseudo = HybridCache(layers=[], next_pos=n, last_seg=torch.zeros(1, dtype=torch.long, device=device),
                         next_rope=torch.tensor([n], device=device))
    return resolve_stream(1, length, device, pseudo)


def _draft_cached(model: M63Omni, mc: MTPCache, new_hidden: torch.Tensor, new_next: list[int], k: int,
                  temperature: float, gen: Optional[torch.Generator]) -> tuple[list[int], list[Optional[torch.Tensor]]]:
    """Commit ``new_hidden`` positions to the MTP cache, then draft ``k`` tokens recursively.

    Equivalent to :func:`_draft` (full recompute) but O(new positions) per step.
    """
    dev = new_hidden.device
    nxt = torch.tensor([new_next], device=dev)
    mh, mc.state = model.mtp(new_hidden.unsqueeze(0), model.embed(nxt), _stream_at(mc.n, new_hidden.shape[0], dev), mc.state)
    mc.n += new_hidden.shape[0]
    h_prev, state, n = mh[:, -1:], mc.state, mc.n
    drafts: list[int] = []
    qs: list[Optional[torch.Tensor]] = []
    for j in range(k):
        logits = model.logits(h_prev[:, -1])[0]
        tok = _sample(logits, temperature, gen)
        drafts.append(int(tok))
        qs.append(torch.softmax(logits.float() / temperature, -1) if temperature > 0 else None)
        if j == k - 1:
            break
        h_prev, state = model.mtp(h_prev, model.embed(tok.view(1, 1)), _stream_at(n, 1, dev), state)
        n += 1
    return drafts, qs


@torch.no_grad()
def generate(
    model: M63Omni,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    draft_tokens: int = 0,
    temperature: float = 0.0,
    memory: Optional[dict[int, MemoryState]] = None,
    memory_enabled: bool = True,
    cache: Optional[HybridCache] = None,
    generator: Optional[torch.Generator] = None,
    eos_id: Optional[int] = None,
    use_mtp_cache: bool = True,
) -> tuple[torch.Tensor, HybridCache, SpecStats]:
    """Generate for a single sequence (B=1). Returns (new token IDs [n], final cache, stats).

    Neural memory is read-only throughout (states are marked frozen).
    ``use_mtp_cache=False`` selects the full-recompute drafter (the reference).
    """
    if prompt_ids.shape[0] != 1:
        raise ValueError("reference speculative decoder supports batch size 1")
    if draft_tokens and model.mtp is None:
        raise ValueError("model has no MTP module")
    frozen = None
    if memory is not None:
        frozen = {i: MemoryState(**{**s.__dict__, "frozen": True}) for i, s in memory.items()}
    stats = SpecStats()
    out = model.forward(prompt_ids, cache=cache, use_cache=True, memory=frozen, memory_enabled=memory_enabled)
    stats.target_forwards += 1
    cache = out.cache
    if draft_tokens and cache is not None and cache.next_pos != prompt_ids.shape[1]:
        raise ValueError("speculative drafting needs the full transcript; pass the whole prompt without a prior cache")
    hist_ids = list(prompt_ids[0].tolist())
    hist_h = [out.hidden_raw[0]]
    next_tok = _sample(out.logits[0, -1], temperature, generator)
    mtp_cache = MTPCache()
    produced: list[int] = []
    while len(produced) < max_new_tokens:
        produced.append(int(next_tok))
        if eos_id is not None and int(next_tok) == eos_id:
            break
        if len(produced) >= max_new_tokens:
            # commit the final token to the cache so the cache matches the transcript
            o = model.forward(next_tok.view(1, 1), cache=cache, use_cache=True, memory=frozen, memory_enabled=memory_enabled)
            stats.target_forwards += 1
            cache = o.cache
            hist_ids.append(int(next_tok))
            break
        k = min(draft_tokens, max_new_tokens - len(produced))
        if use_mtp_cache:
            H = torch.cat(hist_h, 0)
            new_next = hist_ids[mtp_cache.n + 1 :] + [int(next_tok)]
            stats.mtp_positions += (H.shape[0] - mtp_cache.n) + max(k - 1, 0)
            drafts, q_probs = _draft_cached(model, mtp_cache, H[mtp_cache.n :], new_next, k, temperature, generator)
        else:
            L = sum(h.shape[0] for h in hist_h)
            stats.mtp_positions += sum(L + j for j in range(k))
            drafts, q_probs = _draft(model, torch.cat(hist_h, 0), hist_ids + [int(next_tok)], k, temperature, generator)
        stats.drafted += len(drafts)
        snapshot = cache.clone()
        block = torch.tensor([[int(next_tok)] + drafts], device=prompt_ids.device)
        o = model.forward(block, cache=cache, use_cache=True, memory=frozen, memory_enabled=memory_enabled)
        stats.target_forwards += 1
        stats.steps += 1
        n_acc, replacement = _verify(o.logits[0], drafts, q_probs, temperature, generator)
        stats.accepted += n_acc
        stats.accept_history.append(n_acc)
        accepted = [int(next_tok)] + drafts[:n_acc]
        if n_acc == len(drafts):
            cache = o.cache
            hist_h.append(o.hidden_raw[0])
        else:
            stats.rollbacks += 1
            cache = snapshot
            o2 = model.forward(torch.tensor([accepted], device=prompt_ids.device), cache=cache, use_cache=True,
                               memory=frozen, memory_enabled=memory_enabled)
            stats.target_forwards += 1
            cache = o2.cache
            hist_h.append(o2.hidden_raw[0])
        hist_ids.extend(accepted)
        produced.extend(drafts[:n_acc])
        next_tok = replacement
        if eos_id is not None and eos_id in drafts[:n_acc]:
            produced = produced[: produced.index(eos_id) + 1]
            break
    produced = produced[:max_new_tokens]
    return torch.tensor(produced, dtype=torch.long), cache, stats


def _draft(model: M63Omni, hidden: torch.Tensor, ids: list[int], k: int, temperature: float,
           gen: Optional[torch.Generator]) -> tuple[list[int], list[Optional[torch.Tensor]]]:
    """Recursive MTP drafting. ``hidden``: [L, d] target hidden_raw for positions 0..L-1;
    ``ids``: L+1 tokens (the last is the target's chosen next token)."""
    drafts: list[int] = []
    qs: list[Optional[torch.Tensor]] = []
    if k == 0:
        return drafts, qs
    h = hidden.unsqueeze(0)
    nxt = torch.tensor([ids[1:]], device=hidden.device)
    for _ in range(k):
        L = h.shape[1]
        idx = resolve_stream(1, L, h.device)
        mh, _ = model.mtp(h, model.embed(nxt), idx)
        logits = model.logits(mh[:, -1])[0]
        tok = _sample(logits, temperature, gen)
        drafts.append(int(tok))
        qs.append(torch.softmax(logits.float() / temperature, -1) if temperature > 0 else None)
        h = torch.cat([h, mh[:, -1:]], dim=1)
        nxt = torch.cat([nxt, tok.view(1, 1)], dim=1)
    return drafts, qs


def _verify(target_logits: torch.Tensor, drafts: list[int], q_probs: list[Optional[torch.Tensor]],
            temperature: float, gen: Optional[torch.Generator]) -> tuple[int, torch.Tensor]:
    """Returns (number of accepted drafts, the token to emit after them).

    Greedy: accept while the draft equals the target argmax. Stochastic: standard
    speculative sampling (accept with min(1, p/q), else resample from
    norm(max(0, p - q))), which preserves the target distribution exactly.
    """
    for j, d in enumerate(drafts):
        if temperature <= 0:
            pred = target_logits[j].argmax(-1)
            if int(pred) != d:
                return j, pred
            continue
        p = torch.softmax(target_logits[j].float() / temperature, -1)
        q = q_probs[j]
        ratio = (p[d] / q[d].clamp(min=1e-12)).clamp(max=1.0)
        u = torch.rand((), generator=gen)
        if u > ratio:
            resid = (p - q).clamp(min=0)
            resid = resid / resid.sum() if resid.sum() > 0 else p
            return j, torch.multinomial(resid, 1, generator=gen).squeeze(-1)
    last = target_logits[len(drafts)]
    return len(drafts), _sample(last, temperature, gen)
