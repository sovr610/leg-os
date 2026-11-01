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

from ..model.cache import HybridCache, resolve_stream
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
    """Per-depth MTP attention KV and outputs for *committed* positions only.

    Depth ``j`` (1-based) at position ``i`` is committed once its inputs are final:
    depth ``j-1``'s output at ``i`` and token ``x_{i+j}``. With ``L`` committed
    transcript positions plus the target's next token ``x_L``, depth ``j`` commits
    positions ``< L - j + 1``; its last ``j - 1`` positions depend on drafts and
    are computed functionally on top of the cache, never stored, so a rejected
    draft needs no MTP rollback.
    """

    states: list = field(default_factory=list)  # per depth: Optional[KVLayerState]
    hidden: list = field(default_factory=list)  # per depth: committed outputs [n_j, d] (input to depth j+1)
    n: list = field(default_factory=list)  # per depth: committed positions

    def ensure(self, depth: int) -> None:
        while len(self.n) < depth:
            self.states.append(None)
            self.hidden.append(None)
            self.n.append(0)


def _stream_at(n: int, length: int, device: torch.device):
    if n == 0:
        return resolve_stream(1, length, device)
    pseudo = HybridCache(layers=[], next_pos=n, last_seg=torch.zeros(1, dtype=torch.long, device=device),
                         next_rope=torch.tensor([n], device=device))
    return resolve_stream(1, length, device, pseudo)


def _emit(model: M63Omni, h: torch.Tensor, temperature: float, gen: Optional[torch.Generator]):
    logits = model.logits(h)
    tok = _sample(logits, temperature, gen)
    return tok, (torch.softmax(logits.float() / temperature, -1) if temperature > 0 else None)


def _draft_cached(model: M63Omni, mc: MTPCache, hidden: torch.Tensor, ids: list[int], k: int, temperature: float,
                  gen: Optional[torch.Generator]) -> tuple[list[int], list[Optional[torch.Tensor]], int]:
    """Draft ``k`` tokens, one per MTP depth, reusing committed MTP state.

    ``hidden``: [L, d] target hidden_raw for the committed positions; ``ids``: the
    L+1 transcript tokens (the last is the target's chosen next token). Equal to
    :func:`_draft` (full recompute); returns ``(drafts, draft probs, MTP positions computed)``.
    """
    L, dev = hidden.shape[0], hidden.device
    mc.ensure(k)
    ext = list(ids)
    drafts: list[int] = []
    qs: list[Optional[torch.Tensor]] = []
    spec_prev = hidden.new_zeros(0, hidden.shape[1])  # depth j-1 outputs at its speculative positions
    computed = 0
    for j in range(1, k + 1):
        mod = model.mtp.at(j)
        prev = hidden if j == 1 else mc.hidden[j - 2]
        c, n = L - j + 1, mc.n[j - 1]  # depth j commits positions [0, c)
        if c > n:
            tok = torch.tensor([ext[n + j : c + j]], device=dev)
            out, mc.states[j - 1] = mod(prev[n:c].unsqueeze(0), model.embed(tok), _stream_at(n, c - n, dev),
                                        mc.states[j - 1])
            mc.hidden[j - 1] = out[0] if mc.hidden[j - 1] is None else torch.cat([mc.hidden[j - 1], out[0]], 0)
            mc.n[j - 1] = c
            computed += c - n
        if j == 1:
            spec, h_last = spec_prev, mc.hidden[0][-1]
        else:  # positions c..L-1: depth j-1 committed at c, speculative after
            spec_in = torch.cat([prev[c:], spec_prev], 0)
            tok = torch.tensor([ext[c + j : L + j]], device=dev)
            out, _ = mod(spec_in.unsqueeze(0), model.embed(tok), _stream_at(c, j - 1, dev), mc.states[j - 1])
            spec, h_last = out[0], out[0, -1]
            computed += j - 1
        t, q = _emit(model, h_last, temperature, gen)
        drafts.append(int(t))
        qs.append(q)
        ext.append(int(t))
        spec_prev = spec
    return drafts, qs, computed


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
    loop_passes: Optional[int] = None,
) -> tuple[torch.Tensor, HybridCache, SpecStats]:
    """Generate for a single sequence (B=1). Returns (new token IDs [n], final cache, stats).

    Neural memory is read-only throughout (states are marked frozen).
    ``use_mtp_cache=False`` selects the full-recompute drafter (the reference).
    """
    if prompt_ids.shape[0] != 1:
        raise ValueError("reference speculative decoder supports batch size 1")
    if draft_tokens and model.mtp is None:
        raise ValueError("model has no MTP module")
    if model.mtp is not None and draft_tokens > model.mtp.depth:
        raise ValueError(f"draft_tokens={draft_tokens} exceeds the trained MTP depth {model.mtp.depth}")
    frozen = None
    if memory is not None:
        frozen = {i: MemoryState(**{**s.__dict__, "frozen": True}) for i, s in memory.items()}
    stats = SpecStats()
    out = model.forward(prompt_ids, cache=cache, use_cache=True, memory=frozen, memory_enabled=memory_enabled,
                        loop_passes=loop_passes)
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
        H = torch.cat(hist_h, 0)
        if k == 0:
            drafts, q_probs = [], []
        elif use_mtp_cache:
            drafts, q_probs, computed = _draft_cached(model, mtp_cache, H, hist_ids + [int(next_tok)], k, temperature,
                                                      generator)
            stats.mtp_positions += computed
        else:
            stats.mtp_positions += H.shape[0] * k
            drafts, q_probs = _draft(model, H, hist_ids + [int(next_tok)], k, temperature, generator)
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
    """Reference drafter: recompute every MTP depth over the whole transcript.

    ``hidden``: [L, d] target hidden_raw for positions 0..L-1; ``ids``: L+1 tokens
    (the last is the target's chosen next token). Draft ``j`` comes from depth
    ``j`` at the last position, which fuses depth ``j-1``'s output with token
    ``x_{L-1+j}`` (the target's token for ``j = 1``, draft ``j-1`` after that).
    """
    drafts: list[int] = []
    qs: list[Optional[torch.Tensor]] = []
    L, dev = hidden.shape[0], hidden.device
    h, ext = hidden.unsqueeze(0), list(ids)
    for j in range(1, k + 1):
        tok = torch.tensor([ext[j : L + j]], device=dev)
        h, _ = model.mtp.at(j)(h, model.embed(tok), resolve_stream(1, L, dev))
        t, q = _emit(model, h[0, -1], temperature, gen)
        drafts.append(int(t))
        qs.append(q)
        ext.append(int(t))
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
