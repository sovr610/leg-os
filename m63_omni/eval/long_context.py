"""Synthetic long-context probes (RULER/BABILong-style, blueprint §13).

Tasks are generated directly in token-ID space so they run on any vocabulary,
including tiny test configs. Every long task has a matched short-context
control so that difficult reasoning is not mislabelled as a length problem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ..model.m63 import M63Omni


@dataclass
class RetrievalTask:
    prompt: list[int]
    answer: list[int]
    context_tokens: int
    depth: float
    kind: str


def passkey_task(vocab: int, context_tokens: int, depth: float, *, key_len: int = 4, seed: int = 0,
                 marker: int = 1, query: int = 2) -> RetrievalTask:
    """Haystack of random filler with ``marker key`` inserted at ``depth``; the prompt ends with ``query marker``."""
    rng = random.Random(seed)
    low = 3
    key = [rng.randrange(low, vocab) for _ in range(key_len)]
    filler_len = max(0, context_tokens - key_len - 3)
    filler = [rng.randrange(low, vocab) for _ in range(filler_len)]
    at = int(depth * filler_len)
    prompt = filler[:at] + [marker] + key + filler[at:] + [query, marker]
    return RetrievalTask(prompt, key, len(prompt), depth, "passkey")


def multikey_task(vocab: int, context_tokens: int, n_keys: int, *, key_len: int = 2, seed: int = 0) -> RetrievalTask:
    """Several ``(name -> value)`` pairs; query one name (distractor complexity grows with ``n_keys``)."""
    rng = random.Random(seed)
    low = 3
    names = rng.sample(range(low, vocab), n_keys)
    values = [[rng.randrange(low, vocab) for _ in range(key_len)] for _ in range(n_keys)]
    filler_len = max(0, context_tokens - n_keys * (key_len + 2) - 3)
    seq = [rng.randrange(low, vocab) for _ in range(filler_len)]
    for nm, val in zip(names, values):
        pos = rng.randrange(0, len(seq) + 1)
        seq[pos:pos] = [1, nm] + val
    target = rng.randrange(n_keys)
    prompt = seq + [2, 1, names[target]]
    return RetrievalTask(prompt, values[target], len(prompt), -1.0, f"multikey_{n_keys}")


@torch.no_grad()
def greedy_answer(model: M63Omni, prompt: list[int], n: int, *, memory_enabled: bool = True,
                  segmented: bool = False) -> list[int]:
    model.eval()
    ids = torch.tensor([prompt], device=model.embed.weight.device)
    if segmented and model.cfg.memory.enabled:
        S = model.cfg.memory.update_segment_tokens
        cut = (len(prompt) // S) * S
        mem = None
        cache = None
        if cut:
            out = model.forward_segmented(ids[:, :cut], segment_len=S)
            cache, mem = out.cache, out.memory_states
        out = model(ids[:, cut:], cache=cache, use_cache=True, memory=mem) if cut < len(prompt) else out
    else:
        out = model(ids, use_cache=True, memory_enabled=memory_enabled)
        mem = None
    cache = out.cache
    tok = out.logits[0, -1].argmax()
    ans = [int(tok)]
    for _ in range(n - 1):
        out = model.step(tok.view(1, 1), cache, memory=mem, memory_enabled=memory_enabled)
        cache = out.cache
        tok = out.logits[0, -1].argmax()
        ans.append(int(tok))
    return ans


def score_tasks(model: M63Omni, tasks: list[RetrievalTask], **kw) -> float:
    return sum(greedy_answer(model, t.prompt, len(t.answer), **kw) == t.answer for t in tasks) / max(len(tasks), 1)


def length_sweep(model: M63Omni, vocab: int, lengths: list[int], *, depths=(0.1, 0.5, 0.9), seeds=range(3),
                 short_control: int = 64, **kw) -> dict[int, float]:
    """Passkey accuracy per length, plus ``-short_control`` for the matched short control."""
    res = {L: score_tasks(model, [passkey_task(vocab, L, d, seed=s) for d in depths for s in seeds], **kw) for L in lengths}
    res[-short_control] = score_tasks(model, [passkey_task(vocab, short_control, d, seed=s) for d in depths for s in seeds], **kw)
    return res
