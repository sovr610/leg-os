"""Efficiency measurements (blueprint §13 "Efficiency", §14).

Reports TTFT (prefill), TPOT (per-output-token decode), throughput, and measured
cache bytes per token compared with the analytic value. On CPU these numbers
only validate the harness; production numbers come from the serving stack
under mixed traffic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..model.m63 import M63Omni
from ..runtime.capacity import kv_bytes_per_token


@dataclass
class EfficiencyReport:
    prompt_tokens: int
    new_tokens: int
    ttft_ms: float
    tpot_ms: float
    prefill_tok_s: float
    decode_tok_s: float
    kv_bytes_per_token_measured: float
    kv_bytes_per_token_analytic: float
    recurrent_bytes: int


@torch.no_grad()
def measure(model: M63Omni, prompt_tokens: int = 128, new_tokens: int = 16, seed: int = 0) -> EfficiencyReport:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    V = model.cfg.model.vocab_size - model.cfg.model.special_tokens
    ids = torch.randint(0, V, (1, prompt_tokens), generator=g).to(model.embed.weight.device)
    t0 = time.perf_counter()
    out = model.prefill(ids)
    t1 = time.perf_counter()
    cache = out.cache
    tok = out.logits[:, -1:].argmax(-1)
    for _ in range(new_tokens):
        o = model.step(tok, cache)
        cache = o.cache
        tok = o.logits[:, -1:].argmax(-1)
    t2 = time.perf_counter()
    elem = next(model.parameters()).element_size()
    return EfficiencyReport(
        prompt_tokens, new_tokens,
        ttft_ms=(t1 - t0) * 1000, tpot_ms=(t2 - t1) * 1000 / max(new_tokens, 1),
        prefill_tok_s=prompt_tokens / max(t1 - t0, 1e-9), decode_tok_s=new_tokens / max(t2 - t1, 1e-9),
        kv_bytes_per_token_measured=cache.kv_bytes() / cache.next_pos,
        kv_bytes_per_token_analytic=kv_bytes_per_token(model.cfg, elem),
        recurrent_bytes=cache.recurrent_bytes(),
    )
