"""Proxy tournament runner (blueprint §12, step 6).

Trains every candidate under **matched conditions** — same base shape, data
order, tokenizer/vocabulary, optimizer settings, and either the same number of
positions or the same active-parameter FLOPs — then evaluates each on held-out
batches and on systems metrics. Winners must clear systems gates *and* sit on
the quality/latency/memory Pareto front; a single average score never decides.

Variant files (``configs/tournament/*.yaml``) are small overrides of the proxy
config. The runner applies each variant's overrides to a chosen *base* (the
1.3B proxy on real hardware, ``configs/tiny.yaml`` for a CPU smoke run), and
re-fits memory anchors so they always sit on attention layers of the variant's
hybrid pattern.

The runner trains on text batches with the multimodal front end disabled, so a
variant that only changes ``multimodal`` settings (timestamp mode, AV window)
cannot differ from its base here: it is reported as *skipped* with the reason,
never ranked as a tie. Mixed variants run, with a note that their multimodal
overrides were not exercised; score those on media data. Likewise, memory
overrides other than ``enabled`` only act in ``--segmented-memory`` mode.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch

from ..config import GATED_GQA, M63Config, _deep_merge, _read, from_dict
from ..eval.efficiency import measure
from ..eval.harness import bootstrap_ci
from ..model.m63 import M63Omni
from ..model.manifest import build_manifest
from .trainer import Trainer, as_batch, synthetic_batch

BatchFn = Callable[[M63Config, int, int, int, int], Any]  # (cfg, cursor, batch, seq_len, seed) -> batch


@dataclass
class Candidate:
    name: str
    cfg: M63Config
    overrides: dict


@dataclass
class CandidateResult:
    name: str
    overrides: dict
    resident_params: int
    active_params: float
    steps: int
    positions: int
    train_flops: float
    first_loss: float
    final_loss: float
    eval_loss: float
    eval_ci: tuple[float, float]
    train_tokens_per_s: float
    prefill_tok_s: float
    tpot_ms: float
    kv_bytes_per_token: float
    recurrent_bytes: int
    router_max_over_mean: float
    alarms: list[str] = field(default_factory=list)
    systems_pass: bool = True
    gate_failures: list[str] = field(default_factory=list)
    pareto: bool = False


@dataclass
class TournamentReport:
    base_fingerprint: str
    budget: str
    results: list[CandidateResult]
    tolerances: dict = field(default_factory=dict)
    mode: str = "packed"  # "packed" (memory reads M0 only) or "segmented memory" (memory updates happen)
    skipped: dict[str, str] = field(default_factory=dict)  # candidates this runner cannot distinguish, with the reason
    notes: dict[str, list[str]] = field(default_factory=dict)  # overrides a candidate carried but this run did not exercise

    @property
    def winners(self) -> list[str]:
        return [r.name for r in self.results if r.pareto and r.systems_pass]

    def to_dict(self) -> dict:
        return {"base_fingerprint": self.base_fingerprint, "budget": self.budget, "mode": self.mode, "winners": self.winners,
                "tolerances": self.tolerances, "results": [asdict(r) for r in self.results], "skipped": self.skipped,
                "notes": self.notes}

    def to_markdown(self) -> str:
        lines = [f"# Proxy tournament ({self.budget}-matched, {self.mode} training)", "",
                 "| candidate | resident | active | steps | eval loss (95% CI) | TPOT ms | KV B/token | state B | load max/mean | systems | Pareto |",
                 "|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|"]
        for r in sorted(self.results, key=lambda r: r.eval_loss):
            lines.append(
                f"| {r.name} | {r.resident_params / 1e6:.2f}M | {r.active_params / 1e6:.2f}M | {r.steps} | "
                f"{r.eval_loss:.4f} ({r.eval_ci[0]:.4f}–{r.eval_ci[1]:.4f}) | {r.tpot_ms:.2f} | {r.kv_bytes_per_token:.0f} | "
                f"{r.recurrent_bytes} | {r.router_max_over_mean:.2f} | {'pass' if r.systems_pass else 'FAIL: ' + '; '.join(r.gate_failures)} | "
                f"{'★' if r.pareto else ''} |")
        tol = self.tolerances
        lines += ["", f"Winners (systems gates passed and Pareto-optimal on eval loss, TPOT, KV bytes): "
                      f"{', '.join(self.winners) or 'none'}",
                  f"Dominance tolerances: eval loss ±{tol.get('eval_loss', 0):.4f}, TPOT ±{100 * tol.get('tpot_relative', 0):.0f}%."]
        if self.skipped:
            lines += ["", "Skipped (not distinguishable by this runner):"]
            lines += [f"- {name}: {why}" for name, why in sorted(self.skipped.items())]
        if self.notes:
            lines += ["", "Not exercised by this run:"]
            lines += [f"- {name}: {'; '.join(n)}" for name, n in sorted(self.notes.items())]
        lines += ["", "Small-scale results select candidates for the pilot; they are not capability claims."]
        return "\n".join(lines)


# ----------------------------------------------------------------- candidates
def variant_overrides(path: str | Path) -> dict:
    data = _read(Path(path))
    data.pop("extends", None)
    return data


def fit_anchors(cfg_dict: dict) -> dict:
    """Keep memory anchors that are attention layers of the pattern; else use the last attention layer."""
    m, mem = cfg_dict["model"], cfg_dict["memory"]
    pat = m["hybrid_pattern"]
    attn = [i + 1 for i in range(m["layers"]) if pat[i % len(pat)] == GATED_GQA]
    if mem.get("enabled", True) and attn:
        kept = [a for a in mem["anchors"] if a in attn]
        mem["anchors"] = kept or [attn[-1]]
    return cfg_dict


def build_candidates(base: M63Config, variants: Sequence[str | Path]) -> list[Candidate]:
    out = []
    for v in variants:
        ov = variant_overrides(v)
        merged = fit_anchors(_deep_merge(base.to_dict(), ov))
        out.append(Candidate(Path(v).stem, from_dict(merged), ov))
    return out


#: Sections the text tournament never exercises (the multimodal front end is disabled).
TEXT_INERT_SECTIONS = frozenset({"multimodal"})
#: Memory settings that only act when memory updates happen (``segmented_memory=True``).
PACKED_INERT_MEMORY = frozenset({"decay_target", "max_eta", "max_theta", "max_alpha", "grad_clip", "update_segment_tokens"})


def _unexercised(overrides: dict, segmented_memory: bool) -> list[str]:
    out = [f"{sec} overrides (needs media data)" for sec in sorted(TEXT_INERT_SECTIONS & set(overrides))]
    mem = overrides.get("memory") or {}
    if not segmented_memory and isinstance(mem, dict) and PACKED_INERT_MEMORY & set(mem):
        out.append(f"memory.{', memory.'.join(sorted(PACKED_INERT_MEMORY & set(mem)))} (needs --segmented-memory)")
    return out


def _default_batch(cfg: M63Config, cursor: int, batch: int, seq_len: int, seed: int):
    return synthetic_batch(cfg, cursor, batch, seq_len, seed)


# -------------------------------------------------------------------- running
def _pareto(rows: list[CandidateResult], loss_tol: float, tpot_rel_tol: float) -> None:
    """Noise-aware Pareto front on (eval loss, TPOT, KV bytes/token).

    ``j`` dominates ``i`` only if it is no worse on every metric (within tolerance) and better than
    ``i`` by more than the tolerance on at least one — so measurement noise cannot pick a winner.
    """
    def tols(r: CandidateResult) -> tuple[float, float, float]:
        return (loss_tol, tpot_rel_tol * r.tpot_ms, 0.0)

    pts = [(r.eval_loss, r.tpot_ms, r.kv_bytes_per_token) for r in rows]
    for i, r in enumerate(rows):
        t = tols(r)
        r.pareto = r.systems_pass and not any(
            rows[j].systems_pass
            and all(a <= b + tk for a, b, tk in zip(pts[j], pts[i], t))
            and any(a < b - tk for a, b, tk in zip(pts[j], pts[i], t))
            for j in range(len(rows)) if j != i)


def run_tournament(
    candidates: Sequence[Candidate],
    *,
    steps: int = 50,
    batch: int = 4,
    seq_len: Optional[int] = None,
    seed: int = 0,
    eval_batches: int = 8,
    budget: str = "positions",
    reference: Optional[str] = None,
    max_kv_bytes_per_token: Optional[float] = None,
    max_tpot_ms: Optional[float] = None,
    batch_fn: BatchFn = _default_batch,
    segmented_memory: bool = False,
    loss_tolerance: Optional[float] = None,
    tpot_rel_tolerance: float = 0.15,
    timing_repeats: int = 3,
    log: Optional[Callable[[str], None]] = None,
) -> TournamentReport:
    """Run the tournament.

    ``segmented_memory=True`` trains and evaluates every candidate segment by segment so
    neural-memory updates actually happen (required for memory ablations such as
    ``decay_target``; packed training only ever reads the initial memory). Pareto dominance
    uses ``loss_tolerance`` (default: the median bootstrap CI half-width) and a relative TPOT
    tolerance, with TPOT the median of ``timing_repeats`` measurements.
    """
    if budget not in ("positions", "flops"):
        raise ValueError("budget must be 'positions' or 'flops'")
    skipped: dict[str, str] = {}
    notes: dict[str, list[str]] = {}
    runnable = []
    for c in candidates:
        if c.overrides and set(c.overrides) <= TEXT_INERT_SECTIONS:
            skipped[c.name] = ("only changes " + ", ".join(sorted(c.overrides)) + "; this runner trains text with the "
                               "front end disabled, so it would tie its base. Score it on media data.")
            continue
        unexercised = _unexercised(c.overrides, segmented_memory)
        if unexercised:
            notes[c.name] = unexercised
        runnable.append(c)
    candidates = runnable
    if not candidates:
        raise ValueError("no candidates" + (f" this runner can distinguish (skipped: {sorted(skipped)})" if skipped else ""))
    vocab = {c.cfg.model.vocab_size for c in candidates}
    if len(vocab) != 1:
        raise ValueError("candidates must share a vocabulary (matched tokenizer)")
    manifests = {c.name: build_manifest(c.cfg) for c in candidates}
    ref = reference or candidates[0].name
    ref_active = manifests[ref].active_excl_head
    results: list[CandidateResult] = []
    for c in candidates:
        man = manifests[c.name]
        S = seq_len or c.cfg.training.seq_len
        n_steps = steps if budget == "positions" else max(1, round(steps * ref_active / man.active_excl_head))
        torch.manual_seed(seed)
        cfg = c.cfg.replace(multimodal={"enabled": False})
        model = M63Omni(cfg)
        trainer = Trainer(model, lambda cur, cfg=cfg, S=S: batch_fn(cfg, cur, batch, S, seed), total_positions=n_steps * batch * S,
                          segmented_memory=segmented_memory)
        t0 = time.perf_counter()
        hist = trainer.fit(n_steps)
        train_s = time.perf_counter() - t0
        model.eval()
        losses = []
        with torch.no_grad():
            for j in range(eval_batches):
                b = as_batch(batch_fn(cfg, 10**7 + j, batch, S, seed + 1))
                if segmented_memory:
                    out = model.forward_segmented(b.input_ids, b.labels)
                else:
                    out = model(b.input_ids, labels=b.labels, segment_ids=b.segment_ids, position_ids=b.position_ids,
                                reset_mask=b.reset_mask)
                losses.append(float(out.lm_loss))
        mean, lo, hi = bootstrap_ci(losses, seed=seed)
        effs = [measure(model, prompt_tokens=min(64, max(cfg.model.context_targets) - 16), new_tokens=8, seed=seed)
                for _ in range(max(1, timing_repeats))]
        eff = sorted(effs, key=lambda e: e.tpot_ms)[len(effs) // 2]
        alarms = sorted({a for h in hist for a in h["alarms"]})
        r = CandidateResult(
            name=c.name, overrides=c.overrides, resident_params=man.total, active_params=man.active_excl_head,
            steps=n_steps, positions=trainer.state.positions, train_flops=6.0 * man.active_excl_head * trainer.state.positions,
            first_loss=hist[0]["lm"], final_loss=hist[-1]["lm"], eval_loss=mean, eval_ci=(lo, hi),
            train_tokens_per_s=trainer.state.positions / max(train_s, 1e-9), prefill_tok_s=eff.prefill_tok_s,
            tpot_ms=eff.tpot_ms, kv_bytes_per_token=eff.kv_bytes_per_token_measured, recurrent_bytes=eff.recurrent_bytes,
            router_max_over_mean=hist[-1]["max_over_mean"], alarms=alarms,
        )
        if not math.isfinite(r.eval_loss):
            r.gate_failures.append("non-finite eval loss")
        if max_kv_bytes_per_token is not None and r.kv_bytes_per_token > max_kv_bytes_per_token:
            r.gate_failures.append(f"KV {r.kv_bytes_per_token:.0f} B/token > {max_kv_bytes_per_token:.0f}")
        if max_tpot_ms is not None and r.tpot_ms > max_tpot_ms:
            r.gate_failures.append(f"TPOT {r.tpot_ms:.2f} ms > {max_tpot_ms:.2f}")
        if "dead_expert" in alarms:
            r.gate_failures.append("dead experts")
        r.systems_pass = not r.gate_failures
        results.append(r)
        if log:
            log(f"{c.name}: eval {mean:.4f} [{lo:.4f}, {hi:.4f}] tpot {r.tpot_ms:.2f} ms kv {r.kv_bytes_per_token:.0f} B/tok")
    if loss_tolerance is None:
        halves = sorted((r.eval_ci[1] - r.eval_ci[0]) / 2 for r in results)
        loss_tolerance = halves[len(halves) // 2]
    _pareto(results, loss_tolerance, tpot_rel_tolerance)
    base = candidates[0].cfg
    rep = TournamentReport(base.fingerprint(), budget, results, mode="segmented memory" if segmented_memory else "packed",
                           skipped=skipped, notes=notes)
    rep.tolerances = {"eval_loss": loss_tolerance, "tpot_relative": tpot_rel_tolerance}
    return rep


def write_report(rep: TournamentReport, json_path: Optional[str] = None, md_path: Optional[str] = None) -> None:
    if json_path:
        Path(json_path).write_text(json.dumps(rep.to_dict(), indent=2, default=float) + "\n")
    if md_path:
        Path(md_path).write_text(rep.to_markdown() + "\n")
