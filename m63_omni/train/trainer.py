"""Single-process reference training loop (blueprint §7, §9, §12).

This loop is the numerical reference for the distributed implementation: same
objective, schedule, router-bias update, gradient clipping, numerics guards,
and checkpoint contents. Data order is a pure function of the data cursor, so
resume reproduces the uninterrupted run exactly on CPU.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from ..config import M63Config
from ..model.m63 import IGNORE_INDEX, M63Omni
from .checkpoint import CheckpointManager, load_checkpoint
from .monitor import RouterMonitor, SpikeDetector, check_finite
from .objectives import compute_loss
from .schedule import wsd_lr


@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    segment_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    reset_mask: Optional[torch.Tensor] = None
    modality_ids: Optional[torch.Tensor] = None


def as_batch(obj: Any) -> Batch:
    """Accept a :class:`Batch`, a ``data.packing.PackedBatch``, or a dict with the same fields."""
    get = (lambda k: obj.get(k)) if isinstance(obj, dict) else (lambda k: getattr(obj, k, None))
    return Batch(get("input_ids"), get("labels"), get("segment_ids"), get("position_ids"), get("reset_mask"), get("modality_ids"))


def build_optimizer(model: torch.nn.Module, cfg: M63Config) -> torch.optim.AdamW:
    """AdamW; no weight decay on norms, biases, embeddings, decay parameters, or memory initial state."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or name.startswith("embed") or name.endswith(("A_log", "dt_bias", "W1_0", "W2_0", "slots")):
            no_decay.append(p)
        else:
            decay.append(p)
    t = cfg.training
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": t.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=t.peak_lr, betas=tuple(t.betas), eps=t.eps,
    )


def synthetic_batch(cfg: M63Config, cursor: int, batch_size: int, seq_len: int, seed: int = 0) -> Batch:
    """Deterministic, learnable packed batch (documents of modular arithmetic progressions).

    Each row packs several documents; labels never cross document boundaries and
    RoPE positions restart per document — the same contract as ``data.packing``.
    """
    g = torch.Generator().manual_seed(seed * 1_000_003 + cursor)
    V = cfg.model.vocab_size - cfg.model.special_tokens
    ids = torch.empty(batch_size, seq_len, dtype=torch.long)
    seg = torch.empty_like(ids)
    pos = torch.empty_like(ids)
    for b in range(batch_size):
        t, d = 0, 0
        while t < seq_len:
            n = int(torch.randint(seq_len // 4, seq_len + 1, (1,), generator=g))
            n = min(n, seq_len - t)
            start = int(torch.randint(0, V, (1,), generator=g))
            step = int(torch.randint(1, 7, (1,), generator=g))
            ids[b, t : t + n] = (start + step * torch.arange(n)) % V
            seg[b, t : t + n] = d
            pos[b, t : t + n] = torch.arange(n)
            t, d = t + n, d + 1
    labels = torch.cat([ids[:, 1:], torch.full((batch_size, 1), IGNORE_INDEX)], dim=1)
    same = torch.cat([seg[:, 1:] == seg[:, :-1], torch.zeros(batch_size, 1, dtype=torch.bool)], dim=1)
    labels = labels.masked_fill(~same, IGNORE_INDEX)
    reset = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool), seg[:, 1:] != seg[:, :-1]], dim=1)
    return Batch(ids, labels, seg, pos, reset, None)


@dataclass
class TrainState:
    step: int = 0
    positions: int = 0
    cursor: int = 0
    history: list[dict] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: M63Omni,
        batch_fn: Callable[[int], Any],
        *,
        total_positions: int,
        checkpoints: Optional[CheckpointManager] = None,
        checkpoint_every: int = 0,
        segmented_memory: bool = False,
        autocast: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.cfg = model.cfg
        self.batch_fn = batch_fn
        self.total_positions = total_positions
        self.opt = build_optimizer(model, self.cfg)
        self.state = TrainState()
        self.ckpt = checkpoints
        self.checkpoint_every = checkpoint_every
        self.segmented_memory = segmented_memory
        device = next(model.parameters()).device
        self.autocast = (device.type == "cuda" and self.cfg.training.precision == "bf16") if autocast is None else autocast
        self.monitor = RouterMonitor(len(model.blocks), self.cfg.moe.routed_experts, window_tokens=10**12)
        self.spikes = SpikeDetector()

    def _ctx(self):
        if not self.autocast:
            return contextlib.nullcontext()
        dev = next(self.model.parameters()).device.type
        return torch.autocast(device_type=dev, dtype=torch.bfloat16)

    def train_step(self) -> dict:
        self.model.train()
        batch = as_batch(self.batch_fn(self.state.cursor))
        lr = wsd_lr(self.state.positions, self.total_positions, self.cfg.training)
        for group in self.opt.param_groups:
            group["lr"] = lr
        t0 = time.perf_counter()
        use_mtp = self.cfg.mtp.enabled
        with self._ctx():
            if self.segmented_memory:
                out = self.model.forward_segmented(batch.input_ids, batch.labels, compute_mtp=use_mtp)
            else:
                out = self.model(batch.input_ids, labels=batch.labels, segment_ids=batch.segment_ids,
                                 position_ids=batch.position_ids, reset_mask=batch.reset_mask,
                                 modality_ids=batch.modality_ids, compute_mtp=use_mtp)
            losses = compute_loss(self.cfg, out)
        self.opt.zero_grad(set_to_none=True)
        losses.total.backward()
        grad_norm = check_finite(losses.total, self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.clip_grad_norm)
        self.opt.step()
        self.model.update_router_biases(out.router_stats)  # global-batch load (all-reduced in distributed runs)
        alarms = self.monitor.update([s.detach() for s in out.router_stats])
        n_pos = int(batch.input_ids.numel())
        self.state.step += 1
        self.state.positions += n_pos
        self.state.cursor += 1
        metrics = {"step": self.state.step, "positions": self.state.positions, "lr": lr, "grad_norm": grad_norm,
                   "tokens_per_s": n_pos / max(time.perf_counter() - t0, 1e-9), **losses.terms,
                   **RouterMonitor.load_summary(out.router_stats)}
        metrics["loss_spike"] = self.spikes.update(self.state.step, losses.terms["lm"])
        metrics["alarms"] = [a.kind for a in alarms]
        self.state.history.append(metrics)
        if self.ckpt is not None and self.checkpoint_every and self.state.step % self.checkpoint_every == 0:
            self.save()
        return metrics

    def fit(self, steps: int) -> list[dict]:
        return [self.train_step() for _ in range(steps)]

    def save(self) -> list[Path]:
        if self.ckpt is None:
            raise RuntimeError("no checkpoint manager configured")
        return self.ckpt.save(model=self.model, optimizer=self.opt, config_fingerprint=self.cfg.fingerprint(),
                              step=self.state.step, positions=self.state.positions, data_cursor=self.state.cursor,
                              extra={"spikes": list(self.spikes.spikes)})

    def resume(self, path: str | Path) -> None:
        payload = load_checkpoint(path, model=self.model, optimizer=self.opt, config_fingerprint=self.cfg.fingerprint())
        self.state.step = payload["step"]
        self.state.positions = payload["positions"]
        self.state.cursor = payload["data_cursor"]
