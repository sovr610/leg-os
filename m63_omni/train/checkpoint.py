"""Checkpoint/resume (blueprint §7, §12).

A checkpoint contains weights, optimizer state, router biases (model buffers),
RNG states, the data-loader cursor, the training counters, and the config
fingerprint. Writes are atomic (temp file + rename). Resume must reproduce the
uninterrupted run within a defined tolerance — exactly, in this CPU reference.

Two retention tiers: *rolling* checkpoints for immediate recovery (keep the
last N) and *milestone* checkpoints frozen at a position interval for
architecture comparisons (never rotated).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

FORMAT_VERSION = 1


class CheckpointMismatchError(RuntimeError):
    pass


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    config_fingerprint: str, step: int, positions: int, data_cursor: Any,
                    extra: Optional[dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "config_fingerprint": config_fingerprint,
        "model": model.state_dict(),  # includes router load-balancing biases
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": step,
        "positions": positions,
        "data_cursor": data_cursor,
        "rng": rng_state(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    config_fingerprint: Optional[str] = None, restore_rng: bool = True) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION:
        raise CheckpointMismatchError("unsupported checkpoint format")
    if config_fingerprint is not None and payload["config_fingerprint"] != config_fingerprint:
        raise CheckpointMismatchError("checkpoint was produced by a different configuration")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        set_rng_state(payload["rng"])
    return payload


@dataclass
class CheckpointManager:
    directory: Path
    keep_rolling: int = 2
    milestone_every_positions: int = 0

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        (self.directory / "rolling").mkdir(parents=True, exist_ok=True)
        (self.directory / "milestones").mkdir(parents=True, exist_ok=True)
        self._last_milestone = 0

    def save(self, *, positions: int, step: int, **kw) -> list[Path]:
        written = [save_checkpoint(self.directory / "rolling" / f"step_{step:09d}.pt", positions=positions, step=step, **kw)]
        rolling = sorted((self.directory / "rolling").glob("step_*.pt"))
        for old in rolling[: -self.keep_rolling]:
            old.unlink()
        if self.milestone_every_positions and positions // self.milestone_every_positions > self._last_milestone:
            self._last_milestone = positions // self.milestone_every_positions
            written.append(save_checkpoint(self.directory / "milestones" / f"pos_{positions:015d}.pt",
                                           positions=positions, step=step, **kw))
        return written

    def latest(self) -> Optional[Path]:
        rolling = sorted((self.directory / "rolling").glob("step_*.pt"))
        return rolling[-1] if rolling else None

    def milestones(self) -> list[Path]:
        return sorted((self.directory / "milestones").glob("pos_*.pt"))
