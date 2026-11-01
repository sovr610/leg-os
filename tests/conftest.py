"""Shared fixtures: CPU-sized models that keep every structural feature of the reference."""

from __future__ import annotations

import pytest
import torch

from m63_omni.config import tiny
from m63_omni.model.m63 import M63Omni


@pytest.fixture
def cfg():
    return tiny(multimodal={"enabled": False})


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    return M63Omni(cfg).eval()


@pytest.fixture
def mm_model():
    torch.manual_seed(0)
    return M63Omni(tiny()).eval()


def _rand_ids(cfg, B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.model.vocab_size - cfg.model.special_tokens, (B, T), generator=g)


@pytest.fixture
def rand_ids():
    """``rand_ids(cfg, B, T, seed=0)`` -> non-special token IDs."""
    return _rand_ids
