"""Derived capacity numbers reproduce the blueprint's arithmetic (§4, §14)."""

import pytest

from m63_omni.config import reference
from m63_omni.runtime import capacity as cap

GiB, MiB = cap.GiB, cap.MiB


def test_kv_cache_lower_bound():
    c = reference()
    assert cap.kv_bytes_per_token(c) == 12 * 2 * 8 * 128 * 2 == 49152
    assert cap.kv_bytes(c, 32 * 1024) / GiB == 1.5
    assert cap.kv_bytes(c, 128 * 1024) / GiB == 6.0
    assert cap.kv_bytes(c, 256 * 1024) / GiB == 12.0
    assert cap.full_attention_kv_bytes_per_token(c) == 4 * cap.kv_bytes_per_token(c)


def test_recurrent_and_memory_state():
    c = reference()
    assert cap.gdn_state_bytes(c) == 36 * 16 * 128 * 128 * 2 == 18 * MiB
    assert cap.memory_state_bytes(c) == 3 * MiB


def test_weight_storage():
    total = 62.48e9
    assert cap.weight_bytes(total, "bfloat16") / 1e9 == pytest.approx(124.96)
    assert cap.weight_bytes(total, "bfloat16") / GiB == pytest.approx(116.38, abs=0.01)
    assert cap.weight_bytes(total, "fp8") / GiB == pytest.approx(58.19, abs=0.02)
    assert cap.weight_bytes(total, "int4") / GiB == pytest.approx(29.10, abs=0.01)


def test_training_compute_estimate():
    base, allow = cap.training_flops(5e9, 4.75e12)
    assert base == pytest.approx(1.425e23)
    assert allow == pytest.approx(1.8525e23)
    table = {(256, 250): 33.5, (256, 350): 23.9, (512, 250): 16.8, (512, 350): 12.0}
    for (g, tf), days in table.items():
        assert round(cap.training_days(allow, g, tf), 1) == days


def test_sessions_per_deployment_bounds():
    c = reference()
    n = cap.sessions_per_deployment(c, 62.48e9, 4, 80e9, 131072)
    assert 0 < n < 50
    assert cap.sessions_per_deployment(c, 62.48e9, 1, 80e9, 131072) == 0  # BF16 weights do not fit one GPU
