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


# ------------------------------------------------- corrected estimates (docs/IMPROVEMENTS.md §9)
@pytest.fixture(scope="module")
def ref_manifest():
    from m63_omni.model.manifest import build_manifest

    return build_manifest(reference())


def test_training_compute_adds_the_terms_6nd_omits(ref_manifest):
    c, m = reference(), ref_manifest
    tc = cap.training_compute(c, manifest=m)
    assert tc.positions == 4.75e12
    assert tc.backbone == pytest.approx(6 * m.active_incl_head * 4.75e12)  # the tied head runs on every position
    assert tc.mtp / tc.backbone == pytest.approx((m.resident["mtp"] + m.head) / m.active_incl_head)
    assert 0.12 < tc.mtp / tc.backbone < 0.14
    # attention: 6 * (12 backbone + 1 MTP GQA layers) * mean packed length * 4096 per position, per stage
    stages = [(3.5e12, 4096), (0.75e12, 20480), (0.35e12, 81920), (0.15e12, 196608)]
    assert tc.attention == pytest.approx(sum(6 * 13 * ctx * 4096 * pos for pos, ctx in stages))
    assert 0 < tc.recurrent / tc.backbone < 0.01
    assert 2.0e23 < tc.total < 2.3e23 and 0.3 < tc.overhead_vs_6nd < 0.45
    assert tc.encoders == pytest.approx(3 * cap.training_compute(c, manifest=m, encoders_trainable=False).encoders)
    # named hardware: ~53 days on 256 H100 at the typical BF16 profile, ~20 on B200 MXFP8
    assert 50 < tc.days(256, "h100_bf16") < 57 and 18 < tc.days(256, "b200_mxfp8") < 22
    assert tc.days(256, "h100_bf16", 0) > tc.days(256, "h100_bf16") > tc.days(256, "h100_bf16", 2)


def test_training_compute_counts_mtp_compute_per_depth():
    from m63_omni.config import tiny
    from m63_omni.model.manifest import build_manifest

    one = tiny(multimodal={"enabled": False})
    shared = one.replace(mtp={"layers": 2, "draft_tokens": 2, "share_weights": True})
    separate = one.replace(mtp={"layers": 2, "draft_tokens": 2})
    t1, ts, tsep = (cap.training_compute(c) for c in (one, shared, separate))
    assert build_manifest(shared).resident["mtp"] == build_manifest(one).resident["mtp"]  # weights shared
    assert ts.mtp == pytest.approx(2 * t1.mtp) and tsep.mtp == pytest.approx(2 * t1.mtp)  # compute is not
    assert t1.encoders == 0  # no multimodal front end


def test_session_footprint_separates_kv_state_and_checkpoints():
    c = reference()
    base = cap.session_footprint(c, 131072)
    assert base.total == cap.kv_bytes(c, 131072) + 18 * MiB + 3 * MiB  # the blueprint lower bound is unchanged
    fp8 = cap.session_footprint(c, 131072, kv_dtype_bytes=1, state_dtype_bytes=2, memory_dtype_bytes=2)
    assert fp8.kv == base.kv / 2 and fp8.gdn == base.gdn and fp8.memory == base.memory
    with pytest.raises(ValueError, match="FP8 state"):
        cap.session_footprint(c, 131072, kv_dtype_bytes=1, state_dtype_bytes=1)
    s = cap.serving_footprint(c, 131072)
    assert s.mtp_kv == cap.kv_bytes(c, 131072, 1) / 12  # one MTP GQA layer next to 12 backbone ones
    assert s.checkpoints == 16 * cap.gdn_state_bytes(c, 2, include_conv=True)  # every 8K tokens
    assert 3.5e9 < s.total < 4.2e9
    padded = cap.session_footprint(c, 1000, page_tokens=256)
    assert padded.kv == cap.kv_bytes(c, 1024)
    n_lb = cap.sessions_per_deployment(c, 62.48e9, 4, 80e9, 131072)
    n_srv = cap.sessions_per_deployment(c, 62.48e9, 4, 80e9, 131072, footprint=cap.serving_footprint(c, 131072, kv_dtype="bfloat16"))
    assert n_srv < n_lb  # MTP KV and checkpoints cost capacity


def test_recurrent_state_counts_kda_layers_like_gdn():
    c = reference()
    k = c.replace(model={"hybrid_pattern": ["kda", "kda", "kda", "gated_gqa"]})
    assert cap.gdn_state_bytes(k) == cap.gdn_state_bytes(c) == 18 * MiB
    assert cap.kv_bytes_per_token(c.replace(attention={"kv_heads": 4})) == cap.kv_bytes_per_token(c) / 2
