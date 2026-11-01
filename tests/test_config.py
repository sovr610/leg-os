"""Configuration: strict loading, validation rules, and the reference YAML (blueprint §11)."""

from pathlib import Path

import pytest

from m63_omni.config import ConfigError, load_config, reference, tiny

ROOT = Path(__file__).resolve().parent.parent


def test_full_yaml_is_the_reference_config():
    assert load_config(ROOT / "configs" / "full.yaml") == reference()
    assert load_config(ROOT / "configs" / "tiny.yaml") == tiny()


def test_reference_shape_matches_blueprint():
    c = reference()
    assert c.layer_kinds() == ["gdn", "gdn", "gdn", "gated_gqa"] * 12
    assert c.memory_anchor_indices() == [15, 31, 47]
    assert (c.moe.routed_experts, c.moe.shared_experts, c.moe.top_k, c.moe.expert_hidden) == (64, 1, 2, 1536)
    assert (c.attention.query_heads, c.attention.kv_heads, c.attention.head_dim, c.attention.rotary_dims) == (32, 8, 128, 32)


def test_fingerprint_is_stable_and_sensitive():
    assert reference().fingerprint() == reference().fingerprint()
    assert reference().replace(moe={"top_k": 4}).fingerprint() != reference().fingerprint()


@pytest.mark.parametrize(
    "override, message",
    [
        ({"moe": {"topk": 2}}, "unknown key"),
        ({"moe": {"token_dropping": True}}, "token dropping is forbidden"),
        ({"memory": {"anchors": [15]}}, "not an attention layer"),
        ({"memory": {"untrusted_persistent_writes": True}}, "untrusted_persistent_writes"),
        ({"gdn": {"inner_width": 1024}}, "inner_width"),
        ({"multimodal": {"audio_tokens_per_second": 12.5}}, "audio_tokens_per_second"),
        ({"model": {"layers": 46}}, "multiple of the hybrid pattern"),
        ({"gdn": {"decay_gate_dtype": "bfloat16"}}, "float32"),
        ({"moe": {"top_k": "2"}}, "expected int"),
    ],
)
def test_invalid_configs_are_rejected(override, message):
    with pytest.raises(ConfigError, match=message):
        reference().replace(**override)


def test_every_shipped_config_validates():
    paths = sorted((ROOT / "configs").rglob("*.yaml"))
    assert len(paths) >= 10
    for p in paths:
        load_config(p)


def test_extends_applies_parent_first():
    c = load_config(ROOT / "configs" / "tournament" / "ratio_7to1.yaml")
    assert c.model.hidden_size == 1024  # from proxy_1b3
    assert c.model.hybrid_pattern.count("gdn") == 7
    assert c.memory.anchors == (8, 16)
