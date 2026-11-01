"""Step 1 gate: the manifest reproduces the 62.48B reference when instantiated at full shape (§3, §12)."""

from pathlib import Path

import pytest
import torch

from m63_omni.config import load_config, reference, tiny
from m63_omni.model.card import model_card, model_card_markdown
from m63_omni.model.m63 import M63Omni
from m63_omni.model.manifest import BLUEPRINT_REFERENCE, analytic_backbone, build_manifest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def full():
    return build_manifest(reference())


def test_total_and_active_match_blueprint(full):
    assert full.total == pytest.approx(BLUEPRINT_REFERENCE["total"], rel=5e-4)
    assert full.active_excl_head == pytest.approx(BLUEPRINT_REFERENCE["active_excl_head"], rel=5e-3)
    assert full.active_incl_head == pytest.approx(BLUEPRINT_REFERENCE["active_incl_head"], rel=1e-3)


@pytest.mark.parametrize("cat", ["routed_experts", "shared_experts", "gdn_mixers", "gqa_mixers", "embedding", "routers_memory"])
def test_backbone_categories_match_table(full, cat):
    assert full.resident[cat] == pytest.approx(BLUEPRINT_REFERENCE[cat], rel=2e-3)


def test_encoders_and_mtp_close_to_reference(full):
    # Encoders and MTP depend on design details the blueprint leaves open; stay within 2%.
    assert full.resident["multimodal"] == pytest.approx(BLUEPRINT_REFERENCE["multimodal"], rel=0.02)
    assert full.resident["mtp"] == pytest.approx(BLUEPRINT_REFERENCE["mtp"], rel=0.02)


def test_module_counts_equal_closed_form(full):
    a = analytic_backbone(reference())
    for k in ("routed_experts", "shared_experts", "gdn_mixers", "gqa_mixers", "embedding"):
        assert full.resident[k] == a[k], k
    assert full.resident["routed_experts"] == 48 * 64 * 3 * 4096 * 1536


def test_manifest_counts_real_instantiated_model():
    cfg = tiny()
    torch.manual_seed(0)
    m = M63Omni(cfg)
    assert build_manifest(cfg).total == sum(p.numel() for p in m.parameters())


def test_pilot_and_proxy_sizes_in_blueprint_ranges():
    pilot = build_manifest(load_config(ROOT / "configs" / "pilot.yaml"))
    assert 8e9 <= pilot.total <= 12e9
    assert 1.5e9 <= pilot.active_excl_head <= 2.6e9
    for name in ("proxy_300m", "proxy_1b3"):
        p = build_manifest(load_config(ROOT / "configs" / f"{name}.yaml"))
        assert 0.3e9 <= p.total <= 1.3e9, name


def test_model_card_generated_from_same_source():
    card = model_card(reference())
    assert card["kv_bytes_per_token_bf16"] == 49152
    assert card["parameters"]["total"] == build_manifest(reference()).total
    md = model_card_markdown(reference())
    assert reference().fingerprint() in md and "48 KiB" in md


def test_kda_manifest_matches_closed_form_and_card():
    cfg = reference().replace(model={"hybrid_pattern": ["kda", "kda", "kda", "gated_gqa"]})
    m = build_manifest(cfg)
    assert m.resident["kda_mixers"] == analytic_backbone(cfg)["kda_mixers"] and m.resident["gdn_mixers"] == 0
    assert 0.80 < m.resident["kda_mixers"] / build_manifest(reference()).resident["gdn_mixers"] < 0.90  # low-rank gates
    assert "kda_mixers" in m.to_markdown() and "kda_mixers" not in build_manifest(reference()).to_markdown()
    assert "36 KDA" in model_card_markdown(cfg)
    small = tiny(model={"hybrid_pattern": ["kda", "gdn", "kda", "gated_gqa"]})
    torch.manual_seed(0)
    assert build_manifest(small).total == sum(p.numel() for p in M63Omni(small).parameters())
