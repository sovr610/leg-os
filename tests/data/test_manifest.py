"""Provenance ledger, rights refusal, immutable manifests, reproducible order (§8, §15, §16)."""

from __future__ import annotations

import dataclasses

import pytest

from m63_omni.common import Modality
from m63_omni.data.manifest import (
    ConsentStatus,
    DataManifest,
    ManifestEntry,
    ManifestError,
    ProvenanceLedger,
    RightsStatus,
    SourceRecord,
    TakedownStatus,
    TransformRecord,
    batch_order,
)

TRANSFORMS = [("inventory_provenance", {}), ("dedup", {"near_threshold": 0.8}), ("tokenize_pack", {"seq_len": 64})]


def _src(sid: str = "web", **kw) -> SourceRecord:
    base = dict(source_id=sid, version="2026-10-01", license="CC-BY-4.0", rights_status="cleared",
                consent="not_required", takedown="none", modality=Modality.TEXT, uri=f"s3://bucket/{sid}")
    base.update(kw)
    return SourceRecord(**base)


def _entries(src: SourceRecord, n: int) -> list[ManifestEntry]:
    return [ManifestEntry(f"{src.source_id}-{i}", src.source_id, src.version, f"sha256:{i:064x}", 0, 10 + i) for i in range(n)]


def _manifest(n: int = 10, seed: int = 7, **ordering) -> DataManifest:
    s = _src()
    return DataManifest.build([s], TRANSFORMS, seed, _entries(s, n), ordering=ordering or None)


@pytest.mark.parametrize(
    "bad,reason",
    [
        ({"rights_status": "pending"}, "rights_pending"),
        ({"rights_status": "rejected"}, "rights_rejected"),
        ({"takedown": "requested"}, "takedown_requested"),
        ({"takedown": "enforced"}, "takedown_enforced"),
        ({"consent": "withdrawn"}, "consent_withdrawn"),
        ({"consent": "missing"}, "consent_missing"),
    ],
)
def test_manifest_refuses_uncleared_or_taken_down_sources(bad, reason):
    good, src = _src("good"), _src("bad", **bad)
    assert not src.eligible and reason in src.ineligibility_reasons()
    with pytest.raises(ManifestError, match=reason):
        DataManifest.build([good, src], TRANSFORMS, 0, _entries(good, 2))


def test_manifest_refuses_inconsistent_entries():
    s = _src()
    with pytest.raises(ManifestError, match="unlisted source"):
        DataManifest.build([s], TRANSFORMS, 0, [ManifestEntry("d", "other", "v", "sha256:0")])
    with pytest.raises(ManifestError, match="pins"):
        DataManifest.build([s], TRANSFORMS, 0, [ManifestEntry("d", "web", "old-version", "sha256:0")])
    with pytest.raises(ManifestError, match="duplicate document"):
        DataManifest.build([s], TRANSFORMS, 0, _entries(s, 1) * 2)
    with pytest.raises(ManifestError, match="transformations"):
        DataManifest.build([s], [], 0, _entries(s, 1))


def test_manifest_is_immutable_and_hashed(tmp_path):
    m = _manifest()
    assert m.manifest_hash.startswith("sha256:") and m.verify()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.seed = 8  # type: ignore[misc]
    assert m.transformations[1].params == {"near_threshold": 0.8}
    m.transformations[1].params["near_threshold"] = 0.1  # a copy; the manifest cannot change
    assert m.transformations[1].params == {"near_threshold": 0.8}
    assert _manifest().manifest_hash == m.manifest_hash
    assert _manifest(seed=8).manifest_hash != m.manifest_hash
    # persistence round trip and tamper detection
    loaded = DataManifest.load(m.save(tmp_path / "manifest.json"))
    assert loaded == m and loaded.manifest_hash == m.manifest_hash
    d = m.to_dict()
    d["seed"] = 99
    with pytest.raises(ManifestError, match="hash mismatch"):
        DataManifest.from_dict(d)


def test_batch_order_is_reproducible_from_manifest_alone(tmp_path):
    m = _manifest(n=23, batch_size=4)
    e0 = batch_order(m, 0)
    assert e0 == batch_order(m, 0)
    assert e0 == batch_order(DataManifest.from_json(m.to_json()), 0)
    flat = [d for b in e0 for d in b]
    assert sorted(flat) == sorted(m.doc_ids) and len(flat) == 23
    assert [len(b) for b in e0] == [4, 4, 4, 4, 4, 3]
    assert batch_order(m, 1) != e0  # each epoch reshuffles
    assert flat != list(m.doc_ids)  # actually permuted
    dropped = _manifest(n=23, batch_size=4, drop_last=True)
    assert [len(b) for b in batch_order(dropped, 0)] == [4] * 5
    with pytest.raises(ValueError):
        batch_order(m, -1)


def test_ledger_rights_lifecycle_and_lineage():
    led = ProvenanceLedger()
    led.register(_src("web"))
    led.register(_src("licensed_av", rights_status="pending", modality=Modality.VIDEO, consent="obtained"))
    assert [s.source_id for s in led.eligible_sources()] == ["web"]
    led.update("licensed_av", rights_status=RightsStatus.CLEARED, reason="license signed")
    assert led.get("licensed_av").eligible
    led.record_decision("doc-1", "licensed_av", "inventory_provenance", "admitted")
    led.record_decision("doc-2", "licensed_av", "dedup", "dropped", "exact_duplicate", {"original": "doc-1"})
    led.request_takedown("licensed_av", reason="rights holder request")
    rec = led.get("licensed_av")
    assert rec.takedown is TakedownStatus.REQUESTED and not rec.eligible
    assert led.documents_from("licensed_av") == ["doc-1"]
    kinds = [e.kind for e in led.history("licensed_av")]
    assert kinds == ["register", "update", "takedown"]
    assert led.history("licensed_av")[-1].payload["affected_documents"] == ["doc-1"]
    drops = led.decisions(action="dropped")
    assert len(drops) == 1 and drops[0].reason == "exact_duplicate" and drops[0].payload == {"original": "doc-1"}
    assert led.verify_chain()
    with pytest.raises(ManifestError, match="takedown_requested"):
        DataManifest.build(led.sources(), TRANSFORMS, 0)
    with pytest.raises(KeyError):
        led.get("nope")
    assert led.get("web").consent is ConsentStatus.NOT_REQUIRED


def test_transform_record_params_are_canonical():
    a = TransformRecord.create("x", {"b": 1, "a": [1, 2]})
    b = TransformRecord.create("x", {"a": [1, 2], "b": 1})
    assert a == b and a.params_json == '{"a":[1,2],"b":1}'
