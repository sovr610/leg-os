"""Media through sessions: artifact-store blobs, replay, governance, deletion (§5, §6, §10)."""

import json

import pytest
import torch

from m63_omni.agent.artifacts import ArtifactNotFound, ArtifactStore
from m63_omni.common import Trust
from m63_omni.model.interleave import AVPart, ImagePart, TextPart
from m63_omni.runtime.session import Session


def _parts(mm_model, trust=Trust.USER, seed=0):
    g = torch.Generator().manual_seed(seed)
    sr = mm_model.cfg.multimodal.audio_sample_rate
    return [
        TextPart([5, 6, 7]),
        ImagePart(torch.randn(3, 16, 16, generator=g), 4, "img_1", trust),
        AVPart("vid_1", torch.randn(3, 3, 16, 16, generator=g), torch.tensor([0, 700, 1300]),
               torch.randn(int(sr * 1.5), generator=g), tokens_per_frame=2, trust=trust),
    ]


def test_events_hold_references_not_media(mm_model):
    s = Session(mm_model, "t", "s")
    ev = s.ingest_media(_parts(mm_model))
    payload = json.dumps(ev.payload)
    assert len(payload) < 2000  # hashes and parameters only
    assert len(ev.payload["asset_refs"]) == 3 and all(h.startswith("sha256:") for h in ev.payload["asset_refs"])
    assert all(s.store.contains("t", h) for h in ev.payload["asset_refs"])
    assert s.tokens == ev.payload["positions"]


def test_media_replay_reproduces_live_state(mm_model):
    s = Session(mm_model, "t", "s")
    s.ingest_media(_parts(mm_model))
    s.ingest([9, 10, 11])
    s.ingest_media(_parts(mm_model, seed=1)[1:2])
    live, mem = s.next_logits().clone(), s.memory_hash()
    r = Session(mm_model, "t", "s", event_log=s.log, artifact_store=s.store)
    r.replay()
    torch.testing.assert_close(r.next_logits(), live, atol=1e-5, rtol=1e-5)
    assert r.memory_hash() == mem and r.tokens == s.tokens


def test_fork_and_reclaim_with_media(mm_model):
    s = Session(mm_model, "t", "s")
    first = s.ingest_media(_parts(mm_model))
    s.ingest([1, 2, 3])
    child = s.fork(first.event_id, "alt")
    assert child.tokens == first.payload["positions"]
    s.checkpoints.clear()
    s.reclaim()
    s.ingest([4])  # transparent rebuild from the log + artifact store
    ref = Session(mm_model, "t", "ref", artifact_store=s.store)
    ref.ingest_media(_parts(mm_model))
    ref.ingest([1, 2, 3])
    ref.ingest([4])
    torch.testing.assert_close(s.next_logits(), ref.next_logits(), atol=1e-5, rtol=1e-5)


def test_untrusted_media_never_writes_memory(mm_model):
    s = Session(mm_model, "t", "s")
    h0 = s.memory_hash()
    s.ingest_media(_parts(mm_model, trust=Trust.UNTRUSTED))
    assert s.memory_hash() == h0
    assert any(e.payload.get("reason") == "untrusted_content_quarantined" for e in s.log.replay() if e.kind == "note")
    s.ingest_media(_parts(mm_model, trust=Trust.USER))
    assert s.memory_hash() != h0


def test_delete_releases_media_but_not_other_sessions_copy(mm_model):
    store = ArtifactStore()
    a = Session(mm_model, "t", "a", artifact_store=store)
    b = Session(mm_model, "t", "b", artifact_store=store)
    ev = a.ingest_media(_parts(mm_model))
    b.ingest_media(_parts(mm_model)[1:2])  # same image bytes
    image_hash = ev.payload["multimodal"][1]["blob"]
    a.delete()
    assert a.verify_deleted()
    assert store.contains("t", image_hash)  # still owned by session b
    for h in ev.payload["asset_refs"]:
        if h != image_hash:
            assert not store.contains("t", h)
    b.delete()
    assert b.verify_deleted() and not store.contains("t", image_hash)


def test_media_blobs_are_tenant_scoped(mm_model):
    s = Session(mm_model, "t", "s")
    ev = s.ingest_media(_parts(mm_model)[1:2])
    with pytest.raises(ArtifactNotFound):
        s.store.get_bytes("other-tenant", ev.payload["asset_refs"][0])
