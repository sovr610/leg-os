"""Native multimodal front end and time-aligned interleaving (§5, Appendix B "Multimodal alignment")."""

import pytest
import torch

from m63_omni.common import Modality, Trust
from m63_omni.data.tokenizer import special_ids_for_config
from m63_omni.model.interleave import AVPart, ImagePart, TextPart, build_sequence


@pytest.fixture
def av(mm_model):
    cfg = mm_model.cfg.multimodal
    g = torch.Generator().manual_seed(0)
    audio = torch.randn(int(cfg.audio_sample_rate * 2.5), generator=g)
    frames = torch.randn(4, 3, 16, 16, generator=g)
    times = torch.tensor([0, 400, 800, 1600])
    return audio, frames, times


def test_audio_rates(mm_model):
    a = mm_model.multimodal.audio
    assert a.tokens_per_second() == 6.25 and a.tokens_per_second(True) == 12.5
    with torch.no_grad():
        feats, ctc, ends = mm_model.multimodal.encode_audio(torch.randn(16000 * 2))
    assert feats.shape[1] == 12 and ctc.shape[-1] == mm_model.cfg.multimodal.ctc_vocab
    assert ends[0] == pytest.approx(160.0)


def test_interleaving_order_and_markers(mm_model, av):
    audio, frames, times = av
    sp = special_ids_for_config(mm_model.cfg)
    with torch.no_grad():
        seq = build_sequence(mm_model, [TextPart([5, 6]), AVPart("vid", frames, times, audio, tokens_per_frame=2)])
    ids, mods, ts = seq.input_ids[0].tolist(), seq.modality_ids[0].tolist(), seq.timestamps_ms[0].tolist()
    window = 640  # tiny config: 4 audio tokens x 160 ms
    n_windows = -(-2500 // window)
    assert ids.count(sp["<|av_chunk|>"]) == n_windows and ids.count(sp["<|ts|>"]) == n_windows
    starts = [i for i, t in enumerate(ids) if t == sp["<|av_chunk|>"]]
    for w, s in enumerate(starts):
        end = starts[w + 1] if w + 1 < len(starts) else ids.index(sp["<|/video|>"])
        block = mods[s + 2 : end]
        # video tokens precede audio tokens inside a window
        assert block == sorted(block, key=lambda m: 0 if m == Modality.VIDEO else 1)
        assert ts[s] == w * window
    assert mods.count(int(Modality.VIDEO)) == 4 * 2
    assert mods.count(int(Modality.AUDIO)) == 15  # floor(2.5 s * 6.25)


def test_later_media_cannot_change_earlier_windows(mm_model, av):
    audio, frames, times = av
    sr = mm_model.cfg.multimodal.audio_sample_rate
    a2, f2 = audio.clone(), frames.clone()
    a2[int(sr * 1.3) :] = torch.randn(a2[int(sr * 1.3) :].shape)
    f2[3] = torch.randn(3, 16, 16)
    with torch.no_grad():
        s1 = build_sequence(mm_model, [AVPart("v", frames, times, audio, tokens_per_frame=2)])
        s2 = build_sequence(mm_model, [AVPart("v", f2, times, a2, tokens_per_frame=2)])
        o1 = mm_model(inputs_embeds=s1.inputs_embeds, modality_ids=s1.modality_ids).logits
        o2 = mm_model(inputs_embeds=s2.inputs_embeds, modality_ids=s2.modality_ids).logits
    ts = s1.timestamps_ms[0]
    first_window_changed = 1280  # 1.3 s lies in window [1280, 1920)
    pos = int(((ts >= first_window_changed) | (ts == -1)).nonzero()[1])  # first position at/after that window
    torch.testing.assert_close(o1[:, :pos], o2[:, :pos], atol=0, rtol=0)
    assert s1.content_hash != s2.content_hash


@pytest.mark.parametrize("which", ["no_audio", "no_video"])
def test_missing_channels(mm_model, av, which):
    audio, frames, times = av
    part = AVPart("v", None if which == "no_video" else frames, None if which == "no_video" else times,
                  None if which == "no_audio" else audio, tokens_per_frame=2)
    with torch.no_grad():
        seq = build_sequence(mm_model, [part])
    mods = seq.modality_ids[0].tolist()
    assert (int(Modality.AUDIO) in mods) == (which != "no_audio")
    assert (int(Modality.VIDEO) in mods) == (which != "no_video")


def test_same_placeholders_different_content_hash(mm_model):
    a, b = torch.zeros(3, 16, 16), torch.ones(3, 16, 16)
    with torch.no_grad():
        s1 = build_sequence(mm_model, [ImagePart(a, 4, "img")])
        s2 = build_sequence(mm_model, [ImagePart(b, 4, "img")])
    assert s1.input_ids.tolist() == s2.input_ids.tolist()
    assert s1.content_hash != s2.content_hash


def test_untrusted_text_is_marked_and_labels_skip_media(mm_model):
    sp = special_ids_for_config(mm_model.cfg)
    with torch.no_grad():
        seq = build_sequence(mm_model, [TextPart([1, 2]), ImagePart(torch.zeros(3, 8, 8), 3, "i"),
                                        TextPart([3, 4], trust=Trust.UNTRUSTED)])
    ids = seq.input_ids[0].tolist()
    assert sp["<|trust_untrusted|>"] in ids
    assert seq.trust[-1] == "untrusted"
    labels = seq.text_labels()[0]
    mods = seq.modality_ids[0]
    nxt_media = torch.cat([mods[1:] != int(Modality.TEXT), torch.tensor([False])])
    assert (labels[nxt_media] == -100).all()


def test_vision_input_validation(mm_model):
    with pytest.raises(ValueError):
        mm_model.multimodal.vision(torch.zeros(1, 3, 15, 16))
    with pytest.raises(ValueError):
        mm_model.multimodal.vision(torch.zeros(1, 3, 4 * 9, 16))  # grid 9 > max 8
