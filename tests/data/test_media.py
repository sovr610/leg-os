"""Token economics, explicit policy selection, AV chunk interleaving (§5, Appendix B)."""

from __future__ import annotations

import pytest

from m63_omni.common import Modality
from m63_omni.data.media import (
    HIGH_DETAIL,
    HIGH_FIDELITY_AUDIO,
    LONG_PROXY,
    TokenBudgetError,
    estimate_session_tokens,
    image_tokens,
    interleave_av_tokens,
    plan_av_chunks,
    select_policy,
)
from m63_omni.data.tokenizer import special_id

WINDOW_128K = 131072


def test_blueprint_two_hour_budget_exact():
    est = estimate_session_tokens(2 * 3600, LONG_PROXY, chunk_seconds=4.0, markers_per_chunk=2, text_allowance=8000)
    assert est.video == 7200 * 0.5 * 16 == 57_600
    assert est.audio == 7200 * 6.25 == 45_000
    assert est.markers == 3_600
    assert est.text == 8_000
    assert est.total == 114_200
    assert est.remaining(WINDOW_128K) == 16_872
    assert est.fits(WINDOW_128K)


@pytest.mark.parametrize("duration", [1.0, 4.0, 10.5, 61.3, 7200.0, 3.999])
@pytest.mark.parametrize("policy", [LONG_PROXY, HIGH_DETAIL, HIGH_FIDELITY_AUDIO])
def test_estimate_matches_plan(duration, policy):
    est = estimate_session_tokens(duration, policy, text_allowance=0)
    chunks = plan_av_chunks("asset", duration, policy)
    assert est.video == sum(c.n_tokens for c in chunks if c.modality is Modality.VIDEO)
    assert est.audio == sum(c.n_tokens for c in chunks if c.modality is Modality.AUDIO)
    assert est.n_windows == len({c.window for c in chunks})
    toks, mods, _ = interleave_av_tokens(chunks, 512, 64)
    assert len(toks) == len(mods) == est.total  # video + audio + 2 markers per window


def test_select_policy_order_and_explicit_choice():
    two_hours = select_policy(7200, WINDOW_128K)
    assert two_hours.policy is LONG_PROXY and two_hours.mode == "full" and two_hours.remaining == 16_872
    # preferred high-detail first, then the default long proxy
    d = select_policy(7200, WINDOW_128K, (HIGH_DETAIL, LONG_PROXY))
    assert d.policy is LONG_PROXY and [n for n, _ in d.considered] == ["high_detail", "long_proxy"]
    short = select_policy(60, WINDOW_128K, (HIGH_DETAIL, LONG_PROXY))
    assert short.policy is HIGH_DETAIL and short.covers_full_duration


def test_select_policy_raises_instead_of_truncating():
    # §5: 12.5 tokens/s audio over two hours does not fit 128K ...
    with pytest.raises(TokenBudgetError) as exc:
        select_policy(7200, WINDOW_128K, (HIGH_FIDELITY_AUDIO,))
    err = exc.value
    assert "Refusing to truncate" in str(err) and "256K" in str(err)
    assert err.estimates[0].total > WINDOW_128K and 0 < err.recommended_interval_s < 7200
    # ... but does fit the 256K configuration
    assert select_policy(7200, 262144, (HIGH_FIDELITY_AUDIO,)).mode == "full"
    with pytest.raises(TokenBudgetError):
        select_policy(4 * 3600, WINDOW_128K)
    with pytest.raises(TokenBudgetError, match="even one chunk"):
        select_policy(60, 8000, fallback="working_interval")  # text allowance alone fills the window


@pytest.mark.parametrize("fallback,covers", [("working_interval", False), ("chunked_ingestion", True)])
def test_select_policy_explicit_fallbacks(fallback, covers):
    d = select_policy(4 * 3600, WINDOW_128K, fallback=fallback)
    assert d.mode == fallback and d.covers_full_duration is covers
    assert d.estimate.fits(WINDOW_128K) and d.interval_s % 4 == 0 and d.interval_s < 4 * 3600
    assert d.n_segments * d.interval_s >= 4 * 3600
    assert not estimate_session_tokens(d.interval_s + 4).fits(WINDOW_128K)  # the longest interval that fits


def test_av_chunks_interleave_video_then_audio():
    chunks = plan_av_chunks("a1", 12.0, LONG_PROXY)
    assert [(c.modality, c.start_ms) for c in chunks] == [
        (Modality.VIDEO, 0), (Modality.AUDIO, 0),
        (Modality.VIDEO, 4000), (Modality.AUDIO, 4000),
        (Modality.VIDEO, 8000), (Modality.AUDIO, 8000),
    ]
    assert all(c.asset_id == "a1" for c in chunks)
    assert all(c.n_tokens == 32 for c in chunks if c.modality is Modality.VIDEO)
    assert all(c.n_tokens == 25 for c in chunks if c.modality is Modality.AUDIO)
    assert chunks[2].frame_times_ms == (4000, 6000)
    starts = [c.start_ms for c in chunks]
    assert starts == sorted(starts)


def test_av_chunks_missing_channels_and_partial_window():
    no_audio = plan_av_chunks("v", 10.5, LONG_PROXY, has_audio=False)
    assert {c.modality for c in no_audio} == {Modality.VIDEO}
    assert [(c.start_ms, c.end_ms, c.n_frames) for c in no_audio] == [(0, 4000, 2), (4000, 8000, 2), (8000, 10500, 2)]
    no_video = plan_av_chunks("a", 10.5, LONG_PROXY, has_video=False)
    assert {c.modality for c in no_video} == {Modality.AUDIO}
    assert [c.n_tokens for c in no_video] == [25, 25, 16]  # 2.5 s * 6.25 = 15.625 -> 16 (partial window kept)
    assert sum(c.n_tokens for c in no_video) == 66
    tail = plan_av_chunks("x", 8.7, LONG_PROXY)[-2:]
    assert tail[0].modality is Modality.VIDEO and (tail[0].start_ms, tail[0].end_ms, tail[0].n_frames) == (8000, 8700, 1)
    assert tail[1].modality is Modality.AUDIO and tail[1].n_tokens == 5  # ceil(54.375) - 50
    with pytest.raises(ValueError):
        plan_av_chunks("none", 5.0, LONG_PROXY, has_video=False, has_audio=False)


def test_av_chunks_variable_frame_rate():
    # bursty VFR source: 30 fps for the first 3 s, a gap until 9 s, then sparse frames
    times = [int(i * 1000 / 30) for i in range(90)] + [9000, 9100, 11500]
    chunks = plan_av_chunks("vfr", 12.0, LONG_PROXY, frame_times_ms=times)
    video = [c for c in chunks if c.modality is Modality.VIDEO]
    # at most one frame per 2 s sampling cell; the gap window [4, 8) has no video chunk
    assert [(c.start_ms, c.frame_times_ms) for c in video] == [(0, (0, 2000)), (8000, (9000, 11500))]
    assert [c.window for c in chunks if c.modality is Modality.AUDIO] == [0, 1, 2]
    # a constant source slower than the policy cannot be oversampled
    slow = plan_av_chunks("slow", 8.0, LONG_PROXY, has_audio=False, source_fps=0.25)
    assert [c.n_frames for c in slow] == [1, 1]


def test_interleave_tokens_layout():
    chunks = plan_av_chunks("a", 6.0, LONG_PROXY)
    toks, mods, times = interleave_av_tokens(chunks, 512, 64)
    av, ts, media = (special_id(n, 512, 64) for n in ("<|av_chunk|>", "<|ts|>", "<|media|>"))
    assert toks[:2] == [av, ts] and mods[:2] == [Modality.TEXT] * 2
    assert toks[2:34] == [media] * 32 and set(mods[2:34]) == {Modality.VIDEO}
    assert set(mods[34:59]) == {Modality.AUDIO}
    assert toks[59:61] == [av, ts] and times[59] == 4000


def test_image_tokens_range():
    assert image_tokens(224, 224) == 16
    assert image_tokens(448, 448) == 64
    assert image_tokens(896, 896) == 256
    assert image_tokens(8000, 8000) == 256  # downscaled to 896 px first
    assert image_tokens(32, 32) == 16
    assert image_tokens(896, 896, detail="low") == 16
    with pytest.raises(ValueError):
        image_tokens(0, 10)
