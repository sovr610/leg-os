"""Multimodal token economics, sampling policies, and AV chunk planning (blueprint §5).

§5 fixes the representation budget: video proxies at 0.5 fps x 16 tokens per
frame, short high-detail intervals at 1 fps x up to 64 tokens per frame, audio
at ~6.25 tokens/s (12.5 tokens/s for short high-fidelity intervals), images at
16-256 tokens by resolution, and audio/video chunks *interleaved in timestamp
order* every 2-4 s. Its worked example — a two-hour session costs
57,600 video + 45,000 audio + 3,600 marker + 8,000 text = 114,200 positions,
leaving 16,872 of a 131,072 window — is reproduced exactly by
:func:`estimate_session_tokens`.

"The loader must estimate token cost before encoding and select a sampling
policy explicitly rather than truncating silently": :func:`select_policy`
walks the candidate policies in a defined order, then the caller-enabled
fallback (shorter working interval or memory-backed chunked ingestion), and
raises :class:`TokenBudgetError` if nothing fits.

Counting rules (shared by the estimator and the planner, so they always
agree): a session of ``D`` ms is split into windows ``[k*C, min((k+1)*C, D))``.
Visual frames are sampled at times ``j / fps`` and audio representations start
at ``j / rate``; each belongs to the window containing its start time. Hence a
window ``[s, e)`` holds ``ceil(e*r) - ceil(s*r)`` samples (exact rational
arithmetic), and a partial final window is counted, never dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Iterable, Sequence

from ..common import Modality
from .tokenizer import special_id

__all__ = [
    "SamplingPolicy",
    "LONG_PROXY",
    "HIGH_FIDELITY_AUDIO",
    "HIGH_DETAIL",
    "POLICIES",
    "TokenEstimate",
    "estimate_session_tokens",
    "PolicyDecision",
    "TokenBudgetError",
    "select_policy",
    "AVChunk",
    "plan_av_chunks",
    "image_tokens",
    "interleave_av_tokens",
    "AV_MARKERS",
    "FALLBACKS",
    "with_tokens_per_frame",
]


class TokenBudgetError(ValueError):
    """No sampling policy fits the context window; nothing is truncated silently."""

    def __init__(self, message: str, *, estimates: Sequence["TokenEstimate"] = (), context_window: int = 0,
                 recommended_interval_s: float = 0.0) -> None:
        super().__init__(message)
        self.estimates = tuple(estimates)
        self.context_window = context_window
        self.recommended_interval_s = recommended_interval_s


@dataclass(frozen=True)
class SamplingPolicy:
    """A representation-budget policy (§5 table)."""

    name: str
    video_fps: float
    tokens_per_frame: int
    audio_tokens_per_second: float
    image_min_tokens: int = 16
    image_max_tokens: int = 256

    def __post_init__(self) -> None:
        if self.video_fps <= 0 or self.tokens_per_frame <= 0 or self.audio_tokens_per_second <= 0:
            raise ValueError(f"policy {self.name}: rates and tokens per frame must be positive")
        if not 1 <= self.image_min_tokens <= self.image_max_tokens:
            raise ValueError(f"policy {self.name}: need 1 <= image_min_tokens <= image_max_tokens")

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


#: §5 default long proxy: 0.5 fps x 16 tokens/frame, 6.25 audio tokens/s.
LONG_PROXY = SamplingPolicy("long_proxy", 0.5, 16, 6.25)
#: Default visual proxy with the 12.5 tokens/s high-fidelity audio rate.
HIGH_FIDELITY_AUDIO = SamplingPolicy("high_fidelity_audio", 0.5, 16, 12.5)
#: §5 short high-detail intervals: 1 fps x 64 tokens/frame, 12.5 audio tokens/s.
HIGH_DETAIL = SamplingPolicy("high_detail", 1.0, 64, 12.5)

POLICIES: dict[str, SamplingPolicy] = {p.name: p for p in (LONG_PROXY, HIGH_FIDELITY_AUDIO, HIGH_DETAIL)}

#: Two marker tokens per AV window (§5: "Two marker tokens every four seconds").
AV_MARKERS: tuple[str, ...] = ("<|av_chunk|>", "<|ts|>")


# ------------------------------------------------------------------ counting
def _frac(x: float) -> Fraction:
    return Fraction(str(x)) if isinstance(x, float) else Fraction(x)


def _ceil(q: Fraction) -> int:
    return -((-q.numerator) // q.denominator)


def _samples_in(start_ms: int, end_ms: int, rate: Fraction) -> int:
    """Number of samples at times ``j / rate`` seconds with start in ``[start_ms, end_ms)``."""
    return _ceil(Fraction(end_ms, 1000) * rate) - _ceil(Fraction(start_ms, 1000) * rate)


def _windows(duration_ms: int, chunk_ms: int, start_ms: int = 0) -> list[tuple[int, int]]:
    out = []
    s = start_ms
    end = start_ms + duration_ms
    while s < end:
        e = min(s + chunk_ms, end)
        out.append((s, e))
        s = e
    return out


def _to_ms(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"invalid duration/time {seconds!r}")
    return int(round(seconds * 1000))


# ------------------------------------------------------------------ estimate
@dataclass(frozen=True)
class TokenEstimate:
    """Position budget for one session (§5)."""

    policy: str
    duration_s: float
    video: int
    audio: int
    markers: int
    text: int
    n_windows: int
    n_assets: int = 1

    @property
    def media(self) -> int:
        return self.video + self.audio

    @property
    def total(self) -> int:
        return self.video + self.audio + self.markers + self.text

    def remaining(self, context_window: int) -> int:
        return context_window - self.total

    def fits(self, context_window: int) -> bool:
        return self.total <= context_window


def estimate_session_tokens(
    duration_s: float,
    policy: SamplingPolicy = LONG_PROXY,
    chunk_seconds: float = 4.0,
    markers_per_chunk: int = 2,
    text_allowance: int = 8000,
    *,
    has_video: bool = True,
    has_audio: bool = True,
    n_assets: int = 1,
) -> TokenEstimate:
    """Estimate positions for a session BEFORE encoding (§5).

    ``estimate_session_tokens(7200)`` gives 57,600 video + 45,000 audio +
    3,600 markers + 8,000 text = 114,200 positions. Media and marker counts are
    per asset and multiplied by ``n_assets``. The per-window arithmetic matches
    :func:`plan_av_chunks` exactly for constant-rate sources.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if markers_per_chunk < 0 or text_allowance < 0 or n_assets < 1:
        raise ValueError("markers_per_chunk and text_allowance must be >= 0, n_assets >= 1")
    duration_ms = _to_ms(duration_s)
    windows = _windows(duration_ms, _to_ms(chunk_seconds))
    vr, ar = _frac(policy.video_fps), _frac(policy.audio_tokens_per_second)
    video = sum(_samples_in(s, e, vr) for s, e in windows) * policy.tokens_per_frame if has_video else 0
    audio = sum(_samples_in(s, e, ar) for s, e in windows) if has_audio else 0
    markers = len(windows) * markers_per_chunk if (has_video or has_audio) else 0
    return TokenEstimate(
        policy=policy.name,
        duration_s=float(duration_s),
        video=video * n_assets,
        audio=audio * n_assets,
        markers=markers * n_assets,
        text=int(text_allowance),
        n_windows=len(windows),
        n_assets=n_assets,
    )


# ------------------------------------------------------------ policy choice
@dataclass(frozen=True)
class PolicyDecision:
    """An explicit sampling decision; ``covers_full_duration`` is False only for
    a working interval, which the caller must handle deliberately."""

    policy: SamplingPolicy
    mode: str  # "full" | "working_interval" | "chunked_ingestion"
    estimate: TokenEstimate  # positions for ONE context (the interval actually encoded)
    interval_s: float
    n_segments: int
    context_window: int
    covers_full_duration: bool
    considered: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @property
    def remaining(self) -> int:
        return self.estimate.remaining(self.context_window)


FALLBACKS = ("working_interval", "chunked_ingestion")


def _max_interval_windows(
    duration_s: float, context_window: int, policy: SamplingPolicy, chunk_seconds: float, **kw: object
) -> int:
    """Largest number of whole windows whose estimate fits (binary search; cost is monotone)."""
    total_windows = max(1, math.ceil(_to_ms(duration_s) / _to_ms(chunk_seconds)))
    lo, hi = 0, total_windows
    while lo < hi:
        mid = (lo + hi + 1) // 2
        est = estimate_session_tokens(mid * chunk_seconds, policy, chunk_seconds, **kw)  # type: ignore[arg-type]
        if est.fits(context_window):
            lo = mid
        else:
            hi = mid - 1
    return lo


def select_policy(
    duration_s: float,
    context_window: int,
    policies: Sequence[SamplingPolicy] = (LONG_PROXY,),
    *,
    fallback: str | None = None,
    chunk_seconds: float = 4.0,
    markers_per_chunk: int = 2,
    text_allowance: int = 8000,
    has_video: bool = True,
    has_audio: bool = True,
    n_assets: int = 1,
) -> PolicyDecision:
    """Choose a sampling policy explicitly, in a defined order (§5).

    1. Each policy in ``policies`` (e.g. ``(HIGH_DETAIL, LONG_PROXY)``) over the
       full duration; the first that fits wins (``mode="full"``).
    2. If ``fallback="working_interval"``: the longest whole-window interval that
       fits under the cheapest candidate (``covers_full_duration=False``).
       If ``fallback="chunked_ingestion"``: the same interval length, ingested
       segment by segment through memory so the whole session is covered.
    3. Otherwise raise :class:`TokenBudgetError`, whose message names the
       estimate, the largest interval that would fit, and the alternatives
       (256K configuration, working interval, chunked ingestion).
    """
    if not policies:
        raise ValueError("at least one candidate policy is required")
    if fallback is not None and fallback not in FALLBACKS:
        raise ValueError(f"fallback must be one of {FALLBACKS} or None")
    kw = dict(
        markers_per_chunk=markers_per_chunk,
        text_allowance=text_allowance,
        has_video=has_video,
        has_audio=has_audio,
        n_assets=n_assets,
    )
    estimates = [estimate_session_tokens(duration_s, p, chunk_seconds, **kw) for p in policies]  # type: ignore[arg-type]
    considered = tuple((e.policy, e.total) for e in estimates)
    for p, est in zip(policies, estimates):
        if est.fits(context_window):
            return PolicyDecision(p, "full", est, float(duration_s), 1, context_window, True, considered)

    cheapest_i = min(range(len(policies)), key=lambda i: (estimates[i].total, i))
    cheapest = policies[cheapest_i]
    n_win = _max_interval_windows(duration_s, context_window, cheapest, chunk_seconds, **kw)
    interval_s = n_win * chunk_seconds
    if fallback is not None and n_win > 0:
        est = estimate_session_tokens(interval_s, cheapest, chunk_seconds, **kw)  # type: ignore[arg-type]
        n_segments = math.ceil(_to_ms(duration_s) / _to_ms(interval_s))
        return PolicyDecision(
            cheapest,
            fallback,
            est,
            interval_s,
            n_segments,
            context_window,
            fallback == "chunked_ingestion",
            considered + ((f"{cheapest.name}@{fallback}", est.total),),
        )
    tried = ", ".join(f"{name}={total:,}" for name, total in considered)
    if n_win > 0:
        hint = (
            f"largest fitting working interval under {cheapest.name} is {interval_s:g} s "
            f"({math.ceil(_to_ms(duration_s) / _to_ms(interval_s))} segments for chunked ingestion); "
            "pass fallback='working_interval' or 'chunked_ingestion', or use the 256K configuration"
        )
    else:
        hint = "even one chunk does not fit (reduce text_allowance or use the 256K configuration)"
    raise TokenBudgetError(
        f"{duration_s:g} s session does not fit a {context_window:,}-position window "
        f"(estimates: {tried}); {hint}. Refusing to truncate silently.",
        estimates=estimates,
        context_window=context_window,
        recommended_interval_s=interval_s,
    )


# ------------------------------------------------------------- chunk planning
@dataclass(frozen=True)
class AVChunk:
    """One modality chunk of an asset in a timestamp window (§5 interleaving)."""

    asset_id: str
    modality: Modality
    start_ms: int
    end_ms: int
    n_tokens: int
    window: int = 0
    n_frames: int = 0
    frame_times_ms: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        d = dict(self.__dict__)
        d["modality"] = int(self.modality)
        d["frame_times_ms"] = list(self.frame_times_ms)
        return d


def _sample_vfr(frame_times_ms: Sequence[int], rate: Fraction, end_ms: int, start_ms: int) -> list[int]:
    """First source frame in each sampling cell ``[j/rate, (j+1)/rate)`` (variable frame rate)."""
    picked: list[int] = []
    last_cell = None
    for t in sorted(frame_times_ms):
        if t < start_ms or t >= end_ms:
            continue
        q = Fraction(t, 1000) * rate
        cell = q.numerator // q.denominator
        if cell != last_cell:
            picked.append(int(t))
            last_cell = cell
    return picked


def plan_av_chunks(
    asset_id: str,
    duration_s: float,
    policy: SamplingPolicy = LONG_PROXY,
    chunk_seconds: float = 4.0,
    has_video: bool = True,
    has_audio: bool = True,
    *,
    start_s: float = 0.0,
    source_fps: float | None = None,
    frame_times_ms: Sequence[int] | None = None,
) -> list[AVChunk]:
    """Plan interleaved AV chunks in timestamp order (§5).

    For each ``chunk_seconds`` window the video chunk precedes the audio chunk.
    Missing channels are simply absent; a partial final window is kept; a
    window whose video has no sampled frame (e.g. a VFR gap) emits no video
    chunk. Frame sampling:

    * constant rate: ``min(policy.video_fps, source_fps)`` (cannot sample more
      frames than exist);
    * variable rate (``frame_times_ms`` given): the first source frame in each
      policy sampling cell, so gaps and bursts are handled faithfully.
    """
    if not (has_video or has_audio):
        raise ValueError(f"asset {asset_id!r} has neither video nor audio")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    duration_ms = _to_ms(duration_s)
    start_ms = _to_ms(start_s)
    end_ms = start_ms + duration_ms
    windows = _windows(duration_ms, _to_ms(chunk_seconds), start_ms)
    rate = _frac(policy.video_fps)
    if source_fps is not None:
        if source_fps <= 0:
            raise ValueError("source_fps must be positive")
        rate = min(rate, _frac(source_fps))
    vfr_frames: list[int] | None = None
    if has_video and frame_times_ms is not None:
        vfr_frames = _sample_vfr(frame_times_ms, _frac(policy.video_fps), end_ms, start_ms)
    arate = _frac(policy.audio_tokens_per_second)
    chunks: list[AVChunk] = []
    fi = 0
    for w, (s, e) in enumerate(windows):
        if has_video:
            if vfr_frames is not None:
                times = []
                while fi < len(vfr_frames) and vfr_frames[fi] < e:
                    times.append(vfr_frames[fi])
                    fi += 1
            else:
                j0, j1 = _ceil(Fraction(s, 1000) * rate), _ceil(Fraction(e, 1000) * rate)
                times = [int(round(float(Fraction(j * 1000) / rate))) for j in range(j0, j1)]
            if times:
                chunks.append(
                    AVChunk(asset_id, Modality.VIDEO, s, e, len(times) * policy.tokens_per_frame, w, len(times), tuple(times))
                )
        if has_audio:
            n = _samples_in(s, e, arate)
            if n > 0:
                chunks.append(AVChunk(asset_id, Modality.AUDIO, s, e, n, w))
    return chunks


# --------------------------------------------------------------------- images
def image_tokens(
    width: int,
    height: int,
    policy: SamplingPolicy = LONG_PROXY,
    *,
    patch_size: int = 14,
    max_side: int = 896,
    detail: str = "auto",
) -> int:
    """Visual tokens for an image: one resampler query per 4x4 patch block,
    clamped to the policy's 16-256 range (§5). Images larger than ``max_side``
    (64 patches x 14 px) are downscaled preserving aspect ratio first.
    ``detail="low"`` forces the minimum."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if detail == "low":
        return policy.image_min_tokens
    if detail != "auto":
        raise ValueError("detail must be 'auto' or 'low'")
    scale = min(1.0, max_side / max(width, height))
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    patches = math.ceil(w / patch_size) * math.ceil(h / patch_size)
    return int(min(policy.image_max_tokens, max(policy.image_min_tokens, math.ceil(patches / 16))))


# ----------------------------------------------------------------- sequencing
def interleave_av_tokens(
    chunks: Iterable[AVChunk],
    vocab_size: int,
    special_tokens: int,
    *,
    markers: Sequence[str] = AV_MARKERS,
) -> tuple[list[int], list[int], list[int]]:
    """Token IDs, modality IDs, and per-position timestamps (ms) for planned chunks.

    Each window emits its marker tokens (TEXT modality; ``len(markers)`` = the
    ``markers_per_chunk`` of the estimate) followed by ``<|media|>`` placeholder
    positions for its video chunk then its audio chunk, so the total length is
    exactly ``video + audio + markers`` from :func:`estimate_session_tokens`.
    """
    media_id = special_id("<|media|>", vocab_size, special_tokens)
    marker_ids = [special_id(m, vocab_size, special_tokens) for m in markers]
    ordered = sorted(chunks, key=lambda c: (c.start_ms, c.asset_id, c.modality != Modality.VIDEO))
    tokens: list[int] = []
    modality: list[int] = []
    times: list[int] = []
    current: tuple[str, int] | None = None
    for c in ordered:
        key = (c.asset_id, c.start_ms)
        if key != current:
            current = key
            tokens.extend(marker_ids)
            modality.extend([int(Modality.TEXT)] * len(marker_ids))
            times.extend([c.start_ms] * len(marker_ids))
        tokens.extend([media_id] * c.n_tokens)
        modality.extend([int(c.modality)] * c.n_tokens)
        times.extend([c.start_ms] * c.n_tokens)
    return tokens, modality, times


def with_tokens_per_frame(policy: SamplingPolicy, tokens_per_frame: int) -> SamplingPolicy:
    """A high-detail variant with fewer tokens per frame ("up to 64", §5)."""
    if not 1 <= tokens_per_frame <= 64:
        raise ValueError("tokens_per_frame must be in [1, 64]")
    return replace(policy, name=f"{policy.name}@{tokens_per_frame}tpf", tokens_per_frame=tokens_per_frame)
