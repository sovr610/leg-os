"""Time-aligned multimodal sequence construction (blueprint §5).

All modalities map into the residual stream annotated with modality IDs, asset
IDs, trust labels, and explicit timestamps. Audio and video are interleaved in
timestamp order per AV window (default 4 s)::

    <|video|> ( <|av_chunk|> <|ts|> [video tokens of window w] [audio tokens of window w] )* <|/video|>

which costs exactly two marker positions per window (§5's 3,600 markers for a
two-hour session). ``<|ts|>`` carries a learned time embedding of the window
start. Media positions hold ``<|media|>`` placeholder IDs whose embeddings are
replaced by encoder outputs. Raw binary assets never enter the stream — only
encoder representations plus typed references (asset IDs, content hashes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import torch

from ..common import Modality, Trust, content_hash
from .m63 import M63Omni

NO_TIME = -1


@dataclass
class TextPart:
    ids: Sequence[int]
    trust: Trust = Trust.USER


@dataclass
class ImagePart:
    image: torch.Tensor  # [3, H, W]
    n_tokens: int = 64
    asset_id: str = ""
    trust: Trust = Trust.USER


@dataclass
class AVPart:
    asset_id: str
    frames: Optional[torch.Tensor] = None  # [F, 3, H, W]
    frame_times_ms: Optional[torch.Tensor] = None  # [F]
    audio: Optional[torch.Tensor] = None  # [S] at the configured sample rate
    tokens_per_frame: int = 16
    window_ms: Optional[int] = None  # default: the audio encoder's chunk duration (4 s in the reference)
    high_fidelity_audio: bool = False
    trust: Trust = Trust.USER


Part = Union[TextPart, ImagePart, AVPart]


@dataclass
class MultimodalSequence:
    input_ids: torch.Tensor  # [1, T] with <|media|> placeholders
    inputs_embeds: torch.Tensor  # [1, T, d]
    modality_ids: torch.Tensor  # [1, T]
    timestamps_ms: torch.Tensor  # [1, T] (-1 where not applicable)
    trust: list[str]
    asset_ids: list[Optional[str]]
    content_hash: str
    ctc_logits: list[torch.Tensor] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.input_ids.shape[1]

    def text_labels(self) -> torch.Tensor:
        """Next-token labels that only supervise text/tool positions (media targets are ignored)."""
        ids = self.input_ids[0]
        mod = self.modality_ids[0]
        lab = torch.cat([ids[1:], ids.new_full((1,), -100)])
        nxt_mod = torch.cat([mod[1:], mod.new_full((1,), int(Modality.TEXT))])
        lab = lab.masked_fill(nxt_mod != int(Modality.TEXT), -100)
        return lab.unsqueeze(0)


class _Builder:
    def __init__(self, model: M63Omni) -> None:
        from ..data.tokenizer import special_ids_for_config

        self.model = model
        self.sp = special_ids_for_config(model.cfg)
        self.ids: list[int] = []
        self.mod: list[int] = []
        self.ts: list[int] = []
        self.trust: list[str] = []
        self.assets: list[Optional[str]] = []
        self.embeds: list[torch.Tensor] = []  # [n, d] runs
        self.pending_text: list[int] = []
        self.hash_parts: list = []

    def _flush_text(self) -> None:
        if self.pending_text:
            t = torch.tensor(self.pending_text, dtype=torch.long, device=self.model.embed.weight.device)
            self.embeds.append(self.model.embed(t))
            self.pending_text = []

    def token(self, name_or_id: Union[str, int], trust: Trust, modality: Modality = Modality.TEXT,
              asset: Optional[str] = None, ts: int = NO_TIME, extra: Optional[torch.Tensor] = None) -> None:
        tid = self.sp[name_or_id] if isinstance(name_or_id, str) else int(name_or_id)
        self.ids.append(tid)
        self.mod.append(int(modality))
        self.ts.append(ts)
        self.trust.append(trust.value)
        self.assets.append(asset)
        if extra is None:
            self.pending_text.append(tid)
        else:
            self._flush_text()
            t = torch.tensor([tid], dtype=torch.long, device=self.model.embed.weight.device)
            self.embeds.append(self.model.embed(t) + extra.unsqueeze(0))

    def media(self, feats: torch.Tensor, modality: Modality, trust: Trust, asset: str, ts: int) -> None:
        self._flush_text()
        mm = self.model.multimodal
        n = feats.shape[0]
        self.embeds.append(feats + mm.modality_embed.weight[int(modality)])
        self.ids.extend([self.sp["<|media|>"]] * n)
        self.mod.extend([int(modality)] * n)
        self.ts.extend([ts] * n)
        self.trust.extend([trust.value] * n)
        self.assets.extend([asset] * n)


def build_sequence(model: M63Omni, parts: Sequence[Part]) -> MultimodalSequence:
    if model.multimodal is None:
        raise ValueError("model was built without the multimodal front end")
    mm = model.multimodal
    b = _Builder(model)
    ctc: list[torch.Tensor] = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.trust is Trust.UNTRUSTED:
                b.token("<|trust_untrusted|>", part.trust)
            for t in part.ids:
                b.token(int(t), part.trust)
            b.hash_parts.append(["text", list(map(int, part.ids)), part.trust.value])
        elif isinstance(part, ImagePart):
            b.token("<|image|>", part.trust, asset=part.asset_id)
            feats = mm.encode_image(part.image.unsqueeze(0), part.n_tokens)[0]
            b.media(feats, Modality.IMAGE, part.trust, part.asset_id, NO_TIME)
            b.token("<|/image|>", part.trust, asset=part.asset_id)
            b.hash_parts.append(["image", part.asset_id, content_hash(part.image.detach().cpu().numpy().tobytes()),
                                 part.n_tokens, part.trust.value])
        elif isinstance(part, AVPart):
            ctc.extend(_add_av(b, mm, part))
        else:
            raise TypeError(f"unknown part {type(part).__name__}")
    b._flush_text()
    dev = model.embed.weight.device
    preprocessing = {"patch": mm.cfg.patch_size, "sr": mm.cfg.audio_sample_rate, "hop_ms": mm.cfg.hop_ms}
    return MultimodalSequence(
        input_ids=torch.tensor([b.ids], dtype=torch.long, device=dev),
        inputs_embeds=torch.cat(b.embeds, dim=0).unsqueeze(0),
        modality_ids=torch.tensor([b.mod], dtype=torch.long, device=dev),
        timestamps_ms=torch.tensor([b.ts], dtype=torch.long, device=dev),
        trust=b.trust,
        asset_ids=b.assets,
        content_hash=content_hash([model.cfg.fingerprint(), preprocessing, b.hash_parts]),
        ctc_logits=ctc,
    )


def _add_av(b: _Builder, mm, part: AVPart) -> list[torch.Tensor]:
    cfg = mm.cfg
    has_v, has_a = part.frames is not None, part.audio is not None
    if not (has_v or has_a):
        raise ValueError(f"AV asset {part.asset_id} has neither frames nor audio")
    a_rate = mm.audio.tokens_per_second(part.high_fidelity_audio)
    chunk_ms = cfg.audio_chunk_tokens * (2 if part.high_fidelity_audio else 1) / a_rate * 1000.0
    window = part.window_ms or int(round(chunk_ms))
    if has_a and abs(window / chunk_ms - round(window / chunk_ms)) > 1e-6:
        raise ValueError("AV window must be a multiple of the audio encoder chunk so windows stay causal")
    dur = 0.0
    vfeats = afeats = None
    ctc: list[torch.Tensor] = []
    if has_v:
        if part.frame_times_ms is None or len(part.frame_times_ms) != len(part.frames):
            raise ValueError("frame_times_ms must align with frames")
        times = part.frame_times_ms.float()
        if bool((times[1:] < times[:-1]).any()):
            raise ValueError("frame timestamps must be non-decreasing")
        vfeats = mm.encode_video(part.frames, times, part.tokens_per_frame, window)
        dur = max(dur, float(times.max()) + 1.0)
    if has_a:
        afeats, actc, a_end = mm.encode_audio(part.audio, part.high_fidelity_audio)
        afeats = afeats[0]
        ctc.append(actc[0])
        dur = max(dur, part.audio.shape[-1] / cfg.audio_sample_rate * 1000.0)
        a_start = a_end - 1000.0 / a_rate
    open_tok, close_tok = ("<|video|>", "<|/video|>") if has_v else ("<|audio|>", "<|/audio|>")
    b.token(open_tok, part.trust, asset=part.asset_id)
    n_windows = max(1, math.ceil(dur / window))
    for w in range(n_windows):
        lo, hi = w * window, (w + 1) * window
        b.token("<|av_chunk|>", part.trust, asset=part.asset_id, ts=lo)
        b.token("<|ts|>", part.trust, asset=part.asset_id, ts=lo,
                extra=mm.time_embedding(torch.tensor([float(lo)], device=b.model.embed.weight.device))[0])
        if has_v:
            sel = ((part.frame_times_ms >= lo) & (part.frame_times_ms < hi)).nonzero().squeeze(-1)
            for f in sel.tolist():
                b.media(vfeats[f], Modality.VIDEO, part.trust, part.asset_id, int(part.frame_times_ms[f]))
        if has_a:
            sel = ((a_start >= lo) & (a_start < hi)).nonzero().squeeze(-1)
            if sel.numel():
                b.media(afeats[sel], Modality.AUDIO, part.trust, part.asset_id, lo)
    b.token(close_tok, part.trust, asset=part.asset_id)
    meta = {"frames": None if not has_v else part.frame_times_ms.tolist(), "audio_len": None if not has_a else int(part.audio.shape[-1]),
            "tpf": part.tokens_per_frame, "window": window, "hifi": part.high_fidelity_audio}
    digest = [content_hash(part.frames.detach().cpu().numpy().tobytes()) if has_v else None,
              content_hash(part.audio.detach().cpu().numpy().tobytes()) if has_a else None]
    b.hash_parts.append(["av", part.asset_id, digest, meta, part.trust.value])
    return ctc
