"""Native multimodal encoders (blueprint §5).

* Vision: 24-layer, width-1,024 ViT (16 heads, patch 14) with factorised 2-D
  learned positions, followed by a spatial resampler (learned queries,
  cross-attention) producing 16-256 tokens per image/frame.
* Video: the shared ViT + resampler per frame, then a 4-layer temporal
  aggregator that is *frame-causal within each AV window* (a window's tokens
  never depend on later frames).
* Audio: 16 kHz input, causal 128-bin log-mel frontend (25 ms window, 10 ms
  hop), causal stride-2 convolutions downsampling by 16 (6.25 tokens/s; by 8
  for the 12.5 tokens/s high-fidelity mode), and a 20-layer width-1,024
  chunk-aware transformer (block-causal attention over the current and previous
  chunk; ``audio_attention: causal`` makes it token-causal with the same
  two-chunk lookback, so AV windows need not align with chunks). A small CTC
  head supports the ASR auxiliary loss. Token ``j`` depends only on audio
  before its end time ``(j + 1) / rate``.

"Native" means these encoders are trained jointly with the backbone from the
first full run (§5); it does not mean encoder-free.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import Modality
from ..config import MultimodalConfig
from .layers import sinusoidal_features

TIME_FEATURES = 64


class EncoderLayer(nn.Module):
    """Pre-LayerNorm transformer layer (MHA + GELU MLP), biases on."""

    def __init__(self, width: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, mlp_ratio * width)
        self.fc2 = nn.Linear(mlp_ratio * width, width)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        N, L, W = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(N, L, 3, self.heads, W // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(N, L, W))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class CrossAttnLayer(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln_q = nn.LayerNorm(width)
        self.ln_kv = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, mlp_ratio * width)
        self.fc2 = nn.Linear(mlp_ratio * width, width)

    def forward(self, queries: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        N, Lq, W = queries.shape
        Lk = ctx.shape[1]
        hd = W // self.heads
        q = self.q(self.ln_q(queries)).view(N, Lq, self.heads, hd).transpose(1, 2)
        k, v = self.kv(self.ln_kv(ctx)).view(N, Lk, 2, self.heads, hd).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = queries + self.proj(a.transpose(1, 2).reshape(N, Lq, W))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class VisionEncoder(nn.Module):
    def __init__(self, cfg: MultimodalConfig) -> None:
        super().__init__()
        W, p, G = cfg.vision_width, cfg.patch_size, cfg.max_patch_grid
        self.patch, self.grid = p, G
        self.patch_embed = nn.Conv2d(3, W, kernel_size=p, stride=p)
        self.row_pos = nn.Parameter(torch.zeros(G, W))
        self.col_pos = nn.Parameter(torch.zeros(G, W))
        self.layers = nn.ModuleList([EncoderLayer(W, cfg.vision_heads, cfg.vision_mlp_ratio) for _ in range(cfg.vision_layers)])
        self.ln_post = nn.LayerNorm(W)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [N, 3, H, W] with H, W multiples of the patch size -> [N, h*w, width]."""
        N, _, H, Wd = images.shape
        if H % self.patch or Wd % self.patch:
            raise ValueError(f"image size {H}x{Wd} not a multiple of patch {self.patch}")
        h, w = H // self.patch, Wd // self.patch
        if h > self.grid or w > self.grid:
            raise ValueError(f"patch grid {h}x{w} exceeds max {self.grid}; resize or tile explicitly")
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        pos = (self.row_pos[:h].unsqueeze(1) + self.col_pos[:w].unsqueeze(0)).reshape(h * w, -1)
        x = x + pos
        for layer in self.layers:
            x = layer(x)
        return self.ln_post(x)


class SpatialResampler(nn.Module):
    def __init__(self, cfg: MultimodalConfig) -> None:
        super().__init__()
        W = cfg.vision_width
        self.max_queries = cfg.resampler_max_queries
        self.queries = nn.Parameter(torch.zeros(cfg.resampler_max_queries, W))
        self.layers = nn.ModuleList([CrossAttnLayer(W, cfg.vision_heads, cfg.vision_mlp_ratio) for _ in range(cfg.resampler_layers)])

    def forward(self, patches: torch.Tensor, n_tokens: int) -> torch.Tensor:
        if not 1 <= n_tokens <= self.max_queries:
            raise ValueError(f"n_tokens must be in [1, {self.max_queries}]")
        q = self.queries[:n_tokens].unsqueeze(0).expand(patches.shape[0], -1, -1)
        for layer in self.layers:
            q = layer(q, patches)
        return q


class TemporalAggregator(nn.Module):
    def __init__(self, cfg: MultimodalConfig) -> None:
        super().__init__()
        W = cfg.vision_width
        self.time_proj = nn.Linear(TIME_FEATURES, W, bias=False)
        self.layers = nn.ModuleList([EncoderLayer(W, cfg.vision_heads, cfg.vision_mlp_ratio) for _ in range(cfg.temporal_layers)])

    def forward(self, frame_tokens: torch.Tensor, frame_times_ms: torch.Tensor, window_ms: int) -> torch.Tensor:
        """frame_tokens: [F, n, W]; frames are grouped into AV windows; attention is
        frame-causal inside a window and never crosses windows."""
        Fn, n, W = frame_tokens.shape
        x = frame_tokens + self.time_proj(sinusoidal_features(frame_times_ms / 1000.0, TIME_FEATURES)).unsqueeze(1)
        out = torch.empty_like(x)
        window = torch.div(frame_times_ms, window_ms, rounding_mode="floor").long()
        for w in torch.unique(window).tolist():
            sel = (window == w).nonzero().squeeze(-1)
            xs = x[sel].reshape(1, -1, W)  # [1, f*n, W]
            frame_idx = torch.arange(len(sel), device=x.device).repeat_interleave(n)
            mask = frame_idx.unsqueeze(0) <= frame_idx.unsqueeze(1)
            for layer in self.layers:
                xs = layer(xs, mask)
            out[sel] = xs.view(len(sel), n, W)
        return out


class MelFrontend(nn.Module):
    """Causal log-mel features: frame ``i`` depends only on samples before ``(i+1)*hop``."""

    def __init__(self, sample_rate: int, n_mels: int, window_ms: float, hop_ms: float) -> None:
        super().__init__()
        self.win = int(round(sample_rate * window_ms / 1000))
        self.hop = int(round(sample_rate * hop_ms / 1000))
        self.n_fft = 1 << (self.win - 1).bit_length()
        self.register_buffer("window", torch.hann_window(self.win, periodic=True), persistent=False)
        self.register_buffer("mel_fb", _mel_filterbank(sample_rate, self.n_fft, n_mels), persistent=False)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: [B, S] -> log-mel [B, n_mels, S // hop]."""
        B, S = wave.shape
        n_frames = S // self.hop
        x = F.pad(wave.float(), (self.win - self.hop, 0))
        frames = x.unfold(-1, self.win, self.hop)[:, :n_frames] * self.window
        spec = torch.fft.rfft(frames, n=self.n_fft).abs().pow(2)  # [B, frames, n_fft/2+1]
        mel = spec @ self.mel_fb
        return torch.log10(mel.clamp(min=1e-10)).transpose(1, 2).clamp(min=-8.0) / 4.0 + 1.0


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> torch.Tensor:
    def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    n_freq = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, n_freq)
    mels = torch.linspace(hz_to_mel(torch.tensor(0.0)), hz_to_mel(torch.tensor(sr / 2)), n_mels + 2)
    hz = mel_to_hz(mels)
    fb = torch.zeros(n_freq, n_mels)
    for m in range(n_mels):
        lo, c, hi = hz[m], hz[m + 1], hz[m + 2]
        up = (freqs - lo) / (c - lo).clamp(min=1e-6)
        down = (hi - freqs) / (hi - c).clamp(min=1e-6)
        fb[:, m] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


class AudioEncoder(nn.Module):
    def __init__(self, cfg: MultimodalConfig) -> None:
        super().__init__()
        W = cfg.audio_width
        self.cfg = cfg
        self.mel = MelFrontend(cfg.audio_sample_rate, cfg.mel_bins, cfg.window_ms, cfg.hop_ms)
        n_down = int(math.log2(cfg.audio_downsample))
        self.convs = nn.ModuleList([nn.Conv1d(cfg.mel_bins if i == 0 else W, W, kernel_size=3, stride=2) for i in range(n_down)])
        self.layers = nn.ModuleList([EncoderLayer(W, cfg.audio_heads, cfg.audio_mlp_ratio) for _ in range(cfg.audio_layers)])
        self.ln_post = nn.LayerNorm(W)
        self.ctc_head = nn.Linear(W, cfg.ctc_vocab)

    def forward(self, wave: torch.Tensor, high_fidelity: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """wave: [B, S] at the configured sample rate -> (features [B, L, W], ctc_logits [B, L, V])."""
        x = self.mel(wave)
        convs = self.convs[:-1] if high_fidelity else self.convs
        complete = x.shape[-1] // (2 ** len(convs))  # emit only tokens whose full frame span exists
        for conv in convs:
            x = F.gelu(conv(F.pad(x, (2, 0))))  # causal: left padding only
        x = x[..., :complete].transpose(1, 2)
        L = x.shape[1]
        x = x + sinusoidal_features(torch.arange(L, device=x.device).float(), x.shape[-1]).to(x.dtype)
        chunk = self.cfg.audio_chunk_tokens * (2 if high_fidelity else 1)
        pos = torch.arange(L, device=x.device)
        if self.cfg.audio_attention == "causal":  # mask[q, k]: k in (q - 2*chunk, q]
            lag = pos.unsqueeze(1) - pos.unsqueeze(0)
            mask = (lag >= 0) & (lag < 2 * chunk)
        else:  # mask[q, k]: k in the query's chunk or the previous one
            c = pos // chunk
            mask = (c.unsqueeze(0) <= c.unsqueeze(1)) & (c.unsqueeze(0) >= c.unsqueeze(1) - 1)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_post(x)
        return x, self.ctc_head(x)

    def tokens_per_second(self, high_fidelity: bool = False) -> float:
        rate = 1000.0 / self.cfg.hop_ms / self.cfg.audio_downsample
        return rate * (2 if high_fidelity else 1)


class MultimodalFrontEnd(nn.Module):
    """Encoders + projections into the shared residual stream."""

    def __init__(self, cfg: MultimodalConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg)
        self.resampler = SpatialResampler(cfg)
        self.temporal = TemporalAggregator(cfg)
        self.audio = AudioEncoder(cfg)
        self.vision_proj = nn.Sequential(nn.LayerNorm(cfg.vision_width), nn.Linear(cfg.vision_width, d_model))
        self.audio_proj = nn.Sequential(nn.LayerNorm(cfg.audio_width), nn.Linear(cfg.audio_width, d_model))
        self.modality_embed = nn.Embedding(len(Modality), d_model)
        self.time_proj = nn.Linear(TIME_FEATURES, d_model, bias=False)

    def reset_parameters(self, std: float) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=std, a=-3 * std, b=3 * std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.modality_embed.weight, std=std)
        for p in (self.vision.row_pos, self.vision.col_pos, self.resampler.queries):
            nn.init.normal_(p, std=std)

    def encode_image(self, images: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """[N, 3, H, W] -> [N, n_tokens, d_model]."""
        return self.vision_proj(self.resampler(self.vision(images), n_tokens))

    def encode_video(self, frames: torch.Tensor, frame_times_ms: torch.Tensor, tokens_per_frame: int, window_ms: int) -> torch.Tensor:
        """[F, 3, H, W] -> [F, tokens_per_frame, d_model] (frame-causal within AV windows)."""
        per_frame = self.resampler(self.vision(frames), tokens_per_frame)
        return self.vision_proj(self.temporal(per_frame, frame_times_ms.to(per_frame.device).float(), window_ms))

    def encode_audio(self, wave: torch.Tensor, high_fidelity: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[S] or [B, S] -> (tokens [B, L, d_model], ctc_logits [B, L, V], token_end_ms [L])."""
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        feats, ctc = self.audio(wave, high_fidelity)
        rate = self.audio.tokens_per_second(high_fidelity)
        ends = (torch.arange(feats.shape[1], device=feats.device).float() + 1) * (1000.0 / rate)
        return self.audio_proj(feats), ctc, ends

    def time_embedding(self, ms: torch.Tensor) -> torch.Tensor:
        return self.time_proj(sinusoidal_features(ms.float() / 1000.0, TIME_FEATURES))
