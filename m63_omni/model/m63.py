"""M63-Omni backbone (blueprint §3, §4, §6, Appendix A).

48 pre-RMSNorm residual blocks arranged as 12 x ``[GDN, GDN, GDN, Gated GQA]``
(``kda`` may replace ``gdn`` in the pattern for the proxy tournament);
every block's mixer is followed by a sparse MoE FFN (64 routed + 1 shared,
top-2). Embeddings are tied with the LM head. With ``loop.enabled`` a middle
block of layers can run a second time with the same weights; each execution
keeps its own cache slot (see :meth:`M63Config.loop_schedule`). Neural-memory anchors sit at
attention layers 16/32/48 (1-indexed) and append their read tokens to that
layer's attention context. Per block::

    h_l     = x_l + Mixer_l(RMSNorm(x_l))
    x_{l+1} = h_l + MoE_l(RMSNorm(h_l))

This module is the *naive reference implementation*: optimised kernels may
replace pieces only after matching it (§11).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GATED_GQA, GDN, KDA, RECURRENT_KINDS, M63Config
from .attention import GatedGQA
from .cache import HybridCache, StreamIndex, resolve_stream
from .encoders import MultimodalFrontEnd
from .gdn import GatedDeltaNet
from .kda import KimiDeltaAttention
from .layers import RMSNorm
from .memory import MemoryState, NeuralMemoryAnchor
from .moe import MoELayer, RouterStats
from .mtp import MTPStack

IGNORE_INDEX = -100


class ContextLengthError(ValueError):
    """The stream would exceed the largest configured context target (never truncated silently, §5)."""


class HybridBlock(nn.Module):
    def __init__(self, cfg: M63Config, layer_idx: int, kind: str) -> None:
        super().__init__()
        d, eps = cfg.model.hidden_size, cfg.model.norm_eps
        self.layer_idx, self.kind = layer_idx, kind
        self.mixer_norm = RMSNorm(d, eps)
        if kind == GDN:
            self.mixer = GatedDeltaNet(d, cfg.gdn, eps)
        elif kind == KDA:
            self.mixer = KimiDeltaAttention(d, cfg.gdn, eps)
        else:
            self.mixer = GatedGQA(d, cfg.attention, eps)
        self.moe_norm = RMSNorm(d, eps)
        self.moe = MoELayer(d, cfg.moe)

    def forward(
        self,
        x: torch.Tensor,
        idx: StreamIndex,
        state,
        extra_kv: Optional[torch.Tensor],
        modality_ids: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        gdn_mode: str,
    ):
        if self.kind in RECURRENT_KINDS:
            m, new_state = self.mixer(self.mixer_norm(x), state, idx.reset, gdn_mode)
        else:
            m, new_state = self.mixer(self.mixer_norm(x), idx, state, extra_kv)
        h = x + m
        moe_out, stats = self.moe(self.moe_norm(h), modality_ids, valid)
        return h + moe_out, new_state, stats


@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor]
    hidden: torch.Tensor  # after final norm
    hidden_raw: torch.Tensor  # before final norm (MTP input)
    router_stats: list[RouterStats]
    anchor_hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    cache: Optional[HybridCache] = None
    stream: Optional[StreamIndex] = None
    lm_loss: Optional[torch.Tensor] = None
    lm_tokens: int = 0
    mtp_logits: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    mtp_tokens: int = 0
    memory_loss: Optional[torch.Tensor] = None
    memory_states: Optional[dict[int, MemoryState]] = None
    memory_info: list = field(default_factory=list)
    loop_passes: int = 1

    def router_aux_loss(self) -> torch.Tensor:
        return torch.stack([s.aux_loss for s in self.router_stats]).sum()

    def router_z_loss(self) -> torch.Tensor:
        return torch.stack([s.z_loss for s in self.router_stats]).sum()


class M63Omni(nn.Module):
    def __init__(self, cfg: M63Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        d = m.hidden_size
        self.embed = nn.Embedding(m.vocab_size, d)
        self.blocks = nn.ModuleList([HybridBlock(cfg, i, k) for i, k in enumerate(cfg.layer_kinds())])
        self.final_norm = RMSNorm(d, m.norm_eps)
        self.lm_head = None if m.tie_embeddings else nn.Linear(d, m.vocab_size, bias=False)
        self.memory = nn.ModuleDict({str(i): NeuralMemoryAnchor(d, cfg.memory, m.norm_eps) for i in cfg.memory_anchor_indices()})
        self.mtp = MTPStack(cfg) if cfg.mtp.enabled else None
        self.multimodal = MultimodalFrontEnd(cfg.multimodal, d) if cfg.multimodal.enabled else None
        self.reset_parameters()

    # ------------------------------------------------------------------ init
    def reset_parameters(self) -> None:
        std = self.cfg.model.init_std
        out_std = std / math.sqrt(2 * self.cfg.model.layers)
        nn.init.normal_(self.embed.weight, std=std)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, std=std)
        for block in self.blocks:
            for name, mod in block.mixer.named_modules():
                if isinstance(mod, nn.Linear):
                    s = out_std if name == "o_proj" else std
                    nn.init.trunc_normal_(mod.weight, std=s, a=-3 * s, b=3 * s)
            block.moe.reset_parameters(std, out_std)
            for sh in block.moe.shared:
                nn.init.trunc_normal_(sh.gate.weight, std=std, a=-3 * std, b=3 * std)
                nn.init.trunc_normal_(sh.up.weight, std=std, a=-3 * std, b=3 * std)
                nn.init.trunc_normal_(sh.down.weight, std=out_std, a=-3 * out_std, b=3 * out_std)
        for anchor in self.memory.values():
            anchor.reset_parameters(std)
        if self.mtp is not None:
            for name, mod in self.mtp.named_modules():
                if isinstance(mod, nn.Linear):
                    s = out_std if name.endswith(("o_proj", "down")) else std
                    nn.init.trunc_normal_(mod.weight, std=s, a=-3 * s, b=3 * s)
        if self.multimodal is not None:
            self.multimodal.reset_parameters(std)

    # ----------------------------------------------------------------- utils
    @property
    def anchor_indices(self) -> list[int]:
        return [int(k) for k in self.memory.keys()]

    def head_weight(self) -> torch.Tensor:
        return self.embed.weight if self.lm_head is None else self.lm_head.weight

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.head_weight().t()

    def init_memory(self, batch: int, dtype: torch.dtype = torch.float32) -> dict[int, MemoryState]:
        return {i: self.memory[str(i)].init_state(batch, self.embed.weight.device, dtype) for i in self.anchor_indices}

    def set_kernel_mode(self, *, moe_impl: Optional[str] = None, reference_attention: Optional[bool] = None,
                        attention_backend: Optional[str] = None, gdn_backend: Optional[str] = None) -> None:
        """Select kernels (never numerics): every backend must match the naive reference in tests."""
        from ..kernels.moe import IMPLEMENTATIONS
        from .attention import ATTENTION_BACKENDS
        from .gdn import GDN_BACKENDS

        if moe_impl is not None and moe_impl not in IMPLEMENTATIONS:
            raise ValueError(f"unknown MoE implementation {moe_impl!r}; available: {sorted(IMPLEMENTATIONS)}")
        if attention_backend is not None and attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"unknown attention backend {attention_backend!r}")
        if gdn_backend is not None and gdn_backend not in GDN_BACKENDS:
            raise ValueError(f"unknown GDN backend {gdn_backend!r}")
        attn = attention_backend or (None if reference_attention is None else ("reference" if reference_attention else "sdpa"))
        mtp_attn = [m.attn for m in self.mtp.depths] if self.mtp is not None else []
        for b in self.blocks:
            if moe_impl is not None:
                b.moe.impl = moe_impl
            if attn is not None and b.kind == GATED_GQA:
                b.mixer.backend = attn
            if gdn_backend is not None and b.kind == GDN:
                b.mixer.backend = gdn_backend
        for a in mtp_attn:
            if attn is not None:
                a.backend = attn

    def resolve_loop_passes(self, loop_passes: Optional[int], cache: Optional[HybridCache]) -> int:
        """Pass count for a forward call: fixed by an existing cache, else ``loop_passes`` or the default."""
        lp = self.cfg.loop
        if cache is not None:
            if loop_passes is not None and loop_passes != cache.loop_passes:
                raise ValueError(f"this cache was built with {cache.loop_passes} loop pass(es); re-read the context "
                                 f"(re-prefill) to run {loop_passes}")
            passes = cache.loop_passes
        else:
            passes = loop_passes if loop_passes is not None else (lp.default_passes if lp.enabled else 1)
        self.cfg.loop_schedule(passes)  # validates the count
        return passes

    @torch.no_grad()
    def update_router_biases(self, stats: list[RouterStats]) -> None:
        """Apply the auxiliary-loss-free bias update from (globally reduced) per-layer loads."""
        for block, s in zip(self.blocks, stats):
            block.moe.router.update_bias(s.counts)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        *,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
        cache: Optional[HybridCache] = None,
        use_cache: bool = False,
        memory: Optional[dict[int, MemoryState]] = None,
        memory_enabled: bool = True,
        gdn_mode: str = "auto",
        compute_logits: bool = True,
        compute_mtp: bool = False,
        loop_passes: Optional[int] = None,
    ) -> ModelOutput:
        """Run the backbone.

        ``labels[b, t]`` is the target for position ``t`` (already shifted, see
        ``data.packing``), ``-100`` where no loss applies. ``memory`` maps anchor
        layer index -> :class:`MemoryState`; when omitted, anchors read from the
        learned initial state ``M0``. ``memory_enabled=False`` is the memory-off
        fallback path (no memory tokens at all). ``loop_passes`` selects how often
        the looped block runs (fixed by ``cache`` once a session has started).
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids / inputs_embeds")
        x = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        B, T, _ = x.shape
        end = (cache.next_pos if cache is not None else 0) + T
        max_ctx = max(self.cfg.model.context_targets)
        if end > max_ctx:
            raise ContextLengthError(
                f"stream position {end} exceeds the supported context of {max_ctx}; use the extended-context "
                "configuration, a shorter working interval, or memory-backed chunked ingestion (§5)")
        idx = resolve_stream(B, T, x.device, cache, reset_mask, segment_ids, position_ids)
        valid = idx.seg >= 0
        use_memory = memory_enabled and self.cfg.memory.enabled and len(self.memory) > 0
        mem_states = None
        if use_memory:
            mem_states = memory if memory is not None else {
                i: self.memory[str(i)].initial_state_differentiable(B) for i in self.anchor_indices
            }
        passes = self.resolve_loop_passes(loop_passes, cache)
        schedule = self.cfg.loop_schedule(passes)
        if cache is not None and len(cache.layers) != len(schedule):
            raise ValueError(f"cache has {len(cache.layers)} state slots; {passes} loop pass(es) need {len(schedule)}")
        new_layers: list = []
        by_layer: list[list[RouterStats]] = [[] for _ in self.blocks]
        anchor_hidden: dict[int, torch.Tensor] = {}
        for slot, i in enumerate(schedule):
            block = self.blocks[i]
            extra = None
            if use_memory and i in mem_states:
                extra = self.memory[str(i)].read(mem_states[i]).to(x.dtype)
            state = cache.layers[slot] if cache is not None else None
            x, new_state, st = block(x, idx, state, extra, modality_ids, valid, gdn_mode)
            new_layers.append(new_state)
            by_layer[i].append(st)
            if use_memory and i in mem_states:
                anchor_hidden[i] = x
        stats = [s[0] if len(s) == 1 else _merge_stats(s) for s in by_layer]  # per layer, passes combined
        hidden_raw = x
        hidden = self.final_norm(x)
        out = ModelOutput(logits=None, hidden=hidden, hidden_raw=hidden_raw, router_stats=stats,
                          anchor_hidden=anchor_hidden, stream=idx, memory_states=mem_states, loop_passes=passes)
        if use_cache or cache is not None:
            last_rope = idx.rope_pos[:, -1] + 1
            out.cache = HybridCache(layers=new_layers, next_pos=(cache.next_pos if cache else 0) + T,
                                    last_seg=idx.seg[:, -1].clone(), next_rope=last_rope,
                                    mtp=cache.mtp if cache is not None else None,
                                    meta=dict(cache.meta) if cache is not None else {}, loop_passes=passes)
        if compute_logits or labels is not None:
            out.logits = self.logits(hidden)
        if labels is not None:
            mask = labels != IGNORE_INDEX
            out.lm_tokens = int(mask.sum())
            out.lm_loss = F.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]), labels.reshape(-1),
                                          ignore_index=IGNORE_INDEX, reduction="sum") / max(out.lm_tokens, 1)
        if compute_mtp and self.mtp is not None:
            if input_ids is None:
                raise ValueError("MTP training needs input_ids for next-token embeddings")
            # Depth j at position t: fuse depth j-1's output with Emb(x_{t+j}), predict x_{t+j+1} = labels[t+j].
            # A target counts only if every intermediate label is valid, i.e. inside the same document.
            h, losses = hidden_raw, []
            valid = None if labels is None else labels != IGNORE_INDEX
            for j in range(1, self.mtp.depth + 1):
                nxt = torch.cat([input_ids[:, j:], input_ids[:, -1:].expand(-1, min(j, T))], dim=1)[:, :T]
                h, _ = self.mtp.at(j)(h, self.embed(nxt), idx)
                logits_j = self.logits(h)
                if j == 1:
                    out.mtp_logits = logits_j
                if labels is None:
                    continue
                lab = torch.cat([labels[:, j:], torch.full_like(labels[:, : min(j, T)], IGNORE_INDEX)], dim=1)[:, :T]
                valid = valid & (lab != IGNORE_INDEX)
                n = int(valid.sum())
                if j == 1:
                    out.mtp_tokens = n
                losses.append(F.cross_entropy(logits_j.float().reshape(-1, logits_j.shape[-1]),
                                              lab.masked_fill(~valid, IGNORE_INDEX).reshape(-1),
                                              ignore_index=IGNORE_INDEX, reduction="sum") / max(n, 1))
            if losses:
                out.mtp_loss = torch.stack(losses).mean()  # L_MTP = (1/D) sum_j L_j
        return out

    # ------------------------------------------------------ segmented memory
    def forward_segmented(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        segment_len: Optional[int] = None,
        bptt_segments: int = 2,
        memory: Optional[dict[int, MemoryState]] = None,
        reset_prob: float = 0.0,
        rng: Optional[random.Random] = None,
        compute_mtp: bool = False,
        loop_passes: Optional[int] = None,
    ) -> ModelOutput:
        """Process an *unpacked* session sequence segment by segment (Stage D, §6, §8).

        Segment ``j`` reads memory written by segments ``< j`` only. After each
        complete segment, each anchor updates from its own detached hidden
        states. Gradients flow through at most ``bptt_segments`` memory updates
        (truncated BPTT); ``reset_prob`` randomly resets memory to ``M0`` to
        teach forgetting as well as retention.
        """
        if not (self.cfg.memory.enabled and len(self.memory)):
            return self.forward(input_ids, labels=labels, compute_mtp=compute_mtp, loop_passes=loop_passes)
        passes = self.resolve_loop_passes(loop_passes, None)
        B, T = input_ids.shape
        S = segment_len or self.cfg.memory.update_segment_tokens
        rng = rng or random.Random(0)
        mem = memory if memory is not None else {i: self.memory[str(i)].initial_state_differentiable(B) for i in self.anchor_indices}
        cache: Optional[HybridCache] = HybridCache.empty(len(self.cfg.loop_schedule(passes)), passes)
        logits, hidden, raw, stats_all, infos = [], [], [], [], []
        lm_sum = input_ids.new_zeros((), dtype=torch.float32)
        mtp_sum = input_ids.new_zeros((), dtype=torch.float32)
        lm_tokens = mtp_tokens = 0
        mem_losses = []
        for j, s in enumerate(range(0, T, S)):
            ids = input_ids[:, s : s + S]
            lab = None if labels is None else labels[:, s : s + S]
            if j and bptt_segments and j % bptt_segments == 0:
                mem = {i: st.detach() for i, st in mem.items()}
                cache = cache.detach_()
            if j and reset_prob and rng.random() < reset_prob:
                mem = {i: self.memory[str(i)].initial_state_differentiable(B) for i in self.anchor_indices}
            out = self.forward(ids, labels=lab, cache=cache, use_cache=True, memory=mem, compute_mtp=compute_mtp)
            cache = out.cache
            logits.append(out.logits)
            hidden.append(out.hidden)
            raw.append(out.hidden_raw)
            stats_all.append(out.router_stats)
            if out.lm_loss is not None:
                lm_sum = lm_sum + out.lm_loss * out.lm_tokens
                lm_tokens += out.lm_tokens
            if out.mtp_loss is not None:
                mtp_sum = mtp_sum + out.mtp_loss * out.mtp_tokens
                mtp_tokens += out.mtp_tokens
            if ids.shape[1] == S:  # update only at complete segment boundaries
                new_mem = {}
                for i, st in mem.items():
                    h = out.anchor_hidden[i].detach()
                    if j:  # predictive retrieval loss: memory from earlier segments vs this segment
                        mem_losses.append(self.memory[str(i)].retrieval_loss(st, h).mean())
                    new_mem[i], info = self.memory[str(i)].update(st, h)
                    infos.append({"segment": j, "anchor": i, **{k: v.detach() for k, v in info.items()}})
                mem = new_mem
        merged_stats = [_merge_stats([s[l] for s in stats_all]) for l in range(len(self.blocks))]
        out = ModelOutput(
            logits=torch.cat(logits, dim=1), hidden=torch.cat(hidden, 1), hidden_raw=torch.cat(raw, 1),
            router_stats=merged_stats, cache=cache, memory_states=mem, memory_info=infos, loop_passes=passes,
        )
        if labels is not None:
            out.lm_tokens = lm_tokens
            out.lm_loss = lm_sum / max(lm_tokens, 1)
        if compute_mtp and mtp_tokens:
            out.mtp_loss = mtp_sum / mtp_tokens
        if mem_losses:
            out.memory_loss = torch.stack(mem_losses).mean()
        return out

    # ---------------------------------------------------------------- decode
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, memory: Optional[dict[int, MemoryState]] = None, **kw) -> ModelOutput:
        """Prefill; pass ``loop_passes`` to choose the session's pass count (fixed for its cache)."""
        return self.forward(input_ids, use_cache=True, memory=memory, **kw)

    @torch.no_grad()
    def step(self, token_ids: torch.Tensor, cache: HybridCache, memory: Optional[dict[int, MemoryState]] = None,
             memory_enabled: bool = True) -> ModelOutput:
        """Decode ``token_ids [B, n]`` continuing ``cache`` (recurrent GDN path when n == 1)."""
        return self.forward(token_ids, cache=cache, use_cache=True, memory=memory, memory_enabled=memory_enabled)


def _merge_stats(stats: list[RouterStats]) -> RouterStats:
    counts = torch.stack([s.counts for s in stats]).sum(0)
    mc = torch.stack([s.modality_counts for s in stats]).sum(0)
    tokens = sum(s.tokens for s in stats)
    w = [s.tokens / max(tokens, 1) for s in stats]
    aux = sum(wi * s.aux_loss for wi, s in zip(w, stats))
    z = sum(wi * s.z_loss for wi, s in zip(w, stats))
    ent = sum(wi * s.entropy for wi, s in zip(w, stats))
    return RouterStats(counts, mc, tokens, aux, z, ent)
