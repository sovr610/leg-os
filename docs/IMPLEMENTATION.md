# Blueprint → code map

This repository implements the reference design in [`docs/blueprint.md`](blueprint.md)
(*M63-Omni: A Mid-Tier Hybrid Multimodal Orchestrator*, 2026-10-27). It covers the
blueprint's **step 0–4 software deliverables** (§12): the naive numerical reference for
every component, the parameter manifest, the data engine, the orchestrator contracts, the
runtime state model, and the verification inventory. It contains **no trained weights and
makes no capability claims**.

| Blueprint section | Code | Verified by |
|---|---|---|
| §3 Reference architecture & parameter budget | `m63_omni/model/m63.py`, `model/manifest.py`, `model/card.py`, `configs/full.yaml` | `tests/test_manifest.py` (62.486B resident / 4.965B active / 5.501B incl. head vs 62.481 / 4.96 / 5.501) |
| §4 Block math (GDN, MoE router) | `kernels/gdn.py`, `model/gdn.py`, `model/moe.py`, `kernels/moe.py` | `tests/kernels/*`, `tests/model/test_backbone.py`, `tests/model/test_moe_routing.py` |
| KDA tournament mixer (`hybrid_pattern` kind `kda`) | `kernels/kda.py`, `model/kda.py` | `tests/kernels/test_kda_kernels.py`, `tests/model/test_backbone.py`, `tests/test_manifest.py` |
| §4 KV / state arithmetic, §14 training compute | `runtime/capacity.py` (blueprint arithmetic + corrected `training_compute`, `serving_footprint`) | `tests/runtime/test_capacity.py` (48 KiB/token, 1.5/6/12 GiB, 18 MiB, 3 MiB; 2.14e23 FLOPs) |
| §5 Native multimodal front end & token economics | `model/encoders.py`, `model/interleave.py`, `data/media.py` | `tests/model/test_multimodal.py`, `tests/data/` (114,200-position budget) |
| §6 Titans-style memory & state governance | `model/memory.py`, `kernels/memory.py`, `runtime/session.py`, `runtime/event_log.py` | `tests/model/test_memory.py`, `tests/runtime/test_session.py`, `eval/security.py` |
| §7 MoE routing, stability, capacity controls | `model/moe.py`, `train/monitor.py` | `tests/model/test_moe_routing.py`, `tests/train/test_training.py` |
| §8 Data engine & curriculum | `m63_omni/data/*` (`data.mixture.STAGES` = text-heavy Stage A; `BLUEPRINT_STAGES` = the §8 table); dataset survey and sourcing plan in [`DATASETS.md`](DATASETS.md) | `tests/data/*` |
| §9 Objectives, hyperparameters | `train/objectives.py`, `train/schedule.py`, `train/trainer.py` | `tests/train/test_training.py` |
| §10 Orchestrator contracts, verification, escalation | `m63_omni/agent/*` | `tests/agent/*` |
| §11 Distributed layout, repository layout | `train/parallel.py`, `configs/*` | `tests/train/test_training.py::test_parallel_layouts`, `tests/test_config.py` |
| §12 Build program / proxy tournament | `configs/proxy_*.yaml`, `configs/pilot.yaml`, `configs/tournament/*.yaml`, `configs/recommended.yaml` | `tests/test_manifest.py::test_pilot_and_proxy_sizes_in_blueprint_ranges` |
| §13 Evaluation matrix & release gates | `eval/harness.py`, `eval/gates.py`, `eval/long_context.py`, `eval/efficiency.py`, `eval/security.py` | `tests/eval/test_eval.py` |
| §14 Serving & capacity | `runtime/paged_kv.py`, `runtime/prefix_cache.py`, `runtime/speculative.py`, `runtime/capacity.py` | `tests/runtime/*` |
| Appendix A pseudocode (ingest / generate / fork) | `runtime/session.py` (`ingest`, `ingest_media`, `generate`, `fork`, `attach_branch`, `replay`) | `tests/runtime/test_session.py`, `tests/runtime/test_media_session.py` |
| §10 model as planner (valid plans and tool calls) | `runtime/structured.py` (schema-guided decoding), `runtime/planner.py` (`ModelPlanner`, `open_model_session`) | `tests/runtime/test_planner.py` |
| §11 optimised kernels behind references | `kernels/attention.py` (`attention_flex`), `kernels/moe.py` (`moe_grouped_mm`), `kernels/gdn_fla.py` (fla varlen adapter), `model.set_kernel_mode` | `tests/kernels/test_backends.py` |
| §12 proxy tournament (step 6) | `train/tournament.py`, `m63 tournament`, `configs/tournament/*.yaml` | `tests/train/test_tournament.py` |
| §13/§16 release gates, end to end | `eval/release.py`, `eval/agent_security.py`, `m63 release-check` | `tests/eval/test_release.py` (incl. negative tests) |
| Appendix B verification inventory | `tests/` (organised by test family) | — |

## Appendix B coverage

| Test family | Where |
|---|---|
| Boundary lengths 1/63/64/65/2,047/2,048/2,049/32K/max context | `tests/model/test_boundaries.py` (32K at kernel level, marked `slow`) |
| Float64 finite-difference gradients for every kernel and both mixer layers | `tests/kernels/test_gradcheck.py` |
| Sampled speculative decoding preserves the target distribution (chi-square + negative control) | `tests/runtime/test_speculative_distribution.py` |
| Causality (future-suffix replacement, document resets, packed sequences, memory read timing) | `tests/model/test_backbone.py`, `tests/model/test_memory.py`, `tests/model/test_multimodal.py` |
| GDN/KDA equivalence (chunked vs recurrent, odd/even chunks, lengths 1/63/64/65/127/129, gradients) | `tests/kernels/test_gdn_kernels.py`, `tests/kernels/test_kda_kernels.py`, `tests/model/test_backbone.py` |
| GQA / memory masks (extra memory KV, RoPE positions, branch rollback) | `tests/kernels/test_moe_memory_attention_kernels.py`, `tests/runtime/test_caches_and_speculative.py` |
| MoE routing (top-k gradients, zero dropped tokens, node-limited routing) | `tests/kernels/test_moe_memory_attention_kernels.py`, `tests/model/test_moe_routing.py` |
| Multimodal alignment (missing channels, timestamps, identical placeholders / different hashes) | `tests/model/test_multimodal.py`, `tests/data/` |
| Checkpoint/resume (exact) | `tests/train/test_training.py` |
| Serving cache (prefix keys, tenant changes, eviction, CoW pages) | `tests/runtime/test_caches_and_speculative.py` |
| MTP (accept/reject, full state rollback, target-only fallback, one trained depth per draft) | `tests/runtime/test_caches_and_speculative.py` |
| Memory governance (trusted/untrusted writes, TTL, reset, deletion, model upgrade) | `tests/runtime/test_session.py`, `tests/eval/test_eval.py` |
| Agent security (direct/indirect injection, excessive agency) | `tests/agent/` |
| Evaluation harness (known-good/bad fixtures) | `tests/eval/test_eval.py` |

## Decisions where the blueprint leaves room (documented deviations)

1. **Parameter totals.** The manifest is generated from code: 62.486B resident (+0.008% vs
   62.481B). Backbone categories match the blueprint table exactly; the encoders (0.637B vs
   0.630B) and MTP (0.177B vs 0.180B) differ by ~1–2% because the blueprint does not fix
   their internals (CTC head, time projections, resampler width; MTP fusion is a
   parameter-free sum of two RMS-normalised streams).
2. **Timestamps.** Each AV window costs exactly two marker positions (`<|av_chunk|>`,
   `<|ts|>`), matching §5's 3,600-marker budget; `<|ts|>` carries a learned embedding of the
   window start time. `multimodal.timestamp_mode: text` instead writes the start in seconds after
   `<|av_chunk|>` (via the session's `text_encoder`, else raw byte IDs), costing one position per
   ≤3-digit group; `data.media.estimate_session_tokens(timestamp_mode="text")` counts it.
3. **Encoder causality.** The mel frontend and conv downsampling are causal, the audio
   encoder attends block-causally over the current and previous chunk (or token-causally with
   the same lookback under `audio_attention: causal`, which windows shorter than a chunk — such as
   2 s — require), the temporal aggregator is frame-causal within an AV window, audio emits only
   complete tokens, and each audio token joins the window containing its *end* time. Earlier
   windows are therefore bit-for-bit invariant to later media for any window length.
4. **Memory reads.** Read queries are learned slots plus a projection of the *previous*
   segment's summary, so a segment can never read information derived from itself. Value
   targets are L2-normalised to prevent the associative objective collapsing to zero.
5. **Recurrent state precision.** The reference keeps GDN state in FP32 (36 MiB/session at
   reference shape); the blueprint's 18 MiB figure assumes 2-byte storage. Both are exposed
   by `runtime.capacity.gdn_state_bytes(dtype_bytes=...)`.
6. **Schema-guided decoding.** The model chooses values (message, number of calls, tool,
   arguments, escalation); runtime code emits all JSON structure and fills what the model
   must not invent (plan ID, idempotency keys, timeout class, verifiers, budget). Generated
   strings exclude `"`, `\` and control characters, so no escaping is needed; optional
   object properties are emitted only when requested; only discriminated `oneOf` unions and
   nullable strings are supported. Token masks are computed per NFA state set and cached —
   production would precompile them (Outlines/XGrammar style).
7. **Token estimates are upper bounds.** `data.media` rounds a partial final AV window up
   when estimating cost, while the audio encoder emits only complete tokens, so a plan can
   exceed the encoded count by at most one audio token per asset — the safe direction for
   the §5 "estimate before encoding, never truncate silently" rule.
8. **Maximum context is enforced.** A forward pass that would exceed the largest
   `context_targets` entry raises `ContextLengthError` naming the §5 options, instead of
   running past the configured window.
9. **Precision policy.** Sensitive arithmetic runs in max(FP32, input dtype)
   (`kernels/precision.py`): BF16/FP16 accumulate in FP32 and FP64 passes through, so
   float64 gradient checks exercise the real code.
10. **Memory forgetting target.** The blueprint's rule decays memory toward zero, which
    erodes the learned `M0` over long sessions; `memory.decay_target: init` decays toward
    `M0` instead. The default stays `zero`; the tournament decides — but only in
    `--segmented-memory` mode, because packed training never updates memory (it reads `M0`),
    so memory ablations are identical there by construction.
11. **Speculative rollback.** Rejected MTP drafts restore the pre-verification snapshot and
    replay the accepted tokens — exact by construction; production kernels may checkpoint
    per token instead, but must match this reference.
12. **Tournament selection is noise-aware.** TPOT is the median of repeated measurements and
    Pareto dominance requires a margin beyond tolerance (eval loss: median bootstrap CI
    half-width; TPOT: 15% relative), so timing jitter cannot crown a winner among
    equivalent candidates.
13. **MTP depth.** The blueprint's §11 config drafts 2 tokens with a 1-layer MTP, which would reuse
    a depth-1 module recursively on inputs it was never trained on. MTP here is sequential
    (DeepSeek-V3): depth `j` fuses depth `j-1`'s output with token `t+j` and is trained to predict
    `t+j+1`; drafting uses one trained depth per token, and `draft_tokens ≤ mtp.layers` is
    validated. `full.yaml` therefore drafts 1 token; drafting 2 needs `mtp.layers: 2` (optionally
    `share_weights: true`).
14. **Node-limited routing** is skipped when `top_k ≤ max_nodes_per_token` (the reference top-2 /
    2 nodes), where it provably cannot change a selection.
15. **KDA.** The `kda` kind reads its shape from the `gdn` section (same state, conv, and chunk), with
    the channel-wise forget gate and sigmoid output gate as rank-`key_dim` projections (Kimi Linear);
    at reference shape the 36 mixers hold 1.268B parameters vs 1.516B for GDN. The chunked kernel
    forms pairwise per-channel decays explicitly per chunk, so it is exact for any decay strength.
16. **Stage A mix.** The default curriculum replaces the §8 Stage A starting point
    (60 / 10 / 10 / 10 / 10) with a text-heavy one: the first 0.35T positions are text-only,
    then 70 / 12 / 5 / 8 / 5 (73% text over the stage). Stages B–D are unchanged and the blueprint
    table remains available as `BLUEPRINT_STAGES`. The warm-up length is a starting point for the
    proxy ablation, not a measured value.

## Design review

[`IMPROVEMENTS.md`](IMPROVEMENTS.md) is a researched review of the architecture, data, training
cost, and serving cost, with a prioritised roadmap. Its §9 lists concrete corrections to this
repository and where each landed. Its low-risk defaults (KDA, `kv_heads: 4`, memory off, text
timestamps, 2-s AV windows) are collected in [`configs/recommended.yaml`](../configs/recommended.yaml)
and split into proxy-tournament variants; `configs/full.yaml` remains the blueprint baseline.

## Not implemented here (next steps on the build program)

* Megatron-Core distributed training (`train/parallel.py` only validates layouts) and a fused
  memory-update kernel. Optional GPU backends exist for attention (FlexAttention), GDN
  (flash-linear-attention adapter; its layout conversion is tested on CPU, the Triton kernel
  itself only runs when a GPU and `fla` are present) and MoE (`torch._grouped_mm`); none is
  benchmarked here.
* GDN-2 / Mamba-3 recurrent alternatives (KDA is implemented; add further kinds next to `gdn`
  and `kda` in `hybrid_pattern`). A media-data tournament runner: the text tournament cannot
  score timestamp or AV-window variants and reports them as skipped.
* FP8 training, quantised serving, vLLM integration, prefill/decode disaggregation.
* Process/container sandboxing for tools. The reference control plane enforces timeouts on
  a worker thread with a cooperative cancel flag; a tool that ignores the flag cannot be
  killed this way, so production needs process or container isolation.
* Real data acquisition, rights review, annotation, and every training run (steps 5–16).
