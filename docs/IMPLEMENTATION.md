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
| §4 KV / state arithmetic | `runtime/capacity.py` | `tests/runtime/test_capacity.py` (48 KiB/token, 1.5/6/12 GiB, 18 MiB, 3 MiB) |
| §5 Native multimodal front end & token economics | `model/encoders.py`, `model/interleave.py`, `data/media.py` | `tests/model/test_multimodal.py`, `tests/data/` (114,200-position budget) |
| §6 Titans-style memory & state governance | `model/memory.py`, `kernels/memory.py`, `runtime/session.py`, `runtime/event_log.py` | `tests/model/test_memory.py`, `tests/runtime/test_session.py`, `eval/security.py` |
| §7 MoE routing, stability, capacity controls | `model/moe.py`, `train/monitor.py` | `tests/model/test_moe_routing.py`, `tests/train/test_training.py` |
| §8 Data engine & curriculum | `m63_omni/data/*` | `tests/data/*` |
| §9 Objectives, hyperparameters | `train/objectives.py`, `train/schedule.py`, `train/trainer.py` | `tests/train/test_training.py` |
| §10 Orchestrator contracts, verification, escalation | `m63_omni/agent/*` | `tests/agent/*` |
| §11 Distributed layout, repository layout | `train/parallel.py`, `configs/*` | `tests/train/test_training.py::test_parallel_layouts`, `tests/test_config.py` |
| §12 Build program / proxy tournament | `configs/proxy_*.yaml`, `configs/pilot.yaml`, `configs/tournament/*.yaml` | `tests/test_manifest.py::test_pilot_and_proxy_sizes_in_blueprint_ranges` |
| §13 Evaluation matrix & release gates | `eval/harness.py`, `eval/gates.py`, `eval/long_context.py`, `eval/efficiency.py`, `eval/security.py` | `tests/eval/test_eval.py` |
| §14 Serving & capacity | `runtime/paged_kv.py`, `runtime/prefix_cache.py`, `runtime/speculative.py`, `runtime/capacity.py` | `tests/runtime/*` |
| Appendix A pseudocode (ingest / generate / fork) | `runtime/session.py` (`ingest`, `ingest_media`, `generate`, `fork`, `attach_branch`, `replay`) | `tests/runtime/test_session.py`, `tests/runtime/test_media_session.py` |
| §10 model as planner (valid plans and tool calls) | `runtime/structured.py` (schema-guided decoding), `runtime/planner.py` (`ModelPlanner`, `open_model_session`) | `tests/runtime/test_planner.py` |
| §13/§16 release gates, end to end | `eval/release.py`, `eval/agent_security.py`, `m63 release-check` | `tests/eval/test_release.py` (incl. negative tests) |
| Appendix B verification inventory | `tests/` (organised by test family) | — |

## Appendix B coverage

| Test family | Where |
|---|---|
| Causality (future-suffix replacement, document resets, packed sequences, memory read timing) | `tests/model/test_backbone.py`, `tests/model/test_memory.py`, `tests/model/test_multimodal.py` |
| GDN equivalence (chunked vs recurrent, odd/even chunks, lengths 1/63/64/65/127/129, gradients) | `tests/kernels/test_gdn_kernels.py`, `tests/model/test_backbone.py` |
| GQA / memory masks (extra memory KV, RoPE positions, branch rollback) | `tests/kernels/test_moe_memory_attention_kernels.py`, `tests/runtime/test_caches_and_speculative.py` |
| MoE routing (top-k gradients, zero dropped tokens, node-limited routing) | `tests/kernels/test_moe_memory_attention_kernels.py`, `tests/model/test_moe_routing.py` |
| Multimodal alignment (missing channels, timestamps, identical placeholders / different hashes) | `tests/model/test_multimodal.py`, `tests/data/` |
| Checkpoint/resume (exact) | `tests/train/test_training.py` |
| Serving cache (prefix keys, tenant changes, eviction, CoW pages) | `tests/runtime/test_caches_and_speculative.py` |
| MTP (accept/reject, full state rollback, target-only fallback) | `tests/runtime/test_caches_and_speculative.py` |
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
   window start time instead of a multi-token text timestamp.
3. **Encoder causality.** The mel frontend and conv downsampling are causal, the audio
   encoder attends block-causally over the current and previous chunk, the temporal
   aggregator is frame-causal within an AV window, and audio emits only complete tokens.
   Earlier windows are therefore bit-for-bit invariant to later media.
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
8. **Speculative rollback.** Rejected MTP drafts restore the pre-verification snapshot and
   replay the accepted tokens — exact by construction; production kernels may checkpoint
   per token instead, but must match this reference.

## Not implemented here (next steps on the build program)

* Megatron-Core distributed training and optimised Triton/CUDA GDN, grouped-GEMM, and
  memory-update kernels (`train/parallel.py` only validates layouts).
* KDA / GDN-2 / Mamba-3 recurrent alternatives for the proxy tournament (the mixer is
  selected by `hybrid_pattern`; add a new kind next to `gdn`).
* FP8 training, quantised serving, vLLM integration, prefill/decode disaggregation.
* Process/container sandboxing for tools. The reference control plane enforces timeouts on
  a worker thread with a cooperative cancel flag; a tool that ignores the flag cannot be
  killed this way, so production needs process or container isolation.
* Real data acquisition, rights review, annotation, and every training run (steps 5–16).
