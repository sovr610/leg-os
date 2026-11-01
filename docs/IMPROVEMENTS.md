# M63-Omni design review: improvements to architecture, data, training cost, and serving cost

*Research snapshot: 2026-11-01. Companion to [`blueprint.md`](blueprint.md) and [`DATASETS.md`](DATASETS.md).*

> **How this was produced.** Five parallel research surveys covered: backbone architecture, MoE and
> training efficiency, multimodal design, serving, and data / post-training / licensing. Most
> evidence was read directly from GitHub (tech-report PDFs, configs, READMEs, engine docs).
> huggingface.co and arxiv.org are blocked in this environment.
>
> Tags:
> - **[GH]** read on GitHub
> - **[S]** search extract of the cited primary page
> - **[✓]** independently re-checked for this document
> - **[D]** derived arithmetic from this repository's config, manifest, and `runtime/capacity.py`
> - **[M]** measured on this repository's code
> - **[U]** unverified
>
> Recommendations marked **ablate** go to the proxy tournament (`m63 tournament`, blueprint §12),
> not straight into the defaults.

## Contents

1. [Summary: the ten changes that matter most](#1-summary-the-ten-changes-that-matter-most)
2. [The cost picture, corrected](#2-the-cost-picture-corrected)
3. [Backbone architecture](#3-backbone-architecture)
4. [Multimodal design and curriculum](#4-multimodal-design-and-curriculum)
5. [Training efficiency](#5-training-efficiency)
6. [Data efficiency and media procurement](#6-data-efficiency-and-media-procurement)
7. [Post-training, RL, and escalation](#7-post-training-rl-and-escalation)
8. [Serving](#8-serving)
9. [Corrections to this repository and the blueprint](#9-corrections-to-this-repository-and-the-blueprint)
10. [Prioritised roadmap](#10-prioritised-roadmap)
11. [Evidence status](#11-evidence-status)

---

## 1. Summary: the ten changes that matter most

1. **Re-plan around data, not FLOPs.**
   - The main pretraining run costs roughly **$0.5–1.4M of GPU time** [D].
   - Licensing the media the current curriculum implies would cost **$72–288M** (§2).
   - Cap each media share by what can actually be licensed. At realistic procurement, text rises
     to ~79% of positions and video + AV falls to ~4%.
2. **Switch the default recurrent mixer from Gated DeltaNet to Kimi Delta Attention (KDA).**
   Same parameters, recipe, and 1.4T tokens. On MMLU-Pro the KDA hybrid scores 51.0 vs 47.9 for
   the GDN hybrid; on RULER@128K, 84.3 vs 80.5. Prefill speed is about the same [✓ from the Kimi
   Linear tech report]. Keep the 3:1 ratio.
3. **Cut attention KV.**
   - Set `kv_heads` 8 → 4 and store KV in FP8: 48 → 12 KiB/token, so a 128K session drops from
     6 to 1.5 GiB [D].
   - Ablate block-sparse attention (MSA/DSA) as a Stage C conversion. At 128K the 12 attention
     layers account for ~54% of prefill FLOPs; sparse attention could cut prefill by ~45% [D].
4. **Start the ViT from Meta's Perception Encoder PE-L/14.**
   - Apache-2.0, and an exact shape match: 24 layers, width 1024, MLP 4096, 16 heads, patch
     14 [✓].
   - It avoids ~0.4× the backbone's compute and ~5B uncleared image–text pairs.
   - This conflicts with the "from scratch" requirement, so it needs a decision (§4). Keep a
     from-scratch arm in the tournament.
5. **Train media at richer per-second rates, infer at lean ones.**
   - Video: ~32 positions/s in training vs 8/s at inference.
   - Audio: 12.5/s in training vs 6.25/s at inference.
   - Same position budget, ~4× fewer unique video hours and ~2× fewer audio hours needed (§4).
6. **Start text-heavy and gate on instruction-following parity.**
   - Every peer omni model started from a pretrained text model.
   - Instruction following regressed after omni training: IFEval −3.7 to −7.2 points, COLLIE
     −11.4 [GH].
   - Instruction following is the orchestrator's core skill. Add an IFEval / BFCL parity gate
     against a text-only control.
7. **Use interleaved text timestamps.** The 36 recurrent layers never see RoPE or time embeddings.
   Qwen3-VL moved from time-RoPE to text timestamps [GH].
8. **Make the neural memory a research track, off by default.**
   - No production model ships Titans-style session memory [U].
   - "Titans Revisited" could not isolate a gain [S].
   - Serving engines cannot run per-session gradient updates [GH].
   - The memory adds a poisoning surface.
   - Exact context plus the event log already covers 128–256K.
9. **Serve on one GPU per replica in FP8 instead of 4×80 GB BF16 TP4.**
   - About 13 full-128K sessions per GPU vs ~5 [D]: roughly 2.5–3× cheaper per session.
   - Then move to B200 with NVFP4 experts after quantization-aware training (~29 sessions/GPU [D]).
10. **Post-train with on-policy distillation and GSPO.**
    - On-policy distillation (OPD) from a permissively licensed open teacher comes before RL.
      Choose the tokenizer now so this is possible.
    - Use GSPO (+ DAPO / Dr.GRPO pieces) with routing replay for MoE RL stability.
    - Escalate through a verifier-gated cascade with a calibrated router.

---

## 2. The cost picture, corrected

**Training compute.** The blueprint's 1.85e23 FLOPs at 250 TFLOP/s (33.5 days on 256 GPUs) leaves
out two terms:
- the MTP module plus its share of the 131K-vocab head (**+13.6%**) [D];
- long-context attention across Stages A–D (**~+15%**) [D].

A re-derived total is **≈2.1e23 FLOPs** [D]. The 250 TFLOP/s figure also names no GPU. Comparable
5B-active MoEs measure **~150–210 TFLOP/s/GPU on H100 BF16**, even with ideal routing [GH,
Megatron-Bridge performance tables].

| Scenario (2.1e23 FLOPs) | Sustained TFLOP/s/GPU | GPU-hours | Days on 256 GPUs | Cost at listed $/GPU-h [S] |
|---|---:|---:|---:|---|
| H100, BF16, TP4 (realistic baseline) | ~180 | 324K | ~53 | $0.65–1.36M |
| H100, FP8 blockwise + parallel re-layout | ~250 | 233K | ~38 | $0.47–0.98M |
| B200, MXFP8 + re-layout | ~480 | 122K | ~20 | $0.49–1.00M |
| … + Muon (if the ~1.15× token efficiency holds) | — | — | ~17–18 | — |

The whole program costs more: encoder work, the proxy tournament, the pilot, post-training, and
failed runs plausibly add 2–4×. That is still **single-digit millions of dollars of compute**.

**Media licensing.** Reported market rates are about $1–4 per minute of video, or $60–240 per
hour, in 2024–25 [✓ Bloomberg]. Adobe paid about $3/min on average, up to $7.25 [✓].

| Target | Hours | Cost |
|---|---|---|
| `DATASETS.md` procurement target | 1.2M h | $72–288M |
| Resized target | 150–300K h | $9–72M |
| Pilot | 25K h | $1.5–6M |

**Conclusion: media rights dominate cost by one to two orders of magnitude.** Every design choice
that lowers the unique media hours needed is worth more than one that saves FLOPs.

**A supply-capped curriculum (illustrative).** Assumptions [D]:
- 250K licensed AV hours, trained at 46 positions/s for 4 epochs;
- 2.5M unique speech hours at 12.5/s, averaging ~3 epochs across quality tiers;
- images unchanged.

| Modality | Blueprint positions | Supply-capped positions | Share |
|---|---:|---:|---:|
| Text / code | 2.74T | **3.76T** | 79% |
| Image | 0.48T | 0.48T | 10% |
| Audio | 0.49T | 0.34T | 7% |
| Video + sync AV | 1.04T | **0.17T** | 4% |

Text is not data-limited: there are more than 10T filtered tokens, and top tiers can repeat ≤4
times. Edit-structure skill comes mostly from synthetic timeline edits on the licensed footage
(§6), not from more raw hours. A smaller total run is also a legitimate option. Let the
mixture-search pilot (§6) decide.

**Serving.** Two estimates, both bandwidth-floor arithmetic rather than benchmarks [D]:
- Full-128K traffic on one H200 with FP8: about **$0.7–1.6 per million output tokens**.
- Short-context traffic: about **$0.1–0.2/M**.

At long context, KV reads dominate decode, so KV size (§3) is the main serving cost lever.

---

## 3. Backbone architecture

| Change | Evidence | Effect on M63 | Rec. |
|---|---|---|---|
| **GDN → KDA** (keep GDN as the matched control) | Kimi Linear 48B-A3B, same recipe, 1.4T tokens. KDA vs GDN hybrid: MMLU-Pro 51.0 vs 47.9; RULER@128K 84.3 vs 80.5; prefill time "virtually indistinguishable" [✓]. Kimi K3 ships KDA + gated-MLA [GH]. FLA implements KDA under MIT [GH]. | Better recall and long context at the same 18 MiB state | **Adopt** |
| GDN-2 / Mamba-3 / DeltaProduct | At 1.3B / 100B tokens, GDN-2 leads (53.97 vs GDN 52.25) [GH]. The official GDN-2 code is NVIDIA non-commercial; FLA has MIT versions [GH]. | Unknown at 62B | **Ablate** GDN-2 only |
| Keep 3:1 KDA:attention | Kimi ratio sweep (validation PPL): 3:1 5.65; 1:1 5.66; 7:1 5.70; 15:1 5.82; pure attention 5.77 [GH]. Qwen3-Next and Qwen3.5 also use 3:1 [GH]. | — | **Keep** |
| Test what MiniMax M2 failed on | M2 went back to full attention: hybrids looked fine on standard benchmarks but differed on longer-context tasks, and infrastructure was immature [✓]. Linear-attention researchers dispute the conclusion [✓]. | Tournament scoring must include multi-hop and in-context-learning tasks, not just needle tests | **Adopt** (evaluation) |
| **`kv_heads` 8 → 4**, FP8 KV | Qwen3.5 global layers use 2 KV heads [GH]; MiniMax MSA model 64q / 4kv [GH] | 48 → 24 KiB/token BF16, 12 KiB FP8 [D] | **Adopt**; ablate kv=2 |
| Block-sparse attention in the 12 global layers (MSA/DSA) | MSA, 109B-A6B multimodal, 3T tokens: matches full GQA (MMLU 67.2 vs 67.0; VideoMME 45.5 vs 41.1). 14.2× prefill speedup at 1M on H800 [GH]. DSA matched V3.1 at k = 2048 [GH]. Kernels are hardware-specific (SM100 for MSA) [GH]. | Prefill −46–49% at 128K, −60–65% at 256K [D] | **Ablate** as a Stage C conversion |
| MLA | At least as expressive as GQA at equal KV [S]. 13.5 KiB/token; ~6.2–6.8 KiB in FP8 [D]. FlashMLA dropped Hopper support [GH]. QK-norm does not apply, so it needs QK-clip [GH]. | Smaller KV than kv=4 + FP8, at kernel risk | **Ablate** only if kv=4 + FP8 misses the concurrency target |
| NoPE in the global layers | Kimi Linear NoPE vs RoPE: RULER 84.3 vs 78.8 [GH]. Granite 4 uses NoPE [S]. | Long-context robustness, no YaRN tuning | **Ablate** `rotary_dims` {32, 0} |
| Keep QK-norm + sigmoid output gate | Gated attention removes sinks and massive activations [S]. Kimi gate ablation: sigmoid 5.65 vs none 5.67 vs swish 5.81 PPL [GH]. | — | **Keep** |
| GDN/KDA value heads 16 → 32 | Qwen3-Next / Qwen3.5 use 16 key / 32 value heads [GH] | State 18 → 36 MiB; ~+0.9B active [D] | **Ablate** |
| Block Attention Residuals (AttnRes) | Matches a baseline trained with 1.25× compute; on Kimi Linear 48B, GPQA-D 36.9 → 44.4 [GH] | Possible quality gain at about the same cost | **Ablate** |
| Untie embeddings | Qwen3-Next / Qwen3.5 untie [GH] | +0.54B resident [D] | **Ablate** |
| **Neural memory → research track** | No production precedent [U]. "Titans Revisited" found no consistent gain [S]. TTT-E2E loses badly to attention on needle-in-a-haystack [S]. Engines can't run per-session updates [GH]. Kimi K3 uses context compaction instead [GH]. | Removes training, serving, and security complexity; the event log stays the source of truth | **Adopt** `memory.enabled: false` by default |

**Fine-grained MoE** (§5) is an architecture change too. 128 experts × 768 width, top-4, with
matched shared capacity, keeps total and active parameters identical (62.50B / 4.98B [D]). Every
2025–26 peer routes at k ≥ 4 [GH]. **Ablate** with 128/4 as the prior.

---

## 4. Multimodal design and curriculum

| Change | Evidence | Effect | Rec. |
|---|---|---|---|
| **ViT init from PE-L/14** (PE-Core or PE-Lang) | Apache-2.0; 24 × 1024, MLP 4096, 16 heads, patch 14 [✓]. None of 7 open omni models trains its ViT from scratch [GH]. Matching PE-L from scratch ≈ 6.4e22 FLOPs plus ~5B image–text pairs [S/D]. | Large compute and data saving; better OCR and detail | **Adopt** as the default, with a from-scratch tournament arm. **Needs a governance decision:** it contradicts "from scratch", and PE's training data is a web crawl [U]. |
| Native-resolution packing + 2D-RoPE in the ViT | Qwen3-Omni ViT, SigLIP 2 NaFlex [GH] | Fewer wasted tokens on documents and UI | **Adopt** |
| DeepStack (ViT layers 8/16/24 → first LLM layers) | Qwen3-VL / Qwen3-Omni [GH] | More detail, no extra positions | **Ablate** |
| **Richer training rates**: video ~32 pos/s, inference 8–16 | Apollo: 8–32 tokens/frame and fps-consistent sampling [S]. Qwen3-VL uses 128–768 tokens per 2-frame group [GH]. | Unique video hours 18.75M → 4.5M; ~1.1M at ≤4 epochs [D] | **Adopt** (multi-rate resampler) |
| Shot-aware sampling (TransNetV2 keyframe per shot + fps floor) | 0.5 fps misses sub-2-second shots | Editing-relevant coverage | **Adopt** |
| **Interleaved text timestamps** as the AV chunk markers | Qwen3-VL replaced T-RoPE with text stamps [GH]. TimeLens: text timestamps beat positional methods [S]. | Works in the recurrent layers. A 2-hour session goes 114,200 → 121,400 positions, still under 128K [D]. | **Adopt** |
| 2-second AV chunks (from 4) | Qwen2.5-Omni uses 2 s [GH]; the blueprint table says "2–4 s" | Finer audio-video sync | **Adopt** |
| Audio trained at 12.5 Hz, inferred at 6.25 Hz via a learned 2× pool | Qwen moved 25 → 12.5 → 6.25 Hz, but its 6.25 Hz encoder saw ~40M hours [S/GH]. Cleared speech is ~2.5–3M h. | Audio hours 21.8M → 10.9M; ~2.7M at ≤4 epochs, about equal to supply [D] | **Ablate** |
| Audio encoder init from Whisper (MIT) | Ming, Ola, Baichuan, and Kimi use Whisper encoders [GH] | Avoids an encoder trained from scratch on limited hours | **Ablate** (shape change; training-data provenance [U]) |
| Timing contract: model gives ±160 ms, sidecars snap to frames or words | 6.25 Hz ≈ 4.8 frames at 30 fps [D] | Frame accuracy becomes a tool guarantee | **Adopt** |
| **Stage A text-heavy**: 70 / 12 / 5 / 8 / 5 (text / image / video / audio / AV), plus a text warm-up | Qwen3-Omni general stage: ~2.5% each for video and AV [S]. Omni models regressed on instruction following [GH]. | Protects tool and JSON adherence; shrinks the media gap | **Adopt**; ablate warm-up length. Supersede with the supply cap in §2. |
| Recaptions ~50/50 with originals; dense human narration on ~10⁵ videos | Recap-DataComp: +3.1% retrieval, best at p ≈ 0.5–0.6 original captions [S] | Skill coverage without 10⁷ hours | **Adopt** |

The unique hours needed, combined [D]:

| Setting | Video | Audio | Sync AV |
|---|---|---|---|
| Blueprint | 18.75M h | 21.8M h | 9.9M h |
| Richer training rates | 4.48M h (1.12M at ≤4 epochs) | 10.9M h (2.72M) | 3.08M h (0.77M) |
| + Stage A share cut | 3.03M h (0.76M) | 9.33M h (2.33M) | 2.02M h (0.51M) |
| Supply-capped plan (§2) | — | 2.5M h at ~3 epochs | 250K licensed AV h at 4 epochs |

---

## 5. Training efficiency

| Change | Evidence | Effect | Rec. |
|---|---|---|---|
| **Fine-grained experts**: 128×768 top-4 (+2 shared ×768) | Ling MoE scaling law: >7× efficiency leverage at 3.1% activation [S]. DeepSeek-V3 256/top-8, Kimi K2 384/top-8, Qwen3 128/top-8, Qwen3-Next 512/top-10, gpt-oss 128/top-4 [GH]. | Same parameters and FLOPs. Expert combinations 2,016 → 1.07e7. Expert-parallel fan-out per token 1.89 → 3.34 GPUs [D]. | **Ablate** (prior 128/4) |
| Balancing: keep the aux-free bias, add global-batch statistics, a tiny sequence aux (1e-4), freeze the bias for the final ~3% | DeepSeek-V3 aux-free ablations improve quality at 16B and 229B [GH]. Qwen global-batch balancing: better perplexity and specialisation [S]. | Small quality gain | **Adopt** |
| **FP8 parity gate moved into the pilot**; blockwise FP8 (H100) or MXFP8 (B200) | DeepSeek-V3: <0.25% loss gap vs BF16 at 16B and 230B; embedding, head, gating, norms, and attention kept higher precision [GH]. Megatron MXFP8 on GB200: +28% [GH]. | +13–28% throughput | **Adopt** (keep GDN/KDA gates in FP32) |
| NVFP4 pretraining | Matches FP8 on a 12B hybrid at 10T tokens [S]; only ~2% MoE speedup on GB200 [GH] | Little gain | **Defer** |
| **Parallel re-layout**: TP1–2, expert-TP 1, EP8 intra-node, PP4 + virtual pipeline, A2A overlap, DeepEP dispatcher | Megatron MoE guidance [GH]; Kimi K2 chose the smallest feasible EP [GH] | +10–25% throughput [U]. TP4 would cut 1536-wide experts into 384-wide shards. | **Adopt** |
| Muon for 2D matrices (AdamW for embedding, head, norms, router) | Moonlight ~2× [GH]; an independent fair comparison found 1.4× at 0.1B falling to 1.1× at 1.2B [S]; K2 MuonClip ran 15.5T tokens with no loss spikes [GH] | Realistically 1.1–1.25× [U] | **Ablate** at two scales against tuned AdamW |
| Hyperparameter transfer (CompleteP) + fitted LR/batch laws | CompleteP +12–34% at ≤1.5B dense [S]. The current LR of 2e-4 is borrowed from 32–37B-active runs [GH]. | Avoids a mis-set LR | **Adopt** proxy fits |
| Measure the critical batch size; ramp to 16M if allowed | Batch warm-up gave equal-or-better loss with 43% fewer steps [S] | Fewer steps, better MFU | **Measure** |
| WSD with a 20–30% decay aligned to Stages B–D | DeepSeek-V3 and K2 use long constant phases then decay [GH] | — | **Adopt** |
| Distillation during pretraining | Wins only with an existing teacher and below a compute threshold [S]. A large teacher costs ~2.5× text compute [D]. | — | **Defer** to post-training |
| Encoder freezing schedule | Not researched [U] | — | — |

---

## 6. Data efficiency and media procurement

| Change | Evidence | Effect | Rec. |
|---|---|---|---|
| **Proxy-swarm mixture search** (Olmix / RegMix; DoReMi baseline) | DoReMi: 2.6× faster to baseline at 8B [GH]. Olmix: 3.05× more data-efficient than the natural mix [S]. AutoScale: optima shift with scale [GH]. | A swarm of 128 × 300M proxies costs ~0.75% of the main run [D] and replaces the fixed shares | **Adopt** as the §8 pilot. Confirm the top 3 mixes at 1B-active, subject to the supply cap. |
| Ensemble quality classifiers, 20 tiers, ≤4 epochs on the top tier | Nemotron-CC recipe [GH]. DCLM: 63.7 MMLU at 2.6T vs Llama 3 66.2 at 15T [GH]. | Better quality per text token | **Adopt**, plus a classifier for orchestration and tool text |
| Semantic dedup on text and media embeddings | SemDeDup [GH]; GPU fuzzy dedup 16× faster [GH] | Honest "unique hours" counts | **Adopt**; report % removed per source |
| Rephrased web text, 0 / 15 / 30% | 1/3 rephrased reached the same loss 5–10× faster at ≤3B [S]. REWIRE +2.5 CORE at 7B [S]. | Quality, not quantity. Generation could cost 50–330K GPU-h, as much as pretraining [D/U]. | **Ablate**. Use an ≤8B permissive generator and cap generation at ~25% of pretraining GPU-h. |
| **Resized media procurement**: pilot 25K h, full 150–300K h | $1–4/min market [✓] | $1.5–6M pilot, $9–72M full [D] | **Adopt**. Prefer broker-supplied footage with original audio (TV, news, sports) and model releases. |
| Synthetic timeline edits on licensed footage | Engine-rendered edits scale with renders [S]. `DATASETS.md` §5 has the verifier plan. | Edit supervision is effectively unlimited | **Adopt** |
| YouTube third-party training opt-in | Off by default, no payment and no delivery mechanism [S] | Not a data source | — |
| Small IRT-based eval subsets for checkpoint tracking | tinyBenchmarks: ~100 items per benchmark, 1–3% absolute error [GH] | Cheap tracking; too noisy for ablation decisions | **Adopt** for tracking only |

---

## 7. Post-training, RL, and escalation

1. **Choose the tokenizer now** to match a permissively licensed open-weight teacher. On-policy
   distillation needs a shared tokenizer; cross-tokenizer distillation is off-policy only [GH].
2. **Add an on-policy distillation stage after SFT**, sized at ~10% of the RL GPU budget.
   - Reported ~10× cheaper than RL for generic reasoning [U, secondary].
   - Spend RL on timeline-state rewards that no teacher has.
3. **RL algorithm.**
   - **GSPO** as the default. Its sequence-level ratios stabilise MoE without routing replay [S].
   - DAPO dynamic sampling (drop zero-variance groups), clip-higher, overlong shaping.
   - Dr.GRPO length normalisation [GH].
   - **Routing replay (R3)** when the rollout/train KL exceeds a threshold. R3 cut the mismatch KL
     ~26× on Moonlight-16B-A3B [GH].
   - **FP32 LM head during RL.** MiniMax-M1 needed it to fix a train/inference probability
     mismatch [GH].
   - Log the recurrent-state mismatch from the first day. No framework supports this architecture
     yet [GH].
4. **Asynchronous RL** (verl / verl-omni, slime, AReaL) with bounded staleness: 2–3× wall-clock [GH].
   verl-omni already ships a Qwen3-Omni GSPO trainer [GH].
5. **Environment scaling.** Turn editing workflow families into NeMo Gym / OpenEnv environments,
   and split holdouts by family [GH].
6. **No hybrid thinking in v1.** Qwen split its Instruct and Thinking models for quality [GH][S].
   Use a bounded plan field instead, and ablate a budgeted "think" toggle on hard families.
7. **A verifier-gated cascade with a calibrated router** (RouteLLM-style matrix factorisation).
   - RouteLLM reports up to 85% cost reduction at 95% of GPT-4 quality on MT-Bench [GH].
   - Labels: paired verifier outcomes (M63 and flagship pass/fail) plus cost.
   - Calibrate to a target escalation rate and report the full cost–quality curve.
   - Illustration: a flagship 15× costlier and 80% M63 resolution gives ~73% savings [D].
8. **Flagship distillation only with an in-house flagship or explicit contractual permission.**
   Major providers' terms reportedly bar using outputs to train competing models [S]; confirm with
   counsel.

---

## 8. Serving

**Recommended configuration** [D, from GH evidence]:

| Area | Setting |
|---|---|
| Hardware | One H200 per replica (TP1), scaled as data-parallel replicas. Benchmark DEP2/DEP4 inside a node. **No cross-node EP, no EPLB** at this size. |
| Phase 2 | B200 with NVFP4 or MXFP4 experts after QAT. gpt-oss ships MXFP4 MoE weights and evaluates them that way [GH]. |
| Experts | FP8 block-scaled at launch |
| Mixers, router, MTP | BF16 at launch |
| Embedding / head | BF16 |
| Encoders | BF16, on separate encoder workers |
| KV cache | **FP8**, with a 128K long-context eval gate per engine version. Hopper FP8-KV once dropped 128K NIAH from 91% to 13% before a fix [GH]. |
| GDN/KDA state | BF16 or FP32, **never FP8** (engines don't support it) [GH] |
| Engine | vLLM or SGLang, reusing the Qwen3-Next / Qwen3.5 Gated-DeltaNet path (FLA kernels, hybrid KV manager). Keep 128-dim heads and chunk 64 for kernel compatibility [GH]. |
| Disaggregation | **Encoder–prefill–decode (EPD) first**, with encoder embeddings cached by asset hash (vLLM: 2× goodput, 30–50% lower P99 TTFT [GH]). Defer P/D disaggregation until ≥4 replicas; hybrid + MTP + P/D is not validated [GH]. |
| Prefix caching | "Align" mode: the memory-segment length (2,048) must be a multiple of the engine block size; recurrent-state checkpoints at block boundaries [GH] |
| KV tiers | HBM → DRAM → Mooncake / SSD, with session-affine, KV-aware routing (Mooncake: 3.8× throughput, hit rate 1.7% → 92.2% [GH]) |
| Speculation | **MTP k=1, load-gated.** DeepSeek-V3 MTP: 60 → 82 tok/s/rank at low batch, only +14% at 128 requests/rank [GH]. Apply the grammar to drafts too: acceptance length 2.33 → 2.55 [GH]. Speculation on JSON is fine (acceptance length 2.5–2.9 [GH]). |
| Structured output | **Engine grammar** (llguidance: <50 µs average mask [GH]; XGrammar). Keep `runtime/structured.py` as the correctness oracle. It measured **0.85–0.94 s per uncached mask** at a 131K vocabulary [M], roughly 10⁴× slower. |
| Neural memory | Off for v1 serving (no engine support) |

**Capacity, 128K sessions** (KV + recurrent + checkpoints ≈ 3.9 GB per session in FP8) [D]:

| Setup | Full-128K sessions per GPU |
|---|---|
| 4×H100, BF16, TP4 (blueprint) | ~5 |
| H200, FP8 weights + FP8 KV | ~13 |
| B200, NVFP4 experts | ~29 |

---

## 9. Corrections to this repository and the blueprint

These came out of the research. Each is small and concrete.

1. **The training FLOP model** in `runtime/capacity.py` and blueprint §14:
   - omits MTP (+13.6%) and long-context attention (+15%);
   - double-counts inefficiency, since "sustained TFLOP/s" already includes it;
   - names no hardware.

   → Add the terms, name H100 / B200, and use measured TFLOP/s ranges.
2. **Session footprint** in `capacity.py`:
   - omits the MTP layer's KV (+8.3%), recurrent-state checkpoints for prefix reuse and branches,
     and engine page padding;
   - `dtype_bytes=1` also quantizes GDN and memory state, which no engine supports.

   → Separate the KV, state, and checkpoint dtypes.
3. **MTP drafting vs training.** `mtp.layers: 1` with `draft_tokens: 2` reuses one depth-1 module
   recursively, which it was not trained for. DeepSeek-V3 trains D=1 [GH].
   → Draft 1 token by default, or train a 2-layer shared-weight MTP (Nemotron 3 [GH]).
4. **`max_nodes_per_token: 2` does nothing at top-2**: a top-2 token can never reach more than 2
   nodes. → Keep it only with top-k ≥ 4 and cross-node EP.
5. **The LR of 2e-4 is borrowed** from 32–37B-active runs [GH]. → Fit LR and batch on proxies.
6. **The FP8 "deferred" plan has no trigger.** → Make it a pilot gate.
7. **The Step 5 encoder gate is too weak** ("beat frozen/random controls"). → Compare against frozen
   PE-L / Whisper.
8. **The blueprint's chunk length is inconsistent**: 4 s in the budget, "2–4 s" in the table.
   → 2 s with text timestamps.
9. **Reference-only runtime pieces:**
   - `KVLayerState.append` copies the whole KV each token (O(L));
   - the structured decoder builds masks in Python.

   Both are fine as references; a serving port (vLLM/SGLang) is a prerequisite, not an option.
10. **The proxy tournament** must score multi-hop reasoning and in-context learning, not only RULER.
    That is the lesson from MiniMax M2.

---

## 10. Prioritised roadmap

| # | Change | Area | Impact | Effort | Risk | When | Default / ablate |
|---|---|---|---|---|---|---|---|
| 1 | Supply-capped curriculum + resized media procurement | data, cost | ★★★ | M | L | now (step 0) | default |
| 2 | Text-heavy Stage A + IFEval/BFCL parity gate | curriculum | ★★★ | L | L | now | default |
| 3 | ViT init from PE-L/14 (decision required) | multimodal | ★★★ | L | M (governance) | before encoder work | default + from-scratch arm |
| 4 | Richer training rates, lean inference (video/audio) | multimodal | ★★★ | M | M | proxy | default video, ablate audio |
| 5 | KDA mixer (GDN control) | arch | ★★ | L | L | proxy | default |
| 6 | `kv_heads` 4 + FP8 KV | arch, serving | ★★ | L | L | proxy | default |
| 7 | Neural memory off by default | arch, serving | ★★ | L | L | now | default |
| 8 | Text timestamps, 2-s chunks, shot-aware sampling | multimodal | ★★ | L | L | now | default |
| 9 | FP8 training + parallel re-layout, gated in the pilot | training | ★★ | M | M | pilot | default |
| 10 | Correct FLOP / capacity model; name the hardware | planning | ★★ | L | L | now | — |
| 11 | Single-GPU FP8 serving; EPD + embedding cache; engine grammar | serving | ★★★ | M | M | serving bring-up | default |
| 12 | Tokenizer matched to an open teacher; OPD stage | post-training | ★★ | L | L | now (tokenizer) | default |
| 13 | GSPO + R3 + FP32 RL head + async RL | post-training | ★★ | M | M | post-training | default |
| 14 | Verifier-gated cascade router | product | ★★★ | L | L | pilot onward | default |
| 15 | Proxy-swarm mixture search | data | ★★ | M | L | step 6 | — |
| 16 | Fine-grained MoE 128/4 | arch | ★★ | L | M | proxy | ablate |
| 17 | Block-sparse attention (MSA/DSA) at Stage C | arch, cost | ★★ | H | H | pilot+ | ablate |
| 18 | Muon, CompleteP, critical batch size | training | ★ | M | M | proxy | ablate |
| 19 | NoPE globals, GDN-2, AttnRes, value heads 32, untied embeddings, 2-layer MTP, DeepStack, Whisper init | arch, multimodal | ★ | L–M | M | proxy | ablate |
| 20 | Rephrased text 0/15/30% | data | ★ | M | M (cost) | proxy | ablate |
| 21 | NVFP4 experts after QAT, B200 | serving | ★★ | M | M | phase 2 | default later |

---

## 11. Evidence status

- **Re-checked [✓]:**
  - Kimi Linear KDA vs GDN numbers, from the tech report via GitHub;
  - PE-L/14 license and shape, from GitHub;
  - video licensing prices (Bloomberg 2024–25);
  - MiniMax M2's reasons for full attention (MiniMax blog), with critics noted.
- **[D] numbers** are this repository's arithmetic (manifest, `capacity.py`) on the stated
  assumptions, not measurements. **[M]** is one local measurement.
- **Notable unverified items:**
  - on-policy distillation's ~10× cost ratio and Qwen3's GPU-hour figures;
  - NVFP4 bit overhead;
  - the Ling granularity definition;
  - whether FLA's GDN-2 is free of NVIDIA-NC terms;
  - Muon's gain at 5B-active;
  - the providers' anti-competition terms (confirm with counsel);
  - GPU rental prices.
- **To verify further:** allow `huggingface.co` and `arxiv.org` in the environment's network
  settings and re-run the surveys (see `DATASETS.md` §11).
