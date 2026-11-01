# Model card config — m63_omni

Config fingerprint: `sha256:76a9fd259ef827084d9c49cb491c5db97503ae36211681040fd8d883591e0fdc`  
Generated from code; do not edit by hand. Values are derived from configuration, not measured.

## Parameters (code-generated manifest)

| Component | Resident | Active per text token |
|---|---:|---:|
| routed_experts | 57.982B | 1.812B |
| shared_experts | 0.906B | 0.906B |
| gdn_mixers | 1.516B | 1.516B |
| gqa_mixers | 0.705B | 0.705B |
| norms | 0.000B | 0.000B |
| embedding | 0.537B | 0.000B |
| multimodal | 0.637B | 0.000B |
| mtp | 0.177B | 0.000B |
| routers_memory | 0.026B | 0.026B |
| tied LM head (shares embedding) | — | 0.537B when included |
| **total** | **62.486B** | **4.965B** (**5.501B** incl. head) |

Blueprint reference: 62.481B resident, 4.96B active excl. head, 5.501B incl. head.

## Backbone

- 48 blocks, pattern `['gdn', 'gdn', 'gdn', 'gated_gqa']` (36 GDN, 12 gated GQA)
- MoE: 64 routed + 1 shared, top-2, dropless
- Neural memory: anchors [16, 32, 48]

## State and serving arithmetic (BF16)

- GQA KV per token: 49,152 bytes (48 KiB)
- Per-session KV: 32K → 1.50 GiB, 128K → 6.00 GiB, 256K → 12.00 GiB
- Recurrent (GDN/KDA) state: 18.0 MiB per session
- Neural memory weights + momentum: 3.0 MiB per session
- Raw weights: bfloat16 124.97 GB (116.39 GiB), fp8 62.49 GB (58.19 GiB), int4 31.24 GB (29.10 GiB)
- MTP: depth 1, drafts up to 1 token(s) per step

## Training compute (derived, §8 curriculum)

- 6·N·D with the tied head: 1.568e+23 FLOPs over 4.75T positions
- Terms 6·N·D omits: MTP +13.0%, attention +17.9%, recurrent mixers +0.7%, vision encoder +4.7% (assumes 4 patches per visual position)
- Total 2.137e+23 FLOPs: h100_bf16 ~54 days on 256 GPUs, h100_fp8 ~39 days on 256 GPUs, b200_bf16 ~28 days on 256 GPUs, b200_mxfp8 ~20 days on 256 GPUs

## Limitations (always stated)

- Sparse activation reduces per-token expert compute, not resident checkpoint size.
- Adaptive video sampling is a budget policy; long-video detail can be missed.
- Neural memory is approximate and opaque; the event log is the source of truth.
- No benchmark claims: this card reports derived quantities only.
