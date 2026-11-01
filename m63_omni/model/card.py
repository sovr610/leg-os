"""Model-card config generated from the same source of truth as the checkpoint (§11).

Documentation drawn from here cannot disagree with the implemented model: every
number is computed from the config and the code-generated parameter manifest.
"""

from __future__ import annotations

from ..config import GATED_GQA, GDN, KDA, M63Config
from ..runtime import capacity as cap
from .manifest import BLUEPRINT_REFERENCE, ParameterManifest, build_manifest


def _gib(x: float) -> str:
    return f"{x / cap.GiB:.2f} GiB"


def model_card(cfg: M63Config, manifest: ParameterManifest | None = None) -> dict:
    m = manifest or build_manifest(cfg)
    kinds = cfg.layer_kinds()
    total = m.total
    flops, flops_allow = cap.training_flops(5e9, 4.75e12)
    return {
        "name": cfg.model.name,
        "config_fingerprint": cfg.fingerprint(),
        "backbone": {
            "layers": cfg.model.layers,
            "pattern": list(cfg.model.hybrid_pattern),
            "gdn_layers": kinds.count(GDN),
            "kda_layers": kinds.count(KDA),
            "attention_layers": kinds.count(GATED_GQA),
            "hidden_size": cfg.model.hidden_size,
            "vocab_size": cfg.model.vocab_size,
            "context_targets": list(cfg.model.context_targets),
        },
        "moe": {"routed": cfg.moe.routed_experts, "shared": cfg.moe.shared_experts, "top_k": cfg.moe.top_k,
                "expert_hidden": cfg.moe.expert_hidden, "token_dropping": cfg.moe.token_dropping},
        "memory": {"enabled": cfg.memory.enabled, "anchors": list(cfg.memory.anchors),
                   "read_tokens": cfg.memory.read_tokens_per_anchor, "segment_tokens": cfg.memory.update_segment_tokens},
        "parameters": m.to_dict(),
        "kv_bytes_per_token_bf16": cap.kv_bytes_per_token(cfg),
        "session_bytes_bf16": {str(t): cap.session_footprint(cfg, t).total for t in (32768, 131072, 262144)},
        "gdn_state_bytes_bf16": cap.gdn_state_bytes(cfg),
        "memory_state_bytes_bf16": cap.memory_state_bytes(cfg),
        "weights_bytes": {d: cap.weight_bytes(total, d) for d in ("bfloat16", "fp8", "int4")},
        "training_flops_reference": {"dense_6ND": flops, "with_30pct_allowance": flops_allow},
        "training_compute": cap.training_compute(cfg, manifest=m).to_dict(),
        "mtp": {"depth": cfg.mtp.layers if cfg.mtp.enabled else 0, "draft_tokens": cfg.mtp.draft_tokens,
                "share_weights": cfg.mtp.share_weights},
        "loop": None if not cfg.loop.enabled else {
            "layers": [cfg.loop.first_layer, cfg.loop.last_layer], "max_passes": cfg.loop.max_passes,
            "default_passes": cfg.loop.default_passes,
            "active_per_pass_count": {p: m.active_with_passes(cfg, p, include_head=False)
                                      for p in range(1, cfg.loop.max_passes + 1)},
            "kv_bytes_per_token_bf16": {p: cap.kv_bytes_per_token(cfg, loop_passes=p)
                                        for p in range(1, cfg.loop.max_passes + 1)},
        },
    }


def _loop_lines(loop: dict | None) -> list[str]:
    if loop is None:
        return []
    a, kv = loop["active_per_pass_count"], loop["kv_bytes_per_token_bf16"]
    return [f"- Second-pass loop: layers {loop['layers'][0]}-{loop['layers'][1]} run up to {loop['max_passes']}x "
            f"(sessions start at {loop['default_passes']}); "
            + "; ".join(f"{p} pass(es): {a[p] / 1e9:.2f}B active, {kv[p] / 1024:.0f} KiB KV/token" for p in sorted(a))]


def _compute_lines(tc: dict) -> str:
    b = tc["backbone_6ND"]
    lines = [f"- 6·N·D with the tied head: {b:.3e} FLOPs over {tc['positions'] / 1e12:.2f}T positions",
             f"- Terms 6·N·D omits: MTP +{tc['mtp'] / b:.1%}, attention +{tc['attention'] / b:.1%}, "
             f"recurrent mixers +{tc['recurrent'] / b:.1%}, vision encoder +{tc['encoders'] / b:.1%} "
             "(assumes 4 patches per visual position)",
             f"- Total {tc['total']:.3e} FLOPs: "
             + ", ".join(f"{hw} ~{cap.training_days(tc['total'], 256, typ):.0f} days on 256 GPUs"
                         for hw, (_, typ, _) in cap.HARDWARE.items())]
    return "\n".join(lines)


def model_card_markdown(cfg: M63Config) -> str:
    m = build_manifest(cfg)
    c = model_card(cfg, m)
    kv = c["kv_bytes_per_token_bf16"]
    lines = [
        f"# Model card config — {cfg.model.name}",
        "",
        f"Config fingerprint: `{c['config_fingerprint']}`  ",
        "Generated from code; do not edit by hand. Values are derived from configuration, not measured.",
        "",
        "## Parameters (code-generated manifest)",
        "",
        m.to_markdown(),
        "",
        f"Blueprint reference: {BLUEPRINT_REFERENCE['total'] / 1e9:.3f}B resident, "
        f"{BLUEPRINT_REFERENCE['active_excl_head'] / 1e9:.2f}B active excl. head, "
        f"{BLUEPRINT_REFERENCE['active_incl_head'] / 1e9:.3f}B incl. head.",
        "",
        "## Backbone",
        "",
        f"- {c['backbone']['layers']} blocks, pattern `{c['backbone']['pattern']}` "
        f"({c['backbone']['gdn_layers']} GDN, "
        + (f"{c['backbone']['kda_layers']} KDA, " if c["backbone"]["kda_layers"] else "")
        + f"{c['backbone']['attention_layers']} gated GQA)",
        f"- MoE: {c['moe']['routed']} routed + {c['moe']['shared']} shared, top-{c['moe']['top_k']}, dropless",
        f"- Neural memory: {'anchors ' + str(c['memory']['anchors']) if c['memory']['enabled'] else 'disabled'}",
        "",
        "## State and serving arithmetic (BF16)",
        "",
        f"- GQA KV per token: {kv:,.0f} bytes ({kv / 1024:.0f} KiB)",
        "- Per-session KV: " + ", ".join(f"{int(k) // 1024}K → {_gib(cap.kv_bytes(cfg, int(k)))}" for k in c["session_bytes_bf16"]),
        f"- Recurrent (GDN/KDA) state: {c['gdn_state_bytes_bf16'] / cap.MiB:.1f} MiB per session",
        f"- Neural memory weights + momentum: {c['memory_state_bytes_bf16'] / cap.MiB:.1f} MiB per session",
        "- Raw weights: " + ", ".join(f"{k} {v / 1e9:.2f} GB ({_gib(v)})" for k, v in c["weights_bytes"].items()),
        f"- MTP: depth {c['mtp']['depth']}, drafts up to {c['mtp']['draft_tokens']} token(s) per step",
        *_loop_lines(c["loop"]),
        "",
        "## Training compute (derived, §8 curriculum)",
        "",
        _compute_lines(c["training_compute"]),
        "",
        "## Limitations (always stated)",
        "",
        "- Sparse activation reduces per-token expert compute, not resident checkpoint size.",
        "- Adaptive video sampling is a budget policy; long-video detail can be missed.",
        "- Neural memory is approximate and opaque; the event log is the source of truth.",
        "- No benchmark claims: this card reports derived quantities only.",
    ]
    return "\n".join(lines)
