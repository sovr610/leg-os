"""Command-line entry point: ``m63 <command>`` or ``python -m m63_omni.cli <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cfg(path: str | None):
    from .config import load_config, reference

    return reference() if path is None else load_config(path)


def cmd_manifest(a: argparse.Namespace) -> int:
    from .model.manifest import build_manifest

    m = build_manifest(_cfg(a.config))
    print(json.dumps(m.to_dict(), indent=2) if a.json else m.to_markdown())
    return 0


def cmd_card(a: argparse.Namespace) -> int:
    from .model.card import model_card_markdown

    text = model_card_markdown(_cfg(a.config))
    if a.out:
        Path(a.out).write_text(text + "\n")
    else:
        print(text)
    return 0


def cmd_capacity(a: argparse.Namespace) -> int:
    from .model.manifest import build_manifest
    from .runtime import capacity as cap

    cfg = _cfg(a.config)
    manifest = build_manifest(cfg)
    total = manifest.total
    print(f"KV per token (BF16): {cap.kv_bytes_per_token(cfg):,.0f} bytes; full-attention control: "
          f"{cap.full_attention_kv_bytes_per_token(cfg):,.0f} bytes")
    print("per-session lower bound (blueprint §4, BF16 KV + state + memory):")
    for t in (32768, 131072, 262144):
        f = cap.session_footprint(cfg, t)
        print(f"  {t // 1024:>4}K: KV {f.kv / cap.GiB:.2f} GiB + recurrent {f.gdn / cap.MiB:.1f} MiB + memory {f.memory / cap.MiB:.1f} MiB")
    print("per-session serving view (FP8 KV, BF16 state, MTP KV, state checkpoint every 8K tokens, paged):")
    for t in (32768, 131072, 262144):
        f = cap.serving_footprint(cfg, t)
        print(f"  {t // 1024:>4}K: {f.total / 1e9:.2f} GB (KV {f.kv / 1e9:.2f}, MTP KV {f.mtp_kv / 1e9:.2f}, "
              f"checkpoints {f.checkpoints / 1e9:.2f}, state+memory {(f.gdn + f.memory) / 1e6:.0f} MB)")
    for d in ("bfloat16", "fp8", "int4"):
        b = cap.weight_bytes(total, d)
        print(f"weights {d:>8}: {b / 1e9:.2f} GB ({b / cap.GiB:.2f} GiB)")
    n = cap.sessions_per_deployment(cfg, total, a.gpus, a.gpu_gb * 1e9, a.context)
    print(f"upper bound on concurrent {a.context // 1024}K sessions on {a.gpus}x{a.gpu_gb:.0f}GB (BF16, 20% reserve): {n}")
    base, allow = cap.training_flops(5e9, 4.75e12)
    print(f"blueprint §14 training FLOPs: {base:.3e} (6ND), {allow:.3e} with 30% allowance")
    tc = cap.training_compute(cfg, manifest=manifest)
    b = tc.backbone
    print(f"corrected training FLOPs (§8 curriculum): {tc.total:.3e} = 6ND {b:.3e} + MTP {tc.mtp / b:+.1%} "
          f"+ attention {tc.attention / b:+.1%} + recurrent {tc.recurrent / b:+.1%} + encoders {tc.encoders / b:+.1%}")
    for hw, (lo, typ, hi) in cap.HARDWARE.items():
        days = ", ".join(f"{g} GPUs {tc.days(g, hw):.0f} d ({tc.days(g, hw, 2):.0f}-{tc.days(g, hw, 0):.0f})" for g in (256, 512))
        print(f"  {hw:>10} ~{typ:.0f} TFLOP/s sustained ({lo:.0f}-{hi:.0f}): {tc.gpu_hours(hw) / 1e3:,.0f}K GPU-h; {days}")
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    import torch

    from .model.m63 import M63Omni
    from .train.checkpoint import CheckpointManager
    from .train.trainer import Trainer, synthetic_batch

    cfg = _cfg(a.config or str(Path(__file__).resolve().parent.parent / "configs" / "tiny.yaml"))
    cfg = cfg.replace(multimodal={"enabled": False})
    torch.manual_seed(cfg.training.seed)
    model = M63Omni(cfg)
    seq = cfg.training.seq_len
    trainer = Trainer(model, lambda c: synthetic_batch(cfg, c, a.batch, seq, cfg.training.seed),
                      total_positions=a.steps * a.batch * seq,
                      checkpoints=CheckpointManager(Path(a.out)) if a.out else None,
                      checkpoint_every=a.checkpoint_every)
    if a.resume:
        trainer.resume(a.resume)
    for _ in range(a.steps):
        m = trainer.train_step()
        if m["step"] % a.log_every == 0 or m["step"] == 1:
            print(f"step {m['step']:5d} lm {m['lm']:.4f} total {m['total']:.4f} lr {m['lr']:.2e} "
                  f"gnorm {m['grad_norm']:.3f} load max/mean {m['max_over_mean']:.2f}")
    return 0


def cmd_security(a: argparse.Namespace) -> int:
    import torch

    from .model.m63 import M63Omni
    from .eval.security import run_security_suite

    cfg = _cfg(a.config or str(Path(__file__).resolve().parent.parent / "configs" / "tiny.yaml"))
    torch.manual_seed(0)
    rep = run_security_suite(M63Omni(cfg.replace(multimodal={"enabled": False})).eval())
    for r in rep.results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name} {r.detail}")
    return 0 if rep.passed else 1


def cmd_release(a: argparse.Namespace) -> int:
    from .eval.release import run_release_check, write_report

    rep = run_release_check(_cfg(a.config) if a.config else None)
    write_report(rep, a.json, a.md)
    print(rep.to_markdown())
    return 0 if rep.passed else 1


def cmd_tournament(a: argparse.Namespace) -> int:
    from .config import load_config
    from .train.tournament import build_candidates, run_tournament, write_report

    root = Path(__file__).resolve().parent.parent
    base = load_config(a.base or root / "configs" / "tiny.yaml")
    variants = a.variants or sorted(str(p) for p in (root / "configs" / "tournament").glob("*.yaml"))
    rep = run_tournament(build_candidates(base, variants), steps=a.steps, batch=a.batch, seed=a.seed, budget=a.budget,
                         reference=a.reference, max_kv_bytes_per_token=a.max_kv_bytes, max_tpot_ms=a.max_tpot_ms,
                         segmented_memory=a.segmented_memory, log=lambda s: print(s, flush=True))
    write_report(rep, a.json, a.md)
    print(rep.to_markdown())
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    from .config import load_config

    bad = 0
    for p in sorted(Path(a.dir).rglob("*.yaml")):
        try:
            load_config(p)
            print(f"ok    {p}")
        except Exception as e:  # report every bad file
            bad += 1
            print(f"FAIL  {p}: {e}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="m63", description="M63-Omni reference implementation tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("manifest", help="code-generated parameter manifest")
    s.add_argument("--config")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_manifest)
    s = sub.add_parser("card", help="generate the model-card config")
    s.add_argument("--config")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_card)
    s = sub.add_parser("capacity", help="KV/state/weight/training arithmetic")
    s.add_argument("--config")
    s.add_argument("--gpus", type=int, default=4)
    s.add_argument("--gpu-gb", type=float, default=80.0)
    s.add_argument("--context", type=int, default=131072)
    s.set_defaults(fn=cmd_capacity)
    s = sub.add_parser("train", help="reference training loop on synthetic data (CPU-sized configs)")
    s.add_argument("--config")
    s.add_argument("--steps", type=int, default=100)
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--log-every", type=int, default=10)
    s.add_argument("--out")
    s.add_argument("--checkpoint-every", type=int, default=0)
    s.add_argument("--resume")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("security", help="runtime security suite")
    s.add_argument("--config")
    s.set_defaults(fn=cmd_security)
    s = sub.add_parser("release-check", help="evidence for every non-negotiable release gate (exit 1 on failure)")
    s.add_argument("--config", help="CPU-sized config to check (default: tiny)")
    s.add_argument("--json", help="write the JSON report here")
    s.add_argument("--md", help="write the markdown report here")
    s.set_defaults(fn=cmd_release)
    s = sub.add_parser("tournament", help="matched proxy tournament over configs/tournament variants")
    s.add_argument("--base", help="base config the variants override (default: configs/tiny.yaml)")
    s.add_argument("--variants", nargs="*", help="variant YAML files (default: configs/tournament/*.yaml)")
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--budget", choices=["positions", "flops"], default="positions")
    s.add_argument("--reference", help="candidate whose active compute defines the FLOPs budget")
    s.add_argument("--max-kv-bytes", type=float, help="systems gate: max KV bytes per token")
    s.add_argument("--max-tpot-ms", type=float, help="systems gate: max time per output token")
    s.add_argument("--segmented-memory", action="store_true",
                   help="train/evaluate segment by segment so memory updates happen (needed for memory ablations)")
    s.add_argument("--json")
    s.add_argument("--md")
    s.set_defaults(fn=cmd_tournament)
    s = sub.add_parser("validate-configs", help="strictly validate every YAML config")
    s.add_argument("--dir", default="configs")
    s.set_defaults(fn=cmd_validate)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
