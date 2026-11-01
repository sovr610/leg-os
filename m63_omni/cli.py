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
    total = build_manifest(cfg).total
    print(f"KV per token (BF16): {cap.kv_bytes_per_token(cfg):,.0f} bytes; full-attention control: "
          f"{cap.full_attention_kv_bytes_per_token(cfg):,.0f} bytes")
    for t in (32768, 131072, 262144):
        f = cap.session_footprint(cfg, t)
        print(f"  {t // 1024:>4}K session: KV {f.kv / cap.GiB:.2f} GiB + GDN {f.gdn / cap.MiB:.1f} MiB + memory {f.memory / cap.MiB:.1f} MiB")
    for d in ("bfloat16", "fp8", "int4"):
        b = cap.weight_bytes(total, d)
        print(f"weights {d:>8}: {b / 1e9:.2f} GB ({b / cap.GiB:.2f} GiB)")
    n = cap.sessions_per_deployment(cfg, total, a.gpus, a.gpu_gb * 1e9, a.context)
    print(f"upper bound on concurrent {a.context // 1024}K sessions on {a.gpus}x{a.gpu_gb:.0f}GB (BF16, 20% reserve): {n}")
    base, allow = cap.training_flops(5e9, 4.75e12)
    print(f"training FLOPs: {base:.3e} (6ND), {allow:.3e} with 30% allowance")
    for g in (256, 512):
        print(f"  {g} GPUs: {cap.training_days(allow, g, 250):.1f} days @250 TFLOP/s, {cap.training_days(allow, g, 350):.1f} days @350")
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
    s = sub.add_parser("validate-configs", help="strictly validate every YAML config")
    s.add_argument("--dir", default="configs")
    s.set_defaults(fn=cmd_validate)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
