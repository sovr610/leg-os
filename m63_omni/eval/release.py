"""One-command release check over every non-negotiable gate (blueprint §13, §16).

``m63 release-check`` produces evidence for each gate in
:data:`m63_omni.eval.gates.NON_NEGOTIABLE` and fails (non-zero exit) when any is
missing or failed:

* ``zero_token_drops`` — every layer dispatches exactly ``tokens x top_k`` slots
  under natural *and* adversarially skewed routing, and all MoE kernels agree;
* ``zero_cross_tenant_reads`` — runtime (event log, sessions, paged KV, prefix
  cache, memory snapshots) and control-plane isolation scenarios;
* ``zero_unauthorized_tool_executions`` — instrumented agent attack scenarios;
* ``reproducible_checkpoint_resume`` — an interrupted run resumes bit-exactly;
* ``verified_deletion`` — session deletion purges cache, memory, event payloads
  (including on disk) and media blobs;
* ``memory_off_path_functional`` — the memory-off fallback runs end to end.

Checks run on a CPU-sized config that preserves every structural feature; the
same code runs at full scale in the release pipeline.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from ..common import Trust
from ..config import M63Config, tiny
from ..model.interleave import AVPart, ImagePart
from ..model.m63 import M63Omni
from ..runtime.event_log import EventLog
from ..runtime.session import Session
from ..train.checkpoint import CheckpointManager
from ..train.trainer import Trainer, synthetic_batch
from .agent_security import run_agent_security_suite
from .gates import NON_NEGOTIABLE, GateReport, evaluate_release
from .security import run_security_suite


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ReleaseReport:
    config_fingerprint: str
    checks: dict[str, list[Check]] = field(default_factory=dict)
    gates: Optional[GateReport] = None

    def gate_results(self) -> dict[str, bool]:
        return {g: bool(self.checks.get(g)) and all(c.passed for c in self.checks[g]) for g in NON_NEGOTIABLE}

    @property
    def passed(self) -> bool:
        return self.gates is not None and self.gates.passed

    def to_dict(self) -> dict:
        return {
            "config_fingerprint": self.config_fingerprint,
            "passed": self.passed,
            "gates": self.gate_results(),
            "checks": {g: [c.__dict__ for c in cs] for g, cs in self.checks.items()},
            "failures": [] if self.gates is None else self.gates.failures,
            "instruction_parity": None if self.gates is None else self.gates.parity,
        }

    def to_markdown(self) -> str:
        lines = [f"# Release check — {'PASS' if self.passed else 'BLOCKED'}", "",
                 f"Config fingerprint: `{self.config_fingerprint}`", ""]
        for g in NON_NEGOTIABLE:
            cs = self.checks.get(g, [])
            ok = bool(cs) and all(c.passed for c in cs)
            lines.append(f"## {'✅' if ok else '❌'} {g}")
            lines += [f"- {'pass' if c.passed else 'FAIL'}: {c.name}" + (f" — {c.detail}" if c.detail else "") for c in cs] or ["- no evidence"]
            lines.append("")
        if self.gates is not None:
            lines += ["## Product gates", "", self.gates.to_markdown(), ""]
        return "\n".join(lines)


# --------------------------------------------------------------------- gates
def check_token_drops(cfg: M63Config) -> list[Check]:
    out = [Check("config forbids token dropping", not cfg.moe.token_dropping and not cfg.training.token_dropping)]
    torch.manual_seed(0)
    model = M63Omni(cfg.replace(multimodal={"enabled": False})).eval()
    batch = synthetic_batch(model.cfg, 0, 2, model.cfg.training.seq_len)
    k = model.cfg.moe.top_k

    def dispatched_ok(o) -> bool:
        return all(int(s.counts.sum()) == s.tokens * k for s in o.router_stats)

    with torch.no_grad():
        natural = model(batch.input_ids, segment_ids=batch.segment_ids, position_ids=batch.position_ids)
        out.append(Check("natural routing dispatches tokens x top_k slots per layer", dispatched_ok(natural)))
        for b in model.blocks:
            b.moe.router.bias.fill_(0.0)
            b.moe.router.bias[0] = 1e4  # every token wants expert 0
        skewed = {}
        for impl in ("reference", "grouped", "padded"):
            model.set_kernel_mode(moe_impl=impl)
            skewed[impl] = model(batch.input_ids, segment_ids=batch.segment_ids, position_ids=batch.position_ids)
        out.append(Check("skewed routing (all tokens to one expert) stays dropless",
                         all(dispatched_ok(o) for o in skewed.values()) and
                         all(int(s.counts[0]) == s.tokens for s in skewed["grouped"].router_stats)))
        agree = all(torch.allclose(skewed[i].logits, skewed["reference"].logits, atol=1e-5) for i in ("grouped", "padded"))
        out.append(Check("reference, grouped and padded MoE kernels agree under overload", agree))
    return out


def check_resume(cfg: M63Config, steps: int = 4) -> list[Check]:
    cfg = cfg.replace(multimodal={"enabled": False})

    def trainer(seed: int, **kw) -> Trainer:
        torch.manual_seed(seed)
        return Trainer(M63Omni(cfg), lambda c: synthetic_batch(cfg, c, 2, cfg.training.seq_len), total_positions=10**6, **kw)

    a = trainer(1)
    a.fit(steps)
    with tempfile.TemporaryDirectory() as d:
        b = trainer(1, checkpoints=CheckpointManager(Path(d)))
        b.fit(steps // 2)
        path = b.save()[0]
        c = trainer(999)
        c.resume(path)
        c.fit(steps - steps // 2)
    exact = all(torch.equal(x, y) for x, y in zip(a.model.state_dict().values(), c.model.state_dict().values()))
    return [Check(f"{steps}-step run == {steps // 2} steps + checkpoint + resume (bit-exact)", exact),
            Check("data cursor and position counters restored",
                  (c.state.cursor, c.state.positions) == (a.state.cursor, a.state.positions))]


def check_deletion(model: M63Omni) -> list[Check]:
    out = []
    with tempfile.TemporaryDirectory() as d:
        log = EventLog("tenant-a", "del", path=Path(d) / "log.jsonl")
        s = Session(model, "tenant-a", "del", event_log=log)
        s.ingest([5, 6, 7, 8], trust=Trust.USER)
        if model.multimodal is not None:
            g = torch.Generator().manual_seed(0)
            sr = model.cfg.multimodal.audio_sample_rate
            s.ingest_media([ImagePart(torch.randn(3, 8, 8, generator=g), 2, "img"),
                            AVPart("v", None, None, torch.randn(sr, generator=g))])
        s.generate(2)
        blobs = [h for e in s.log.replay() if e.kind == "ingest" for h in e.payload.get("asset_refs", [])]
        s.delete()
        disk = (Path(d) / "log.jsonl").read_text()
        out.append(Check("cache, memory, event payloads and checkpoints purged", s.verify_deleted()))
        out.append(Check("deleted payloads absent from the on-disk log", '"tokens"' not in disk))
        out.append(Check("media blobs released from the artifact store",
                         all(not s.store.contains("tenant-a", h) for h in blobs), f"{len(blobs)} blobs"))
    return out


def run_release_check(cfg: Optional[M63Config] = None, parity=None) -> ReleaseReport:
    """``parity``: an optional :class:`~m63_omni.eval.parity.ParityReport` from trained checkpoints."""
    cfg = cfg or tiny()
    rep = ReleaseReport(cfg.fingerprint())
    torch.manual_seed(0)
    model = M63Omni(cfg).eval()
    runtime = run_security_suite(model)
    by = {r.name: r for r in runtime.results}
    agent = run_agent_security_suite()

    rep.checks["zero_token_drops"] = check_token_drops(cfg)
    rep.checks["zero_cross_tenant_reads"] = (
        [Check(f"runtime: {r.name}", r.passed, r.detail) for r in runtime.results if r.name.startswith("tenant_")]
        + [Check(f"agent: {s.name}", s.passed and s.cross_tenant_violations == 0) for s in agent.scenarios
           if s.name == "cross_tenant_isolation"]
    )
    rep.checks["zero_unauthorized_tool_executions"] = [
        Check(f"agent: {s.name}", s.passed and s.unauthorized_executions == 0, s.detail) for s in agent.scenarios
    ] + [Check("total unauthorized executions == 0", agent.unauthorized_executions == 0, str(agent.unauthorized_executions))]
    rep.checks["reproducible_checkpoint_resume"] = check_resume(cfg)
    rep.checks["verified_deletion"] = [Check("runtime: deletion_verified", by["deletion_verified"].passed)] + check_deletion(model)
    rep.checks["memory_off_path_functional"] = [Check("runtime: memory_off_path", by["memory_off_path"].passed)] + [
        Check(f"runtime: {n}", by[n].passed) for n in ("memory_untrusted_write_blocked", "memory_frozen_during_generation",
                                                        "memory_snapshot_upgrade_invalidated") if n in by
    ]
    rep.gates = evaluate_release(rep.gate_results(), parity=parity)
    return rep


def write_report(rep: ReleaseReport, json_path: Optional[str] = None, md_path: Optional[str] = None) -> None:
    if json_path:
        Path(json_path).write_text(json.dumps(rep.to_dict(), indent=2) + "\n")
    if md_path:
        Path(md_path).write_text(rep.to_markdown() + "\n")
