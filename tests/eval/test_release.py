"""Release check: every gate has evidence, and each check fails when the property is broken."""

import json

import torch

from m63_omni.config import tiny
from m63_omni.eval import agent_security, release
from m63_omni.eval.gates import NON_NEGOTIABLE
from m63_omni.kernels import moe as moe_kernels


def test_release_check_passes_with_evidence_for_every_gate(tmp_path):
    rep = release.run_release_check()
    assert rep.passed, rep.to_markdown()
    assert set(rep.gate_results()) == set(NON_NEGOTIABLE) and all(rep.gate_results().values())
    assert all(len(rep.checks[g]) >= 2 for g in NON_NEGOTIABLE)
    release.write_report(rep, tmp_path / "r.json", tmp_path / "r.md")
    assert json.loads((tmp_path / "r.json").read_text())["passed"] is True


def test_agent_suite_detects_a_weakened_control_plane(monkeypatch):
    original = agent_security._Kit._add

    def weakened(self, spec):  # drop permissions and confirmation requirements
        fields = {f: getattr(spec, f) for f in spec.__dataclass_fields__}
        fields.update(permissions=frozenset(), requires_confirmation=False)
        original(self, type(spec)(**fields))

    monkeypatch.setattr(agent_security._Kit, "_add", weakened)
    rep = agent_security.run_agent_security_suite()
    assert not rep.passed and rep.unauthorized_executions > 0


def test_token_drop_gate_detects_a_dropping_kernel(monkeypatch):
    good = moe_kernels.IMPLEMENTATIONS["grouped"]

    def dropping(x, idx, w, *ws):  # silently drops every token past a capacity of 4 per expert
        keep = torch.zeros_like(idx, dtype=torch.bool)
        for e in range(ws[0].shape[0]):
            pos = (idx == e).nonzero()[:4]
            keep[pos[:, 0], pos[:, 1]] = True
        return good(x, idx, w * keep, *ws)

    monkeypatch.setitem(moe_kernels.IMPLEMENTATIONS, "grouped", dropping)
    checks = release.check_token_drops(tiny())
    assert not all(c.passed for c in checks)


def test_resume_gate_detects_lost_optimizer_state(monkeypatch):
    real = release.Trainer.resume

    def forgetful(self, path):
        opt_state = self.opt.state_dict()
        real(self, path)
        self.opt.load_state_dict(opt_state)  # pretend the optimizer state was not checkpointed

    monkeypatch.setattr(release.Trainer, "resume", forgetful)
    assert not all(c.passed for c in release.check_resume(tiny()))
