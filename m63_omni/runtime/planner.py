"""The M63 model as an orchestrator planner (blueprint §2, §10).

:class:`ModelPlanner` implements the ``ContextBundle -> envelope`` planner
contract of :class:`m63_omni.agent.orchestrator.Orchestrator`:

1. the bundle's request, artifact summaries, tool manifest, and *new*
   observations are ingested into the runtime :class:`Session` with their trust
   labels (tool output stays untrusted and is wrapped in ``<|trust_untrusted|>``);
2. the model decodes an ``m63.actions/v1`` envelope with
   :class:`~m63_omni.runtime.structured.SchemaDecoder`. The model chooses the
   message, the number of calls, each tool, its arguments, and the escalation
   flag/reason; runtime code fills what the model must not invent (plan ID,
   idempotency keys, timeout class, verifier list, budget). Every envelope parses
   and every argument object validates against its tool schema, by construction.

:func:`open_model_session` opens the control-plane session and the runtime
session on **one** :class:`EventLog`, so intake, plans, receipts, model input,
and model output are a single authoritative history. When the control plane
moves to another branch (undo, repair), the planner re-attaches the runtime
session to that branch before planning, restoring the matching model state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from ..agent.control_plane import ControlPlane, Principal
from ..agent.control_plane import Session as ControlSession
from ..agent.orchestrator import ContextBundle
from ..agent.schemas import SCHEMA_VERSION, parse_envelope
from ..common import Trust, canonical_json
from ..model.m63 import M63Omni
from .session import Session
from .structured import SchemaDecoder

MESSAGE_SCHEMA = {"type": ["string", "null"], "maxLength": 160}
REASON_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 160}


@dataclass
class PlannerConfig:
    temperature: float = 0.0
    max_calls: int = 4
    max_repair_attempts: int = 1
    max_string_tokens: int = 24
    max_array_items: int = 4
    max_context_tokens_per_item: int = 512


class ModelPlanner:
    def __init__(self, session: Session, tokenizer, config: PlannerConfig = PlannerConfig(),
                 generator: Optional[torch.Generator] = None) -> None:
        self.session, self.tok, self.cfg, self.generator = session, tokenizer, config, generator
        self._seen: set[str] = set()
        self._runs: set[str] = set()
        self.envelopes: list[dict] = []

    # ----------------------------------------------------------- context
    def _special(self, name: str) -> int:
        return self.tok.special_id(name)

    def _ingest_text(self, opener: str, text: str, trust: Trust, closer: Optional[str] = None) -> None:
        ids = [self._special(opener)]
        if trust is Trust.UNTRUSTED:
            ids.append(self._special("<|trust_untrusted|>"))
        ids += self.tok.encode_ordinary(text)[: self.cfg.max_context_tokens_per_item]
        if closer:
            ids.append(self._special(closer))
        self.session.ingest(ids, trust=trust)

    def _ingest_context(self, b: ContextBundle) -> None:
        if b.run_id not in self._runs:
            self._runs.add(b.run_id)
            tools = [{"tool": t["tool"], "arguments": t["arguments"], "description": t.get("description", "")} for t in b.tools]
            self._ingest_text("<|system|>", canonical_json({"tools": tools, "verifiers": list(b.verifiers)}), Trust.SYSTEM)
            self._ingest_text("<|user|>", b.request["text"], Trust.USER)
        arts = canonical_json({"artifacts": [dict(a) for a in b.artifacts], "budget": dict(b.remaining_budget), "mode": b.mode})
        key = "artifacts:" + arts
        if key not in self._seen:
            self._seen.add(key)
            self._ingest_text("<|tool_result|>", arts, Trust.TOOL_VERIFIED, "<|/tool_result|>")
        for o in b.observations:
            k = "obs:" + canonical_json(o)
            if k in self._seen:
                continue
            self._seen.add(k)
            trust = Trust(o.get("trust", Trust.UNTRUSTED.value))
            if o.get("quarantined"):
                trust = Trust.UNTRUSTED
            self._ingest_text("<|tool_result|>", canonical_json(o.get("content", {})), trust, "<|/tool_result|>")

    # ----------------------------------------------------------- decoding
    def __call__(self, bundle: ContextBundle) -> dict[str, Any]:
        if bundle.branch_id != self.session.branch_id:
            self.session.attach_branch(bundle.branch_id)
        self._ingest_context(bundle)
        env = self.decode(bundle)
        self.envelopes.append(env)
        return env

    def decode(self, bundle: ContextBundle) -> dict[str, Any]:
        s, cfg = self.session, self.cfg
        s.feed([self._special("<|assistant|>"), self._special("<|tool_call|>")])
        d = SchemaDecoder(s, self.tok, temperature=cfg.temperature, generator=self.generator,
                          max_string_tokens=cfg.max_string_tokens, max_array_items=cfg.max_array_items)
        plan_id = bundle.plan_id_hint
        d.force("{" + f'"schema_version": {json.dumps(SCHEMA_VERSION)}, "plan_id": {json.dumps(plan_id)}, "assistant_message": ')
        d.value(MESSAGE_SCHEMA, after=",")
        d.force(', "calls": [')
        tools = list(bundle.tools)
        cap = min(cfg.max_calls, int(bundle.remaining_budget.get("calls", cfg.max_calls))) if tools else 0
        n = 0
        while n < cap:
            stop = d.pick(["]", "{"] if n == 0 else [", ", "]"]) == (0 if n == 0 else 1)
            if stop:
                break
            if n:
                d.force(", ")
            d.force('{"tool": ')
            spec = tools[d.choose([json.dumps(t["tool"]) for t in tools])]
            d.force(', "arguments": ')
            d.value(spec["arguments"], after=",")
            n += 1
            d.force(f', "idempotency_key": {json.dumps(f"{plan_id}/call_{n}")}, '
                    f'"timeout_class": {json.dumps(spec["timeout_class"])}, '
                    f'"verify": {json.dumps(list(spec.get("default_verifiers", [])))}' + "}")
        repairs = min(cfg.max_repair_attempts, int(bundle.remaining_budget.get("repair_attempts", 0)))
        d.force(f'], "budget": {{"max_calls": {n}, "max_repair_attempts": {repairs}}}, "escalation": {{"required": ')
        if d.choose(["false", "true"]) == 1:
            d.force(', "reason": ')
            d.value(REASON_SCHEMA, after="}")
            d.force("}}")
        else:
            d.force(', "reason": null}}')
        s.feed([self._special("<|/tool_call|>")])
        s.commit_assistant({"structured": SCHEMA_VERSION, "plan_id": plan_id})
        env = json.loads(d.text)
        parse_envelope(env)  # by construction this never raises; keep it as an assertion
        return env


def open_model_session(
    control_plane: ControlPlane,
    principal: Principal,
    session_id: str,
    model: M63Omni,
    *,
    tokenizer_version: str = "bpe-v0",
    memory_enabled: bool = True,
) -> tuple[ControlSession, Session]:
    """Open a control-plane session and a runtime session that share one event log and artifact store."""
    cps = control_plane.open_session(principal, session_id)
    rt = Session(model, principal.tenant_id, session_id, event_log=cps.event_log, branch_id=cps.branch_id,
                 tokenizer_version=tokenizer_version, memory_enabled=memory_enabled, artifact_store=control_plane.store,
                 announce_attach=True)
    return cps, rt


def describe(bundle_or_env: Mapping[str, Any]) -> str:
    """Compact human-readable rendering of an envelope (for logs and demos)."""
    calls = ", ".join(c["tool"] for c in bundle_or_env.get("calls", []))
    return f"{bundle_or_env.get('plan_id')}: [{calls}] escalate={bundle_or_env.get('escalation', {}).get('required')}"
