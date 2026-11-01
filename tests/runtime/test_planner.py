"""Model ↔ control plane: schema-guided decoding, ModelPlanner, shared event log (§10)."""

import json
import random
import re

import pytest
import torch

from m63_omni.agent import (BranchRequest, Clip, ControlPlane, Orchestrator, Principal, Timeline, ToolRegistry, ToolSpec,
                            apply_edit_tool)
from m63_omni.agent.jsonschema_lite import validate
from m63_omni.agent.orchestrator import ContextBundle
from m63_omni.agent.schemas import HASH_PATTERN, ID_PATTERN, TOOL_REF_PATTERN, parse_envelope
from m63_omni.config import tiny
from m63_omni.data.tokenizer import train_bpe
from m63_omni.model.m63 import M63Omni
from m63_omni.runtime.planner import ModelPlanner, PlannerConfig, open_model_session
from m63_omni.runtime.session import Session
from m63_omni.runtime.structured import IntegerConstraint, Regex, RegexConstraint, SchemaDecoder, TextConstraint

CORPUS = ['{"tool": "timeline.apply_edit@2", "arguments": {"timeline_id": "asset_91"}}',
          "trim clip c_17 then add a dissolve transition", "render probe verified the cut"] * 6
SECRET = b"planner-test-secret-0123456789abcdef"


@pytest.fixture(scope="module")
def tok():
    cfg = tiny()
    return train_bpe(CORPUS, cfg.model.vocab_size, cfg.model.special_tokens)


@pytest.fixture
def lm():
    torch.manual_seed(0)
    return M63Omni(tiny(multimodal={"enabled": False})).eval()


def _registry():
    reg = ToolRegistry()
    reg.register(apply_edit_tool())
    reg.register(ToolSpec(
        "caption.style", 1,
        {"type": "object", "additionalProperties": False, "required": ["font", "size", "bold", "lang"],
         "properties": {"font": {"enum": ["sans", "serif", "mono"]}, "size": {"type": "integer", "minimum": 8, "maximum": 96},
                        "bold": {"type": "boolean"}, "lang": {"type": "string", "pattern": "^[a-z]{2}(?:-[A-Z]{2})?$"}}},
        lambda args, ctx: {"ok": True}, permissions=frozenset({"captions:write"}), timeout_class="interactive"))
    return reg


def _timeline():
    return Timeline("asset_91", (Clip("c_16", "a0", 0, 8000), Clip("c_17", "a1", 10000, 40000)), (), {"a0": 20000, "a1": 60000})


def _bundle(reg, plan="s/main/run_x/plan_1", calls=4, repairs=2):
    return ContextBundle(run_id="run_x", step=1, mode="initial", request={"text": "trim c_17", "trust": "user"},
                         tenant_id="t", session_id="s", branch_id="main", plan_id_hint=plan, artifacts=(),
                         tools=tuple(reg.manifest()), verifiers=("timeline_schema",), observations=(),
                         remaining_budget={"calls": calls, "repair_attempts": repairs, "cost": 10.0})


# ------------------------------------------------------------------ regex engine
@pytest.mark.parametrize("pattern", [ID_PATTERN, TOOL_REF_PATTERN, HASH_PATTERN, r"^[a-z]{2}(?:-[A-Z]{2})?$", r"(ab|c)+d?", r"x{2,3}"])
def test_regex_engine_agrees_with_re(pattern):
    rng = random.Random(0)
    alphabet = "abcdxyzAZ09_-./:@sha256" + "f" * 3
    ours, ref = Regex(pattern), re.compile(pattern)
    for _ in range(400):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 9)))
        assert ours.fullmatch(s) == bool(ref.fullmatch(s)), s
        if ours.fullmatch(s):  # every prefix of a match is viable
            assert all(ours.viable(s[:i]) for i in range(len(s) + 1))


def test_completions_are_valid():
    c = RegexConstraint(ID_PATTERN, 1, 32)
    for prefix in ("", "a/", "x.y/", "a-"):
        assert c.complete(prefix + c.completion(prefix))
    i = IntegerConstraint(8, 96)
    assert [t for t in ("", "0", "1", "9", "97", "100") if i.viable(t)] == ["", "1", "9"]
    assert i.complete("1" + i.completion("1")) and i.complete(i.completion(""))
    t = TextConstraint(3, 10)
    assert not t.viable('he"y') and t.complete("ab" + t.completion("ab"))


# --------------------------------------------------------------- decoding
@pytest.mark.parametrize("seed", range(6))
def test_model_planner_always_emits_schema_valid_envelopes(lm, tok, seed):
    reg = _registry()
    sess = Session(lm, "t", "s")
    planner = ModelPlanner(sess, tok, PlannerConfig(temperature=1.5, max_calls=3), generator=torch.Generator().manual_seed(seed))
    env = planner(_bundle(reg))
    parsed = parse_envelope(env)
    assert parsed.plan_id == "s/main/run_x/plan_1" and env["budget"]["max_calls"] == len(env["calls"])
    for i, call in enumerate(env["calls"], 1):
        spec = reg.resolve(call["tool"])
        assert validate(call["arguments"], spec.argument_schema) == []
        assert call["idempotency_key"] == f"s/main/run_x/plan_1/call_{i}"  # filled by runtime, not the model
    note = [e for e in sess.log.replay() if e.kind == "note" and e.payload.get("structured")]
    assert note and note[-1].trust == "untrusted"


def test_decoder_respects_zero_call_budget(lm, tok):
    sess = Session(lm, "t", "s")
    env = ModelPlanner(sess, tok)(_bundle(_registry(), calls=0, repairs=0))
    assert env["calls"] == [] and env["budget"] == {"max_calls": 0, "max_repair_attempts": 0}


def test_schema_decoder_handles_nested_union_arrays(lm, tok):
    sess = Session(lm, "t", "s")
    sess.ingest(tok.encode_ordinary("edit:"))
    schema = apply_edit_tool().argument_schema
    for seed in range(3):
        d = SchemaDecoder(sess, tok, temperature=2.0, generator=torch.Generator().manual_seed(seed), max_array_items=3)
        value = d.value(schema)
        assert validate(value, schema) == [] and json.loads(d.text) == value


# ------------------------------------------------------- shared event log
def _control_plane():
    reg = _registry()
    cp = ControlPlane(reg, secret=SECRET)
    who = Principal("editor", "tenant-a", frozenset({"timeline:write", "captions:write"}))
    return cp, who


def test_orchestrator_with_model_planner_shares_one_log(lm, tok):
    cp, who = _control_plane()
    cps, rt = open_model_session(cp, who, "edit-1", lm)
    assert cps.event_log is rt.log
    cp.ingest_artifact(cps, "asset_91", "timeline", _timeline().to_dict())
    planner = ModelPlanner(rt, tok, PlannerConfig(max_calls=2))
    result = Orchestrator(cp, planner).run(cps, "Trim c_17 and add a dissolve")
    kinds = [e.kind for e in rt.log.replay(cps.branch_id)]
    assert "plan" in kinds and "ingest" in kinds and "final" in kinds
    assert any(e.kind == "note" and e.payload.get("structured") for e in rt.log.replay(cps.branch_id))
    assert result.final.grounding_violations(cp.receipts(cps), verify_signature=cp.verify_receipt) == []
    live = rt.next_logits().clone()
    rebuilt = Session(lm, "tenant-a", "edit-1", event_log=rt.log, branch_id=rt.branch_id, artifact_store=cp.store)
    rebuilt.replay()
    torch.testing.assert_close(rebuilt.next_logits(), live, atol=1e-5, rtol=1e-5)
    assert cp.audit(cps).ok


def test_runtime_session_follows_control_plane_branches(lm, tok):
    cp, who = _control_plane()
    cps, rt = open_model_session(cp, who, "edit-2", lm)
    art = cp.ingest_artifact(cps, "asset_91", "timeline", _timeline().to_dict())
    planner = ModelPlanner(rt, tok, PlannerConfig(max_calls=1))
    Orchestrator(cp, planner).run(cps, "Trim c_17")
    tokens_after_run = rt.tokens
    branch = cp.create_branch(cps, BranchRequest(art.event_id, "try another cut", name="alt"))
    assert cps.branch_id == branch == "alt"
    Orchestrator(cp, planner).run(cps, "Use a fade instead")
    assert rt.branch_id == "alt"

    def context_tokens(branch):
        return sum(len(e.payload["tokens"]) for e in rt.log.replay(branch)
                   if (e.kind == "ingest" and "tokens" in e.payload) or (e.kind == "note" and e.payload.get("role") == "assistant"))

    # the model context on "alt" is exactly the alt lineage: the first run's tokens stay on "main"
    assert rt.tokens == context_tokens("alt")
    assert context_tokens("main") == tokens_after_run
    first_run = [e for e in rt.log.replay("main") if e.kind == "note" and e.payload.get("structured")]
    assert first_run and not any(e.event_id == first_run[0].event_id for e in rt.log.replay("alt"))
    fresh = Session(lm, "tenant-a", "edit-2", event_log=rt.log, branch_id="alt", artifact_store=cp.store)
    fresh.replay()
    torch.testing.assert_close(fresh.next_logits(), rt.next_logits(), atol=1e-5, rtol=1e-5)


class _SynthTok:
    """A stub tokenizer with a large synthetic vocabulary (for mask equivalence and timing)."""

    def __init__(self, n: int, seed: int = 0) -> None:
        rng = random.Random(seed)
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/@. :é\"\\{}"
        self.vocab_size = n
        self.texts = [chr(32 + i % 95) for i in range(95)]
        self.texts += ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 9))) for _ in range(n - 95 - 16)]
        self.texts += [None] * 16

    def is_special(self, i: int) -> bool:
        return self.texts[i] is None

    def token_bytes(self, i: int) -> bytes:
        return self.texts[i].encode()


PATTERNS = [ID_PATTERN, HASH_PATTERN, TOOL_REF_PATTERN, "^[a-z]{2}(?:-[A-Z]{2})?$", "^(?:trim|split)_[0-9]+$"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_trie_masks_equal_brute_force(pattern):
    dec = SchemaDecoder(None, _SynthTok(6000))
    c = RegexConstraint(pattern, min_len=1, max_len=40)
    prefixes = [""]
    rng = random.Random(1)
    for _ in range(6):  # grow viable prefixes by sampling allowed tokens
        cur = prefixes[-1]
        allowed = dec._allowed(c, cur).nonzero().flatten().tolist()
        if not allowed:
            break
        prefixes.append(cur + dec.token_text[rng.choice(allowed)])
    for cur in prefixes:
        got = dec._allowed(c, cur)
        want = torch.tensor([t is not None and dec._clean[i].item() and c.viable(cur + t)
                             for i, t in enumerate(dec.token_text)])
        assert torch.equal(got, want), (pattern, cur)


@pytest.mark.parametrize("lo,hi,cur", [(8, 96, ""), (8, 96, "1"), (0, 7, ""), (100, 250, "2"), (5, 10**6, "")])
def test_integer_masks_equal_brute_force(lo, hi, cur):
    dec = SchemaDecoder(None, _SynthTok(4000))
    c = IntegerConstraint(lo, hi)
    want = torch.tensor([t is not None and dec._clean[i].item() and c.viable(cur + t) for i, t in enumerate(dec.token_text)])
    assert torch.equal(dec._allowed(c, cur), want)


def test_regex_mask_is_fast_at_full_vocabulary_size():
    """131,072 tokens: an uncached mask was ~0.9 s with the per-token loop; the class trie is far faster."""
    import time

    dec = SchemaDecoder(None, _SynthTok(131072))
    c = RegexConstraint(TOOL_REF_PATTERN, min_len=1, max_len=64)
    dec._allowed(c, "")  # builds the pattern's trie once
    t0 = time.perf_counter()
    for cur in ("t", "timeline", "timeline.apply", "timeline.apply_edit@"):
        dec._allowed(c, cur)
    per_mask = (time.perf_counter() - t0) / 4
    assert per_mask < 0.25, per_mask


def test_planner_thinks_harder_on_repair(tok):
    import dataclasses

    torch.manual_seed(0)
    cfg = tiny(multimodal={"enabled": False}, model={"layers": 12}, memory={"anchors": [4, 12]},
               loop={"enabled": True, "first_layer": 5, "last_layer": 8})
    sess = Session(M63Omni(cfg).eval(), "t", "s")
    planner = ModelPlanner(sess, tok, PlannerConfig(think_harder_on_repair=True))
    reg = _registry()
    planner(_bundle(reg))
    assert sess.loop_passes == 1  # initial planning stays on the cheap path
    parse_envelope(planner(dataclasses.replace(_bundle(reg, plan="s/main/run_x/plan_2"), mode="repair")))
    assert sess.loop_passes == 2
    assert [e.payload["loop_passes"] for e in sess.log.replay() if e.kind == "note" and "loop_passes" in e.payload] == [2]
