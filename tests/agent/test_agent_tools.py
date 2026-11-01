"""Versioned tool specs and registry resolution (§10)."""

from __future__ import annotations

import pytest

from m63_omni.agent import (
    ToolRefError,
    ToolRegistry,
    ToolSpec,
    UnknownToolError,
    UnknownToolVersionError,
    apply_edit_tool,
)
from m63_omni.agent.jsonschema_lite import SchemaError
from m63_omni.common import Trust

_ARGS = {"type": "object", "additionalProperties": False, "properties": {"q": {"type": "string"}}}


def _noop(args: dict, ctx: object) -> dict:
    return {}


def test_register_and_resolve_exact_versions() -> None:
    reg = ToolRegistry()
    v1 = reg.register(ToolSpec("search.web", 1, _ARGS, _noop))
    v3 = reg.register(ToolSpec("search.web", 3, _ARGS, _noop))
    assert reg.resolve("search.web@1") is v1 and reg.resolve("search.web@3") is v3
    assert reg.versions("search.web") == (1, 3)
    assert "search.web@3" in reg and "search.web@2" not in reg and "garbage" not in reg
    with pytest.raises(UnknownToolVersionError, match=r"available: \[1, 3\]"):
        reg.resolve("search.web@2")
    with pytest.raises(UnknownToolError):
        reg.resolve("search.images@1")
    with pytest.raises(ToolRefError):
        reg.resolve("search.web")
    with pytest.raises(ValueError, match="already registered"):
        reg.register(ToolSpec("search.web", 1, _ARGS, _noop))
    reg.unregister("search.web@1")
    assert reg.refs() == ("search.web@3",)


def test_spec_defaults_and_validation() -> None:
    spec = ToolSpec("search.web", 1, _ARGS, _noop)
    assert spec.ref == "search.web@1"
    assert spec.output_trust is Trust.UNTRUSTED  # untrusted unless the spec says otherwise
    assert spec.side_effect is False
    with pytest.raises(ValueError, match="additionalProperties"):
        ToolSpec("search.web", 1, {"type": "object", "properties": {}}, _noop)
    with pytest.raises(SchemaError):
        ToolSpec("search.web", 1, {"type": "object", "additionalProperties": False, "propertys": {}}, _noop)
    with pytest.raises(ValueError, match="side_effect"):
        ToolSpec("x.y", 1, {**_ARGS, "properties": {"t": {"type": "string"}}}, _noop, writes=("t",))
    with pytest.raises(ValueError, match="not an argument"):
        ToolSpec("x.y", 1, _ARGS, _noop, reads=("missing",))
    with pytest.raises(ValueError):
        ToolSpec("Bad Name", 1, _ARGS, _noop)


def test_manifest_exposes_contract_not_implementation() -> None:
    spec = apply_edit_tool()
    m = spec.manifest()
    assert m["tool"] == "timeline.apply_edit@2"
    assert m["permissions"] == ["timeline:write"]
    assert m["output_trust"] == "tool_verified"
    assert "timeline_schema" in m["default_verifiers"]
    assert "handler" not in m and "precheck" not in m
    assert m["arguments"]["additionalProperties"] is False
