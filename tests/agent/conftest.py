"""Fixtures for the agent test suite (blueprint §10, Appendix B "Agent security")."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make agent_helpers importable in any import mode

from agent_helpers import FAST_TIMEOUTS, SECRET, ToolKit, make_timeline  # noqa: E402

from m63_omni.agent import ControlPlane, Principal  # noqa: E402


@pytest.fixture
def toolkit() -> ToolKit:
    return ToolKit()


@pytest.fixture
def cp(toolkit: ToolKit) -> ControlPlane:
    return ControlPlane(toolkit.registry, secret=SECRET, timeout_classes=FAST_TIMEOUTS, clock=lambda: 1_700_000_000.0)


@pytest.fixture
def editor() -> Principal:
    return Principal("alice", "acme", frozenset({"timeline:write", "web:read"}))


@pytest.fixture
def session(cp: ControlPlane, editor: Principal):
    s = cp.open_session(editor, "session_42")
    cp.ingest_artifact(s, "asset_91", "timeline", make_timeline().to_dict())
    return s
