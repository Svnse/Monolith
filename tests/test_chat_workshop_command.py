from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.pages.chat import PageChat
from core.workflow_registry import Workflow


def _stub():
    flows = [Workflow("genesis", "Genesis", "", "native", None),
             Workflow("alpha", "Alpha", "", "monoline", None)]
    set_calls, blocks = [], []
    reg = types.SimpleNamespace(
        list_workflows=lambda: flows,
        active_id=lambda: "",
        set_active=lambda wid: set_calls.append(wid))
    stub = types.SimpleNamespace(
        _workflow_registry=reg,
        _emit_command_block=lambda t, b: blocks.append((t, b)))
    return stub, set_calls, blocks


def test_set_active_by_name():
    stub, set_calls, _ = _stub()
    assert PageChat._handle_workshop_command(stub, "/workshop Alpha") is True
    assert set_calls == ["alpha"]


def test_set_active_by_id():
    stub, set_calls, _ = _stub()
    PageChat._handle_workshop_command(stub, "/workshop alpha")
    assert set_calls == ["alpha"]


def test_list_when_no_arg():
    stub, _, blocks = _stub()
    PageChat._handle_workshop_command(stub, "/workshop")
    assert blocks and any("Alpha" in b for _, b in blocks)


def test_reset_to_genesis():
    stub, set_calls, _ = _stub()
    PageChat._handle_workshop_command(stub, "/workshop genesis")
    assert set_calls == [""]
    PageChat._handle_workshop_command(stub, "/workshop off")
    assert set_calls == ["", ""]


def test_unknown_name_reports_available():
    stub, set_calls, blocks = _stub()
    PageChat._handle_workshop_command(stub, "/workshop nope")
    assert set_calls == []  # nothing set on a miss
    assert any("not found" in t.lower() for t, _ in blocks)


def test_routing_recognizes_workshop():
    # _handle_world_commands routes "/workshop ..." to _handle_workshop_command
    stub, _, _ = _stub()
    stub.state = types.SimpleNamespace(world_state=None)
    routed = {}

    def _fake(c):
        routed["cmd"] = c
        return True

    stub._handle_workshop_command = _fake
    assert PageChat._handle_world_commands(stub, "/workshop Alpha") is True
    assert routed.get("cmd") == "/workshop Alpha"
