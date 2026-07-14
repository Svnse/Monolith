from __future__ import annotations

import core.skill_runtime as sr
from core.skill_runtime import (
    ToolExecutionContext, L1_PRINCIPAL_TOOLS, L2_WORKER_TOOLS, L3_LEAF_TOOLS,
    derive_child_context, execute_tool_call_enveloped, execute_run_workshop,
)
from core.paths import LOG_DIR


def test_run_workshop_is_l1_only_capability():
    # The governance gate: present at L1, ABSENT at L2/L3 (so children can't call it).
    assert "run_workshop" in L1_PRINCIPAL_TOOLS
    assert "run_workshop" not in L2_WORKER_TOOLS
    assert "run_workshop" not in L3_LEAF_TOOLS


def test_gate_denies_run_workshop_below_l1():
    # A flow's blocks run at L3 (and L2); the gate must deny run_workshop there -> no
    # workflow -> run_workshop -> workflow recursion.
    root = ToolExecutionContext(archive_dir=LOG_DIR, level=1, allowed_tools=L1_PRINCIPAL_TOOLS)
    for lvl in (2, 3):
        child = derive_child_context(root, lvl, label="x")
        assert "run_workshop" not in child.allowed_tools
        env = execute_tool_call_enveloped({"tool": "run_workshop", "name": "alpha"}, child)
        assert env.ok is False
        assert "denied" in env.text.lower() or "not permitted" in env.text.lower()


def test_executor_l1_hands_off_to_host():
    seen = {}
    ctx = ToolExecutionContext(
        archive_dir=LOG_DIR, level=1, allowed_tools=L1_PRINCIPAL_TOOLS,
        on_run_workshop=lambda c: (seen.update(c), "[run_workshop: 'X' PENDING]")[1])
    out = execute_run_workshop({"name": "X"}, ctx)
    assert out == "[run_workshop: 'X' PENDING]"
    assert seen.get("name") == "X"


def test_executor_without_host_denies():
    ctx = ToolExecutionContext(archive_dir=LOG_DIR, level=1,
                               allowed_tools=L1_PRINCIPAL_TOOLS, on_run_workshop=None)
    out = execute_run_workshop({"name": "X"}, ctx)
    assert "only the principal" in out.lower()


def test_run_workshop_registered():
    assert sr._TOOL_EXECUTORS.get("run_workshop") is execute_run_workshop


def test_child_context_drops_run_workshop_hook():
    root = ToolExecutionContext(archive_dir=LOG_DIR, level=1, allowed_tools=L1_PRINCIPAL_TOOLS,
                                on_run_workshop=lambda c: "x")
    child = derive_child_context(root, 2, label="x")
    assert child.on_run_workshop is None  # only L1 carries the host hook
