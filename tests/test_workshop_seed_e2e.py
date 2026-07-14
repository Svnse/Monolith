from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.workshop_seed import seed_workshop_flows
from core.workflow_registry import WorkflowRegistry
from engine import monoline_bridge as br
from tests._monoline_requirement import requires_monoline


pytestmark = requires_monoline


def test_seeded_flow_lists_and_runs(tmp_path, monkeypatch):
    # The whole SP1 loop end-to-end: seed -> registry lists -> bridge runs it -> output.
    seeded = seed_workshop_flows(worlds_dir=tmp_path)
    assert seeded >= 1

    reg = WorkflowRegistry(workflows_dir=tmp_path)
    ids = [w.id for w in reg.list_workflows()]
    assert "sample-two-step" in ids
    wf = reg.get("sample-two-step")
    assert wf is not None and wf.kind == "monoline"

    # run the seeded flow through the REAL bridge with a deterministic engine (no live model)
    m = br.load_monoline()

    def _echo(messages, _cfg):
        for mm in reversed(messages):
            if str(mm.get("role", "")).lower() == "user":
                return f"echo:{mm.get('content', '')}"
        return "echo:"

    monkeypatch.setattr(m["engine"], "engine_call", _echo)
    # isolate: don't write this run's frames/faults into the REAL turn_trace.sqlite3 (which would
    # leak rows into other store-based tests). This test only asserts output, not the trace.
    import core.turn_trace as tt
    monkeypatch.setattr(tt, "_flag_enabled", lambda: False)
    run = br.run_monoline_world(
        wf, user_input="hello", parent_turn_id="",
        spawn_budget=None, should_cancel=lambda: False, is_busy=lambda: False,
        on_step=None, should_stop=None)
    assert run.result.output, "seeded flow produced no output"
