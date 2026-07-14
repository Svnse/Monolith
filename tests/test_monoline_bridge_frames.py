from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.turn_trace as tt
from engine import monoline_bridge as br
from core.workflow_registry import Workflow
from tests._monoline_requirement import requires_monoline


pytestmark = requires_monoline


def _world(tmp_path):
    p = Path(tmp_path) / "echo.monoline"
    p.write_text(json.dumps({
        "name": "Echo",  # NO schema_version -> lenient build_preset path (the world actually runs)
        "blocks": [
            {"id": "input", "kind": "port", "config": {"direction": "in", "label": "request", "source": "user_input"}},
            {"id": "assistant", "kind": "llm"},
            {"id": "output", "kind": "port", "config": {"direction": "out", "label": "response", "source": "subgraph"}}],
        "connections": [["input.value", "assistant.prompt"], ["assistant.response", "output.value"]],
    }), encoding="utf-8")
    return Workflow(id="echo", name="Echo", description="", kind="monoline", source_path=p)


def test_local_llm_block_gets_host_written_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(Path(tmp_path) / "turn_trace.sqlite3")
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: "OUT")
    try:
        # parent_turn_id="" -> run-root is its own NULL-parent governance root (discoverable);
        # the local llm block writes a child frame under it (parent=run_root).
        br.run_monoline_world(_world(tmp_path), user_input="hi", parent_turn_id="",
                              spawn_budget=None, should_cancel=lambda: False,
                              is_busy=lambda: False, on_step=None, should_stop=None)
        root = tt.latest_governance_root()
        kids = tt.list_child_frames(root)
        # the local llm block produced a host-written child frame under the run root
        assert any(getattr(k, "engine_key", "").startswith("monoline:") for k in kids)
    finally:
        tt.set_db_path(None)
