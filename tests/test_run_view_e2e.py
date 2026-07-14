"""End-to-end (model-free) closure for the unified run view (spec 2026-06-14 §10):
a real bridge run -> RunEvent stream -> RunModelBuilder -> RunView (the live chat wiring),
then the SAME run rehydrated from turn_trace -> an equivalent RunView (the browser path).
Uses an echo engine, so no live model is required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

import core.turn_trace as tt
from core.run_model import RunModelBuilder, RunStarted
from core.workflow_registry import Workflow
from engine import monoline_bridge as br
from ui.components.run_view import RunView
from tests._monoline_requirement import requires_monoline


pytestmark = requires_monoline


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _echo_world(tmp_path: Path) -> Workflow:
    bp = {
        "name": "Echo",
        "blocks": [
            {"id": "input", "kind": "port",
             "config": {"direction": "in", "label": "request", "source": "user_input"}},
            {"id": "assistant", "kind": "llm"},
            {"id": "output", "kind": "port",
             "config": {"direction": "out", "label": "response", "source": "subgraph"}},
        ],
        "connections": [["input.value", "assistant.prompt"], ["assistant.response", "output.value"]],
    }
    p = Path(tmp_path) / "echo.monoline"
    p.write_text(json.dumps(bp), encoding="utf-8")
    return Workflow(id="echo", name="Echo", description="", kind="monoline", source_path=p)


def _echo_engine(messages, _cfg):
    for msg in reversed(messages):
        if str(msg.get("role", "")).lower() == "user":
            return f"echo:{msg.get('content', '')}"
    return "echo:"


def test_e2e_live_events_render_then_rehydrate(app, tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    wf = _echo_world(tmp_path)
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", _echo_engine)
    try:
        # LIVE: feed the bridge's RunEvent stream through the builder into a RunView (chat wiring).
        builder = RunModelBuilder()
        view = RunView()

        def _on_event(ev):
            builder.apply(ev)
            if isinstance(ev, RunStarted):
                view.bind(builder.model)

        br.run_monoline_world(
            wf, user_input="hello", parent_turn_id="",
            spawn_budget=None, should_cancel=lambda: False, is_busy=lambda: False,
            on_step=None, should_stop=None, on_event=_on_event)
        view.show()
        assert "assistant" in [b.id for b in builder.model.block_list()]
        view.expand_row("assistant")
        detail = view.row_detail_text("assistant")
        assert "echo:hello" in detail   # the block's echoed output is visible
        assert "hello" in detail        # the derived input (user prompt via the input port)

        # HISTORICAL: the same run rehydrates from turn_trace into an equivalent RunView.
        runs = tt.list_recent_runs(5)
        assert runs and runs[0].flow_id == "echo"
        rm = tt.rehydrate_run(runs[0].run_id)
        assert rm is not None
        v2 = RunView(rm)
        v2.show()
        v2.expand_row("assistant")
        assert "echo:hello" in v2.row_detail_text("assistant")
    finally:
        tt.set_db_path(None)
