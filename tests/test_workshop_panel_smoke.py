from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

import core.turn_trace as tt
from core.run_model import (BlockFinished, RunBlockSpec, RunModelBuilder, RunStarted,
                            live_runs)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _seed_run(db: Path, monkeypatch, run_id="hist1", name="Historical", out="HISTOUT"):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(db)
    tt.record_frame(tt.FrameTraceRecord(
        turn_id=run_id, captured_at=_now(), backend="monoline", engine_key=f"monoline:{name}",
        gen_id=0, final_messages=tuple(), system_prompt_chars=0, user_prompt_chars=2,
        total_chars=2, parent_turn_id=None,
        metadata={"kind": "workflow", "flow": "hist", "name": name, "user_input": "hi",
                  "graph": [{"id": "draft", "label": "draft", "kind": "llm"}], "wires": []}))
    tt.record_fault(tt.FaultTraceRecord(
        turn_id=run_id, parent_turn_id=run_id, seq=0, emitted_at=_now(),
        event_kind="monoline_block", source_kind="kernel", source_name="monoline_bridge",
        authority_tier="observation", fault_kind=None, severity=None,
        payload={"block_id": "draft", "block_label": "draft", "step_kind": "call_llm",
                 "ok": True, "error": "", "outputs": {"response": out},
                 "started_at": 1.0, "completed_at": 2.0}))


def test_browser_lists_historical_runs(app, tmp_path, monkeypatch):
    from ui.panels.workshop import WorkshopPane
    _seed_run(Path(tmp_path) / "tt.sqlite3", monkeypatch)
    pane = None
    try:
        pane = WorkshopPane()
        pane.show()
        pane.refresh()
        assert pane.run_count() >= 1
        assert "hist1" in pane.run_ids()
    finally:
        if pane is not None:
            pane.deleteLater()
        tt.set_db_path(None)


def test_browser_selecting_historical_run_rehydrates_into_runview(app, tmp_path, monkeypatch):
    from ui.panels.workshop import WorkshopPane
    _seed_run(Path(tmp_path) / "tt.sqlite3", monkeypatch)
    pane = None
    try:
        pane = WorkshopPane()
        pane.show()
        pane.select_run("hist1")
        view = pane.current_run_view()
        assert view.row_status("draft") == "done"       # rehydrated block shown
        view.expand_row("draft")
        assert "HISTOUT" in view.row_detail_text("draft")
    finally:
        if pane is not None:
            pane.deleteLater()
        tt.set_db_path(None)


def test_browser_shows_live_run_and_binds_same_model(app, tmp_path, monkeypatch):
    from ui.panels.workshop import WorkshopPane
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    b = RunModelBuilder()
    b.apply(RunStarted(run_id="live1", flow_id="f", name="Live", user_input="hi",
                       graph=[RunBlockSpec(id="draft", label="draft", kind="llm")], wires=[]))
    live_runs.register(b.model)
    pane = None
    try:
        pane = WorkshopPane()
        pane.show()
        pane.refresh()
        assert "live1" in pane.run_ids()
        pane.select_run("live1")
        view = pane.current_run_view()
        assert view.block_row_count() == 1
        # a live update flows to the bound view (it observes the SAME model object)
        b.apply(BlockFinished(run_id="live1", block_id="draft", label="draft", kind="llm",
                              outputs={"response": "LIVEOUT"}, started_at=1.0, completed_at=2.0,
                              status="done"))
        assert view.row_status("draft") == "done"
    finally:
        if pane is not None:
            pane.deleteLater()
        live_runs.drop("live1")
        tt.set_db_path(None)
