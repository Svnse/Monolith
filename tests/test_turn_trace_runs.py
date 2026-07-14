from __future__ import annotations

import datetime as _dt
from pathlib import Path

import core.turn_trace as tt


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _record_run(db: Path, monkeypatch, run_id="run1", flow="two-step", name="Two-Step",
                draft_out="DRAFTED", ok=True):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(db)
    tt.record_frame(tt.FrameTraceRecord(
        turn_id=run_id, captured_at=_now(), backend="monoline",
        engine_key=f"monoline:{name}", gen_id=0, final_messages=tuple(),
        system_prompt_chars=0, user_prompt_chars=2, total_chars=2, parent_turn_id=None,
        metadata={"kind": "workflow", "flow": flow, "name": name, "user_input": "hi",
                  "graph": [{"id": "input", "label": "request", "kind": "port"},
                            {"id": "draft", "label": "draft", "kind": "llm"},
                            {"id": "output", "label": "response", "kind": "port"}],
                  "wires": ["input.value -> draft.prompt", "draft.response -> output.value"]}))
    tt.record_fault(tt.FaultTraceRecord(
        turn_id=run_id, parent_turn_id=run_id, seq=0, emitted_at=_now(),
        event_kind="monoline_block", source_kind="kernel", source_name="monoline_bridge",
        authority_tier="observation", fault_kind=None, severity=None,
        payload={"block_id": "draft", "block_label": "draft", "step_kind": "call_llm",
                 "ok": ok, "error": "" if ok else "boom",
                 "outputs": {"response": draft_out}, "started_at": 1.0, "completed_at": 2.0}))


def test_list_recent_runs_returns_monoline_roots(tmp_path, monkeypatch):
    _record_run(Path(tmp_path) / "tt.sqlite3", monkeypatch)
    try:
        runs = tt.list_recent_runs(10)
        assert runs and runs[0].run_id == "run1"
        assert runs[0].flow_id == "two-step"
        assert runs[0].name == "Two-Step"
    finally:
        tt.set_db_path(None)


def test_list_recent_runs_excludes_non_workflow_frames(tmp_path, monkeypatch):
    db = Path(tmp_path) / "tt.sqlite3"
    _record_run(db, monkeypatch)
    # a per-block monoline frame (kind != workflow) must NOT appear as a run
    tt.record_frame(tt.FrameTraceRecord(
        turn_id="blk1", captured_at=_now(), backend="monoline", engine_key="monoline:draft",
        gen_id=0, final_messages=tuple(), system_prompt_chars=0, user_prompt_chars=0,
        total_chars=5, parent_turn_id="run1",
        metadata={"kind": "monoline_block", "block_label": "draft"}))
    try:
        ids = [r.run_id for r in tt.list_recent_runs(10)]
        assert "run1" in ids and "blk1" not in ids
    finally:
        tt.set_db_path(None)


def test_rehydrate_run_rebuilds_model_with_io(tmp_path, monkeypatch):
    _record_run(Path(tmp_path) / "tt.sqlite3", monkeypatch)
    try:
        m = tt.rehydrate_run("run1")
        assert m is not None
        assert m.name == "Two-Step" and m.status == "done"
        draft = m.block("draft")
        assert draft.status == "done"
        assert draft.outputs.get("response") == "DRAFTED"
        # derived input: draft.prompt comes from the input port -> user_input "hi"
        assert m.inputs_for("draft") == {"prompt": "hi"}
    finally:
        tt.set_db_path(None)


def test_rehydrate_run_reflects_block_error(tmp_path, monkeypatch):
    _record_run(Path(tmp_path) / "tt.sqlite3", monkeypatch, ok=False)
    try:
        m = tt.rehydrate_run("run1")
        assert m is not None and m.status == "error"
        assert m.block("draft").status == "error"
    finally:
        tt.set_db_path(None)


def test_rehydrate_unknown_root_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    try:
        assert tt.rehydrate_run("does-not-exist") is None
    finally:
        tt.set_db_path(None)
