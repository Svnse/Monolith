from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_on_run_workshop_dispatches_and_returns_pending(app, monkeypatch):
    from ui.pages.chat import PageChat, _PipelineWorker
    import engine.monoline_bridge as br
    from core.workflow_registry import Workflow

    monkeypatch.setattr(br, "load_monoline", lambda: {})           # skip the real swap
    monkeypatch.setattr(_PipelineWorker, "run", lambda self: None)  # don't actually run the flow

    wf = Workflow("alpha", "Alpha", "", "monoline", source_path=Path("x.monoline"))
    stub = types.SimpleNamespace(
        _workflow_registry=types.SimpleNamespace(get=lambda n: wf, list_workflows=lambda: [wf]),
        _last_task_id="", _tool_cancel_requested=False, _spawn_budget=None,
        _pipeline_workers=[], _monoline_is_busy=lambda: False,
        _set_workshop_active=lambda a: None)
    out = PageChat._on_run_workshop(stub, {"name": "alpha", "input": "hi"})
    assert "PENDING" in out and "Alpha" in out
    assert len(stub._pipeline_workers) == 1


def test_on_run_workshop_unknown_name(app):
    from ui.pages.chat import PageChat
    stub = types.SimpleNamespace(
        _workflow_registry=types.SimpleNamespace(get=lambda n: None, list_workflows=lambda: []))
    out = PageChat._on_run_workshop(stub, {"name": "nope"})
    assert "no workflow named 'nope'" in out


def test_workshop_tool_done_and_error_fold_back(app):
    from ui.pages.chat import PageChat
    queued = []
    stub = types.SimpleNamespace(
        _workshop_run_finished=lambda w: None,  # tested separately below
        _queue_tool_followup=lambda txt, rewrite_index=None: queued.append(txt))
    PageChat._on_workshop_tool_done(stub, "Alpha", "the result", None)
    assert queued and "completed" in queued[0] and "the result" in queued[0]
    PageChat._on_workshop_tool_error(stub, "Alpha", "boom", None)
    assert "failed" in queued[1] and "boom" in queued[1]


def test_workshop_flag_cleared_unconditionally_when_last_run_finishes(app):
    # C2 regression: _workshop_run_finished must clear the flag WITHOUT gating on _engine_is_busy
    # (which counts our own 'workshop=RUNNING' as busy). Clear only when the LAST run finishes.
    from ui.pages.chat import PageChat
    cleared = []
    stub = types.SimpleNamespace(
        _workshop_inflight=1, _pipeline_workers=[],
        _set_workshop_active=lambda a: cleared.append(a))
    PageChat._workshop_run_finished(stub, None)
    assert cleared == [False]   # inflight 1 -> 0 -> cleared, unconditionally

    cleared.clear()
    stub._workshop_inflight = 2  # two concurrent runs
    PageChat._workshop_run_finished(stub, None)
    assert cleared == []         # one still running -> not cleared
    PageChat._workshop_run_finished(stub, None)
    assert cleared == [False]    # last one done -> cleared


def test_run_workshop_per_turn_cap(app, monkeypatch):
    from ui.pages.chat import PageChat, _PipelineWorker
    import engine.monoline_bridge as br
    from core.workflow_registry import Workflow

    monkeypatch.setattr(br, "load_monoline", lambda: {})
    monkeypatch.setattr(_PipelineWorker, "run", lambda self: None)
    wf = Workflow("alpha", "Alpha", "", "monoline", source_path=Path("x.monoline"))
    stub = types.SimpleNamespace(
        _MAX_WORKSHOP_RUNS_PER_TURN=2, _workshop_turn_count=0, _workshop_inflight=0,
        _workflow_registry=types.SimpleNamespace(get=lambda n: wf, list_workflows=lambda: [wf]),
        _last_task_id="", _tool_cancel_requested=False, _spawn_budget=None,
        _pipeline_workers=[], _monoline_is_busy=lambda: False,
        _set_workshop_active=lambda a: None)
    assert "PENDING" in PageChat._on_run_workshop(stub, {"name": "alpha"})
    assert "PENDING" in PageChat._on_run_workshop(stub, {"name": "alpha"})
    out3 = PageChat._on_run_workshop(stub, {"name": "alpha"})
    assert "limit" in out3.lower()           # 3rd call (cap=2) refused
    assert len(stub._pipeline_workers) == 2  # only 2 actually launched


def _finished_run_builder():
    from core.run_model import (RunModelBuilder, RunStarted, RunBlockSpec,
                                 BlockFinished, RunFinished)
    b = RunModelBuilder()
    b.apply(RunStarted(run_id="r", flow_id="f", name="F", user_input="hi",
                       graph=[RunBlockSpec(id="draft", label="Draft", kind="llm")], wires=[]))
    b.apply(BlockFinished(run_id="r", block_id="draft", label="Draft", kind="llm",
                          outputs={"response": "INNER STEP OUTPUT"},
                          started_at=1.0, completed_at=2.0, status="done", error=""))
    b.apply(RunFinished(run_id="r", output="FINAL ANSWER", error=""))
    return b


def test_on_pipeline_done_embeds_trace_and_syncs_history(app):
    from ui.pages.chat import PageChat, ORIGIN_PIPELINE
    inserted: dict = {}
    synced: dict = {}
    worker = types.SimpleNamespace(builder=_finished_run_builder())

    class _Sessions:
        def insert_message(self, pos, role, content, extra=None):
            inserted.update(pos=pos, role=role, content=content, extra=extra)
            return 0

    stub = types.SimpleNamespace(
        _sessions=_Sessions(),
        _current_session={"messages": []},
        _append_message_widget=lambda idx: None,
        _set_send_button_state=lambda **kw: None,
        _set_workshop_active=lambda v: None,
        sig_sync_history=types.SimpleNamespace(emit=lambda h: synced.update(h=h)),
        _build_engine_history_from_session=lambda: ["HISTORY"],
    )

    PageChat._on_pipeline_done(stub, "FINAL ANSWER", worker)

    assert inserted["role"] == "assistant"
    assert inserted["extra"] == {"origin": ORIGIN_PIPELINE}
    assert "FINAL ANSWER" in inserted["content"]                 # the visible answer
    assert "[ATTACHED: workshop trace" in inserted["content"]    # block trace embedded (hidden in bubble)
    assert "- Draft: INNER STEP OUTPUT" in inserted["content"]
    assert synced["h"] == ["HISTORY"]                            # history synced immediately (timing fix)


def test_on_pipeline_done_without_worker_still_inserts_plain_answer(app):
    # Defensive: a caller that does not pass the worker still gets the answer + an immediate sync.
    from ui.pages.chat import PageChat
    inserted: dict = {}
    synced: dict = {}

    class _Sessions:
        def insert_message(self, pos, role, content, extra=None):
            inserted.update(content=content)
            return 0

    stub = types.SimpleNamespace(
        _sessions=_Sessions(), _current_session={"messages": []},
        _append_message_widget=lambda idx: None,
        _set_send_button_state=lambda **kw: None,
        _set_workshop_active=lambda v: None,
        sig_sync_history=types.SimpleNamespace(emit=lambda h: synced.update(h=h)),
        _build_engine_history_from_session=lambda: [])

    PageChat._on_pipeline_done(stub, "PLAIN ANSWER")

    assert inserted["content"] == "PLAIN ANSWER"     # no trace, no crash
    assert "h" in synced                             # still synced immediately


def test_pipeline_worker_emits_stopped_not_error_when_run_was_cancelled(app, monkeypatch):
    # A user STOP must take the clean stopped path: emit sig_pipeline_stopped, NOT sig_pipeline_error
    # (even though summarize_run_failure would report the runtime's stop sentinel) and NOT done.
    from ui.pages.chat import _PipelineWorker
    import engine.monoline_bridge as br
    from core.workflow_registry import Workflow

    fake_run = types.SimpleNamespace(result=types.SimpleNamespace(output="", error="Activation stopped."))
    monkeypatch.setattr(br, "run_monoline_world", lambda *a, **k: fake_run)
    monkeypatch.setattr(br, "summarize_run_failure", lambda run: "Activation stopped.")

    wf = Workflow("a", "A", "", "monoline", source_path=Path("x.monoline"))
    w = _PipelineWorker(wf, {"prompt": "hi"}, parent_turn_id="", spawn_budget=None,
                        should_cancel=lambda: True, is_busy=lambda: False)  # user pressed STOP
    seen = {"stopped": 0, "error": 0, "done": 0}
    w.sig_pipeline_stopped.connect(lambda: seen.__setitem__("stopped", seen["stopped"] + 1))
    w.sig_pipeline_error.connect(lambda _m: seen.__setitem__("error", seen["error"] + 1))
    w.sig_pipeline_done.connect(lambda _m: seen.__setitem__("done", seen["done"] + 1))

    w.run()

    assert seen == {"stopped": 1, "error": 0, "done": 0}


def test_on_pipeline_stopped_resets_state_without_inserting_a_message(app):
    from ui.pages.chat import PageChat
    inserted = {"n": 0}
    flags = {}
    stub = types.SimpleNamespace(
        _sessions=types.SimpleNamespace(
            insert_message=lambda *a, **k: inserted.__setitem__("n", inserted["n"] + 1)),
        _set_send_button_state=lambda **k: flags.update(k),
        _set_workshop_active=lambda v: flags.__setitem__("workshop", v))

    PageChat._on_pipeline_stopped(stub)

    assert inserted["n"] == 0                   # NOTHING inserted into the chat ("shouldn't say anything")
    assert flags.get("is_running") is False     # send button reset
    assert flags.get("workshop") is False       # workshop activity flag cleared
