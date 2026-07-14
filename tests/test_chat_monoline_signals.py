from __future__ import annotations

import os
import types
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_dry_run_does_not_set_active(app, monkeypatch):
    # decision 5: Test = in-panel dry-run, NO Set-Active. It runs off-thread and
    # streams into the Running tab, but the active flow stays Genesis.
    from ui.pages.chat import PageChat, _PipelineWorker
    from core.workflow_registry import Workflow
    import engine.monoline_bridge as monoline_bridge
    wf = Workflow(id="alpha", name="Alpha", description="", kind="monoline",
                  source_path=None)
    stub = types.SimpleNamespace()
    started = {}
    stub._workflow_registry = types.SimpleNamespace(get=lambda wid: wf,
                                                     active_id=lambda: "")
    stub._last_task_id = None
    stub._pipeline_workers = []
    stub.state = types.SimpleNamespace(world_state=None)
    stub._set_workshop_active = lambda a: None
    stub._monoline_is_busy = lambda: False
    # _on_pipeline_error / _on_pipeline_stopped must exist on the stub because _run_monoline_dry
    # connects them (real PageChat has both; the stub is a SimpleNamespace).
    stub._on_pipeline_error = lambda *a: None
    stub._on_pipeline_stopped = lambda *a: None
    stub._trace_html = lambda *a, **k: None
    monkeypatch.setattr(monoline_bridge, "load_monoline", lambda: {})
    # capture the worker start without launching a real thread
    monkeypatch.setattr(_PipelineWorker, "run", lambda self: started.setdefault("ran", True))
    PageChat._run_monoline_dry(stub, "alpha")
    assert stub._workflow_registry.active_id() == ""  # Set-Active was NOT called
    assert len(stub._pipeline_workers) == 1           # a worker was dispatched


def test_pipeline_worker_emits_step_error_as_pipeline_error(app, monkeypatch):
    import engine.monoline_bridge as br
    from ui.pages.chat import _PipelineWorker

    class _Step:
        block_label = "Assistant"
        step_kind = "call_llm"
        error = "Local provider requires a model_path (path to .gguf file)"

    class _Result:
        error = ""
        output = ""
        step_log = [_Step()]
        block_status = {"assistant": "error"}

    class _Run:
        result = _Result()

    monkeypatch.setattr(br, "run_monoline_world", lambda *a, **k: _Run())

    worker = _PipelineWorker(
        types.SimpleNamespace(name="Broken"),
        {"prompt": "Hey"},
        parent_turn_id="",
        spawn_budget=None,
        should_cancel=lambda: False,
        is_busy=lambda: False,
    )
    errors, dones = [], []
    worker.sig_pipeline_error.connect(errors.append)
    worker.sig_pipeline_done.connect(dones.append)

    worker.run()

    assert dones == []
    assert errors == [
        "Assistant [call_llm]: Local provider requires a model_path (path to .gguf file)"
    ]
