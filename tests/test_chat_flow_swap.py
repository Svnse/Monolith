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


class _FakeWS:
    def __init__(self, active=""):
        self._active = active

    def get_active_workflow(self):
        return self._active


class _FakeRegistry:
    def __init__(self, wf=None):
        self._wf = wf

    def get(self, wid):
        return self._wf


def _make_chat_stub(active_id, wf):
    """Build a minimal object exposing exactly what the leading guard touches, then
    bind the REAL PageChat._dispatch_generation to it. This isolates the guard from
    PageChat's heavy __init__ (model load, surfaces) for a deterministic INV-#1 test."""
    from ui.pages.chat import PageChat
    stub = types.SimpleNamespace()
    stub.state = types.SimpleNamespace(world_state=_FakeWS(active_id))
    stub._workflow_registry = _FakeRegistry(wf)
    stub._MONOLINE_ENTRY_SOURCES = PageChat._MONOLINE_ENTRY_SOURCES
    stub._emitted = []
    stub._monoline_called = []

    # Genesis path collaborators -> record-only stubs (must run byte-for-byte order).
    stub._set_send_button_state = lambda **k: None
    stub._assistant_box = types.SimpleNamespace(start_rewrite_stream=lambda i: None)
    stub._start_assistant_stream = lambda: stub.__dict__.setdefault("_started", True)
    stub.message_list = types.SimpleNamespace(scrollToBottom=lambda: None)
    stub.sig_debug = types.SimpleNamespace(emit=lambda *a: None)
    stub.sig_generate = types.SimpleNamespace(emit=lambda p: stub._emitted.append(p))

    def _fake_monoline_run(self_wf, payload, *, rewrite_index=None, source="chat"):
        stub._monoline_called.append((self_wf, payload, source))
    stub._dispatch_monoline_run = _fake_monoline_run

    stub._dispatch_generation = types.MethodType(PageChat._dispatch_generation, stub)
    return stub


def test_genesis_default_emits_sig_generate_identical_payload(app):
    # INV-#1: no active flow -> branch NOT taken; sig_generate emitted with the payload.
    stub = _make_chat_stub(active_id="", wf=None)
    payload = {"prompt": "hi", "ephemeral": False}
    stub._dispatch_generation(payload, source="send:hi")  # REAL prefixed form, not the clean token
    assert stub._monoline_called == []        # monoline path never entered
    assert stub._emitted == [payload]         # sig_generate emitted, identical object
    assert stub.__dict__.get("_started") is True  # Genesis stream setup ran


def test_active_monoline_flow_diverts_and_does_not_emit(app):
    from core.workflow_registry import Workflow
    wf = Workflow(id="alpha", name="Alpha", description="", kind="monoline",
                  source_path=None)
    stub = _make_chat_stub(active_id="alpha", wf=wf)
    payload = {"prompt": "hi"}
    # REAL prefixed source: this only diverts under PREFIX matching (source.split(":")[0]),
    # so it is the regression that catches an exact-equality guard against the live vocabulary.
    stub._dispatch_generation(payload, source="send:hi")
    assert len(stub._monoline_called) == 1     # diverted to the monoline lane
    assert stub._emitted == []                 # sig_generate NOT emitted


def test_internal_sources_never_diverted(app):
    from core.workflow_registry import Workflow
    wf = Workflow(id="alpha", name="Alpha", description="", kind="monoline",
                  source_path=None)
    stub = _make_chat_stub(active_id="alpha", wf=wf)
    # tool_followup is Genesis-internal machinery -> must NOT divert even with active flow.
    stub._dispatch_generation({"prompt": "x"}, source="tool_followup")
    assert stub._monoline_called == []
    assert len(stub._emitted) == 1


def test_monoline_is_busy_ignores_own_workshop_flag(app):
    # Self-deadlock guard: _dispatch_monoline_run sets engines['workshop']='RUNNING';
    # the monolith-provider atom would refuse (Arm 2) if that counted as busy.
    import types
    from ui.pages.chat import PageChat

    class _WS:
        def snapshot(self):
            return {"engines": {"workshop": {"status": "RUNNING"}}}
    stub = types.SimpleNamespace()
    stub.state = types.SimpleNamespace(world_state=_WS())
    fn = types.MethodType(PageChat._monoline_is_busy, stub)
    assert fn() is False  # our own workshop flag must NOT make the run busy

def test_monoline_is_busy_true_on_external_generation(app):
    import types
    from ui.pages.chat import PageChat

    class _WS:
        def snapshot(self):
            return {"engines": {"workshop": {"status": "RUNNING"},
                                "main": {"status": "generating"}}}
    stub = types.SimpleNamespace()
    stub.state = types.SimpleNamespace(world_state=_WS())
    fn = types.MethodType(PageChat._monoline_is_busy, stub)
    assert fn() is True  # a GENUINE external generation still counts as busy


def test_on_pipeline_done_inserts_pipeline_origin_and_resets(app):
    # State-cleanliness (the real INV-#1 risk beyond the guard): the result lane must leave
    # PageChat equivalent to a Genesis turn-end -- message inserted with origin=ORIGIN_PIPELINE,
    # send button reset, workshop activity flag cleared. Also pins the live insert_message
    # signature (index, role, text, extra=) so a mismatch can't pass behind the stubbed worker.
    # (The RunView already reflects done via the RunFinished event -- no card to finalize here.)
    import types
    from ui.pages.chat import PageChat, ORIGIN_PIPELINE
    inserted, flags = {}, {}
    stub = types.SimpleNamespace()
    stub._current_session = {"messages": [{"role": "user", "content": "hi"}]}
    stub._sessions = types.SimpleNamespace(
        insert_message=lambda idx, role, text, extra=None: (
            inserted.update(idx=idx, role=role, text=text, extra=extra) or 0))
    stub._append_message_widget = lambda idx: flags.__setitem__("appended", idx)
    stub._set_send_button_state = lambda **k: flags.update(k)
    stub._set_workshop_active = lambda a: flags.__setitem__("workshop", a)
    stub.sig_sync_history = types.SimpleNamespace(emit=lambda h: flags.__setitem__("synced", h))
    stub._build_engine_history_from_session = lambda: ["H"]
    PageChat._on_pipeline_done(stub, "the answer")  # no worker -> plain answer, no trace block
    assert inserted["role"] == "assistant" and inserted["text"] == "the answer"
    assert inserted["idx"] == 1                       # appended after the 1 existing message
    assert inserted["extra"] == {"origin": ORIGIN_PIPELINE}
    assert flags.get("is_running") is False           # send button reset (turn-end equivalent)
    assert flags.get("workshop") is False             # workshop activity flag cleared
    assert flags.get("appended") == 0                 # append just the new row (no full re-render)
    assert flags.get("synced") == ["H"]               # workshop output synced to engine immediately


def test_on_pipeline_error_resets_button_and_workshop(app):
    # The error slot must leave PageChat at a clean idle turn-end: send button reset, workshop
    # flag cleared, and the failure surfaced in the trace. (No card to finalize -- a completed-
    # with-error run flips its RunView via the RunFinished event; this is the reset half.)
    from ui.pages.chat import PageChat
    flags, traces = {}, []
    stub = types.SimpleNamespace()
    stub._trace_html = lambda text, *a, **k: traces.append(text)
    stub._set_send_button_state = lambda **k: flags.update(k)
    stub._set_workshop_active = lambda a: flags.__setitem__("workshop", a)
    PageChat._on_pipeline_error(stub, "boom")
    assert flags.get("is_running") is False
    assert flags.get("workshop") is False
    assert any("boom" in t for t in traces)


def test_apply_run_event_registers_model_and_binds_inline_view(app):
    # Wiring contract: on RunStarted the live model is registered (so the run browser can bind
    # it) and, for an equipped run, the inline RunView is bound to that same model.
    import types
    from ui.pages.chat import PageChat
    from core.run_model import RunModelBuilder, RunStarted, live_runs
    bound = {}
    worker = types.SimpleNamespace(
        builder=RunModelBuilder(),
        inline_view=types.SimpleNamespace(bind=lambda m: bound.__setitem__("model", m)))
    PageChat._apply_run_event(types.SimpleNamespace(), worker,
                              RunStarted(run_id="rX", flow_id="f", name="F",
                                         user_input="hi", graph=[], wires=[]))
    assert worker.builder.model is not None
    assert live_runs.get("rX") is worker.builder.model    # registered for the browser
    assert bound.get("model") is worker.builder.model      # inline view bound to the same model
    live_runs.drop("rX")


def test_apply_run_event_dry_run_registers_without_inline_view(app):
    # A dry-run has no chat row (inline_view=None): it must still register the live model for the
    # browser, and must not crash on the missing view.
    import types
    from ui.pages.chat import PageChat
    from core.run_model import RunModelBuilder, RunStarted, live_runs
    worker = types.SimpleNamespace(builder=RunModelBuilder(), inline_view=None)
    PageChat._apply_run_event(types.SimpleNamespace(), worker,
                              RunStarted(run_id="rDry", flow_id="f", name="F",
                                         user_input="", graph=[], wires=[]))
    assert live_runs.get("rDry") is worker.builder.model
    live_runs.drop("rDry")


def test_dispatch_monoline_run_setup_failure_leaves_no_stuck_state(app, monkeypatch):
    # INV-#1 state-cleanliness (review IMPORTANT-1): if setup throws before the worker launches,
    # the run-state flags must NOT be left stuck -- a stuck send button / workshop flag would
    # corrupt the NEXT Genesis turn. The running-state is set only AFTER the thread starts, so a
    # mid-setup failure (here: a failing load_monoline) must reset to idle and never flip to True.
    import engine.monoline_bridge as br
    from ui.pages.chat import PageChat
    from core.workflow_registry import Workflow
    wf = Workflow(id="alpha", name="Alpha", description="", kind="monoline", source_path=None)

    def _boom():
        raise RuntimeError("plugin missing")
    monkeypatch.setattr(br, "load_monoline", _boom)

    btn, ws_flag = [], []
    stub = types.SimpleNamespace()
    stub._surface = types.SimpleNamespace()          # no _append_card_widget -> hasattr False, skipped
    stub._last_task_id = ""
    stub._tool_cancel_requested = False
    stub._pipeline_workers = []
    stub._monoline_is_busy = lambda: False
    stub._on_pipeline_block = lambda *a: None
    stub._on_pipeline_done = lambda *a: None
    stub._on_pipeline_error = lambda *a: None
    stub._set_send_button_state = lambda **k: btn.append(k.get("is_running"))
    stub._set_workshop_active = lambda a: ws_flag.append(a)
    stub._trace_html = lambda *a, **k: None

    PageChat._dispatch_monoline_run(stub, wf, {"prompt": "x"}, source="send:hi")
    assert btn == [False]          # button reset on failure, NEVER flipped to a stuck True
    assert ws_flag and ws_flag[-1] is False   # workshop flag left idle


def test_dispatch_monoline_run_preflight_failure_traces_and_makes_no_row(app, monkeypatch):
    # New preflight discipline: validation runs BEFORE the run row is built, so a bad config
    # leaves NO empty/error row, NO worker, and never flips the running flags -- it just traces.
    import engine.monoline_bridge as br
    from ui.pages.chat import PageChat
    from core.workflow_registry import Workflow

    wf = Workflow(id="alpha", name="Alpha", description="", kind="monoline", source_path=None)
    monkeypatch.setattr(br, "load_monoline", lambda: {})
    monkeypatch.setattr(br, "validate_chat_workflow", lambda workflow: "Assistant: bad config")

    flags, traces, appended = {}, [], []
    stub = types.SimpleNamespace()
    stub._surface = types.SimpleNamespace(_append_card_widget=lambda w: appended.append(w))
    stub._last_task_id = ""
    stub._tool_cancel_requested = False
    stub._pipeline_workers = []
    stub._monoline_is_busy = lambda: False
    stub._set_send_button_state = lambda **k: flags.update(k)
    stub._set_workshop_active = lambda a: flags.__setitem__("workshop", a)
    stub._trace_html = lambda text, *a, **k: traces.append(text)

    PageChat._dispatch_monoline_run(stub, wf, {"prompt": "x"}, source="send:hi")

    assert stub._pipeline_workers == []              # no worker created
    assert appended == []                            # no run row appended on preflight fail
    assert any("Assistant: bad config" in t for t in traces)
    assert "workshop" not in flags                   # running flags never flipped
