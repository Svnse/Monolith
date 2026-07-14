from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal

from core.state import AppState, SystemStatus
from core.task import Task, TaskStatus
from monokernel.bridge import MonoBridge
from monokernel.dock import MonoDock
from monokernel.guard import MonoGuard
from ui.addons.context import AddonContext
from ui.addons.host import AddonHost
from ui.addons.registry import AddonRegistry
from ui.addons.spec import AddonSpec
from ui.bridge import UIBridge


class DummyEngine(QObject):
    sig_status = Signal(SystemStatus)
    sig_trace = Signal(str)
    sig_token = Signal(str)
    sig_finished = Signal()
    sig_model_capabilities = Signal(dict)

    def __init__(self, auto_ready: bool = True) -> None:
        super().__init__()
        self.auto_ready = auto_ready
        self.calls: list[tuple[str, Any]] = []

    def set_model_path(self, payload: dict) -> None:
        self.calls.append(("set_model_path", payload))
        self.sig_status.emit(SystemStatus.READY)

    def set_history(self, payload: dict) -> None:
        self.calls.append(("set_history", payload))

    def set_ctx_limit(self, payload: dict) -> None:
        self.calls.append(("set_ctx_limit", payload))

    def load_model(self) -> None:
        self.calls.append(("load_model", None))
        self.sig_status.emit(SystemStatus.LOADING)
        self.sig_status.emit(SystemStatus.READY)

    def unload_model(self) -> None:
        self.calls.append(("unload_model", None))
        self.sig_status.emit(SystemStatus.UNLOADING)
        self.sig_status.emit(SystemStatus.READY)

    def generate(self, payload: dict) -> None:
        self.calls.append(("generate", payload))
        self.sig_status.emit(SystemStatus.RUNNING)
        if self.auto_ready:
            self.sig_finished.emit()
            self.sig_status.emit(SystemStatus.READY)

    def stop_generation(self) -> None:
        self.calls.append(("stop_generation", None))
        self.sig_status.emit(SystemStatus.READY)

    def shutdown(self) -> None:
        self.calls.append(("shutdown", None))

    def finish(self) -> None:
        self.sig_finished.emit()
        self.sig_status.emit(SystemStatus.READY)


class _EventSink:
    _addon_id = "terminal"

    def __init__(self) -> None:
        self.events: list[tuple[str, str, object]] = []

    def on_engine_event(self, engine_key: str, event: str, payload: object) -> None:
        self.events.append((engine_key, event, payload))


class _StackStub:
    def __init__(self, widgets: list[object]) -> None:
        self._widgets = widgets

    def count(self) -> int:
        return len(self._widgets)

    def widget(self, index: int) -> object:
        return self._widgets[index]


class _UIStub:
    def __init__(self, widgets: list[object]) -> None:
        self.stack = _StackStub(widgets)


def _make_task(command: str, payload: dict | None = None) -> Task:
    return Task.new(
        addon_pid="test",
        target="llm",
        command=command,
        payload=payload or {},
        priority=2,
    )


def test_guard_rejects_unknown_engine() -> None:
    state = AppState()
    guard = MonoGuard(state, {})
    task = Task.new(
        addon_pid="test",
        target="llm",
        command="generate",
        payload={"prompt": "hi"},
    )
    accepted = guard.submit(task)
    assert accepted is False
    assert task.status == TaskStatus.FAILED


def test_dock_processes_queue_on_ready() -> None:
    state = AppState()
    engine = DummyEngine(auto_ready=True)
    guard = MonoGuard(state, {"llm": engine})
    dock = MonoDock(guard)
    bridge = MonoBridge(dock)

    t1 = bridge.wrap("test", "generate", "llm", payload={"prompt": "one"})
    t2 = bridge.wrap("test", "generate", "llm", payload={"prompt": "two"})
    bridge.submit(t1)
    bridge.submit(t2)

    calls = [c for c in engine.calls if c[0] == "generate"]
    assert len(calls) == 2


def test_bridge_wrap_stamps_task_id_into_payload() -> None:
    """turn_trace v1 join key: payload['task_id'] must equal str(task.id) so
    the engine reuses the canonical id instead of minting a fresh hex uuid.
    Without this, /rating outcomes and frame_traces use different turn_ids
    and the monothink hook can never match the rated turn's reasoning_mode."""
    state = AppState()
    engine = DummyEngine(auto_ready=True)
    guard = MonoGuard(state, {"llm": engine})
    dock = MonoDock(guard)
    bridge = MonoBridge(dock)

    task = bridge.wrap("test", "generate", "llm", payload={"prompt": "one"})
    assert task.payload.get("task_id") == str(task.id)
    # Dashed UUID format — not .hex — so str(task.id) round-trips.
    assert "-" in task.payload["task_id"]


def test_bridge_wrap_does_not_override_explicit_task_id() -> None:
    """If a caller passes task_id explicitly (foreign correlation id), the
    bridge stamp must not clobber it. Defensive — the bridge owns the
    DEFAULT, not the field."""
    state = AppState()
    engine = DummyEngine(auto_ready=True)
    guard = MonoGuard(state, {"llm": engine})
    dock = MonoDock(guard)
    bridge = MonoBridge(dock)

    task = bridge.wrap(
        "test", "generate", "llm",
        payload={"prompt": "x", "task_id": "external-correlation-id"},
    )
    assert task.payload["task_id"] == "external-correlation-id"


def test_cancelled_task_is_skipped() -> None:
    state = AppState()
    engine = DummyEngine(auto_ready=False)
    guard = MonoGuard(state, {"llm": engine})
    dock = MonoDock(guard)
    bridge = MonoBridge(dock)

    first = _make_task("generate", {"prompt": "one"})
    second = _make_task("generate", {"prompt": "two"})

    dock.enqueue(first)
    dock.enqueue(second)
    dock.cancel_task(str(second.id))

    engine.finish()

    calls = [c for c in engine.calls if c[0] == "generate"]
    assert len(calls) == 1


def test_dock_drains_immediate_queue_before_next_generate() -> None:
    state = AppState()
    engine = DummyEngine(auto_ready=False)
    guard = MonoGuard(state, {"llm": engine})
    dock = MonoDock(guard)
    bridge = MonoBridge(dock)

    first = bridge.wrap("test", "generate", "llm", payload={"prompt": "one"})
    queued_history = bridge.wrap("test", "set_history", "llm", payload={"history": []})
    second = bridge.wrap("test", "generate", "llm", payload={"prompt": "two"})

    bridge.submit(first)
    bridge.submit(queued_history)
    bridge.submit(second)

    # Finishing the first generation clears the active task. In tests, trigger
    # submit directly (the ready signal is queued via QTimer in guard).
    engine.finish()
    dock._try_submit("llm")

    calls = [c for c in engine.calls if c[0] == "generate"]
    assert len(calls) == 2
    assert any(c[0] == "set_history" for c in engine.calls)


def test_addon_host_routes_engine_events_for_text_backends() -> None:
    state = AppState()
    engine = DummyEngine(auto_ready=False)
    guard = MonoGuard(state, {"llm": engine})
    dock = MonoDock(guard)
    bridge = MonoBridge(dock)
    sink = _EventSink()
    ui = _UIStub([sink])
    ctx = AddonContext(
        state=state,
        guard=guard,
        bridge=bridge,
        ui=ui,
        host=None,
        ui_bridge=UIBridge(),
    )
    AddonHost(AddonRegistry(), ctx)

    engine.sig_token.emit("hello")
    engine.sig_trace.emit("trace")
    engine.sig_status.emit(SystemStatus.RUNNING)
    engine.sig_model_capabilities.emit({"ctx_limit": 8192})

    assert ("llm", "token", "hello") in sink.events
    assert ("llm", "trace", "trace") in sink.events
    assert ("llm", "status", SystemStatus.RUNNING) in sink.events
    assert ("llm", "model_capabilities", {"ctx_limit": 8192}) in sink.events


def test_addon_host_replaces_blank_module_id_on_launch() -> None:
    class _BlankIdModule:
        _mod_id = ""

    class _LaunchStack:
        def __init__(self) -> None:
            self.widgets: list[object] = []

        def addWidget(self, widget: object) -> None:  # noqa: N802
            self.widgets.append(widget)

        def removeWidget(self, widget: object) -> None:  # noqa: N802
            self.widgets.remove(widget)

    class _LaunchUI:
        def __init__(self) -> None:
            self.stack = _LaunchStack()
            self.registered: list[tuple[str, str, str, str]] = []
            self.switched: list[str] = []

        def register_module(self, mod_id: str, addon_id: str, icon: str, title: str) -> None:
            self.registered.append((mod_id, addon_id, icon, title))

        def unregister_module(self, mod_id: str) -> None:
            return

        def switch_to_module(self, mod_id: str) -> None:
            self.switched.append(mod_id)

    state = AppState()
    guard = MonoGuard(state, {})
    dock = MonoDock(guard)
    registry = AddonRegistry()
    registry.register(
        AddonSpec(
            id="blank",
            kind="module",
            title="BLANK",
            icon="B",
            factory=lambda _ctx: _BlankIdModule(),
        )
    )
    ui = _LaunchUI()
    ctx = AddonContext(
        state=state,
        guard=guard,
        bridge=MonoBridge(dock),
        ui=ui,
        host=None,
        ui_bridge=UIBridge(),
    )
    host = AddonHost(registry, ctx)

    mod_id = host.launch_module("blank")

    assert mod_id
    assert ui.registered == [(mod_id, "blank", "B", "BLANK")]
    assert ui.switched == [mod_id]
    assert getattr(ui.stack.widgets[0], "_mod_id") == mod_id
