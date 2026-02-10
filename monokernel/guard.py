from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.paths import LOG_DIR

from PySide6.QtCore import QObject, Signal, QTimer

from core.state import AppState, SystemStatus
from core.task import Task, TaskStatus
from engine.base import EnginePort

ENGINE_DISPATCH = {
    "set_path": "set_model_path",
    "set_history": "set_history",
    "set_ctx_limit": "set_ctx_limit",
    "load": "load_model",
    "unload": "unload_model",
    "generate": "generate",
}

IMMEDIATE_COMMANDS = {"set_history", "set_path", "set_ctx_limit"}
PAYLOAD_COMMANDS = {"generate"}


class MonoGuard(QObject):
    sig_token = Signal(str, str)
    sig_trace = Signal(str, str)
    sig_status = Signal(str, SystemStatus)
    sig_engine_ready = Signal(str)
    sig_usage = Signal(str, int)
    sig_image = Signal(object)
    sig_finished = Signal(str, str)

    def __init__(self, state: AppState, engines: dict[str, EnginePort]):
        super().__init__()
        self.state = state
        self.engines = engines
        self.active_tasks: dict[str, Optional[Task]] = {
            key: None for key in engines.keys()
        }
        self._stop_requested: dict[str, bool] = {key: False for key in engines.keys()}
        self._viztracer = None

        self._engine_connections: dict[str, dict[str, object]] = {}
        for key, engine in engines.items():
            self._connect_engine_signals(key, engine)


    def _connect_engine_signals(self, key: str, engine: EnginePort) -> None:
        status_slot = lambda status, engine_key=key: self._on_status_changed(engine_key, status)
        token_slot = lambda t, ek=key: self.sig_token.emit(ek, t)
        trace_slot = lambda m, ek=key: self.sig_trace.emit(ek, m)
        finished_slot = lambda engine_key=key: self._on_engine_finished(engine_key)

        engine.sig_status.connect(status_slot)
        engine.sig_token.connect(token_slot)
        engine.sig_trace.connect(trace_slot)

        usage_slot = None
        if hasattr(engine, "sig_usage"):
            usage_slot = lambda used, ek=key: self.sig_usage.emit(ek, used)
            engine.sig_usage.connect(usage_slot)

        image_slot = None
        if hasattr(engine, "sig_image"):
            image_slot = self.sig_image.emit
            engine.sig_image.connect(image_slot)

        has_finished = hasattr(engine, "sig_finished")
        if has_finished:
            engine.sig_finished.connect(finished_slot)

        self._engine_connections[key] = {
            "status": status_slot,
            "token": token_slot,
            "trace": trace_slot,
            "usage": usage_slot,
            "image": image_slot,
            "finished": finished_slot if has_finished else None,
        }

    def _disconnect_engine_signals(self, key: str, engine: EnginePort) -> None:
        slots = self._engine_connections.pop(key, None)
        if not slots:
            return
        for signal_name, slot in slots.items():
            if slot is None:
                continue
            signal = getattr(engine, f"sig_{signal_name}", None)
            if signal is None:
                continue
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def register_engine(self, key: str, engine: EnginePort):
        if key in self.engines:
            self.unregister_engine(key)
        self.engines[key] = engine
        self.active_tasks[key] = None
        self._stop_requested[key] = False
        self._connect_engine_signals(key, engine)

    def unregister_engine(self, key: str):
        engine = self.engines.get(key)
        if engine is None:
            return
        self._disconnect_engine_signals(key, engine)
        self.active_tasks.pop(key, None)
        self._stop_requested.pop(key, None)
        del self.engines[key]

    def get_active_task_id(self, engine_key: str) -> str | None:
        task = self.active_tasks.get(engine_key)
        return str(task.id) if task else None

    def get_active_task(self, engine_key: str) -> Task | None:
        return self.active_tasks.get(engine_key)

    def submit(self, task: Task) -> bool:
        engine = self.engines.get(task.target)
        if engine is None:
            self.sig_trace.emit("system", f"[GUARD] submit: REJECTED unknown engine target={task.target}")
            self.sig_trace.emit("system", f"ERROR: Unknown engine target: {task.target}")
            return False

        method_name = ENGINE_DISPATCH.get(task.command)
        if not method_name:
            self.sig_trace.emit("system", f"[GUARD] submit: REJECTED unknown command={task.command}")
            self.sig_trace.emit("system", f"ERROR: Unknown command: {task.command}")
            task.status = TaskStatus.FAILED
            return False

        handler = getattr(engine, method_name, None)
        if not handler:
            self.sig_trace.emit("system", f"[GUARD] submit: REJECTED no handler for {method_name}")
            self.sig_trace.emit("system", f"ERROR: Engine lacks handler: {method_name}")
            task.status = TaskStatus.FAILED
            return False

        if task.command in IMMEDIATE_COMMANDS:
            self.sig_trace.emit("system", f"[GUARD] submit: IMMEDIATE {task.command} task={task.id}")
            self.sig_trace.emit("system", f"GUARD: IMMEDIATE {task.command} task={task.id}")
            task.status = TaskStatus.RUNNING
            handler(task.payload)
            task.status = TaskStatus.DONE
            return True

        if self.active_tasks.get(task.target) is not None:
            active = self.active_tasks.get(task.target)
            self.sig_trace.emit("system", f"[GUARD] submit: REJECTED BUSY target={task.target}, active_task={active.id if active else None}, active_cmd={active.command if active else None}")
            self.sig_trace.emit("system", f"GUARD: rejected task={task.id} target={task.target} (busy)")
            return False

        self.sig_trace.emit("system", f"[GUARD] submit: ACCEPTED task={task.id} cmd={task.command} target={task.target}")
        self.sig_trace.emit("system", f"GUARD: accepted task={task.id} target={task.target} command={task.command}")
        self.active_tasks[task.target] = task
        task.status = TaskStatus.RUNNING

        if task.command in PAYLOAD_COMMANDS:
            handler(task.payload)
        else:
            handler()
        return True

    def stop(self, target: str = "all") -> None:
        self.sig_trace.emit("system", f"GUARD: STOP target={target}")
        if target == "all":
            keys = list(self.engines.keys())
        else:
            keys = [target]

        for key in keys:
            engine = self.engines.get(key)
            if not engine:
                continue
            task = self.active_tasks.get(key)
            if task is not None:
                self._stop_requested[key] = True
            engine.stop_generation()

    def _on_engine_finished(self, engine_key: str) -> None:
        task = self.active_tasks.get(engine_key)
        if task:
            self.sig_finished.emit(engine_key, str(task.id))
            self.sig_trace.emit(engine_key, f"GUARD: finished engine={engine_key} task={task.id}")

    def _on_status_changed(self, engine_key: str, new_status: SystemStatus) -> None:
        self.sig_trace.emit("system", f"[GUARD] _on_status_changed: engine={engine_key}, status={new_status}, active_task={self.active_tasks.get(engine_key) is not None}")
        self.sig_status.emit(engine_key, new_status)

        if new_status == SystemStatus.ERROR:
            task = self.active_tasks.get(engine_key)
            had_task = task is not None
            if task:
                task.status = TaskStatus.FAILED
            self.active_tasks[engine_key] = None
            self._stop_requested[engine_key] = False
            self.sig_status.emit(engine_key, SystemStatus.READY)
            if had_task:
                QTimer.singleShot(0, lambda: self.sig_engine_ready.emit(engine_key))
            return

        if new_status == SystemStatus.READY:
            task = self.active_tasks.get(engine_key)
            had_task = task is not None
            if task and task.status == TaskStatus.RUNNING:
                if self._stop_requested.get(engine_key, False):
                    task.status = TaskStatus.CANCELLED
                else:
                    task.status = TaskStatus.DONE
            self.active_tasks[engine_key] = None
            self._stop_requested[engine_key] = False
            if had_task:
                QTimer.singleShot(0, lambda: self.sig_engine_ready.emit(engine_key))


    def enable_viztracer(self, enabled: bool) -> None:
        if enabled:
            if self._viztracer is not None:
                return
            try:
                from viztracer import VizTracer
            except Exception as exc:
                self.sig_trace.emit("system", f"OVERSEER: viztracer unavailable: {exc}")
                return
            try:
                self._viztracer = VizTracer(
                    min_duration=5000,
                    ignore_frozen=True,
                    exclude_files=["*/site-packages/*"],
                )
            except TypeError:
                self._viztracer = VizTracer(
                    min_duration=5000,
                    ignore_frozen=True,
                )
            except Exception:
                self._viztracer = VizTracer()
            self._viztracer.start()
            self.sig_trace.emit("system", "OVERSEER: viztracer started")
            return

        tracer = self._viztracer
        if tracer is None:
            return
        tracer.stop()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = LOG_DIR / f"viztrace_{ts}.json"
        tracer.save(str(out_path))
        self.sig_trace.emit("system", f"OVERSEER: viztracer saved {out_path}")
        self._viztracer = None
