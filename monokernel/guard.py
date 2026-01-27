from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Signal, QTimer

from core.state import AppState, SystemStatus
from core.task import Task, TaskStatus
from engine.base import EnginePort

ENGINE_DISPATCH = {
    "set_path": "set_model_path",
    "load": "load_model",
    "unload": "unload_model",
    "generate": "generate",
}


class MonoGuard(QObject):
    sig_token = Signal(str)
    sig_trace = Signal(str)
    sig_status = Signal(str, SystemStatus)
    sig_engine_ready = Signal(str)
    sig_usage = Signal(int)
    sig_image = Signal(object)
    sig_finished = Signal(str, str)

    def __init__(self, state: AppState, engines: dict[str, EnginePort]):
        super().__init__()
        self.state = state
        self.engines = engines
        self.active_tasks: dict[str, Optional[Task]] = {
            key: None for key in engines.keys()
        }

        for key, engine in engines.items():
            engine.sig_status.connect(
                lambda status, engine_key=key: self._on_status_changed(
                    engine_key, status
                )
            )
            engine.sig_token.connect(self.sig_token)
            engine.sig_trace.connect(self.sig_trace)
            if hasattr(engine, "sig_usage"):
                engine.sig_usage.connect(self.sig_usage)
            if hasattr(engine, "sig_image"):
                engine.sig_image.connect(self.sig_image)
            if hasattr(engine, "sig_finished"):
                engine.sig_finished.connect(
                    lambda engine_key=key: self._on_engine_finished(engine_key)
                )

    def get_active_task_id(self, engine_key: str) -> str | None:
        task = self.active_tasks.get(engine_key)
        return str(task.id) if task else None

    def get_active_task(self, engine_key: str) -> Task | None:
        return self.active_tasks.get(engine_key)

    def submit(self, task: Task) -> bool:
        engine = self.engines.get(task.target)
        if engine is None:
            self.sig_trace.emit(f"ERROR: Unknown engine target: {task.target}")
            return False

        if self.active_tasks.get(task.target) is not None:
            return False

        self.active_tasks[task.target] = task
        task.status = TaskStatus.RUNNING
        method_name = ENGINE_DISPATCH.get(task.command)
        if not method_name:
            self.sig_trace.emit(f"ERROR: Unknown command: {task.command}")
            self.active_tasks[task.target] = None
            task.status = TaskStatus.FAILED
            return False

        handler = getattr(engine, method_name, None)
        if not handler:
            self.sig_trace.emit(f"ERROR: Engine lacks handler: {method_name}")
            self.active_tasks[task.target] = None
            task.status = TaskStatus.FAILED
            return False

        if task.command == "set_path":
            handler(task.payload.get("path"))
        elif task.command == "generate":
            handler(task.payload)
        else:
            handler()
        return True

    def stop(self, target: str = "all") -> None:
        if target == "all":
            keys = list(self.engines.keys())
        else:
            keys = [target]

        for key in keys:
            engine = self.engines.get(key)
            if not engine:
                continue
            engine.stop_generation()
            task = self.active_tasks.get(key)
            if task:
                task.status = TaskStatus.CANCELLED
            self.active_tasks[key] = None

    def _on_engine_finished(self, engine_key: str) -> None:
        task = self.active_tasks.get(engine_key)
        if task:
            self.sig_finished.emit(engine_key, str(task.id))

    def _on_status_changed(self, engine_key: str, new_status: SystemStatus) -> None:
        self.sig_status.emit(engine_key, new_status)

        if new_status == SystemStatus.ERROR:
            task = self.active_tasks.get(engine_key)
            had_task = task is not None
            if task:
                task.status = TaskStatus.FAILED
            self.active_tasks[engine_key] = None
            self.sig_status.emit(engine_key, SystemStatus.READY)
            if had_task:
                QTimer.singleShot(0, lambda: self.sig_engine_ready.emit(engine_key))
            return

        if new_status == SystemStatus.READY:
            task = self.active_tasks.get(engine_key)
            had_task = task is not None
            if task and task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.DONE
            self.active_tasks[engine_key] = None
            if had_task:
                QTimer.singleShot(0, lambda: self.sig_engine_ready.emit(engine_key))
