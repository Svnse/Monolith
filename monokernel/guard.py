from __future__ import annotations

import time
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
    "stop": "stop_generation",
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
    sig_model_capabilities = Signal(str, dict)

    def __init__(self, state: AppState, engines: dict[str, EnginePort]):
        super().__init__()
        self.state = state
        self.world_state = getattr(state, "world_state", None)
        self.engines = engines
        self.active_tasks: dict[str, Optional[Task]] = {
            key: None for key in engines.keys()
        }
        self._stop_requested: dict[str, bool] = {key: False for key in engines.keys()}
        self._viztracer = None

        # Circuit breaker
        self._error_counts: dict[str, int] = {k: 0 for k in engines}
        self._circuit_open_until: dict[str, float] = {}
        self._cb_threshold: int = 3
        self._cb_cooldown_sec: int = 60

        # VRAM quotas: engine_key → min free MB required for GPU commands
        self._vram_min_free_mb: dict[str, int] = {}

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
            image_slot = lambda img, ek=key: self.sig_image.emit(ek, img)
            engine.sig_image.connect(image_slot)

        model_capabilities_slot = None
        if hasattr(engine, "sig_model_capabilities"):
            model_capabilities_slot = lambda payload, ek=key: self.sig_model_capabilities.emit(ek, payload)
            engine.sig_model_capabilities.connect(model_capabilities_slot)

        has_finished = hasattr(engine, "sig_finished")
        if has_finished:
            engine.sig_finished.connect(finished_slot)

        self._engine_connections[key] = {
            "status": status_slot,
            "token": token_slot,
            "trace": trace_slot,
            "usage": usage_slot,
            "image": image_slot,
            "model_capabilities": model_capabilities_slot,
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
        # Drop the world_state tombstone for this engine. Without this, the
        # vitals footer keeps rendering rows for engines that no longer
        # exist (one row per chat tab the user has ever opened).
        if self.world_state is not None and hasattr(self.world_state, "clear_engine"):
            try:
                self.world_state.clear_engine(key)
            except Exception:
                pass

    def configure_circuit_breaker(self, threshold: int, cooldown_sec: int) -> None:
        """Update circuit breaker parameters at runtime."""
        self._cb_threshold = max(1, int(threshold))
        self._cb_cooldown_sec = max(1, int(cooldown_sec))

    def set_vram_quota(self, engine_key: str, min_free_mb: int) -> None:
        """Require at least *min_free_mb* VRAM free before GPU commands on *engine_key*."""
        self._vram_min_free_mb[engine_key] = max(0, int(min_free_mb))

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
            task.status = TaskStatus.FAILED
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

        if task.command == "stop":
            self.sig_trace.emit("system", f"[GUARD] submit: STOP task={task.id} target={task.target}")
            task.status = TaskStatus.DONE
            self.stop(task.target)
            if self.world_state is not None:
                self.world_state.append_action_log({
                    "type": "engine_stop",
                    "engine": task.target,
                    "source": task.addon_pid,
                    "outcome": "requested",
                })
            return True

        if task.command in IMMEDIATE_COMMANDS:
            self.sig_trace.emit("system", f"[GUARD] submit: IMMEDIATE {task.command} task={task.id}")
            self.sig_trace.emit("system", f"GUARD: IMMEDIATE {task.command} task={task.id}")
            task.status = TaskStatus.RUNNING
            handler(task.payload)
            task.status = TaskStatus.DONE
            return True

        # --- Circuit breaker ---
        open_until = self._circuit_open_until.get(task.target, 0.0)
        if open_until > time.monotonic():
            remaining = int(open_until - time.monotonic())
            self.sig_trace.emit(
                "system",
                f"[GUARD] CIRCUIT OPEN — engine={task.target} retry in {remaining}s",
            )
            task.status = TaskStatus.FAILED
            return False

        # --- VRAM quota ---
        _GPU_COMMANDS = {"generate", "load"}
        if task.command in _GPU_COMMANDS and task.target in self._vram_min_free_mb:
            quota = self._vram_min_free_mb[task.target]
            resources = (
                self.world_state.snapshot().get("resources", {})
                if self.world_state else {}
            )
            vram_free = resources.get("vram_free_mb")
            if vram_free is not None and vram_free < quota:
                self.sig_trace.emit(
                    "system",
                    f"[GUARD] VRAM QUOTA engine={task.target} "
                    f"free={vram_free}MB required≥{quota}MB — rejected",
                )
                task.status = TaskStatus.FAILED
                return False

        if self.active_tasks.get(task.target) is not None:
            active = self.active_tasks.get(task.target)
            self.sig_trace.emit("system", f"[GUARD] submit: REJECTED BUSY target={task.target}, active_task={active.id if active else None}, active_cmd={active.command if active else None}")
            self.sig_trace.emit("system", f"GUARD: rejected task={task.id} target={task.target} (busy)")
            return False

        self.sig_trace.emit("system", f"[GUARD] submit: ACCEPTED task={task.id} cmd={task.command} target={task.target}")
        self.sig_trace.emit("system", f"GUARD: accepted task={task.id} target={task.target} command={task.command}")
        self.active_tasks[task.target] = task
        task.status = TaskStatus.RUNNING
        if self.world_state is not None:
            status_val = task.status.value
            if task.command == "unload":
                status_val = SystemStatus.UNLOADING.value
            self.world_state.set_active_task(
                task.target,
                {
                    "id": str(task.id),
                    "addon_pid": task.addon_pid,
                    "command": task.command,
                    "priority": task.priority,
                    "status": status_val,
                },
            )

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
                if self.world_state is not None:
                    self.world_state.set_active_task(
                        key,
                        {
                            "id": str(task.id),
                            "addon_pid": task.addon_pid,
                            "command": task.command,
                            "priority": task.priority,
                            "status": TaskStatus.CANCELLED.value,
                            "stop_requested": True,
                        },
                    )
            engine.stop_generation()

    def _on_engine_finished(self, engine_key: str) -> None:
        task = self.active_tasks.get(engine_key)
        if task:
            self.sig_finished.emit(engine_key, str(task.id))
            self.sig_trace.emit(engine_key, f"GUARD: finished engine={engine_key} task={task.id}")

    def _on_status_changed(self, engine_key: str, new_status: SystemStatus) -> None:
        self.sig_trace.emit("system", f"[GUARD] _on_status_changed: engine={engine_key}, status={new_status}, active_task={self.active_tasks.get(engine_key) is not None}")
        self.sig_status.emit(engine_key, new_status)
        if self.world_state is not None:
            self.world_state.set_engine_status(engine_key, new_status.value)

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
            if self.world_state is not None:
                self.world_state.set_active_task(engine_key, None)

            # Circuit breaker: count error, open circuit if threshold reached
            count = self._error_counts.get(engine_key, 0) + 1
            self._error_counts[engine_key] = count
            if count >= self._cb_threshold:
                open_until = time.monotonic() + self._cb_cooldown_sec
                self._circuit_open_until[engine_key] = open_until
                self.sig_trace.emit(
                    "system",
                    f"[GUARD] CIRCUIT OPEN engine={engine_key} "
                    f"errors={count} cooldown={self._cb_cooldown_sec}s",
                )
            return

        if new_status == SystemStatus.READY:
            task = self.active_tasks.get(engine_key)
            had_task = task is not None
            stop_was_requested = self._stop_requested.get(engine_key, False)
            if task and task.status == TaskStatus.RUNNING:
                if stop_was_requested:
                    task.status = TaskStatus.CANCELLED
                else:
                    task.status = TaskStatus.DONE
            self.active_tasks[engine_key] = None
            self._stop_requested[engine_key] = False
            if had_task:
                QTimer.singleShot(0, lambda: self.sig_engine_ready.emit(engine_key))
            if self.world_state is not None:
                self.world_state.set_active_task(engine_key, None)

            # Circuit breaker: reset on clean task completion (not a user stop)
            if had_task and not stop_was_requested:
                self._error_counts[engine_key] = 0
                self._circuit_open_until.pop(engine_key, None)


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
