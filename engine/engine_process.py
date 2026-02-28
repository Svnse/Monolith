"""
engine/engine_process.py
Base class for subprocess-isolated ML engines.

Architecture
────────────
Each heavy ML engine (vision, audio) inherits EngineProcess and runs its
model in a separate OS process spawned via multiprocessing (spawn context).

Benefits
  • OOM / crash in the worker → worker dies, main app stays alive.
  • GPU CUDA context is owned by the worker; kill it to fully free VRAM.
  • Main thread never blocks on model inference.

IPC protocol   (both directions: plain dicts via multiprocessing.Queue)
──────────────
Parent → worker ("ops"):
  {"op": "load",     "model_path": str, "config": dict}
  {"op": "generate", "gen_id": int,     "config": dict}
  {"op": "stop"}
  {"op": "unload"}
  {"op": "shutdown"}

Worker → parent ("events"):
  {"event": "status",   "status": "loading"|"ready"|"running"|"error"|"unloaded"}
  {"event": "progress", "gen_id": int, "step": int, "total": int}
  {"event": "result",   "gen_id": int, ...mode-specific fields}
  {"event": "resource", "vram_used_mb": int, "vram_free_mb": int}
  {"event": "trace",    "message": str}
  {"event": "error",    "message": str}
  {"event": "stopped"}
  {"event": "unloaded"}

Addon event bus
───────────────
sig_event(dict) re-emits every raw worker event to the Qt signal layer.
AddonEventBus (ui/addons/bus.py) subscribes here and fans out to addons
by channel name ("vision:image", "audio:transcription", etc.).
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from core.state import SystemStatus

# ── multiprocessing context ────────────────────────────────────────────────
# "spawn" is required on Windows and safe on all platforms; avoids forking
# a CUDA-initialised process which would corrupt the GPU state.
_MP_CTX: mp.context.BaseContext | None = None


def _get_ctx() -> mp.context.BaseContext:
    global _MP_CTX
    if _MP_CTX is None:
        _MP_CTX = mp.get_context("spawn")
    return _MP_CTX


# ── base ───────────────────────────────────────────────────────────────────

class EngineProcess(QObject):
    """
    Manages a subprocess-isolated ML engine worker.

    Subclasses must override ``_worker_fn`` — a *static* (picklable) function
    that is the entry point of the child process.  It receives the two Queue
    objects and must loop until it receives {"op": "shutdown"}.

    The standard EnginePort interface (set_model_path / load_model /
    unload_model / generate / stop_generation / shutdown) is implemented here
    so subclasses only need to override what differs.
    """

    # ── Qt signals ─────────────────────────────────────────────────────────
    sig_status = Signal(SystemStatus)
    sig_trace  = Signal(str)
    sig_token  = Signal(str)       # kept for EnginePort protocol compat
    sig_event  = Signal(dict)      # raw event bus — all worker events bubble here

    # ── tuning ─────────────────────────────────────────────────────────────
    _POLL_MS       = 30    # queue drain interval (ms)
    _DEAD_CHECK_MS = 500   # worker liveness check interval (ms)

    def __init__(self) -> None:
        super().__init__()

        self._proc: mp.Process | None = None
        self._to_worker: mp.Queue | None   = None
        self._from_worker: mp.Queue | None = None

        self._gen_id        = 0
        self._active_gen_id = 0
        self._shutdown_requested = False
        self._pending_model_path: str = ""

        # Drain queue on a timer — never blocks the Qt event loop
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self._POLL_MS)
        self._poll_timer.timeout.connect(self._drain_queue)

        # Detect unexpected worker death
        self._dead_timer = QTimer(self)
        self._dead_timer.setInterval(self._DEAD_CHECK_MS)
        self._dead_timer.timeout.connect(self._check_liveness)

    # ── subclass interface ─────────────────────────────────────────────────

    @staticmethod
    def _worker_fn(to_worker: mp.Queue, from_worker: mp.Queue) -> None:
        """
        Override in subclass.  Runs entirely inside the child process.
        Must be a static/module-level function so it can be pickled by spawn.
        """
        raise NotImplementedError

    # ── EnginePort interface ───────────────────────────────────────────────

    def set_model_path(self, payload: dict) -> None:
        self._pending_model_path = str(payload.get("path") or "")
        QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))

    def load_model(self) -> None:
        path = self._pending_model_path
        if not path:
            self.sig_trace.emit("[ENGINE] load_model: no model path set")
            self.sig_status.emit(SystemStatus.ERROR)
            return
        if not self._ensure_proc():
            return
        self.sig_status.emit(SystemStatus.LOADING)
        self._send("load", model_path=path, config=getattr(self, "_load_config", {}))

    def unload_model(self) -> None:
        if self._proc and self._proc.is_alive():
            self.sig_status.emit(SystemStatus.UNLOADING)
            self._send("unload")
        else:
            QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))

    def generate(self, payload: dict) -> None:
        if not (self._proc and self._proc.is_alive()):
            self.sig_trace.emit("[ENGINE] generate: worker not running")
            self.sig_status.emit(SystemStatus.ERROR)
            return
        self._gen_id += 1
        self._active_gen_id = self._gen_id
        self.sig_status.emit(SystemStatus.RUNNING)
        self._send("generate", gen_id=self._gen_id, config=payload)

    def stop_generation(self) -> None:
        self._active_gen_id = 0
        if self._proc and self._proc.is_alive():
            self._send("stop")

    def runtime_command(self, command: str, payload: dict | None = None) -> dict:
        return {"ok": False, "error": "runtime_command not supported on EngineProcess"}

    def shutdown(self) -> None:
        self._shutdown_requested = True
        self._poll_timer.stop()
        self._dead_timer.stop()
        if self._proc and self._proc.is_alive():
            try:
                self._send("shutdown")
                self._proc.join(timeout=3.0)
            except Exception:
                pass
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
        self._proc        = None
        self._to_worker   = None
        self._from_worker = None

    # ── process lifecycle ──────────────────────────────────────────────────

    def _ensure_proc(self) -> bool:
        """Spawn the worker if it is not already alive.  Returns True on success."""
        if self._proc is not None and self._proc.is_alive():
            return True

        ctx = _get_ctx()
        self._to_worker   = ctx.Queue()
        self._from_worker = ctx.Queue()

        try:
            self._proc = ctx.Process(
                target=self.__class__._worker_fn,
                args=(self._to_worker, self._from_worker),
                daemon=True,
            )
            self._proc.start()
        except Exception as exc:
            self.sig_trace.emit(f"[ENGINE] spawn failed: {exc}")
            self.sig_status.emit(SystemStatus.ERROR)
            self._to_worker   = None
            self._from_worker = None
            return False

        self._poll_timer.start()
        self._dead_timer.start()
        return True

    # ── IPC helpers ────────────────────────────────────────────────────────

    def _send(self, op: str, **kwargs: Any) -> None:
        """Put an op dict into the to-worker queue (non-blocking)."""
        if self._to_worker is None:
            return
        try:
            self._to_worker.put_nowait({"op": op, **kwargs})
        except Exception as exc:
            self.sig_trace.emit(f"[ENGINE] _send failed: {exc}")

    def _drain_queue(self) -> None:
        """Called on the poll timer — pulls events from the worker queue."""
        if self._from_worker is None:
            return
        budget = 24  # max events per tick to avoid blocking Qt event loop
        while budget > 0:
            budget -= 1
            try:
                event = self._from_worker.get_nowait()
            except Exception:
                break
            self._dispatch_event(event)

    # ── event dispatch (override to handle mode-specific events) ──────────

    def _dispatch_event(self, event: dict) -> None:
        """
        Translate a raw worker event dict into Qt signals.
        Subclasses should call super() then handle their own event kinds.
        """
        self.sig_event.emit(event)   # broadcast for addon bus

        kind = str(event.get("event") or "")

        if kind == "trace":
            self.sig_trace.emit(str(event.get("message") or ""))

        elif kind == "status":
            s = str(event.get("status") or "").lower()
            _STATUS_MAP = {
                "loading":  SystemStatus.LOADING,
                "ready":    SystemStatus.READY,
                "running":  SystemStatus.RUNNING,
                "error":    SystemStatus.ERROR,
                "unloading":SystemStatus.UNLOADING,
                "unloaded": SystemStatus.READY,
            }
            if s in _STATUS_MAP:
                self.sig_status.emit(_STATUS_MAP[s])

        elif kind == "error":
            self.sig_trace.emit(f"[ENGINE] ERROR: {event.get('message', '')}")
            self.sig_status.emit(SystemStatus.ERROR)

        elif kind == "stopped":
            self.sig_status.emit(SystemStatus.READY)

    # ── liveness check ─────────────────────────────────────────────────────

    def _check_liveness(self) -> None:
        if self._shutdown_requested:
            return
        if self._proc is not None and not self._proc.is_alive():
            exit_code = self._proc.exitcode
            self.sig_trace.emit(
                f"[ENGINE] worker died unexpectedly (exit={exit_code})"
            )
            self.sig_status.emit(SystemStatus.ERROR)
            self._poll_timer.stop()
            self._dead_timer.stop()
            self._proc = None
