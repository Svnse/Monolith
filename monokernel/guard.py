from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QObject, Signal

from core.state import AppState, SystemStatus
from engine.base import EnginePort


class MonoGuard(QObject):
    sig_token = Signal(str)
    sig_trace = Signal(str)
    sig_status = Signal(SystemStatus)
    sig_usage = Signal(int)
    sig_image = Signal(object)

    def __init__(self, state: AppState, engine: EnginePort):
        super().__init__()
        self.state = state
        self.engine = engine
        self._status: SystemStatus = SystemStatus.READY
        self._pending: Optional[Tuple[str, tuple, dict]] = None

        # Pass-through connections
        self.engine.sig_token.connect(self.sig_token)
        self.engine.sig_trace.connect(self.sig_trace)
        self.engine.sig_usage.connect(self.sig_usage)
        if hasattr(self.engine, "sig_image"):
            self.engine.sig_image.connect(self.sig_image)
        self.engine.sig_status.connect(self._on_status_changed)

    # -------------------------------
    # Command Slots
    # -------------------------------
    def slot_set_model_path(self, path: str) -> None:
        if hasattr(self.engine, "set_model_path"):
            self.engine.set_model_path(path)

    def slot_load_model(self):
        if self._status in (SystemStatus.RUNNING, SystemStatus.LOADING):
            self._pending = ("load_model", (), {})
            self._request_stop(clear_pending=False)
            return
        self.engine.load_model()

    def slot_unload_model(self):
        if self._status in (SystemStatus.RUNNING, SystemStatus.LOADING):
            self._pending = ("unload_model", (), {})
            self._request_stop(clear_pending=False)
            return
        self.engine.unload_model()

    def slot_generate(self, user_input: str, config: dict | None = None):
        if self._status in (SystemStatus.RUNNING, SystemStatus.LOADING):
            self._pending = ("generate", (user_input, config), {})
            self._request_stop(clear_pending=False)
            return
        self.engine.generate(user_input, config)

    def slot_stop(self):
        self._request_stop(clear_pending=True)

    def _request_stop(self, clear_pending: bool) -> None:
        """Interrupt current execution and optionally clear pending command."""
        if clear_pending:
            self._pending = None
        self.engine.stop_generation()

    # -------------------------------
    # Internal Callbacks
    # -------------------------------
    def _on_status_changed(self, new_status: SystemStatus):
        self._status = new_status
        self.sig_status.emit(new_status)
        if new_status == SystemStatus.READY and self._pending:
            command_name, args, kwargs = self._pending
            self._pending = None

            if command_name == "load_model":
                self.engine.load_model()
            elif command_name == "unload_model":
                self.engine.unload_model()
            elif command_name == "generate":
                self.engine.generate(*args, **kwargs)
