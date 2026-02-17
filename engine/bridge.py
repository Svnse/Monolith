from PySide6.QtCore import QObject, Signal

from core.state import SystemStatus
from engine.base import EnginePort


class EngineBridge(QObject):
    sig_token = Signal(str)
    sig_trace = Signal(str)
    sig_status = Signal(SystemStatus)
    sig_usage = Signal(int)
    sig_image = Signal(object)
    sig_finished = Signal()
    sig_agent_event = Signal(dict)

    def __init__(self, impl: EnginePort):
        super().__init__()
        self.impl = impl
        self._gen_id = 0
        self._active_gid = 0

        impl.sig_status.connect(self.sig_status)
        if hasattr(impl, "sig_finished"):
            impl.sig_finished.connect(self.sig_finished)
        if hasattr(impl, "sig_agent_event"):
            impl.sig_agent_event.connect(self._on_agent_event)

        impl.sig_token.connect(self._on_token)
        impl.sig_trace.connect(self._on_trace)
        if hasattr(impl, "sig_usage"):
            impl.sig_usage.connect(self._on_usage)
        if hasattr(impl, "sig_image"):
            impl.sig_image.connect(self._on_image)

    def _is_current_generation(self) -> bool:
        return self._active_gid == self._gen_id

    def _on_token(self, token: str) -> None:
        if self._is_current_generation():
            self.sig_token.emit(token)

    def _on_trace(self, message: str) -> None:
        if self._is_current_generation():
            self.sig_trace.emit(message)

    def _on_usage(self, usage: int) -> None:
        if self._is_current_generation():
            self.sig_usage.emit(usage)

    def _on_image(self, image: object) -> None:
        if self._is_current_generation():
            self.sig_image.emit(image)

    def _on_agent_event(self, event: dict) -> None:
        if self._is_current_generation():
            self.sig_agent_event.emit(event)

    def set_history(self, payload: dict) -> None:
        if hasattr(self.impl, "set_history"):
            self.impl.set_history(payload)

    def set_model_path(self, payload: dict) -> None:
        if hasattr(self.impl, "set_model_path"):
            self.impl.set_model_path(payload)

    def set_ctx_limit(self, payload: dict) -> None:
        if hasattr(self.impl, "set_ctx_limit"):
            self.impl.set_ctx_limit(payload)

    def load_model(self) -> None:
        self.impl.load_model()

    def unload_model(self) -> None:
        self.impl.unload_model()

    def generate(self, payload: dict) -> None:
        self._gen_id += 1
        self._active_gid = self._gen_id
        self.impl.generate(payload)

    def stop_generation(self) -> None:
        self._gen_id += 1
        self._active_gid = 0
        self.impl.stop_generation()

    def shutdown(self) -> None:
        self.impl.shutdown()
