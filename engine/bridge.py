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

    def __init__(self, impl: EnginePort):
        super().__init__()
        self.impl = impl
        self._gen_id = 0
        self._active_gid = 0
        self._token_conn = None
        self._trace_conn = None
        self._usage_conn = None
        self._image_conn = None

        impl.sig_status.connect(self.sig_status)
        if hasattr(impl, "sig_finished"):
            impl.sig_finished.connect(self.sig_finished)

        self._connect_gated_handlers(self._active_gid)

    def _disconnect_gated_handlers(self) -> None:
        connections = [
            (self.impl.sig_token, self._token_conn),
            (self.impl.sig_trace, self._trace_conn),
            (self.impl.sig_usage, self._usage_conn),
        ]
        if hasattr(self.impl, "sig_image"):
            connections.append((self.impl.sig_image, self._image_conn))
        for signal, conn in connections:
            if conn is not None:
                try:
                    signal.disconnect(conn)
                except (RuntimeError, TypeError):
                    pass
        self._token_conn = None
        self._trace_conn = None
        self._usage_conn = None
        self._image_conn = None

    def _connect_gated_handlers(self, gid: int) -> None:
        self._disconnect_gated_handlers()
        self._token_conn = self.impl.sig_token.connect(
            lambda t, gid=gid: self.sig_token.emit(t)
            if self._active_gid == gid
            else None
        )
        self._trace_conn = self.impl.sig_trace.connect(
            lambda t, gid=gid: self.sig_trace.emit(t)
            if self._active_gid == gid
            else None
        )
        self._usage_conn = self.impl.sig_usage.connect(
            lambda u, gid=gid: self.sig_usage.emit(u)
            if self._active_gid == gid
            else None
        )
        if hasattr(self.impl, "sig_image"):
            self._image_conn = self.impl.sig_image.connect(
                lambda image, gid=gid: self.sig_image.emit(image)
                if self._active_gid == gid
                else None
            )

    def set_model_path(self, path: str) -> None:
        if hasattr(self.impl, "set_model_path"):
            self.impl.set_model_path(path)

    def load_model(self) -> None:
        self.impl.load_model()

    def unload_model(self) -> None:
        self.impl.unload_model()

    def generate(self, user_input: str, config: dict | None = None) -> None:
        self._gen_id += 1
        gid = self._gen_id
        self._active_gid = gid
        self._connect_gated_handlers(gid)
        self.impl.generate(user_input, config)

    def stop_generation(self) -> None:
        self._gen_id += 1
        self._active_gid = self._gen_id
        self._connect_gated_handlers(self._active_gid)
        self.impl.stop_generation()

    def shutdown(self) -> None:
        self.impl.shutdown()
