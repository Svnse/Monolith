from PySide6.QtCore import QObject, Signal

from core.state import SystemStatus
from engine.base import EnginePort


class EngineBridge(QObject, EnginePort):
    sig_token = Signal(str)
    sig_trace = Signal(str)
    sig_status = Signal(SystemStatus)
    sig_usage = Signal(int)

    def __init__(self, impl: EnginePort):
        super().__init__()
        self.impl = impl

        impl.sig_token.connect(self.sig_token.emit)
        impl.sig_trace.connect(self.sig_trace.emit)
        impl.sig_status.connect(self.sig_status.emit)
        impl.sig_usage.connect(self.sig_usage.emit)

    def load_model(self) -> None:
        self.impl.load_model()

    def unload_model(self) -> None:
        self.impl.unload_model()

    def generate(self, user_input: str) -> None:
        self.impl.generate(user_input)

    def stop_generation(self) -> None:
        self.impl.stop_generation()

    def shutdown(self) -> None:
        self.impl.shutdown()
