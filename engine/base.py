from __future__ import annotations

from typing import Protocol, runtime_checkable

from PySide6.QtCore import Signal


@runtime_checkable
class EnginePort(Protocol):
    sig_status: Signal
    sig_trace: Signal
    sig_token: Signal
    sig_image: Signal
    sig_finished: Signal

    def set_model_path(self, path: str) -> None:
        ...

    def load_model(self) -> None:
        ...

    def unload_model(self) -> None:
        ...

    def generate(self, payload: dict) -> None:
        ...

    def stop_generation(self) -> None:
        ...

    def shutdown(self) -> None:
        ...
