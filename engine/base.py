from __future__ import annotations

from typing import Protocol, runtime_checkable

from PySide6.QtCore import Signal


@runtime_checkable
class EnginePort(Protocol):
    sig_token: Signal
    sig_trace: Signal
    sig_status: Signal
    sig_usage: Signal

    def load_model(self) -> None:
        ...

    def unload_model(self) -> None:
        ...

    def generate(self, user_input: str, config: dict | None = None) -> None:
        ...

    def stop_generation(self) -> None:
        ...

    def shutdown(self) -> None:
        ...


@runtime_checkable
class VisionEnginePort(Protocol):
    sig_image: Signal

    def set_model_path(self, path: str) -> None:
        ...
