from __future__ import annotations

from typing import Protocol, runtime_checkable

from PySide6.QtCore import Signal


@runtime_checkable
class EnginePort(Protocol):
    """
    EnginePort protocol defines the minimal interface all engines must implement.

    Required signals:
        sig_token: Text output stream (for LLM-style engines)
        sig_trace: Debug/status messages
        sig_status: SystemStatus transitions (LOADING/RUNNING/READY/ERROR)

    Optional signals (check with hasattr before use):
        sig_usage: Token/step count tracking (LLM-specific)
        sig_image: Image output (vision engines)
        sig_audio: Audio output (audio engines)
        sig_finished: Optional completion notification

    The protocol intentionally keeps optional signals out to prevent forcing
    LLM-centric requirements onto vision/audio engines.
    """
    sig_status: Signal
    sig_trace: Signal
    sig_token: Signal

    def set_model_path(self, payload: dict) -> None:
        ...

    def load_model(self) -> None:
        ...

    def unload_model(self) -> None:
        ...

    def generate(self, payload: dict) -> None:
        ...

    def runtime_command(self, command: str, payload: dict | None = None) -> dict:
        ...

    def stop_generation(self) -> None:
        ...

    def shutdown(self) -> None:
        ...
