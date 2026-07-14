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
        sig_model_capabilities: LLM metadata (ctx length, limits, etc.)

    The protocol intentionally keeps optional signals out to prevent forcing
    LLM-centric requirements onto vision/audio engines.

    Text-producing backends must emit assistant output through sig_token and
    lifecycle through sig_status/sig_trace. They must never push tokens
    directly into PageChat or any other UI object. The kernel/host path is:

        Engine -> EngineBridge -> MonoGuard -> AddonHost.on_engine_event -> UI

    If you add a new model loader or backend, wire it into this contract so
    the terminal module receives output through the same path as every other
    text backend.
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

    def stop_generation(self) -> None:
        ...

    def shutdown(self) -> None:
        ...
