"""MonolineEngine — Monoline as a process-isolated kernel engine (Kernel Contract v2 §9 / D10).

Mirrors VisionProcess/audio: a thin EngineProcess subclass whose static _worker_fn runs the
Monoline flow worker in a spawned child process. The kernel routes it like any engine — register
under key "monoline" in guard.engines via EngineBridge, "no MonoGuard logic changes required"
(Contract v2 §9). Only dicts cross the IPC boundary, so Monoline's colliding core/ui stay
isolated in the worker. See docs/reports/GENESIS_CARD_BUILD_LOG.md.
"""
from __future__ import annotations

from engine.engine_process import EngineProcess


class MonolineEngine(EngineProcess):
    """Subprocess-isolated Monoline flow engine. EnginePort surface comes from EngineProcess;
    the flow-running logic lives in engine/_workers/monoline_worker.py (the child process)."""

    @staticmethod
    def _worker_fn(to_worker, from_worker) -> None:
        from engine._workers import monoline_worker
        monoline_worker.main(to_worker, from_worker)
