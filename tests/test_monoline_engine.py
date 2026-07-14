"""MonolineEngine = a process-isolated kernel engine (subclass of EngineProcess) per Kernel
Contract v2 §9 / D10: Monoline runs in its own spawned worker, only dicts cross IPC, the kernel
routes it like vision/audio with no MonoGuard changes. Thin class; the worker holds the logic.
See docs/reports/GENESIS_CARD_BUILD_LOG.md."""
from __future__ import annotations

from engine.engine_process import EngineProcess
from engine.monoline_engine import MonolineEngine


def test_is_engineprocess_subclass():
    assert issubclass(MonolineEngine, EngineProcess)


def test_worker_fn_delegates_to_worker_main(monkeypatch):
    import engine._workers.monoline_worker as mw
    seen: dict = {}
    monkeypatch.setattr(mw, "main", lambda tw, fw: seen.update(tw=tw, fw=fw))
    MonolineEngine._worker_fn("TO_Q", "FROM_Q")
    assert seen == {"tw": "TO_Q", "fw": "FROM_Q"}
