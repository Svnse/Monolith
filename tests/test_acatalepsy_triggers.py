from __future__ import annotations

import threading
import time
from types import SimpleNamespace


def test_stop_preserves_thread_reference_while_run_is_still_unwinding(monkeypatch):
    from core.acatalepsy import triggers

    entered = threading.Event()
    release = threading.Event()

    def slow_run_audit(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=2.0)
        return SimpleNamespace(status="success")

    monkeypatch.setattr(triggers._auditor, "run_audit", slow_run_audit)

    worker = triggers.AuditorWorker(
        llm=lambda **_kwargs: "{}",
        source="auditor_test",
        poll_interval_secs=0.01,
    )
    worker.start()
    worker.queue_handle.enqueue("manual")
    assert entered.wait(timeout=1.0)

    worker.stop(timeout=0.01)

    thread = worker._thread
    try:
        assert thread is not None
        assert thread.is_alive()
        assert worker._stop_event.is_set()
    finally:
        release.set()
        if thread is not None:
            thread.join(timeout=1.0)
        worker.stop(timeout=0.5)


def test_stop_clears_thread_reference_after_clean_shutdown():
    from core.acatalepsy import triggers

    worker = triggers.AuditorWorker(
        llm=lambda **_kwargs: "{}",
        source="auditor_test",
        poll_interval_secs=1.0,
    )
    worker.start()

    worker.stop(timeout=1.0)

    assert worker._thread is None
    assert worker._stop_event.is_set()
