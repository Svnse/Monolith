"""Acatalepsy runtime singleton — holds the active worker handle.

Bootstrap creates the worker (behind the v1 flag) and registers it here;
the dev panel and other consumers (slash commands, addons) fetch it via
``get_active_worker()``. Decouples worker creation from UI code so the
panel doesn't need to know how the worker was instantiated.

If the worker isn't running (flag off, sidecar misconfigured, etc.),
``get_active_worker()`` returns None and consumers should degrade
gracefully (dev panel disables the "Audit now" button, etc.).
"""
from __future__ import annotations

import threading

from core.acatalepsy.triggers import AuditorWorker


__all__ = (
    "get_active_worker",
    "register_worker",
    "deregister_worker",
)


_lock = threading.Lock()
_active_worker: AuditorWorker | None = None


def register_worker(worker: AuditorWorker) -> None:
    """Set the active worker. Replaces any prior registration."""
    global _active_worker
    with _lock:
        _active_worker = worker


def deregister_worker(worker: AuditorWorker | None = None) -> None:
    """Clear the active worker registration.

    If ``worker`` is provided, only clears if it matches the active one
    (avoids races where a new worker has been registered between stop
    and deregister calls).
    """
    global _active_worker
    with _lock:
        if worker is None or _active_worker is worker:
            _active_worker = None


def get_active_worker() -> AuditorWorker | None:
    """Return the active worker, or None if no worker is running."""
    with _lock:
        return _active_worker
