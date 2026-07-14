"""MonoSearch consumer-facing service — the layer that ties the registry, the
salience ledger, and the router together, and (critically) POPULATES the ledger
before serving the self-directed modes.

THE POPULATION TRIGGER (1b #1): the salience ledger is derived/regenerable and is
NOT written on the producer side. `failing`/`recurring` therefore `refresh()`
(rebuild from the registered adapters) before reading — otherwise the ledger is
empty and the selector is a dark store (the exact open-loop MonoSearch exists to
kill). Both the model skill and a future autonomous reader go through here.

`search`/`get` go straight to the router (they read adapters live; they do not
depend on the ledger), so they do not pay the rebuild cost.
"""
from __future__ import annotations

from datetime import datetime

from core.monosearch import registry, router, salience
from core.monosearch.record import Record


def _now() -> float:
    return datetime.now().timestamp()


def refresh(now: float | None = None, adapters: list | None = None) -> int:
    """Rebuild the salience ledger from the given adapters (default: all
    registered). Per-source clear means a scoped refresh only touches its own
    sources. Returns the number of observations recorded."""
    adapters = adapters if adapters is not None else registry.all_adapters()
    return salience.rebuild(adapters, now if now is not None else _now())


def failing(limit: int = 10, now: float | None = None) -> list[dict]:
    """'What I keep failing' — refresh ONLY fault_traces (the only source `failing`
    reads), then the top self-sourced fault keys. Scoped so this headline query
    does not rescan — or couple its latency to — the seven unrelated stores."""
    now = now if now is not None else _now()
    fault = registry.get_adapter("fault_traces")
    refresh(now, [fault] if fault is not None else [])
    return salience.failing(now, limit)


def recurring(limit: int = 10, now: float | None = None) -> list[dict]:
    """'What keeps recurring' — refresh ALL sources, then top keys across them."""
    now = now if now is not None else _now()
    refresh(now)
    return salience.recurring(now, limit)


def search(query: str, filters: dict | None = None, limit: int = 20) -> list[Record]:
    return router.search(query, filters, limit)


def get(namespaced_id: str) -> Record | None:
    return router.get(namespaced_id)


def _signal_records(prefix: str, limit: int) -> list[Record]:
    """Records from the identity_signals adapter (the current curiosity/emergence
    signals), filtered by id prefix. These carry recurrence_key=None (current-state
    signals, not ledger-ranked), so they are read DIRECTLY from the adapter — not
    via salience. Returns [] gracefully if the adapter isn't registered."""
    adapter = registry.get_adapter("identity_signals")
    if adapter is None:
        return []
    try:
        recs = adapter.list({}, max(int(limit), 50))
    except Exception:
        return []
    return [r for r in recs if r.namespaced_id.startswith(prefix)][: max(1, int(limit))]


def pulling(limit: int = 10) -> list[Record]:
    """What's drawing my curiosity — the current curiosity pulls (real claims,
    not a 'run the skill' nudge), in the signal's own pull-strength order."""
    return _signal_records("curiosity:", limit)


def unresolved(limit: int = 10) -> list[Record]:
    """Self-derived claims that have emerged and await review/integration —
    the open identity question, not yet resolved into Origin-0/Emergent."""
    return _signal_records("emergence:", limit)
