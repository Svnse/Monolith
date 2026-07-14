"""Turn-scoped handle map for cite-grounded selection.

`render_recall_lane` assigns a stable within-turn handle (R1, R2, ...) to each
recalled belief it shows the model, and registers handle -> ACU row here. The
grounded_verdict resolver turns a cited handle back into an Authority via
``compute_authority(row)`` — the same authority the recall lane already rendered.

A cited handle that was never shown this turn (a stale or fabricated cite)
resolves to None, so it earns no grounding — the resolution-time laundering guard
that pairs with the emission-time guards in the scaffold (cite optional + an
explicit no-ground token).

Per-turn + ephemeral: ``reset()`` runs at the top of every recall render. V1
assumes single-flight turns (the agent server processes one turn at a time);
when a turn-context/TurnClock lands, scope this to it instead of module state.
"""
from __future__ import annotations

_MAP: dict[str, dict] = {}


def reset() -> None:
    """Clear the map — called at the start of each recall render so handles never
    carry across turns."""
    _MAP.clear()


def register(handle: str, acu: dict) -> None:
    _MAP[handle] = acu


def resolve(handle: str) -> int | None:
    """Cited handle -> Authority level, or None if it isn't a ground shown this
    turn (unresolvable / fabricated cite earns no authority)."""
    acu = _MAP.get(handle)
    if acu is None:
        return None
    from core.acatalepsy.authority import compute_authority
    return compute_authority(acu)


def snapshot() -> dict[str, dict]:
    """Read-only copy of the current handle->row map (telemetry/debug)."""
    return dict(_MAP)
