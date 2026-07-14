"""Canonical per-turn wall-clock instant — one 'now' captured once per outer turn.

Mirrors core/turn_counter.py. The engine captures a single timezone-aware UTC
instant at the SAME outer-turn boundary as the turn counter, freezes it across
tool-followup / continuation generations within the turn, and stamps it on
``config["_now_iso"]``. Downstream consumers derive whatever representation they
need FROM this one instant instead of each calling ``datetime.now()`` — killing
the "now everywhere, since nowhere" divergence (WHEN_PLANE_AUDIT Tier-1 #3).

Canonical representation is **UTC ISO** (unambiguous, matches the counter's clock).
Each consumer converts: the absolute temporal lane renders LOCAL via ``parse_local``
(``temporal_context.format_temporal_value`` strftimes the datetime it receives
directly, so it must be handed a local-aware value); the relative lane subtracts
aware datetimes (zone-agnostic), so the same local-aware value serves both lanes.

Dark sub-flag ``MONOLITH_TURN_CLOCK_V1`` (default off). Flag off → ``resolve_turn_now``
returns ``""`` → ``config["_now_iso"]`` never written → the key is absent →
every consumer falls through to its existing OS-clock read → prompt byte-identical.

Best-effort: every path degrades to a safe value and NEVER raises into generation.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

_FLAG_ENV = "MONOLITH_TURN_CLOCK_V1"
_TRUTHY = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True iff the TurnClock feature flag is set."""
    return str(os.environ.get(_FLAG_ENV, "")).strip().lower() in _TRUTHY


def capture_now_iso() -> str:
    """The canonical per-turn instant: timezone-aware UTC, ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def resolve_turn_now(prev_iso: str, is_outer: bool) -> str:
    """Decide the captured instant for the current dispatch (mirrors resolve_turn_n).

      * feature off            → ""  (so config['_now_iso'] is never written —
                                       flag-off byte-identical, even on a mid-session
                                       ON->OFF toggle)
      * on + outer turn        → a fresh capture (a new instant per user turn)
      * on + inner/followup     → prev_iso reused verbatim (so 'now' is constant
                                       within a multi-step tool turn — coherent with
                                       the counter, KV-cache-safe)

    On an inner turn with no prior capture (e.g. the flag was toggled on mid-turn),
    capture fresh rather than emit "" — emitting "" would mislabel the turn as off.
    """
    if not enabled():
        return ""
    if is_outer or not prev_iso:
        return capture_now_iso()
    return prev_iso


def parse_local(now_iso: str | None) -> datetime | None:
    """Parse a stamped ``_now_iso`` back to a LOCAL-aware datetime for display.

    The absolute temporal lane strftimes the datetime it receives directly (it does
    NOT re-astimezone), so a consumer that renders wall-clock time must be handed a
    local-aware value. Naive input is assumed UTC (conservative). Returns ``None`` on
    missing/corrupt input so the caller falls through to its own OS-clock read.
    """
    if not now_iso:
        return None
    try:
        dt = datetime.fromisoformat(str(now_iso))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone()  # to local zone
