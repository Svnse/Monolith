"""Monotonic outer-turn counter — a persisted integer incremented once per
outer conversation turn.

Purpose: give the `[BEARING]` block a *readable* age ("42 turns ago") in place
of the opaque UUID `updated_at_turn`, so the model can judge for itself whether
its frame is stale. This is **metadata only** — the runtime never authors
posture; staleness is the model's call.

Persisted to ``CONFIG_DIR/turn_counter.json`` so the count is monotonic across
restarts. A session-scoped counter would compute a negative age for a bearing
carried over from a prior session (bearing.json outlives the process).

Best-effort: every IO path is wrapped so corruption or an unwritable location
degrades to a safe value and NEVER raises into the generation path.

Dark sub-flag ``MONOLITH_TURN_COUNTER_V1`` (default off). The primitive itself
is harmless; the flag gates whether the engine increments and stamps it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from core.paths import CONFIG_DIR

_FLAG_ENV = "MONOLITH_TURN_COUNTER_V1"
_FILENAME = "turn_counter.json"

_TRUTHY = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True iff the turn-counter feature flag is set."""
    return str(os.environ.get(_FLAG_ENV, "")).strip().lower() in _TRUTHY


def _counter_path() -> Path:
    return CONFIG_DIR / _FILENAME


def current_turn(path: Path | None = None) -> int:
    """Return the current counter value without incrementing.

    Returns 0 when the file is missing, malformed, or holds a negative value.
    """
    p = path or _counter_path()
    try:
        with Path(p).open("r", encoding="utf-8") as f:
            data = json.load(f)
        n = int(data.get("n", 0))
        return n if n >= 0 else 0
    except Exception:
        return 0


def next_turn(path: Path | None = None) -> int:
    """Increment, persist, and return the new counter value.

    Atomic write (temp file + os.replace, matching store.py / continuity.py) so
    an interrupted write can't corrupt the counter and silently reset it to 0.
    Best-effort: if persistence fails (e.g. unwritable path) the returned value
    is still ``current + 1`` so the caller gets a monotonic value within the
    process; the on-disk value simply doesn't advance.
    """
    p = Path(path or _counter_path())
    n = current_turn(p) + 1
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"n": n}, f)
        os.replace(tmp, p)
    except Exception:
        pass
    return n


def resolve_turn_n(prev_n: int, is_outer: bool) -> int:
    """Decide the readable turn-count for the current dispatch.

    The single source of truth for the engine glue (kept here so it is
    unit-testable rather than buried in the Qt-bound generation path):

      * feature off            → 0  (so config['_turn_n'] is never written —
                                      flag-off byte-identical, even if the flag
                                      was toggled ON->OFF mid-session)
      * on + outer turn        → next_turn()  (increment + persist)
      * on + inner/followup    → prev_n        (reuse, so the bearing age render
                                      stays constant within a turn / KV-safe)
    """
    if not enabled():
        return 0
    if is_outer:
        return next_turn()
    return prev_n
