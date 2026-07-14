"""Bearing store — owns CONFIG_DIR/bearing.json.

Persists three things in one file:
  1. The current Bearing snapshot.
  2. Pending rejection (drives next-turn [BEARING_UPDATE_REJECTED] injection).
  3. Rejection streak (consecutive structural failures since last successful
     commit — used by updater.py to escalate via emit_fault at N=3).

Atomic-write pattern mirrors core/continuity.py:104-109 (temp file +
os.replace; Windows-safe).

Cross-session by default. Bearing does NOT clear on session boundary —
this is the key contract difference from WORKING_MEMORY.
"""
from __future__ import annotations

import json
import os
from typing import Any

from core.paths import CONFIG_DIR

from .schema import Bearing

_STORE_PATH = CONFIG_DIR / "bearing.json"

_STORE_VERSION = 1


# ── io ──────────────────────────────────────────────────────────────


def _empty_store() -> dict[str, Any]:
    return {
        "version": _STORE_VERSION,
        "bearing": Bearing().to_dict(),
        "pending_rejection": None,
        "rejection_streak": 0,
        "pending_staleness": None,
    }


def _load() -> dict[str, Any]:
    if not _STORE_PATH.exists():
        return _empty_store()
    try:
        with _STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return _empty_store()
    if not isinstance(data, dict):
        return _empty_store()
    data.setdefault("version", _STORE_VERSION)
    data.setdefault("bearing", Bearing().to_dict())
    data.setdefault("pending_rejection", None)
    data.setdefault("rejection_streak", 0)
    data.setdefault("pending_staleness", None)
    return data


def _save(data: dict[str, Any]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STORE_PATH.with_name(_STORE_PATH.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, _STORE_PATH)


# ── bearing snapshot api ────────────────────────────────────────────


def get_bearing() -> Bearing:
    data = _load()
    return Bearing.from_dict(data.get("bearing") or {})


def set_bearing(bearing: Bearing) -> None:
    data = _load()
    data["bearing"] = bearing.to_dict()
    _save(data)


def clear_bearing() -> None:
    """Reset to empty Bearing. Does NOT touch pending_rejection or streak."""
    data = _load()
    data["bearing"] = Bearing().to_dict()
    _save(data)


# ── pending rejection api ───────────────────────────────────────────


def get_pending_rejection() -> dict[str, Any] | None:
    data = _load()
    raw = data.get("pending_rejection")
    return raw if isinstance(raw, dict) else None


def set_pending_rejection(
    failed_rules: list[str],
    turn_id: str,
    ts: str,
    detail: str = "",
    evidence: str = "",
) -> None:
    """Persist a pending rejection. `detail` is the verifier's failure-mode
    string (e.g. "D1: referents.add[0] missing reason"); `evidence` is a
    truncated snippet of the offending envelope body (parse_error path only).
    Both surface in the compiler's [BEARING_UPDATE_REJECTED] block so the
    model sees diagnostic context for its repair attempt.
    """
    data = _load()
    payload: dict[str, Any] = {
        "failed_rules": list(failed_rules),
        "turn_id": str(turn_id),
        "ts": str(ts),
    }
    if detail:
        payload["detail"] = str(detail)
    if evidence:
        payload["evidence"] = str(evidence)
    data["pending_rejection"] = payload
    _save(data)


def clear_pending_rejection() -> None:
    data = _load()
    data["pending_rejection"] = None
    _save(data)


# ── rejection streak api ────────────────────────────────────────────


def get_rejection_streak() -> int:
    data = _load()
    try:
        return int(data.get("rejection_streak", 0))
    except (TypeError, ValueError):
        return 0


def increment_rejection_streak() -> int:
    data = _load()
    try:
        current = int(data.get("rejection_streak", 0))
    except (TypeError, ValueError):
        current = 0
    current += 1
    data["rejection_streak"] = current
    _save(data)
    return current


def reset_rejection_streak() -> None:
    data = _load()
    data["rejection_streak"] = 0
    _save(data)


# ── pending staleness api ───────────────────────────────────────────
#
# Mirrors pending_rejection, but for the channel-staleness closure loop
# (see staleness.py / compiler.bearing_interceptor). State is the actuator's
# per-signal streak record: {"signal_id": str, "streak": int}. Independent of
# pending_rejection — neither clobbers the other.


def get_pending_staleness() -> dict[str, Any] | None:
    data = _load()
    raw = data.get("pending_staleness")
    return raw if isinstance(raw, dict) else None


def set_pending_staleness(state: dict[str, Any]) -> None:
    """Persist the staleness streak record. `state` is the actuator's
    next_state, e.g. {"signal_id": "channel:user", "streak": 2}."""
    data = _load()
    data["pending_staleness"] = dict(state)
    _save(data)


def clear_pending_staleness() -> None:
    data = _load()
    data["pending_staleness"] = None
    _save(data)
