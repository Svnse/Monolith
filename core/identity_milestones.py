"""Identity milestone ledger (M2 V0).

Small JSON-backed state for the identity-evolution loop, mirroring the
atomic-write pattern of ``core/proposals.py`` (next to it in CONFIG_DIR):

  * ``acu_watermark`` — ACU count at the last emergence check (throttles the
    detector so it only fires after enough new accrual).
  * ``milestone`` — current milestone N (origin-N→N+1 versioning is V1; the
    counter exists now so the ledger shape is stable).
  * ``latest_emergence_signal`` — the detector's last read-only advisory, which
    the Observer surfaces as an ``[OBSERVER]`` line (Observer never writes here).
  * ``origin0_hash`` — sha256 of the frozen Origin-0 region, for diffability.

This ledger is identity-owned storage. The deterministic detector writes it;
the Observer only reads it. Keeps the Observer's no-mutation contract intact.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from core.paths import CONFIG_DIR


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

STORE_PATH = CONFIG_DIR / "identity_milestones.json"

_SCHEMA_VERSION = 1


def _empty() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "milestone": 0,
        "acu_watermark": 0,
        "latest_emergence_signal": None,
        "origin0_hash": "",
        # M3 curiosity: latest surfaced pulls + a seen-set so the heartbeat
        # retires a pull after it has resurfaced enough times.
        "latest_curiosity_signal": None,
        "curiosity_surfaced": {},
        # M3.1 kill-actuator: pulls Monolith retired as noise (key -> {reason, at}).
        # Reversible; excluded from future surfacing while present.
        "curiosity_killed": {},
    }


def load_ledger() -> dict[str, Any]:
    if not STORE_PATH.exists():
        return _empty()
    try:
        with STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    base = _empty()
    base.update({k: data[k] for k in base if k in data})
    return base


def save_ledger(data: dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_name(STORE_PATH.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, STORE_PATH)


# ── watermark ─────────────────────────────────────────────────────────

def get_watermark() -> int:
    return int(load_ledger().get("acu_watermark", 0) or 0)


def set_watermark(count: int) -> None:
    data = load_ledger()
    data["acu_watermark"] = int(count)
    save_ledger(data)


# ── milestone ─────────────────────────────────────────────────────────

def get_milestone() -> int:
    return int(load_ledger().get("milestone", 0) or 0)


def bump_milestone() -> int:
    data = load_ledger()
    data["milestone"] = int(data.get("milestone", 0) or 0) + 1
    save_ledger(data)
    return data["milestone"]


# ── emergence signal (detector writes, Observer reads) ─────────────────

def get_latest_emergence_signal() -> dict[str, Any] | None:
    sig = load_ledger().get("latest_emergence_signal")
    return sig if isinstance(sig, dict) else None


def set_latest_emergence_signal(signal: dict[str, Any] | None) -> None:
    data = load_ledger()
    data["latest_emergence_signal"] = signal
    save_ledger(data)


# ── origin-0 hash (diffability) ───────────────────────────────────────

def get_origin0_hash() -> str:
    return str(load_ledger().get("origin0_hash", "") or "")


def set_origin0_hash(value: str) -> None:
    data = load_ledger()
    data["origin0_hash"] = str(value or "")
    save_ledger(data)


# ── curiosity (M3) — latest pulls + retirement seen-set ───────────────

def get_latest_curiosity_signal() -> dict[str, Any] | None:
    sig = load_ledger().get("latest_curiosity_signal")
    return sig if isinstance(sig, dict) else None


def set_latest_curiosity_signal(signal: dict[str, Any] | None) -> None:
    data = load_ledger()
    data["latest_curiosity_signal"] = signal
    save_ledger(data)


def get_curiosity_surfaced() -> dict[str, int]:
    raw = load_ledger().get("curiosity_surfaced")
    return {str(k): int(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def get_curiosity_killed() -> dict[str, dict]:
    raw = load_ledger().get("curiosity_killed")
    return {str(k): dict(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def kill_curiosity_pull(key: str, reason: str) -> None:
    """Retire a pull as noise (reversible). Excluded from surfacing while killed."""
    data = load_ledger()
    killed = data.get("curiosity_killed")
    if not isinstance(killed, dict):
        killed = {}
    killed[str(key)] = {"reason": str(reason or ""), "killed_at": _now_iso()}
    data["curiosity_killed"] = killed
    save_ledger(data)


def unkill_curiosity_pull(key: str) -> None:
    """Reverse a kill — the pull becomes eligible to surface again."""
    data = load_ledger()
    killed = data.get("curiosity_killed")
    if isinstance(killed, dict) and str(key) in killed:
        del killed[str(key)]
        data["curiosity_killed"] = killed
        save_ledger(data)


def bump_curiosity_surfaced(keys: list[str]) -> None:
    """Increment the surface-count for each pull key (retirement bookkeeping).

    Retired/graduated keys remain in the seen-set indefinitely — safe because
    the stability gate (not this set) is the true curiosity-eligibility check; a
    graduated claim won't re-enter curiosity. Tombstoning old keys is a V1 nicety.
    Single-writer today (the turn_trace/bootstrap heartbeat); if that is ever
    parallelized, this load-mutate-save needs file locking (save is atomic, the
    read-modify-write is not).
    """
    data = load_ledger()
    seen = data.get("curiosity_surfaced")
    if not isinstance(seen, dict):
        seen = {}
    for k in keys:
        seen[str(k)] = int(seen.get(str(k), 0)) + 1
    data["curiosity_surfaced"] = seen
    save_ledger(data)
