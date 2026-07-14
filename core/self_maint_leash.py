"""Self-maintenance wake leash — the persisted, restart-safe daily wake budget.

Gap 5 of the 2026-06-19 trigger-plan audit: the autonomous daemon's per-UTC-day
wake cap must survive crash-restarts (the supervisor restarts the process, which
would reset an in-memory counter) and must not be defeatable by moving the system
clock backward. So the count is persisted to disk, keyed to the UTC date, read
before each wake, advanced only forward, and never reset by a backward clock.

Hardened after a fresh-eyes audit (2026-06-19) that found two robustness holes in
the first cut — neither model-reachable, both runtime/supervisor-exploitable:
  - Persistence was non-atomic and fail-OPEN: a kill mid-write or any corruption
    made `_load` return {} → a fresh full budget. Now: atomic temp+os.replace, and
    a present-but-unparseable state file FAILS CLOSED (a wake is refused), while a
    genuinely-absent file is still treated as a legit first run.
  - The read-modify-write had no lock: concurrent processes (a supervisor double-
    start) each read the same stale count and all passed the cap. Now an inter-
    process file lock serializes the RMW.
A cap typo in the env now fails closed (zero wakes), not open to the default.

Pure-ish: `now` is injectable; the only side effects are the JSON state file and
its sibling `.lock`.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from datetime import datetime, timezone

from core.paths import CONFIG_DIR

_STATE_PATH = CONFIG_DIR / "self_maint_wakes.json"
_CAP_ENV = "MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"
_DEFAULT_CAP = 24  # default wakes/day; E recalibrates
# Absolute ceiling on the daily wake cap — the COST leash. A level-1 model turn can set
# the cap via the self_maint skill; without this ceiling it could self-arm an unbounded
# paid-generation loop (audit 2026-06-22). 2880 = every 30s all day — generous headroom
# over "every minute" (1440), but bounds runaway. E-tunable.
_MAX_CAP = 2880
_CAP_OVERRIDE: int | None = None  # runtime override set by the control skill; None => env/default


def set_cap_override(n) -> None:
    """Runtime daily-cap override (the control skill). None clears it back to env/default.
    Clamped to [0, _MAX_CAP] — the model cannot set an unbounded cost ceiling."""
    global _CAP_OVERRIDE
    if n is None:
        _CAP_OVERRIDE = None
        return
    try:
        _CAP_OVERRIDE = max(0, min(_MAX_CAP, int(n)))
    except (TypeError, ValueError):
        _CAP_OVERRIDE = None

_LOCK_STALE_S = 30.0   # steal a lock-file held longer than this (a crashed holder)
_LOCK_TIMEOUT_S = 10.0  # give up acquiring after this (then fail closed, never deadlock)
_LOCK_POLL_S = 0.02

# In-process serialization. The daemon is single-threaded in production, but this also
# makes the lock-FILE section single-threaded within a process — avoiding the Windows
# create/O_EXCL/unlink race (WinError 32: deleting a file another thread holds open). The
# lock FILE below is then only doing its real job: guarding against a cross-process double
# start. Module-level so all callers in a process share it.
_INPROC_LOCK = threading.Lock()


class _CorruptState(Exception):
    """The state file exists but is unparseable — fail closed, never fresh-budget."""


def max_wakes_per_day() -> int:
    if _CAP_OVERRIDE is not None:
        return _CAP_OVERRIDE  # runtime override (control skill) wins; already clamped to _MAX_CAP
    raw = os.environ.get(_CAP_ENV)
    if raw is None:
        return _DEFAULT_CAP
    try:
        return max(0, min(_MAX_CAP, int(raw)))  # even a launcher typo is bounded by the ceiling
    except (TypeError, ValueError):
        return 0  # a cap typo => zero autonomous wakes (fail closed), not the default


def _today(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")


def _load() -> dict:
    """{} if the file is absent (legit first run); raises _CorruptState if present
    but unparseable (fail closed — must NOT be read as a fresh budget)."""
    if not _STATE_PATH.exists():
        return {}
    try:
        raw = _STATE_PATH.read_text(encoding="utf-8")
    except OSError as e:
        # A present-but-unreadable file (Windows sharing-violation from an AV scan /
        # backup / supervisor double-start) must NOT be read as a fresh budget — fail
        # closed, same as a corrupt parse. (audit #3)
        raise _CorruptState(f"unreadable state file: {e}")
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as e:
        raise _CorruptState(str(e))
    if not isinstance(data, dict):
        raise _CorruptState("state is not an object")
    return data


def _save(state: dict) -> None:
    """Atomic write: a kill mid-write leaves the OLD valid file intact (os.replace)."""
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{_STATE_PATH}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(state))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(_STATE_PATH))
    except Exception:
        pass


@contextlib.contextmanager
def _interprocess_lock():
    """Serialize the read-modify-write. An in-process threading.Lock serializes threads
    within this process (which also makes the lock-FILE section single-threaded, avoiding
    the Windows create/unlink race); a best-effort O_EXCL lock FILE additionally guards a
    cross-process double-start (a crashed holder's file is stolen after _LOCK_STALE_S).
    Yields True if serialization was acquired, False otherwise — the caller MUST fail
    closed on False (proceeding unlocked would permit a cross-process double-spend of the
    daily cap, audit #7). Never deadlocks: both acquisitions are bounded by _LOCK_TIMEOUT_S
    and the daemon retries next interval."""
    if not _INPROC_LOCK.acquire(timeout=_LOCK_TIMEOUT_S):
        yield False
        return
    lock_path = f"{_STATE_PATH}.lock"
    fd = None
    try:
        try:
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        except Exception:
            pass
        waited = 0.0
        while fd is None:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(lock_path) > _LOCK_STALE_S:
                        os.unlink(lock_path)
                        continue
                except OSError:
                    pass
                time.sleep(_LOCK_POLL_S)
                waited += _LOCK_POLL_S
                if waited >= _LOCK_TIMEOUT_S:
                    break  # cross-process holder — caller fails CLOSED (no double-spend)
            except OSError:
                break  # lock-dir unwritable — caller fails CLOSED
        yield fd is not None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                os.unlink(lock_path)
            except Exception:
                pass
        _INPROC_LOCK.release()


def wakes_today(now: datetime | None = None) -> int:
    try:
        st = _load()
    except _CorruptState:
        return -1  # unknown / corrupt
    return int(st.get("count", 0) or 0) if str(st.get("date") or "") == _today(now) else 0


def try_consume_wake(now: datetime | None = None) -> dict:
    """Reserve one wake for today. Returns {ok, count, cap[, reason]}.

    ok=False with reason 'daily_cap' (budget spent) or 'corrupt_state' (fail closed).
    The accounting date only moves FORWARD (effective = max(today, stored_date)); a
    backward clock keeps the stored count, so it cannot grant extra wakes. The whole
    read-modify-write runs under an interprocess lock so concurrent processes cannot
    both pass the cap. Restart-safe (atomically persisted).
    """
    today = _today(now)
    cap = max_wakes_per_day()
    with _interprocess_lock() as locked:
        if not locked:
            # Could not serialize the RMW — fail closed rather than risk a cross-process
            # double-spend. The daemon logs the skip reason and retries next interval.
            return {"ok": False, "reason": "lock_unavailable", "count": -1, "cap": cap}
        try:
            st = _load()
        except _CorruptState:
            return {"ok": False, "reason": "corrupt_state", "count": -1, "cap": cap}
        stored_date = str(st.get("date") or "")
        count = int(st.get("count", 0) or 0)
        # Never move the accounting date backward; reset the count only on a genuine
        # forward day change.
        effective_date = today if today > stored_date else stored_date
        if effective_date != stored_date:
            count = 0
        if count >= cap:
            _save({"date": effective_date, "count": count})
            return {"ok": False, "reason": "daily_cap", "count": count, "cap": cap}
        count += 1
        _save({"date": effective_date, "count": count})
        return {"ok": True, "count": count, "cap": cap}
