"""Tests for the self-maintenance wake leash (core/self_maint_leash.py) — Gap 5.

The autonomous daemon's per-UTC-day wake cap must survive crash-restarts (counter
persisted, not in-memory) and must NOT be defeatable by moving the clock backward.
Pure; injects `now`; monkeypatches the state path. Runs without pytest.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime, timezone

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from core import self_maint_leash as leash  # noqa: E402

_D1 = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
_D2 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)


def _tmp(name):
    p = pathlib.Path(tempfile.gettempdir()) / name
    for stray in (p, pathlib.Path(f"{p}.lock"), pathlib.Path(f"{p}.tmp")):
        try:
            if stray.exists():
                stray.unlink()
        except OSError:
            # A lingering Windows handle / AV scan can briefly hold the .lock file
            # (WinError 32). Harmless: the product's stale-steal reclaims it, and the
            # in-process lock serializes regardless. Don't fail test setup on it.
            pass
    leash._STATE_PATH = p
    return p


def test_first_wake_ok():
    _tmp("smw1.json")
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "5"
    r = leash.try_consume_wake(now=_D1)
    assert r["ok"] is True and r["count"] == 1


def test_daily_cap_enforced():
    _tmp("smw2.json")
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "2"
    assert leash.try_consume_wake(now=_D1)["ok"] is True
    assert leash.try_consume_wake(now=_D1)["ok"] is True
    r = leash.try_consume_wake(now=_D1)
    assert r["ok"] is False and r["reason"] == "daily_cap"


def test_forward_day_resets_budget():
    _tmp("smw3.json")
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "2"
    leash.try_consume_wake(now=_D1)
    leash.try_consume_wake(now=_D1)  # cap reached on D1
    r = leash.try_consume_wake(now=_D2)  # genuine new day
    assert r["ok"] is True and r["count"] == 1


def test_backward_clock_does_not_grant_extra_wakes():
    _tmp("smw4.json")
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "2"
    leash.try_consume_wake(now=_D2)
    leash.try_consume_wake(now=_D2)  # cap reached on D2
    r = leash.try_consume_wake(now=_D1)  # attacker moves the clock BACK
    assert r["ok"] is False  # must NOT reset the count


def test_count_is_persisted_for_restart():
    p = _tmp("smw5.json")
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "5"
    leash.try_consume_wake(now=_D1)
    st = json.loads(p.read_text(encoding="utf-8"))
    assert st["count"] == 1 and st["date"] == "2026-06-19"


def test_corrupt_state_fails_closed():
    # A corrupt/truncated state file must NOT be read as count 0 = fresh budget.
    p = _tmp("smw_corrupt.json")
    p.write_text("{ this is not valid json", encoding="utf-8")
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "5"
    r = leash.try_consume_wake(now=_D1)
    assert r["ok"] is False  # fail closed on corruption


def test_missing_state_is_legit_first_wake():
    _tmp("smw_missing.json")  # ensures it does not exist
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "5"
    r = leash.try_consume_wake(now=_D1)
    assert r["ok"] is True and r["count"] == 1  # genuine first run


def test_garbage_cap_fails_closed():
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "abc"
    try:
        assert leash.max_wakes_per_day() == 0  # a cap typo => zero wakes, not 24
    finally:
        os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "5"


def test_cap_override_takes_precedence_over_env():
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "24"
    leash.set_cap_override(1440)   # the control skill raising the cap for "every minute all day"
    try:
        assert leash.max_wakes_per_day() == 1440   # under the ceiling => honored
    finally:
        leash.set_cap_override(None)
    assert leash.max_wakes_per_day() == 24   # cleared => env again
    os.environ.pop("MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY", None)


def test_cap_override_clamped_to_ceiling():
    # A level-1 model turn must NOT be able to self-arm an unbounded paid loop. (audit blocker)
    leash.set_cap_override(10 ** 9)
    try:
        assert leash.max_wakes_per_day() == leash._MAX_CAP   # clamped, not 1e9
    finally:
        leash.set_cap_override(None)


def test_env_cap_also_clamped():
    # Even a launcher typo (env) is bounded by the absolute ceiling.
    leash.set_cap_override(None)
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "99999"
    try:
        assert leash.max_wakes_per_day() == leash._MAX_CAP
    finally:
        os.environ.pop("MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY", None)


def test_unreadable_state_fails_closed():
    # A present-but-unreadable state file (Windows sharing-violation / a directory) must
    # NOT be read as a fresh budget — it must fail closed like a corrupt parse. (audit #3)
    d = pathlib.Path(tempfile.mkdtemp()) / "as_dir"
    d.mkdir()
    leash._STATE_PATH = d  # reading a directory with read_text raises OSError
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "5"
    r = leash.try_consume_wake(now=_D1)
    assert r["ok"] is False and r["reason"] == "corrupt_state"


def test_lock_unavailable_fails_closed():
    # If the interprocess lock cannot be acquired, the leash must SKIP the wake (fail
    # closed), never proceed unlocked — proceeding unlocked permits a cross-process
    # double-spend of the daily cap. (audit #7)
    import contextlib

    @contextlib.contextmanager
    def _no_lock():
        yield False

    _tmp("smw_locktest.json")
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "5"
    orig = leash._interprocess_lock
    leash._interprocess_lock = _no_lock
    try:
        r = leash.try_consume_wake(now=_D1)
        assert r["ok"] is False and r["reason"] == "lock_unavailable"
    finally:
        leash._interprocess_lock = orig


def test_concurrent_consume_respects_cap():
    import threading
    _tmp("smw_concurrent.json")
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "5"
    grants = []
    glock = threading.Lock()

    def worker():
        r = leash.try_consume_wake(now=_D1)
        if r.get("ok"):
            with glock:
                grants.append(1)

    ts = [threading.Thread(target=worker) for _ in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sum(grants) <= 5  # the cap holds under concurrent read-modify-write


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    os.environ.pop("MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY", None)
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
