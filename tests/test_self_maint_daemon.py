"""Tests for the self-maintenance daemon's SAFETY CORE (core/self_maint_daemon.py).

Not the threaded loop — the testable safety invariants the audit required:
  - Gap 4: the wake's ToolExecutionContext is level=2 (below that the allow-list
    gate is bypassed and every tool is reachable).
  - the allow-list is NARROW: review_act + read-only tools, and explicitly NOT
    scratchpad / write_file / run_command / spawn_subagent / any mutator.
  - the wake is gated by the persisted daily leash (Gap 5 unit, integrated here).
Runs without pytest.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from core import self_maint_daemon as daemon  # noqa: E402
from core import self_maint_leash as leash  # noqa: E402


def test_wake_context_is_level_2():
    # Gap 4 — the allow-list gate only fires when level > 1. Tested on the pure
    # kwargs (the real ToolExecutionContext pulls Qt, absent in the venv).
    assert daemon.wake_context_kwargs(lambda: False)["level"] == 2


def test_wake_tools_are_narrow():
    assert "review_act" in daemon.WAKE_TOOLS
    for forbidden in ("scratchpad", "write_file", "run_command", "spawn_subagent",
                      "edit_file", "propose_amendment", "pin"):
        assert forbidden not in daemon.WAKE_TOOLS


def test_wake_tools_are_review_act_only_in_v1():
    # V1: a maintenance wake is snooze/escalate ONLY — no filesystem reads. Dropping the
    # read tools closes the unconfined arbitrary-path read disclosure by construction. A
    # decision-lock: re-adding reads (behind path-confinement) must update this test. (audit #2)
    assert daemon.WAKE_TOOLS == frozenset({"review_act"})
    for read_tool in ("open_file", "read_file", "grep", "find_files", "list_files"):
        assert read_tool not in daemon.WAKE_TOOLS


def test_wake_context_allowed_tools_are_the_narrow_set():
    allowed = daemon.wake_context_kwargs(lambda: False)["allowed_tools"]
    assert "review_act" in allowed
    assert "scratchpad" not in allowed
    assert "write_file" not in allowed


def test_try_wake_respects_daily_leash():
    import pathlib
    import tempfile
    p = pathlib.Path(tempfile.gettempdir()) / "smd_leash.json"
    if p.exists():
        p.unlink()
    leash._STATE_PATH = p
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "1"
    try:
        assert daemon.try_wake()["ok"] is True       # first wake of the day
        assert daemon.try_wake()["ok"] is False       # cap reached
    finally:
        os.environ.pop("MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY", None)


def test_trigger_disabled_by_default():
    os.environ.pop("MONOLITH_SELF_MAINT_TRIGGER_V1", None)
    assert daemon.trigger_enabled() is False  # the daemon never self-starts unless opted in


def test_trigger_enabled_when_flag_truthy():
    for val in ("1", "true", "YES", "on"):
        os.environ["MONOLITH_SELF_MAINT_TRIGGER_V1"] = val
        assert daemon.trigger_enabled() is True, val
    os.environ["MONOLITH_SELF_MAINT_TRIGGER_V1"] = "0"
    assert daemon.trigger_enabled() is False
    os.environ.pop("MONOLITH_SELF_MAINT_TRIGGER_V1", None)


def test_wake_interval_default_and_override():
    os.environ.pop("MONOLITH_SELF_MAINT_WAKE_INTERVAL_S", None)
    assert daemon.wake_interval_s() == 1800  # conservative default
    os.environ["MONOLITH_SELF_MAINT_WAKE_INTERVAL_S"] = "60"
    assert daemon.wake_interval_s() == 60
    os.environ["MONOLITH_SELF_MAINT_WAKE_INTERVAL_S"] = "garbage"
    assert daemon.wake_interval_s() == 1800  # bad value falls back to default
    for floorless in ("0", "-5"):
        os.environ["MONOLITH_SELF_MAINT_WAKE_INTERVAL_S"] = floorless
        assert daemon.wake_interval_s() == 1800  # <=0 must fail closed to the default floor (audit #5)
    os.environ.pop("MONOLITH_SELF_MAINT_WAKE_INTERVAL_S", None)


def test_read_wake_tail_returns_last_rows_oldest_to_newest():
    import json
    import pathlib
    import tempfile
    p = pathlib.Path(tempfile.gettempdir()) / "smd_tail.jsonl"
    rows = [{"turn_id": f"maint_{i:06d}", "items_seen": i} for i in range(30)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    daemon._WAKE_LEDGER = p
    tail = daemon.read_wake_tail(5)
    assert len(tail) == 5
    assert tail[0]["turn_id"] == "maint_000025"   # oldest of the tail first
    assert tail[-1]["turn_id"] == "maint_000029"  # newest last


def test_read_wake_tail_missing_file_is_empty():
    import pathlib
    import tempfile
    p = pathlib.Path(tempfile.gettempdir()) / "smd_tail_absent.jsonl"
    if p.exists():
        p.unlink()
    daemon._WAKE_LEDGER = p
    assert daemon.read_wake_tail(5) == []  # never raises on a missing ledger


def test_log_wake_appends_row():
    import json
    import pathlib
    import tempfile
    p = pathlib.Path(tempfile.gettempdir()) / "smd_wake_ledger.jsonl"
    if p.exists():
        p.unlink()
    daemon._WAKE_LEDGER = p
    daemon.log_wake({"turn_id": "maint_000001", "items_seen": 3, "tool_calls": ["review_act"]})
    daemon.log_wake({"turn_id": "maint_000002", "items_seen": 0, "tool_calls": []})
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["turn_id"] == "maint_000001" and rows[0]["items_seen"] == 3


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
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
