"""Tests for the self_maint control skill (skills/self-maint/executor.py).

Loads the executor by path and stubs the runner + leash so the op routing is verified
without a live daemon. Mirrors test_review_act_skill. Runs without pytest.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from core import self_maint_leash as leash  # noqa: E402
from engine import self_maint_runner as smr  # noqa: E402

_EXEC = pathlib.Path(_REPO) / "skills" / "self-maint" / "executor.py"

_REAL_GET = smr.get_runner
_REAL_BUSY = smr.engine_is_busy
_REAL_CAP = leash.set_cap_override


def _load():
    spec = importlib.util.spec_from_file_location("self_maint_ctl_exec", _EXEC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeRunner:
    def __init__(self):
        self.calls = []
        self._status = "idle"

    def set_interval(self, s):
        self.calls.append(("set_interval", s))
        return 60

    def set_is_busy(self, f):
        self.calls.append(("set_is_busy", callable(f)))

    def start(self, *, force=False):
        self.calls.append(("start", force))
        self._status = "running"
        return True

    def stop(self, timeout=0.0):
        self.calls.append(("stop", timeout))
        self._status = "stopped"

    def snapshot(self):
        return {"status": self._status, "interval_s": 60, "apply_on": False, "wake": 1}


def _restore():
    smr.get_runner = _REAL_GET
    smr.engine_is_busy = _REAL_BUSY
    leash.set_cap_override = _REAL_CAP


def test_start_forces_and_binds_is_busy():
    fr = _FakeRunner()
    smr.get_runner = lambda: fr
    smr.engine_is_busy = lambda ws: False
    try:
        out = _load().run({"op": "start"}, ctx=None)
        assert ("start", True) in fr.calls                       # force=True (skill authorizes)
        assert any(c[0] == "set_is_busy" for c in fr.calls)       # live-turn guard bound (audit req)
        assert "observe-first" in out                             # the safety reminder is surfaced
    finally:
        _restore()


def test_start_with_cadence_args_sets_interval_and_cap():
    fr = _FakeRunner()
    captured = {}
    smr.get_runner = lambda: fr
    smr.engine_is_busy = lambda ws: False
    leash.set_cap_override = lambda n: captured.__setitem__("cap", n)
    try:
        _load().run({"op": "start", "seconds": 60, "cap": 1440}, ctx=None)
        assert ("set_interval", 60) in fr.calls
        assert captured.get("cap") == 1440
    finally:
        _restore()


def test_set_interval_routes():
    fr = _FakeRunner()
    smr.get_runner = lambda: fr
    try:
        _load().run({"op": "set_interval", "seconds": 60})
        assert ("set_interval", 60) in fr.calls
    finally:
        _restore()


def test_set_cap_routes():
    captured = {}
    smr.get_runner = lambda: _FakeRunner()
    leash.set_cap_override = lambda n: captured.__setitem__("cap", n)
    try:
        _load().run({"op": "set_cap", "cap": 1440})
        assert captured.get("cap") == 1440
    finally:
        _restore()


def test_stop_routes():
    fr = _FakeRunner()
    smr.get_runner = lambda: fr
    try:
        _load().run({"op": "stop"})
        assert any(c[0] == "stop" for c in fr.calls)
    finally:
        _restore()


def test_status_returns_snapshot():
    smr.get_runner = lambda: _FakeRunner()
    try:
        out = _load().run({"op": "status"})
        assert "status=" in out and "interval=" in out
    finally:
        _restore()


def test_unknown_op_refused():
    smr.get_runner = lambda: _FakeRunner()
    try:
        out = _load().run({"op": "nuke"})
        assert "op must be" in out
    finally:
        _restore()


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
