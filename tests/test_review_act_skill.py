"""Tests for the dedicated `review_act` tool (skills/review-act/executor.py).

The keystone fix from the 2026-06-19 trigger-plan audit: a SINGLE-PURPOSE tool the
daemon can allow-list without leaking other ops (vs the 13-op `scratchpad`), routed
through the flag-gated `self_maint.safe_review_act` so observe-first is real. It
exposes ONLY snooze/escalate; resolve/dismiss are refused before they reach anything
(defense in depth: here, AND in self_maint, AND at review_mark's authz).

Loads the executor by path (as skill_runtime does), so it runs without pytest.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from core import self_maint as sm  # noqa: E402

_calls: list = []


def _stub_ok(item_id, action, *, note=None):
    _calls.append((item_id, action, note))
    return {"ok": True, "item_id": item_id, "action": action, "state": {"status": action}}


def _stub_off(item_id, action, *, note=None):
    return {"ok": False, "refused": "flag_off", "item_id": item_id, "action": action}


@contextmanager
def _patched_safe_review_act(stub):
    """Patch the module global without leaking it into later test modules."""
    original = sm.safe_review_act
    sm.safe_review_act = stub
    try:
        yield
    finally:
        sm.safe_review_act = original


def _load():
    p = os.path.join(_REPO, "skills", "review-act", "executor.py")
    spec = importlib.util.spec_from_file_location("review_act_executor_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_refused_before_self_maint():
    _calls.clear()
    with _patched_safe_review_act(_stub_ok):
        out = _load().run({"item_id": "acu:1", "action": "resolve"})
    assert "snooze" in out.lower() and "escalate" in out.lower()
    assert _calls == []  # resolve never reaches the actuator


def test_dismiss_refused_before_self_maint():
    _calls.clear()
    with _patched_safe_review_act(_stub_ok):
        _load().run({"item_id": "acu:1", "action": "dismiss"})
    assert _calls == []


def test_snooze_routes_to_safe_review_act():
    _calls.clear()
    with _patched_safe_review_act(_stub_ok):
        out = _load().run({"item_id": "acu:87", "action": "snooze", "note": "stale"})
    assert "ok" in out.lower() and _calls == [("acu:87", "snooze", "stale")]


def test_escalate_routes_to_safe_review_act():
    _calls.clear()
    with _patched_safe_review_act(_stub_ok):
        _load().run({"item_id": "pin:6", "action": "escalate"})
    assert _calls == [("pin:6", "escalate", None)]


def test_missing_item_id_refused():
    _calls.clear()
    with _patched_safe_review_act(_stub_ok):
        out = _load().run({"action": "snooze"})
    assert "item_id" in out and _calls == []


def test_flag_off_surfaced_not_crashed():
    with _patched_safe_review_act(_stub_off):
        out = _load().run({"item_id": "acu:1", "action": "snooze"})
    assert "flag_off" in out


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
