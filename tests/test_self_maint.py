"""Tests for the safe self-maintenance actuator (core/self_maint.py).

The actuator is the model's SAFE hand on its own review queue: it may snooze
(reversible) and escalate (flag for the operator), and must REFUSE resolve/dismiss (un-checked
judgment calls). Flag-gated (default OFF). Tests monkeypatch review_loop.review_mark
so they exercise the gate logic without the live review stores.

Runnable with the active Python environment:
  python -m pytest tests/test_self_maint.py -q
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import self_maint  # noqa: E402


# ── Task 1: flag ────────────────────────────────────────────────────────────
def test_enabled_defaults_off():
    os.environ.pop("MONOLITH_SELF_MAINT_V1", None)
    assert self_maint.enabled() is False


def test_enabled_when_flag_truthy():
    os.environ["MONOLITH_SELF_MAINT_V1"] = "1"
    try:
        assert self_maint.enabled() is True
    finally:
        os.environ.pop("MONOLITH_SELF_MAINT_V1", None)


# ── Task 2: ledger ──────────────────────────────────────────────────────────
def test_log_appends_one_json_row():
    p = pathlib.Path(tempfile.gettempdir()) / "self_maint_test.jsonl"
    if p.exists():
        p.unlink()
    self_maint._LEDGER = p
    self_maint._log({"ts": "t", "ok": True, "item_id": "x"})
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1 and rows[0]["item_id"] == "x"


# ── Task 3: the safe gate ───────────────────────────────────────────────────
def _stub_mark(calls):
    def _mark(item_id, action, *, actor="monolith", note=None, **kw):
        calls.append((item_id, action, actor, note))
        return {"ok": True, "item_id": item_id, "action": action, "actor": actor,
                "state": {"status": "snoozed" if action == "snooze" else "escalated"}}
    return _mark


def _setup(tmp_name):
    calls = []
    self_maint.review_loop.review_mark = _stub_mark(calls)
    p = pathlib.Path(tempfile.gettempdir()) / tmp_name
    if p.exists():
        p.unlink()
    self_maint._LEDGER = p
    return calls, p


def test_flag_off_refuses_and_does_not_call_mark():
    os.environ.pop("MONOLITH_SELF_MAINT_V1", None)
    calls, _ = _setup("sm_off.jsonl")
    out = self_maint.safe_review_act("acu:87", "snooze")
    assert out["ok"] is False and out["refused"] == "flag_off"
    assert calls == []  # review_mark never called when flag off


def test_unsafe_action_refused_and_not_applied():
    os.environ["MONOLITH_SELF_MAINT_V1"] = "1"
    try:
        calls, _ = _setup("sm_unsafe.jsonl")
        out = self_maint.safe_review_act("acu:87", "resolve")
        assert out["ok"] is False and out["refused"] == "unsafe_action"
        assert calls == []  # resolve/dismiss never reach review_mark
    finally:
        os.environ.pop("MONOLITH_SELF_MAINT_V1", None)


def test_snooze_is_applied_via_review_mark():
    os.environ["MONOLITH_SELF_MAINT_V1"] = "1"
    try:
        calls, _ = _setup("sm_snooze.jsonl")
        out = self_maint.safe_review_act("acu:87", "snooze", note="stale")
        assert out["ok"] is True and out["state"]["status"] == "snoozed"
        assert calls == [("acu:87", "snooze", "monolith", "stale")]
    finally:
        os.environ.pop("MONOLITH_SELF_MAINT_V1", None)


def test_escalate_is_applied_and_logged():
    os.environ["MONOLITH_SELF_MAINT_V1"] = "1"
    try:
        calls, p = _setup("sm_esc.jsonl")
        out = self_maint.safe_review_act("pin:6", "escalate")
        assert out["ok"] is True and calls[0][1] == "escalate"
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert rows and rows[-1]["action"] == "escalate" and rows[-1]["ok"] is True
    finally:
        os.environ.pop("MONOLITH_SELF_MAINT_V1", None)


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
