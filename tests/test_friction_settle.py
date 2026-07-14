"""Tests for core/friction_settle — back-compat shim over intent_settle.

The membership settle logic + interceptor are tested in test_intent_settle.py.
Here: confirm the shim re-exports, and that the bearing single-writer fires under
INJECT while preserving current_frame (the <frame> fastpath's self-field).
"""
from __future__ import annotations

import dataclasses

from core import friction_organ as fo
from core import friction_settle as settle
from core import friction_store as fs


def test_shim_reexports():
    from core import intent_settle
    assert settle.friction_settle_interceptor is intent_settle.friction_settle_interceptor
    assert settle.settle is intent_settle.settle


def test_flag_off_is_noop(tmp_path, monkeypatch):
    db = tmp_path / "tt.sqlite3"
    monkeypatch.setattr(fs, "_DB_PATH", db)
    monkeypatch.delenv(fs._FLAG_ENV, raising=False)
    messages = [{"role": "assistant", "content": "answer"},
                {"role": "user", "content": "no, wrong"}]
    out = settle.friction_settle_interceptor(messages, {})
    assert out is messages
    assert fs.recent_settled(db_path=db) == []


def test_settle_feeds_bearing_under_inject_preserving_current_frame(tmp_path, monkeypatch):
    db = tmp_path / "tt.sqlite3"
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    monkeypatch.setenv("MONOLITH_FRICTION_INJECT_V1", "1")  # bearing write gated on INJECT
    monkeypatch.setattr(fs, "_DB_PATH", db)
    fo.on_turn_ready("here is my long answer about the differ and overlap channels",
                     "design me a world model", "t1", 1, "2026-06-19T00:00:00")

    from addons.system.bearing import store as bstore
    from addons.system.bearing.schema import Bearing
    monkeypatch.setattr(bstore, "_STORE_PATH", tmp_path / "bearing.json")
    bstore.set_bearing(dataclasses.replace(Bearing(), current_frame="holding the design"))

    messages = [
        {"role": "assistant", "content": "the differ scores content overlap and markers"},
        {"role": "user", "content": "No, that's not what I meant — drop the overlap idea."},
    ]
    out = settle.friction_settle_interceptor(messages, {"_now_iso": "2026-06-19T00:01:00"})
    assert out is messages  # side-effect only

    settled = fs.recent_settled(db_path=db)
    assert {s["friction_type"] for s in settled} == {"correction"}  # marker caught
    b = bstore.get_bearing()
    assert b.current_frame == "holding the design"          # fastpath self-field preserved
    assert b.user_model is not None and b.user_model.intent_read  # Other-field written


def test_bearing_not_written_in_observe_phase(tmp_path, monkeypatch):
    """With FRICTION_V1 on but INJECT off (observe phase), settle records the
    artifact but does NOT mutate bearing — observe is read-only on bearing."""
    db = tmp_path / "tt.sqlite3"
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    monkeypatch.delenv("MONOLITH_FRICTION_INJECT_V1", raising=False)
    monkeypatch.setattr(fs, "_DB_PATH", db)
    fo.on_turn_ready("answer about overlap", "design me a world model", "t1", 1, "2026-06-19T00:00:00")

    from addons.system.bearing import store as bstore
    from addons.system.bearing.schema import Bearing
    monkeypatch.setattr(bstore, "_STORE_PATH", tmp_path / "bearing.json")
    bstore.set_bearing(Bearing())  # empty: user_model None

    messages = [{"role": "assistant", "content": "answer about overlap"},
                {"role": "user", "content": "No, wrong, drop overlap"}]
    settle.friction_settle_interceptor(messages, {"_now_iso": "2026-06-19T00:01:00"})
    assert fs.recent_settled(db_path=db)            # artifact recorded
    assert bstore.get_bearing().user_model is None  # bearing untouched in observe
