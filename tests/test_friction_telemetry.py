"""Tests for core/friction_telemetry — the [FRICTION] reader contributor."""
from __future__ import annotations

from core import friction_store as fs
from core import friction_telemetry as ft


def _seed_high(tmp_path, monkeypatch):
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = tmp_path / "tt.sqlite3"
    for i in range(3):
        rid = fs.record_prediction("t", i, "intent", "c", "f", 0.5, "next_turn",
                                   now_iso="x", db_path=db)
        fs.settle_prediction(rid, 0.92, "correction",
                             {"answer_overlap": 0.1, "markers": ["correction"]},
                             "no, that's not what I meant", "t2", now_iso="y", db_path=db)
    return db


def test_inject_flag_off_returns_none(tmp_path, monkeypatch):
    _seed_high(tmp_path, monkeypatch)
    monkeypatch.delenv(ft._INJECT_FLAG, raising=False)
    assert ft.contribute_section([{"role": "user", "content": "hi"}], {}) is None


def test_peer_turn_returns_none(tmp_path, monkeypatch):
    _seed_high(tmp_path, monkeypatch)
    monkeypatch.setenv(ft._INJECT_FLAG, "1")
    peer = [{"role": "user", "content": "[CHANNEL: connect/codex] hello"}]
    assert ft.contribute_section(peer, {}) is None


def test_calm_window_is_silent():
    # all uptake, low score -> nothing worth surfacing
    rows = [{"id": i, "friction_score": 0.1, "friction_type": "uptake",
             "channel_json": {"answer_overlap": 0.5}, "observation": ""} for i in range(5)]
    assert ft.render_friction_block(rows) is None


def test_rising_friction_renders_block():
    rows = [
        {"id": 3, "friction_score": 0.92, "friction_type": "correction",
         "channel_json": {"answer_overlap": 0.1}, "observation": "no, not that"},
        {"id": 2, "friction_score": 0.85, "friction_type": "reframe",
         "channel_json": {"answer_overlap": 0.2}, "observation": "the real view is"},
        {"id": 1, "friction_score": 0.1, "friction_type": "uptake",
         "channel_json": {"answer_overlap": 0.6}, "observation": ""},
    ]
    block = ft.render_friction_block(rows)
    assert block is not None
    assert block.startswith("[FRICTION]")
    assert "correction" in block
    assert "weigh it" in block  # mirror-not-objective framing present


def test_contribute_marks_surfaced_on_commit(tmp_path, monkeypatch):
    db = _seed_high(tmp_path, monkeypatch)
    monkeypatch.setenv(ft._INJECT_FLAG, "1")
    # point the reader at the temp db by monkeypatching the store default
    monkeypatch.setattr(fs, "_DB_PATH", db)
    sec = ft.contribute_section([{"role": "user", "content": "hi"}], {})
    assert sec is not None and sec.name == "friction"
    assert sec.on_commit is not None
    sec.on_commit()
    assert all(r["surfaced"] == 1 for r in fs.recent_settled(db_path=db))
