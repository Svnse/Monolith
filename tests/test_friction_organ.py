"""Tests for core/friction_organ — predict + crystallize beats."""
from __future__ import annotations

from core import friction_organ as fo
from core import friction_store as fs


def test_extract_returns_claim_and_falsifier():
    pred = fo.extract("here is the design", "please refactor the auth module and fix the bug")
    assert pred["claim"]
    assert pred["falsifier"]
    assert isinstance(pred["confidence"], float)


def test_on_turn_ready_noop_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv(fs._FLAG_ENV, raising=False)
    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "tt.sqlite3")
    rid = fo.on_turn_ready("answer", "user msg", "t1", 1, "2026-06-19T00:00:00")
    assert rid == -1
    assert fs.latest_open(db_path=tmp_path / "tt.sqlite3") is None


def test_on_turn_ready_records_open_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = tmp_path / "tt.sqlite3"
    monkeypatch.setattr(fs, "_DB_PATH", db)
    rid = fo.on_turn_ready("answer", "design me a world model", "t1", 3, "2026-06-19T00:00:00")
    assert rid > 0
    intent = fs.latest_open("intent", db_path=db)
    assert intent is not None and intent["kind"] == "intent"
    # trajectory is DEFERRED to v1.1 — must NOT be recorded (no hollow dead state)
    assert fs.latest_open("trajectory", db_path=db) is None


def test_should_crystallize_recurrence_gate():
    rows = [{"friction_type": "topic_drift"} for _ in range(3)]
    assert fo.should_crystallize(rows, "topic_drift", min_count=3) is True
    assert fo.should_crystallize(rows[:2], "topic_drift", min_count=3) is False
    # calm types never crystallize
    assert fo.should_crystallize([{"friction_type": "uptake"}] * 5, "uptake") is False


def test_maybe_crystallize_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv(fs._FLAG_ENV, raising=False)
    assert fo.maybe_crystallize("correction") is False
