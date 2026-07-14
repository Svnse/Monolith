"""Tests for core/pipeline_policies/intent_predict_bridge — the predict beat.

Reads the CLEAN public_answer off TurnReadyEvent (not raw history); self-gates on
the friction flag; records ONE open prediction carrying a frozen prediction_set.
"""
from __future__ import annotations

from core import friction_store as fs
from core import intent_predict_bridge as bridge
from core.turn_pipeline_events import TurnReadyEvent, TurnCompleteEvent


def test_flag_off_records_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(fs._FLAG_ENV, raising=False)
    db = tmp_path / "tt.sqlite3"
    monkeypatch.setattr(fs, "_DB_PATH", db)
    ev = TurnReadyEvent(turn_id="t1", emitted_at="2026-06-22T00:00:00",
                        public_answer="a plan to rotate the session token and manage expiry")
    bridge._handle(ev, None)
    assert fs.latest_open(db_path=db) is None


def test_records_open_prediction_with_frozen_set(tmp_path, monkeypatch):
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = tmp_path / "tt.sqlite3"
    monkeypatch.setattr(fs, "_DB_PATH", db)
    ev = TurnReadyEvent(turn_id="t1", emitted_at="2026-06-22T00:00:00",
                        public_answer="here is the plan: rotate the session token and analyze expiry tradeoffs")
    bridge._handle(ev, None)
    op = fs.latest_open("intent", db_path=db)
    assert op is not None
    assert op["created_at"] == "2026-06-22T00:00:00"   # clean event instant, for the settle guard
    pset = op["prediction_set_json"]
    assert isinstance(pset, dict) and pset.get("referents")   # frozen, non-empty


def test_ignores_non_turnready(tmp_path, monkeypatch):
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = tmp_path / "tt.sqlite3"
    monkeypatch.setattr(fs, "_DB_PATH", db)
    bridge._handle(TurnCompleteEvent(turn_id="t1"), None)
    assert fs.latest_open(db_path=db) is None


def test_supersedes_prior_open_per_outer_turn(tmp_path, monkeypatch):
    """Two emits in one outer turn (initial + tool_followup) leave ONE open
    prediction = the latest (final) answer; no orphan accumulation."""
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = tmp_path / "tt.sqlite3"
    monkeypatch.setattr(fs, "_DB_PATH", db)
    bridge._handle(TurnReadyEvent(turn_id="t1", emitted_at="2026-06-22T00:00:01",
                                  public_answer="initial answer about the auth token"), None)
    bridge._handle(TurnReadyEvent(turn_id="t1", emitted_at="2026-06-22T00:00:02",
                                  public_answer="final answer about the migration rollout plan"), None)
    op = fs.latest_open("intent", db_path=db)
    assert op is not None and op["created_at"] == "2026-06-22T00:00:02"  # final wins
    # the superseded one is abandoned, not open or settled
    assert fs.recent_settled(db_path=db) == []
