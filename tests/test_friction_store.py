"""Tests for core/friction_store — Layer F prediction_traces.

Critical invariants: flag-OFF writes nothing AND never creates the table
(byte-identical guarantee); round-trip record->settle->read works; reads
tolerate a missing table.
"""
from __future__ import annotations

import sqlite3

from core import friction_store as fs


def _db(tmp_path):
    return tmp_path / "tt.sqlite3"


def test_flag_off_writes_nothing_and_no_table(tmp_path, monkeypatch):
    monkeypatch.delenv(fs._FLAG_ENV, raising=False)
    db = _db(tmp_path)
    rid = fs.record_prediction("t1", 1, "intent", "wants depth", "asks for breadth",
                               0.5, "next_turn", now_iso="2026-06-19T00:00:00", db_path=db)
    assert rid == -1
    # the DB file must not even have the table (or not exist at all)
    if db.exists():
        conn = sqlite3.connect(str(db))
        try:
            assert not fs._table_exists(conn)
        finally:
            conn.close()
    assert fs.latest_open(db_path=db) is None
    assert fs.recent_settled(db_path=db) == []


def test_record_settle_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = _db(tmp_path)
    rid = fs.record_prediction("t1", 7, "intent", "wants the deep answer",
                               "asks for a shallow summary", 0.6, "next_turn",
                               now_iso="2026-06-19T00:00:00", db_path=db)
    assert rid > 0
    open_pred = fs.latest_open(db_path=db)
    assert open_pred is not None
    assert open_pred["status"] == "open"
    assert open_pred["claim"] == "wants the deep answer"
    assert open_pred["turn_n"] == 7

    fs.settle_prediction(rid, 0.92, "correction",
                         {"markers": ["correction"], "answer_overlap": 0.1},
                         "no, deeper", "t2", now_iso="2026-06-19T00:01:00", db_path=db)

    assert fs.latest_open(db_path=db) is None  # no longer open
    settled = fs.recent_settled(db_path=db)
    assert len(settled) == 1
    s = settled[0]
    assert s["status"] == "settled"
    assert s["friction_type"] == "correction"
    assert s["friction_score"] == 0.92
    assert s["channel_json"]["markers"] == ["correction"]  # deserialized


def test_settle_only_affects_open(tmp_path, monkeypatch):
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = _db(tmp_path)
    rid = fs.record_prediction("t1", 1, "intent", "c", "f", 0.5, "next_turn",
                               now_iso="x", db_path=db)
    fs.settle_prediction(rid, 0.5, "uptake", {}, "obs", "t2", now_iso="y", db_path=db)
    # second settle is a no-op (status already settled)
    fs.settle_prediction(rid, 0.99, "correction", {}, "obs2", "t3", now_iso="z", db_path=db)
    settled = fs.recent_settled(db_path=db)
    assert len(settled) == 1
    assert settled[0]["friction_type"] == "uptake"  # unchanged


def test_mark_surfaced(tmp_path, monkeypatch):
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = _db(tmp_path)
    rid = fs.record_prediction("t1", 1, "intent", "c", "f", 0.5, "next_turn",
                               now_iso="x", db_path=db)
    fs.settle_prediction(rid, 0.7, "topic_drift", {}, "obs", "t2", now_iso="y", db_path=db)
    fs.mark_surfaced(rid, db_path=db)
    assert fs.recent_settled(db_path=db)[0]["surfaced"] == 1


def test_latest_open_filters_by_kind(tmp_path, monkeypatch):
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = _db(tmp_path)
    fs.record_prediction("t1", 1, "intent", "ci", "f", 0.5, "next_turn", now_iso="x", db_path=db)
    fs.record_prediction("t1", 1, "trajectory", "ct", "f", 0.5, "multi_turn", now_iso="x", db_path=db)
    assert fs.latest_open("intent", db_path=db)["kind"] == "intent"
    assert fs.latest_open("trajectory", db_path=db)["kind"] == "trajectory"


# ── Path B v2: frozen prediction_set_json + abandon_open + migration ──


def test_prediction_set_roundtrip(tmp_path, monkeypatch):
    """The frozen set is persisted as JSON and deserialized on read."""
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = _db(tmp_path)
    pset = {"directions": [{"move": "plan", "referent": "token rotation"}],
            "referents": ["token", "rotation", "session"], "source": "floor"}
    rid = fs.record_prediction("t1", 1, "intent", "wants a plan response",
                               "pulls outside the staked set", 0.6, "next_turn",
                               now_iso="2026-06-22T00:00:00", prediction_set_json=pset, db_path=db)
    assert rid > 0
    op = fs.latest_open(db_path=db)
    assert op["prediction_set_json"]["referents"] == ["token", "rotation", "session"]
    assert op["prediction_set_json"]["directions"][0]["move"] == "plan"


def test_abandon_open_supersedes(tmp_path, monkeypatch):
    """abandon_open closes prior opens so at most one prediction is ever open
    (tool-loop followup supersession; orphan-prevention)."""
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = _db(tmp_path)
    fs.record_prediction("t1", 1, "intent", "first", "f", 0.5, "next_turn", now_iso="a", db_path=db)
    n = fs.abandon_open(db_path=db)
    assert n == 1
    fs.record_prediction("t1", 1, "intent", "second", "f", 0.5, "next_turn", now_iso="b", db_path=db)
    op = fs.latest_open(db_path=db)
    assert op is not None and op["claim"] == "second"  # only the latest is open
    # the abandoned one is neither open nor settled -> not in recent_settled
    assert fs.recent_settled(db_path=db) == []


def test_abandon_open_flag_off_noop(tmp_path, monkeypatch):
    monkeypatch.delenv(fs._FLAG_ENV, raising=False)
    assert fs.abandon_open(db_path=_db(tmp_path)) == 0


def test_v1_schema_migrates_in_place(tmp_path, monkeypatch):
    """A pre-existing v1 table (no prediction_set_json) gains the column on next
    write — E's live DB must upgrade, not be ignored or recreated."""
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = _db(tmp_path)
    # Seed a v1-schema table WITHOUT prediction_set_json.
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE prediction_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT, schema_version INTEGER,
            turn_id TEXT, turn_n INTEGER, created_at TEXT, kind TEXT, claim TEXT,
            falsifier TEXT, confidence REAL, horizon TEXT, status TEXT DEFAULT 'open',
            friction_score REAL, friction_type TEXT, channel_json TEXT,
            settled_at TEXT, settled_turn_id TEXT, observation TEXT, surfaced INTEGER DEFAULT 0)"""
    )
    conn.execute(
        "INSERT INTO prediction_traces (turn_id, kind, claim, status) VALUES ('old','intent','v1 row','open')"
    )
    conn.commit()
    conn.close()
    c0 = fs._connect(db)
    try:
        assert "prediction_set_json" not in fs._column_names(c0)
    finally:
        c0.close()

    # Next write triggers the guarded ALTER; the v1 row survives, new col present.
    rid = fs.record_prediction("t2", 2, "intent", "v2 row", "f", 0.5, "next_turn",
                               now_iso="x", prediction_set_json={"referents": ["a"]}, db_path=db)
    assert rid > 0
    c1 = fs._connect(db)
    try:
        cols = fs._column_names(c1)
    finally:
        c1.close()
    assert "prediction_set_json" in cols
    # the legacy v1 row reads back with a None set (settles 'unresolved' downstream)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    old = conn.execute("SELECT * FROM prediction_traces WHERE claim='v1 row'").fetchone()
    conn.close()
    assert old["prediction_set_json"] is None
