"""Tests for the Affect branch (B3) — valence/arousal readings + recency profile.

Affect models a person (does not rank, has no truth value): immutable readings
accumulate into a recency-weighted, decaying profile. user-sourced is gold.
"""
from __future__ import annotations

import pytest

from core.acatalepsy.affect import extract_affect, append_reading, affect_profile
from core.acatalepsy.normalize import parse_triple


def _aff(form):
    return extract_affect(parse_triple(form))


def test_extract_negative_affect():
    v, a, i = _aff("user | dislikes | safetywrapper")
    assert v < 0 and a > 0 and i > 0


def test_extract_positive_high_arousal():
    v, a, i = _aff("e | loves | precision")
    assert v > 0.5 and a > 0.5


def test_extract_none_for_non_affect_relation():
    assert extract_affect(parse_triple("trump | president_of | usa")) is None
    assert extract_affect(None) is None


def _migrated(tmp_path, monkeypatch):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "aff.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema
    schema.migrate()
    return _dbc


def test_profile_is_recency_weighted(tmp_path, monkeypatch):
    dbc = _migrated(tmp_path, monkeypatch)
    from datetime import datetime, timezone, timedelta
    now = datetime(2026, 5, 30, tzinfo=timezone.utc)
    old = (now - timedelta(days=60)).isoformat()
    recent = (now - timedelta(hours=1)).isoformat()
    conn = dbc.connect_acatalepsy(role="migration")
    append_reading(conn, subject="user", valence=-0.9, arousal=0.8, source="user", ts=old)
    append_reading(conn, subject="user", valence=0.9, arousal=0.5, source="user", ts=recent)
    conn.commit()
    prof = affect_profile("user", conn, now=now.isoformat())
    conn.close()
    assert prof is not None and prof["n"] == 2
    assert prof["valence"] > 0   # recent positive dominates the 60-day-old negative


def test_profile_scan_selects_by_insertion_id_not_wall_clock(tmp_path, monkeypatch):
    """When-plane fix: the bounded scan must keep the most recent readings by
    monotonic insertion id, not by wall-clock ts. If a later reading is written
    with an EARLIER ts (a backward clock step), it must still win the scan
    window instead of being knocked out by a stale-but-later-stamped row."""
    dbc = _migrated(tmp_path, monkeypatch)
    from datetime import datetime, timezone, timedelta
    now = datetime(2026, 5, 30, tzinfo=timezone.utc)
    later_ts = now.isoformat()
    earlier_ts = (now - timedelta(days=3)).isoformat()
    conn = dbc.connect_acatalepsy(role="migration")
    # Reading A inserted FIRST, stamped with the LATER ts.
    append_reading(conn, subject="user", valence=-1.0, arousal=0.1, source="user", ts=later_ts)
    # Reading B inserted SECOND (higher id), stamped with an EARLIER ts — i.e.
    # the wall clock stepped backward between the two writes.
    append_reading(conn, subject="user", valence=1.0, arousal=0.9, source="user", ts=earlier_ts)
    conn.commit()
    # Bound the scan to one row: the genuinely-newest reading (B, highest id)
    # must win. Under the old ORDER BY ts DESC, A (later ts) would win instead.
    prof = affect_profile("user", conn, now=now.isoformat(), scan_limit=1)
    conn.close()
    assert prof is not None and prof["n"] == 1
    assert prof["valence"] > 0   # reading B (insertion-newest) selected, not A


def test_profile_none_when_no_readings(tmp_path, monkeypatch):
    dbc = _migrated(tmp_path, monkeypatch)
    conn = dbc.connect_acatalepsy(role="migration")
    prof = affect_profile("nobody", conn)
    conn.close()
    assert prof is None


def test_intake_emotional_claim_records_a_reading(tmp_path, monkeypatch):
    dbc = _migrated(tmp_path, monkeypatch)
    from core.acatalepsy import intake, canonical_log
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)
    intake.ingest_l1(raw_form="user | dislikes | safetywrapper", provenance="user")
    conn = dbc.connect_acatalepsy(role="migration")
    rows = conn.execute(
        "SELECT subject, valence, target FROM affect_readings WHERE subject='user'").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["valence"] < 0
    assert rows[0]["target"] == "safetywrapper"


def test_reasserting_emotional_claim_adds_to_the_trajectory(tmp_path, monkeypatch):
    dbc = _migrated(tmp_path, monkeypatch)
    from core.acatalepsy import intake, canonical_log
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)
    intake.ingest_l1(raw_form="user | dislikes | safetywrapper", provenance="user")  # novel
    intake.ingest_l1(raw_form="user | dislikes | safetywrapper", provenance="user")  # match
    conn = dbc.connect_acatalepsy(role="migration")
    n = conn.execute(
        "SELECT COUNT(*) FROM affect_readings WHERE subject='user'").fetchone()[0]
    conn.close()
    assert n == 2   # each assertion is a point in the trajectory
