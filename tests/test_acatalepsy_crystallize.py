"""Tests for CID assignment + crystallization + the Mad Cow gate.

CID = hash(canonical_form + cf_version) ONLY — never provenance/l_level/
temporal. Crystallization is the L1->L2 (hot->cold) phase boundary. Mad Cow:
a self-provenance claim cannot crystallize without user/world confirmation.
"""
from __future__ import annotations

import json

import pytest


def _migrated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "cryst.sqlite3"
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", db_path, raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema
    schema.migrate()
    return _dbc


def _insert_l1(dbc, canonical, provenance, *, spans=None, reinforcement=1):
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        cur = conn.execute(
            "INSERT INTO acus(canonical, source, provenance, l_level, reinforcement, "
            "evidence_spans, created_at, last_seen) VALUES(?,?,?,?,?,?,?,?)",
            (canonical, provenance, provenance, "L1", reinforcement,
             json.dumps(spans or []), "now", "now"),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


# ── compute_cid ───────────────────────────────────────────────────────

def test_compute_cid_is_deterministic():
    from core.acatalepsy.crystallize import compute_cid
    assert compute_cid("a | r | b") == compute_cid("a | r | b")
    assert compute_cid("a | r | b").startswith("cid:sha256:")


def test_compute_cid_changes_with_cf_version():
    from core.acatalepsy.crystallize import compute_cid
    assert compute_cid("a | r | b", 1) != compute_cid("a | r | b", 2)


# ── crystallize (L1 -> L2) ────────────────────────────────────────────

def test_crystallize_promotes_l1_to_l2(tmp_path, monkeypatch):
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    acu_id = _insert_l1(dbc, "trump | president_of | usa", "user")
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        res = C.crystallize(acu_id, conn=conn)
        conn.commit()
        row = conn.execute(
            "SELECT cid, l_level, promoted_to_l2_ts FROM acus WHERE id=?", (acu_id,)
        ).fetchone()
    finally:
        conn.close()
    assert res.crystallized is True
    assert res.collided_with is None
    assert row["cid"] == C.compute_cid("trump | president_of | usa")
    assert row["l_level"] == "L2"
    assert row["promoted_to_l2_ts"]


def test_cid_independent_of_provenance_collision_routes_to_existing(tmp_path, monkeypatch):
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    # Same canonical form, different provenance — must mint the SAME cid,
    # so the second crystallize collides and routes to the first (no fork).
    a = _insert_l1(dbc, "x | r | y", "user")
    b = _insert_l1(dbc, "x | r | y", "world")
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        ra = C.crystallize(a, conn=conn)
        rb = C.crystallize(b, conn=conn)
        conn.commit()
        rowb = conn.execute("SELECT cid, l_level, merged_into, state FROM acus WHERE id=?", (b,)).fetchone()
    finally:
        conn.close()
    assert ra.cid == rb.cid                 # provenance does not affect identity
    assert rb.crystallized is False
    assert rb.collided_with == a
    assert rowb["merged_into"] == a
    assert rowb["state"] == "archived"


def test_collision_transfers_reinforcement_to_survivor(tmp_path, monkeypatch):
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    a = _insert_l1(dbc, "x | r | y", "user", reinforcement=2)
    b = _insert_l1(dbc, "x | r | y", "world", reinforcement=5)
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        C.crystallize(a, conn=conn)
        rb = C.crystallize(b, conn=conn)
        conn.commit()
        surv = conn.execute("SELECT reinforcement FROM acus WHERE id=?", (a,)).fetchone()
    finally:
        conn.close()
    assert rb.collided_with == a
    assert surv["reinforcement"] == 7   # 2 + 5 transferred, no signal lost


# ── Mad Cow gate ──────────────────────────────────────────────────────

def test_self_only_cannot_crystallize(tmp_path, monkeypatch):
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    acu_id = _insert_l1(dbc, "model | guesses | thing", "self",
                        spans=[{"text": "...", "provenance": "self", "ts": "now"}])
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        assert C.can_crystallize(acu_id, conn) is False
        with pytest.raises(C.MadCowError):
            C.crystallize(acu_id, conn=conn)
    finally:
        conn.close()


def test_self_with_user_evidence_can_crystallize(tmp_path, monkeypatch):
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    acu_id = _insert_l1(dbc, "model | inferred | thing", "self",
                        spans=[{"text": "a", "provenance": "self", "ts": "now"},
                               {"text": "b", "provenance": "user", "ts": "now"}])
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        assert C.can_crystallize(acu_id, conn) is True
        res = C.crystallize(acu_id, conn=conn)
        conn.commit()
    finally:
        conn.close()
    assert res.crystallized is True


def test_reinforcement_count_alone_does_not_enable_crystallization(tmp_path, monkeypatch):
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    # High reinforcement but all-self evidence: must NOT be crystallizable.
    acu_id = _insert_l1(dbc, "model | repeats | itself", "self",
                        spans=[{"text": "x", "provenance": "self", "ts": "now"}],
                        reinforcement=99)
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        assert C.can_crystallize(acu_id, conn) is False
    finally:
        conn.close()


def test_user_provenance_can_crystallize_without_spans(tmp_path, monkeypatch):
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    acu_id = _insert_l1(dbc, "user | stated | fact", "user")
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        assert C.can_crystallize(acu_id, conn) is True
    finally:
        conn.close()


# ── Self-identity-memory L2 (flag-gated expansion) ────────────────────

def _insert_self(dbc, canonical, *, kind="self", source_events=(), state="active"):
    """Insert a kind=self L1 stub whose spans carry the given distinct source_events."""
    spans = [{"text": f"obs{e}", "provenance": "self", "source_event": e, "ts": "now"}
             for e in source_events]
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        cur = conn.execute(
            "INSERT INTO acus(canonical, source, provenance, kind, l_level, reinforcement, "
            "evidence_spans, state, created_at, last_seen) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (canonical, "self", "self", kind, "L1", max(1, len(source_events)),
             json.dumps(spans), state, "now", "now"),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def test_self_l2_disabled_by_default(tmp_path, monkeypatch):
    # Flag UNSET: a self claim with 3 distinct source_events still cannot crystallize
    # (flag-off path byte-identical to today's Mad Cow).
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    acu_id = _insert_self(dbc, "monolith | values | honesty", source_events=(1, 2, 3))
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        assert C.can_crystallize(acu_id, conn) is False
        with pytest.raises(C.MadCowError):
            C.crystallize(acu_id, conn=conn)
    finally:
        conn.close()


def test_self_crystallizes_at_three_distinct_events_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MONOLITH_SELF_IDENTITY_L2_V1", "1")
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    acu_id = _insert_self(dbc, "monolith | values | honesty", source_events=(11, 22, 33))
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        assert C.can_crystallize(acu_id, conn) is True
        res = C.crystallize(acu_id, conn=conn)
        conn.commit()
        row = conn.execute("SELECT cid, l_level FROM acus WHERE id=?", (acu_id,)).fetchone()
    finally:
        conn.close()
    assert res.crystallized is True
    assert row["l_level"] == "L2"
    assert row["cid"] == C.compute_cid("monolith | values | honesty")


def test_self_below_distinct_threshold_blocked_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MONOLITH_SELF_IDENTITY_L2_V1", "1")
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    acu_id = _insert_self(dbc, "monolith | prefers | brevity", source_events=(1, 2))
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        assert C.can_crystallize(acu_id, conn) is False   # only 2 distinct < 3
    finally:
        conn.close()


def test_self_same_event_repeated_does_not_pass(tmp_path, monkeypatch):
    # Verbatim self-repetition (one occasion, many spans) must NOT pass — the whole
    # point of distinct source_events over raw reinforcement.
    monkeypatch.setenv("MONOLITH_SELF_IDENTITY_L2_V1", "1")
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    acu_id = _insert_self(dbc, "monolith | is | great", source_events=(7, 7, 7, 7, 7))
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        assert C.can_crystallize(acu_id, conn) is False   # distinct = 1
    finally:
        conn.close()


def test_non_self_kind_not_crystallized_by_self_recurrence(tmp_path, monkeypatch):
    # Scope guard: a world-fact with self provenance + 3 distinct events must NOT
    # reach L2 by self-repetition (only kind=self gets the identity-memory path).
    monkeypatch.setenv("MONOLITH_SELF_IDENTITY_L2_V1", "1")
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    acu_id = _insert_self(dbc, "paris | capital_of | france",
                          kind="world-fact", source_events=(1, 2, 3))
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        assert C.can_crystallize(acu_id, conn) is False
    finally:
        conn.close()


def test_self_archived_not_crystallized_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MONOLITH_SELF_IDENTITY_L2_V1", "1")
    dbc = _migrated_db(tmp_path, monkeypatch)
    from core.acatalepsy import crystallize as C
    acu_id = _insert_self(dbc, "monolith | was | wrong",
                          source_events=(1, 2, 3), state="-inf")
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        assert C.can_crystallize(acu_id, conn) is False   # non-active never crystallizes
    finally:
        conn.close()
