"""Tests for the L1 comparison pass / one-writer intake (the spine keystone).

MATCH/PARTIAL/NOVEL is decided purely by normalizer output (no Kind branch —
it doesn't exist until L2). MATCH reinforces (provenance-weighted) without
touching veracity; PARTIAL induces exactly one neutral `overlaps` edge; NOVEL
creates a stub with a reconstruction-complete event payload.
"""
from __future__ import annotations

import json

import pytest


def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "intake.sqlite3"
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", db_path, raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema
    schema.migrate()
    from core.acatalepsy import canonical_log, intake
    for mod in (canonical_log, intake):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, attr):
                    delattr(tl, attr)
    return _dbc


def _row(dbc, acu_id):
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        return conn.execute("SELECT * FROM acus WHERE id=?", (acu_id,)).fetchone()
    finally:
        conn.close()


def _edges(dbc):
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        return conn.execute("SELECT * FROM acu_relations").fetchall()
    finally:
        conn.close()


# ── NOVEL ─────────────────────────────────────────────────────────────

def test_novel_creates_l1_stub_with_structured_span(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    # self-provenance stays L1 (Mad Cow) — so we can inspect the raw stub.
    res = intake.ingest_l1(raw_form="trump | born_in | nyc", provenance="self")
    assert res.outcome == "novel"
    row = _row(dbc, res.acu_id)
    assert row["l_level"] == "L1"
    assert row["cid"] is None
    assert row["provenance"] == "self"
    assert row["reinforcement"] == 1  # self weight
    spans = json.loads(row["evidence_spans"])
    assert len(spans) == 1 and spans[0]["provenance"] == "self" and "ts" in spans[0]


def test_novel_event_payload_is_reconstruction_complete(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake, canonical_log
    # self -> stays L1 so l1_novel_survive is the last event (no crystallize after).
    res = intake.ingest_l1(raw_form="a | r | b", provenance="self")
    ev = canonical_log.read_one(canonical_log.latest_event_id())
    assert ev.kind == "l1_novel_survive"
    p = ev.payload
    # Enough to rebuild the row without joining other tables.
    for key in ("acu_id", "canonical_form", "provenance", "cf_version", "reinforcement", "span"):
        assert key in p
    assert p["canonical_form"] == "a | r | b"


# ── MATCH ─────────────────────────────────────────────────────────────

def test_match_reinforces_without_touching_veracity(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    # Distinct evidence each time -> count increments AND both spans append.
    r1 = intake.ingest_l1(raw_form="water | is | wet", provenance="user",
                          evidence_span="said in session 1")
    r2 = intake.ingest_l1(raw_form="water | is | wet", provenance="user",
                          evidence_span="said in session 2")
    assert r2.outcome == "match"
    assert r1.acu_id == r2.acu_id
    row = _row(dbc, r1.acu_id)
    assert row["reinforcement"] == 4          # 2 + 2 (provenance-weighted)
    assert row["veracity"] == 5.0             # untouched
    spans = json.loads(row["evidence_spans"])
    assert len(spans) == 2                    # distinct spans both kept


def test_match_dedups_identical_evidence_but_still_counts(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    # Identical evidence text twice: count increments, but the span is not
    # duplicated (one piece of evidence seen twice).
    intake.ingest_l1(raw_form="sky | is | blue", provenance="self")
    r2 = intake.ingest_l1(raw_form="sky | is | blue", provenance="self")
    row = _row(dbc, r2.acu_id)
    assert row["reinforcement"] == 2
    assert len(json.loads(row["evidence_spans"])) == 1


def test_span_carries_source_event_key(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    r_ev = intake.ingest_l1(raw_form="a | r | b", provenance="self", source_event=7)
    r_none = intake.ingest_l1(raw_form="c | r | d", provenance="self")  # no source_event
    s_ev = json.loads(_row(dbc, r_ev.acu_id)["evidence_spans"])[0]
    s_none = json.loads(_row(dbc, r_none.acu_id)["evidence_spans"])[0]
    assert s_ev["source_event"] == 7
    assert s_none["source_event"] is None        # None-safe: key present, value None


def test_distinct_source_events_accumulate_distinct_spans(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    # Same self claim asserted on 3 distinct occasions, then a repeat of one occasion.
    for ev in (101, 102, 103):
        intake.ingest_l1(raw_form="monolith | values | honesty", provenance="self",
                         source_event=ev)
    r = intake.ingest_l1(raw_form="monolith | values | honesty", provenance="self",
                         source_event=103)   # dup occasion -> does NOT add a span
    spans = json.loads(_row(dbc, r.acu_id)["evidence_spans"])
    assert {s.get("source_event") for s in spans} == {101, 102, 103}
    assert len(spans) == 3                        # 3 distinct occasions
    # Raw reinforcement counts EVERY assertion (4) — proving distinct-event count is a
    # separate, non-gameable signal from raw reinforcement (the self->L2 rationale).
    assert _row(dbc, r.acu_id)["reinforcement"] == 4


def test_same_source_event_repeated_dedups_span(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    for _ in range(5):
        r = intake.ingest_l1(raw_form="monolith | is | careful", provenance="self",
                             source_event=42)    # same occasion 5x
    spans = json.loads(_row(dbc, r.acu_id)["evidence_spans"])
    assert len(spans) == 1                        # one occasion -> one span
    assert {s.get("source_event") for s in spans} == {42}


def test_match_is_normalized_not_exact(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    r1 = intake.ingest_l1(raw_form="A | r | B", provenance="self")
    r2 = intake.ingest_l1(raw_form="  a |  r | b. ", provenance="self")
    assert r2.outcome == "match"
    assert r1.acu_id == r2.acu_id


def test_match_populates_legacy_null_triple(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    # Simulate a pre_cid_legacy row with NULL canonical_triple.
    conn = dbc.connect_acatalepsy(role="migration")
    conn.execute(
        "INSERT INTO acus(canonical, source, provenance, l_level, reinforcement, "
        "evidence_spans, canonical_triple, pre_cid_legacy, created_at, last_seen) "
        "VALUES('legacy | claim | here','model','self','L1',1,'[]',NULL,1,'now','now')")
    conn.commit()
    conn.close()
    res = intake.ingest_l1(raw_form="legacy | claim | here", provenance="user")
    assert res.outcome == "match"
    row = _row(dbc, res.acu_id)
    assert row["canonical_triple"] is not None
    assert json.loads(row["canonical_triple"])["entity_a"] == "legacy"


# ── PARTIAL ───────────────────────────────────────────────────────────

def test_partial_induces_one_overlaps_edge(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    first = intake.ingest_l1(raw_form="trump | born_in | nyc", provenance="user")
    res = intake.ingest_l1(raw_form="trump | president_of | usa", provenance="user")
    assert res.outcome == "partial"
    assert res.edge_id is not None
    edges = _edges(dbc)
    assert len(edges) == 1
    assert edges[0]["source_id"] == res.acu_id
    assert edges[0]["target_id"] == first.acu_id
    assert edges[0]["relation"] == "overlaps"
    assert edges[0]["axis_tags"] is None       # no edge-typing in the spine


# ── locked rows (B1: must not duplicate) ──────────────────────────────

def test_locked_match_does_not_reinforce_or_duplicate(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    conn = dbc.connect_acatalepsy(role="migration")
    conn.execute(
        "INSERT INTO acus(canonical, source, provenance, l_level, reinforcement, "
        "locked, evidence_spans, created_at, last_seen) "
        "VALUES('origin | is | locked','identity_origin_0','user','L1',1,1,'[]','now','now')")
    conn.commit()
    conn.close()
    res = intake.ingest_l1(raw_form="origin | is | locked", provenance="user")
    assert res.outcome == "match"
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM acus WHERE canonical='origin | is | locked'").fetchone()[0]
        r = conn.execute(
            "SELECT reinforcement FROM acus WHERE canonical='origin | is | locked'").fetchone()[0]
    finally:
        conn.close()
    assert n == 1   # not duplicated
    assert r == 1   # locked: immutable, not reinforced


# ── atomicity reject ──────────────────────────────────────────────────

def test_non_atomic_form_is_rejected(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    res = intake.ingest_l1(raw_form="a | likes | b and c", provenance="user")
    assert res.outcome == "rejected"
    assert res.acu_id == -1


# ── Kind + crystallization trigger (Phase 2) ──────────────────────────

def test_world_fact_crystallizes_on_intake(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    res = intake.ingest_l1(raw_form="trump | president_of | usa", provenance="world")
    row = _row(dbc, res.acu_id)
    assert row["kind"] == "world-fact"
    assert row["l_level"] == "L2"          # Mad Cow passes (world) -> crystallizes
    assert row["cid"] is not None
    assert row["promoted_to_l2_ts"]


def test_self_claim_stays_l1_until_confirmed(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    res = intake.ingest_l1(raw_form="monolith | uses | seven tiers", provenance="self")
    row = _row(dbc, res.acu_id)
    assert row["kind"] == "self"
    assert row["l_level"] == "L1"          # self can't self-promote
    assert row["cid"] is None


def test_self_crystallizes_after_user_reinforcement(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    r1 = intake.ingest_l1(raw_form="monolith | has | kill switch", provenance="self")
    assert _row(dbc, r1.acu_id)["l_level"] == "L1"
    r2 = intake.ingest_l1(raw_form="monolith | has | kill switch", provenance="user")
    assert r2.acu_id == r1.acu_id
    row = _row(dbc, r1.acu_id)
    assert row["l_level"] == "L2"          # user confirmation -> crystallizes
    assert row["cid"] is not None


def test_legacy_null_kind_row_classified_and_crystallizes_on_user_match(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    conn = dbc.connect_acatalepsy(role="migration")
    conn.execute(
        "INSERT INTO acus(canonical, source, provenance, l_level, reinforcement, "
        "evidence_spans, canonical_triple, kind, pre_cid_legacy, created_at, last_seen) "
        "VALUES('paris | capital_of | france','model','self','L1',1,'[]',NULL,NULL,1,'now','now')")
    conn.commit()
    conn.close()
    res = intake.ingest_l1(raw_form="paris | capital_of | france", provenance="user")
    assert res.outcome == "match"
    row = _row(dbc, res.acu_id)
    assert row["kind"] == "world-fact"     # inferred on encounter
    assert row["l_level"] == "L2"          # user match -> crystallizes
    assert row["cid"] is not None


def test_inf_falsehood_recognized_but_not_reinforced(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    conn = dbc.connect_acatalepsy(role="migration")
    conn.execute(
        "INSERT INTO acus(canonical, source, provenance, kind, l_level, reinforcement, "
        "evidence_spans, state, cf_version, created_at, last_seen) "
        "VALUES('trump | president_of | usa | 1800','world','world','world-fact','L2',5,'[]',"
        "'-inf',1,'now','now')")
    conn.commit()
    conn.close()
    res = intake.ingest_l1(raw_form="trump | president_of | usa | 1800", provenance="user")
    assert res.outcome == "match"          # recognized, not duplicated
    row = _row(dbc, res.acu_id)
    assert row["reinforcement"] == 5       # NOT reinforced — known falsehood is immutable
    conn = dbc.connect_acatalepsy(role="migration")
    n = conn.execute(
        "SELECT COUNT(*) FROM acus WHERE canonical='trump | president_of | usa | 1800'"
    ).fetchone()[0]
    conn.close()
    assert n == 1


# ── self -> L2 through the REAL intake trigger (integration) ──────────
# The unit tests exercise can_crystallize directly; these drive the production
# path ingest_l1 -> _run_intake MATCH -> _append_span -> _maybe_crystallize ->
# is_crystallize_eligible -> can_crystallize -> crystallize. This is where
# "silently does nothing" would hide (span source_event must survive the MATCH
# reinforce for the distinct-count to ever reach 3).

def test_self_reaches_l2_through_ingest_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MONOLITH_SELF_IDENTITY_L2_V1", "1")
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    r1 = intake.ingest_l1(raw_form="monolith | values | honesty", provenance="self",
                          source_event=1)
    r2 = intake.ingest_l1(raw_form="monolith | values | honesty", provenance="self",
                          source_event=2)
    row2 = _row(dbc, r2.acu_id)
    assert row2["l_level"] == "L1" and row2["cid"] is None    # 2 distinct < 3 -> still L1
    r3 = intake.ingest_l1(raw_form="monolith | values | honesty", provenance="self",
                          source_event=3)
    row3 = _row(dbc, r3.acu_id)
    assert row3["l_level"] == "L2"                            # crystallized via real trigger
    assert row3["cid"] is not None
    assert r1.acu_id == r2.acu_id == r3.acu_id


def test_self_l2_inert_through_ingest_when_flag_off(tmp_path, monkeypatch):
    # Flag off (dark default): even 3 distinct occasions via the real trigger stay L1.
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    for ev in (1, 2, 3):
        r = intake.ingest_l1(raw_form="monolith | values | honesty", provenance="self",
                             source_event=ev)
    row = _row(dbc, r.acu_id)
    assert row["l_level"] == "L1" and row["cid"] is None


def test_self_verbatim_repetition_through_ingest_never_reaches_l2(tmp_path, monkeypatch):
    # The anti-gaming guarantee at the integration level: same occasion 6x (one
    # source_event) reinforces but never crystallizes, even with the flag on.
    monkeypatch.setenv("MONOLITH_SELF_IDENTITY_L2_V1", "1")
    dbc = _setup(tmp_path, monkeypatch)
    from core.acatalepsy import intake
    for _ in range(6):
        r = intake.ingest_l1(raw_form="monolith | is | great", provenance="self",
                             source_event=99)
    row = _row(dbc, r.acu_id)
    assert row["l_level"] == "L1" and row["cid"] is None
    assert row["reinforcement"] == 6                          # counted, but not crystallized
