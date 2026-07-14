"""Tests for EQID v1 — deterministic inverse-relation equivalence grouping."""
from __future__ import annotations

import pytest

from core.acatalepsy import eqid


def test_inverse_pair_shares_eqid():
    a = eqid.compute_eqid_for_form("paris | capital_of | france")
    b = eqid.compute_eqid_for_form("france | has_capital | paris")
    assert a is not None and a == b


def test_non_inverse_relation_does_not_group_reversed():
    # 'likes' has no inverse and is not symmetric -> "a likes b" != "b likes a".
    a = eqid.compute_eqid_for_form("alice | likes | bob")
    b = eqid.compute_eqid_for_form("bob | likes | alice")
    assert a is not None and b is not None and a != b


def test_symmetric_relation_is_entity_order_independent():
    a = eqid.compute_eqid_for_form("alice | sibling_of | bob")
    b = eqid.compute_eqid_for_form("bob | sibling_of | alice")
    assert a is not None and a == b


def test_unknown_relation_is_singleton_not_grouped():
    a = eqid.compute_eqid_for_form("x | frobnicates | y")
    b = eqid.compute_eqid_for_form("y | frobnicates | x")
    assert a is not None and b is not None and a != b


def test_qualifiers_distinguish_eqid():
    a = eqid.compute_eqid_for_form("trump | president_of | usa | 2017")
    b = eqid.compute_eqid_for_form("trump | president_of | usa | 2005")
    assert a is not None and b is not None and a != b


def test_unparseable_form_returns_none():
    assert eqid.compute_eqid(None) is None
    assert eqid.compute_eqid_for_form("just some prose, not a triple") is None


def test_map_version_is_part_of_hash(monkeypatch):
    before = eqid.compute_eqid_for_form("paris | capital_of | france")
    monkeypatch.setattr(eqid, "EQID_MAP_VERSION", eqid.EQID_MAP_VERSION + 1)
    after = eqid.compute_eqid_for_form("paris | capital_of | france")
    assert before != after


# ── DB-backed: assignment at crystallization + backfill ────────────────

def _setup(tmp_path, monkeypatch):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "eqid.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, canonical_log
    schema.migrate()
    tl = getattr(canonical_log, "_tl", None)
    if tl is not None:
        for a in ("writer_conn", "reader_conn"):
            if hasattr(tl, a):
                delattr(tl, a)
    return _dbc


def _insert_l1(dbc, canonical, *, kind="world-fact", provenance="world"):
    import json
    from core.acatalepsy.normalize import normalize_canonical, parse_triple
    cf = normalize_canonical(canonical)
    t = parse_triple(cf)
    ct = json.dumps({"entity_a": t.entity_a, "relation": t.relation,
                     "entity_b": t.entity_b, "qualifiers": t.qualifiers}) if t else None
    span = json.dumps([{"text": canonical, "provenance": provenance, "ts": "now"}])
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        cur = conn.execute(
            "INSERT INTO acus(canonical, source, provenance, kind, l_level, reinforcement, "
            "evidence_spans, canonical_triple, state, cf_version, created_at, last_seen) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (cf, provenance, provenance, kind, "L1", 1, span, ct, "active", 1, "now", "now"))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _row(dbc, acu_id):
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        return conn.execute("SELECT cid, eqid FROM acus WHERE id=?", (acu_id,)).fetchone()
    finally:
        conn.close()


def _crystallize(dbc, acu_id):
    from core.acatalepsy import crystallize
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        res = crystallize.crystallize(acu_id, conn=conn)
        conn.commit()
        return res
    finally:
        conn.close()


def test_crystallize_assigns_eqid(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert_l1(dbc, "paris | capital_of | france")
    _crystallize(dbc, aid)
    row = _row(dbc, aid)
    assert row["cid"] is not None
    assert row["eqid"] == eqid.compute_eqid_for_form("paris | capital_of | france")


def test_two_inverse_claims_get_distinct_cid_same_eqid(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    a = _insert_l1(dbc, "paris | capital_of | france")
    b = _insert_l1(dbc, "france | has_capital | paris")
    _crystallize(dbc, a)
    _crystallize(dbc, b)
    ra, rb = _row(dbc, a), _row(dbc, b)
    assert ra["cid"] != rb["cid"]      # distinct identities (grouped, not collapsed)
    assert ra["eqid"] == rb["eqid"]    # one equivalence group


def _insert_crystallized(dbc, canonical, cid):
    from core.acatalepsy.normalize import normalize_canonical
    cf = normalize_canonical(canonical)
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        cur = conn.execute(
            "INSERT INTO acus(canonical, source, provenance, kind, l_level, reinforcement, "
            "cid, eqid, state, cf_version, created_at, last_seen) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (cf, "world", "world", "world-fact", "L2", 1, cid, None, "active", 1, "now", "now"))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _backfill(dbc):
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        n = eqid.backfill_eqids(conn)
        conn.commit()
        return n
    finally:
        conn.close()


def test_backfill_assigns_eqid_to_crystallized_rows(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert_crystallized(dbc, "paris | capital_of | france", "cid:sha256:aaa")
    assert _backfill(dbc) == 1
    assert _row(dbc, aid)["eqid"] == eqid.compute_eqid_for_form("paris | capital_of | france")


def test_backfill_skips_l1_stubs_and_is_idempotent(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    l1 = _insert_l1(dbc, "berlin | capital_of | germany")           # cid IS NULL
    _insert_crystallized(dbc, "rome | capital_of | italy", "cid:sha256:bbb")
    assert _backfill(dbc) == 1          # only the crystallized row
    assert _row(dbc, l1)["eqid"] is None
    assert _backfill(dbc) == 0          # idempotent — nothing left to do
