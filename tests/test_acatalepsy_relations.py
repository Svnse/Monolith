"""Tests for core/acatalepsy/relations.py — the acu_relations read primitives
(the reader half of the contradiction/overlap graph open loop). Each read does a
single LEFT JOIN acus (twice) to resolve both endpoints in one query.

See docs/superpowers/specs/2026-06-09-acu-relations-monosearch-adapter-design.md.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    """Isolated acatalepsy DB with two ACUs + edges (contradicts, overlaps, and
    one dangling edge whose target ACU does not exist)."""
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acu.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, attr):
                    delattr(tl, attr)

    from core.acu_store import ACUStore
    store = ACUStore()
    a_id = store.ingest("sky | is | blue", source="model")
    b_id = store.ingest("sky | is | green", source="model")
    locked_id = store.ingest_locked("origin | values | precision")
    assert a_id >= 1 and b_id >= 1 and locked_id >= 1

    missing_id = 99999  # no ACU with this id -> dangling edge
    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        conn.execute(
            "INSERT INTO acu_relations(source_id, target_id, relation, score, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (a_id, b_id, "contradicts", 0.8, "2026-06-02T18:00:00+00:00", None),
        )
        conn.execute(
            "INSERT INTO acu_relations(source_id, target_id, relation, score, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (a_id, locked_id, "overlaps", 1.0, "2026-06-02T18:05:00+00:00", None),
        )
        conn.execute(
            "INSERT INTO acu_relations(source_id, target_id, relation, score, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (a_id, missing_id, "contradicts", 0.5, "2026-06-02T18:10:00+00:00", None),
        )
        conn.commit()
    finally:
        conn.close()
    return {"a": a_id, "b": b_id, "locked": locked_id, "missing": missing_id}


def test_read_recent_resolves_both_endpoints(seeded):
    from core.acatalepsy import relations
    rows = relations.read_recent(limit=10)
    by_rel = {(r["relation"], r["source_id"], r["target_id"]): r for r in rows}
    edge = by_rel[("contradicts", seeded["a"], seeded["b"])]
    assert edge["source_canonical"] == "sky | is | blue"
    assert edge["target_canonical"] == "sky | is | green"
    assert edge["score"] == 0.8
    assert edge["source_state"] == "active"
    assert edge["target_state"] == "active"


def test_read_recent_relation_filter(seeded):
    from core.acatalepsy import relations
    rows = relations.read_recent(limit=10, relation="overlaps")
    assert rows  # at least the overlaps edge
    assert all(r["relation"] == "overlaps" for r in rows)


def test_read_recent_orders_desc_by_id(seeded):
    from core.acatalepsy import relations
    rows = relations.read_recent(limit=10)
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids, reverse=True)


def test_dangling_edge_yields_null_target_canonical(seeded):
    # The read primitive faithfully LEFT-JOINs: a missing endpoint -> NULL canonical.
    # (The SKIP decision lives in the adapter, not here.)
    from core.acatalepsy import relations
    rows = relations.read_recent(limit=10)
    dangling = [r for r in rows if r["target_id"] == seeded["missing"]]
    assert len(dangling) == 1
    assert dangling[0]["target_canonical"] is None


def test_locked_endpoint_flag_surfaced(seeded):
    from core.acatalepsy import relations
    rows = relations.read_recent(limit=10)
    edge = next(r for r in rows if r["target_id"] == seeded["locked"])
    assert edge["target_locked"] == 1
    assert edge["target_canonical"] == "origin | values | precision"


def test_read_one_by_edge_id(seeded):
    from core.acatalepsy import relations
    rows = relations.read_recent(limit=10)
    target = next(r for r in rows if r["relation"] == "contradicts" and r["target_id"] == seeded["b"])
    one = relations.read_one(target["id"])
    assert one is not None
    assert one["id"] == target["id"]
    assert one["source_canonical"] == "sky | is | blue"
    assert relations.read_one(123456) is None


def test_read_since_id_ascending(seeded):
    from core.acatalepsy import relations
    rows = relations.read_since_id(0, limit=10)
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids)  # ascending (salience.rebuild iteration path)
    # >= 3: the three edges seeded here, PLUS any organic overlaps edge intake
    # auto-writes on the PARTIAL-match ingest of "sky | is | green" (the live
    # producer doing its job — not all edges in the table are hand-seeded).
    assert len(ids) >= 3
