"""Tests for the acatalepsy-relations adapter (acu_relations contradiction/overlap
graph → MonoSearch). See docs/superpowers/specs/2026-06-09-acu-relations-monosearch-adapter-design.md.

Two flavours (mirrors test_acus_adapter.py):
  * pure mapping unit tests — hand-built JOIN-shaped rows → _to_record (render,
    tiers, recurrence_key, absent-endpoint skip, locked-endpoint render);
  * one REAL end-to-end against a seeded isolated acatalepsy DB — seed two ACUs +
    a contradicts + an overlaps edge via direct SQL, then exercise the real
    read primitives through get/search/list.
"""
from __future__ import annotations

from core.monosearch.adapters.acu_relations import RelationsAdapter
from core.monosearch.record import EvidenceTier, Provenance


def _rrow(**over):
    """A full JOIN-shaped relation row (what relations.read_* yields)."""
    base = {
        "id": 5,
        "source_id": 12,
        "target_id": 34,
        "relation": "contradicts",
        "score": 0.8,
        "created_at": "2026-06-02T18:00:00+00:00",
        "updated_at": None,
        "source_canonical": "sky | is | blue",
        "source_locked": 0,
        "source_state": "active",
        "target_canonical": "sky | is | green",
        "target_locked": 0,
        "target_state": "active",
    }
    base.update(over)
    return base


# ── pure mapping ──────────────────────────────────────────────────────


def test_to_record_shape():
    a = RelationsAdapter()
    r = a._to_record(_rrow())
    assert r is not None
    assert r.namespaced_id == "relation:5"
    assert r.source == "acatalepsy-relations"
    assert r.evidence_tier == EvidenceTier.DERIVED
    assert r.provenance is Provenance.SELF
    # not a recurring unit -> not salience-eligible
    assert r.recurrence_key is None
    assert r.text == "#12 sky | is | blue --[contradicts]--> #34 sky | is | green"
    assert r.ts == 1780423200.0  # 2026-06-02T18:00:00Z -> epoch
    assert r.metadata["relation"] == "contradicts"
    assert r.metadata["source_id"] == 12
    assert r.metadata["target_id"] == 34
    assert r.metadata["score"] == 0.8
    assert r.metadata["source_state"] == "active"
    assert r.metadata["target_state"] == "active"


def test_to_record_renders_overlaps():
    a = RelationsAdapter()
    r = a._to_record(_rrow(relation="overlaps"))
    assert r is not None
    assert r.text == "#12 sky | is | blue --[overlaps]--> #34 sky | is | green"
    assert r.metadata["relation"] == "overlaps"


def test_to_record_skips_absent_endpoint():
    # LEFT JOIN miss (endpoint ACU deleted/missing) -> NULL canonical -> skip.
    a = RelationsAdapter()
    assert a._to_record(_rrow(target_canonical=None)) is None
    assert a._to_record(_rrow(source_canonical=None)) is None


def test_to_record_renders_locked_endpoint():
    # A locked Origin-0 endpoint is RENDERED (namespaced_id is relation:, not acu:,
    # so we name it in a relationship, we don't double-serve the locked ACU). The
    # locked state is surfaced in metadata, not hidden.
    a = RelationsAdapter()
    r = a._to_record(_rrow(source_locked=1))
    assert r is not None
    assert r.metadata["source_locked"] is True
    assert r.metadata["target_locked"] is False


# ── REAL end-to-end against a seeded isolated acatalepsy DB ────────────


import pytest


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    """Two ACUs + a locked ACU; a contradicts edge (a->b), an overlaps edge to the
    locked ACU, and a dangling edge (target ACU missing). Returns ids."""
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
    a_id = store.ingest("alpha | is | one", source="model")
    b_id = store.ingest("beta | is | two", source="model")
    locked_id = store.ingest_locked("origin | values | precision")
    missing_id = 99999

    edges = {}
    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        cur = conn.execute(
            "INSERT INTO acu_relations(source_id, target_id, relation, score, created_at, updated_at) VALUES(?,?,?,?,?,?)",
            (a_id, b_id, "contradicts", 0.8, "2026-06-02T18:00:00+00:00", None),
        )
        edges["contradicts"] = int(cur.lastrowid)
        cur = conn.execute(
            "INSERT INTO acu_relations(source_id, target_id, relation, score, created_at, updated_at) VALUES(?,?,?,?,?,?)",
            (a_id, locked_id, "overlaps", 1.0, "2026-06-02T18:05:00+00:00", None),
        )
        edges["overlaps_locked"] = int(cur.lastrowid)
        cur = conn.execute(
            "INSERT INTO acu_relations(source_id, target_id, relation, score, created_at, updated_at) VALUES(?,?,?,?,?,?)",
            (a_id, missing_id, "contradicts", 0.5, "2026-06-02T18:10:00+00:00", None),
        )
        edges["dangling"] = int(cur.lastrowid)
        conn.commit()
    finally:
        conn.close()
    return {"a": a_id, "b": b_id, "locked": locked_id, "missing": missing_id, "edges": edges}


def test_e2e_get_resolves_edge(seeded):
    a = RelationsAdapter()
    r = a.get(f"relation:{seeded['edges']['contradicts']}")
    assert r is not None
    assert r.text == "#1 alpha | is | one --[contradicts]--> #2 beta | is | two"
    assert r.evidence_tier == EvidenceTier.DERIVED
    assert a.get("relation:abc") is None        # unparseable
    assert a.get("acu:1") is None               # wrong namespace


def test_e2e_get_skips_dangling_edge(seeded):
    a = RelationsAdapter()
    # The dangling edge exists in the table, but its target ACU is gone -> skip.
    assert a.get(f"relation:{seeded['edges']['dangling']}") is None


def test_e2e_search_query_post_filters_and_skips_dangling(seeded):
    a = RelationsAdapter()
    hits = a.search("beta", {}, 50)
    texts = [r.text for r in hits]
    assert any("beta | is | two" in t for t in texts)        # contradicts a->b matches
    ids = {r.namespaced_id for r in hits}
    assert f"relation:{seeded['edges']['dangling']}" not in ids  # dangling skipped


def test_e2e_search_relation_filter(seeded):
    a = RelationsAdapter()
    hits = a.search("", {"relation": "overlaps"}, 50)
    assert hits
    assert all(r.metadata["relation"] == "overlaps" for r in hits)


def test_e2e_list_includes_locked_excludes_dangling(seeded):
    a = RelationsAdapter()
    recs = a.list({}, 100)
    ids = {r.namespaced_id for r in recs}
    assert f"relation:{seeded['edges']['overlaps_locked']}" in ids   # locked endpoint rendered
    assert f"relation:{seeded['edges']['dangling']}" not in ids      # dangling skipped
    locked_rec = next(r for r in recs if r.namespaced_id == f"relation:{seeded['edges']['overlaps_locked']}")
    assert locked_rec.metadata["target_locked"] is True
