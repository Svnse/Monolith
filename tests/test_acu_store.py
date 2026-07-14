"""ACUStore — now a facade over the unified substrate (intake + global DB).

Writes funnel through the one-writer L1 intake (normalized dedup, reinforcement,
atomicity gate); reads rank by reinforcement+recency excluding merged/archived.
`veracity` is dead. Isolation via monkeypatching db_connect.DB_PATH.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acu.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)
    from core.acu_store import ACUStore
    return ACUStore()


def test_ingest_new_atomic_claim(store) -> None:
    row_id = store.ingest("sky | is | blue")
    assert row_id >= 1
    assert store.count() == 1
    acu = store.get_by_id(row_id)
    assert acu["canonical"] == "sky | is | blue"
    assert acu["reinforcement"] == 1
    assert acu["provenance"] == "self"   # source 'model' -> self


def test_ingest_duplicate_reinforces(store) -> None:
    id1 = store.ingest("water | is | wet")
    id2 = store.ingest("water | is | wet")
    assert id1 == id2
    assert store.count() == 1
    assert store.get_by_id(id1)["reinforcement"] == 2


def test_ingest_normalized_dedup(store) -> None:
    id1 = store.ingest("Water | Is | Wet")
    id2 = store.ingest("  water |  is | wet. ")
    assert id1 == id2          # normalized to one form
    assert store.count() == 1


def test_ingest_empty_returns_negative(store) -> None:
    assert store.ingest("") == -1
    assert store.ingest("   ") == -1
    assert store.count() == 0


def test_ingest_non_atomic_rejected(store) -> None:
    # Free-text / non-pipe forms are rejected by the atomicity gate (this is
    # what stopped the legacy conversational junk).
    assert store.ingest("just some free text") == -1
    assert store.ingest("a | likes | b and c") == -1
    assert store.count() == 0


def test_ingest_many(store) -> None:
    ids = store.ingest_many(["a | r | b", "c | r | d", "e | r | f", ""])
    assert len([i for i in ids if i > 0]) == 3
    assert store.count() == 3


def test_retrieve_ranks_by_reinforcement(store) -> None:
    store.ingest("seen | once | x")
    store.ingest("seen | twice | y")
    store.ingest("seen | twice | y")  # reinforced to 2
    results = store.retrieve()
    assert results[0]["canonical"] == "seen | twice | y"
    assert results[1]["canonical"] == "seen | once | x"


def test_retrieve_excludes_archived(store) -> None:
    keep = store.ingest("keep | this | one")
    drop = store.ingest("drop | this | one")
    from core import db_connect as _dbc
    conn = _dbc.connect_acatalepsy(role="migration")
    conn.execute("UPDATE acus SET state='archived' WHERE id=?", (drop,))
    conn.commit()
    conn.close()
    cans = {r["canonical"] for r in store.retrieve()}
    assert "keep | this | one" in cans
    assert "drop | this | one" not in cans


def test_search_substring(store) -> None:
    store.ingest("python | is | language")
    store.ingest("rust | is | language")
    store.ingest("sun | is | hot")
    assert len(store.search("language")) == 2


def test_ingest_locked_is_immutable(store) -> None:
    lid = store.ingest_locked("Origin | values | precision")
    acu = store.get_by_id(lid)
    assert acu["canonical"] == "origin | values | precision"
    assert acu["locked"] == 1
    assert acu["confidentity"] == 1.0
    # A normalized re-ingest matches the locked row but does NOT reinforce it.
    same = store.ingest("origin | values | precision", source="model")
    assert same == lid
    assert store.count() == 1
    assert store.get_by_id(lid)["reinforcement"] == 1  # unchanged (locked)


def test_close_is_noop(store) -> None:
    store.close()
    store.close()  # idempotent
    assert store.count() == 0


def test_retrieve_reranks_by_effective_reinforcement_when_decay_on(store, monkeypatch) -> None:
    """When-plane decay: with MONOLITH_ACU_DECAY_V1 on, retrieve() re-ranks by
    EFFECTIVE (time-decayed) reinforcement — a fresh low-raw claim outranks a
    stale high-raw one. Flag off keeps the raw reinforcement DESC order."""
    from datetime import datetime, timezone, timedelta
    from core import db_connect as _dbc

    now = datetime(2026, 6, 2, tzinfo=timezone.utc)
    stale = (now - timedelta(days=400)).isoformat()
    fresh = now.isoformat()
    cols = ("canonical, source, provenance, kind, l_level, reinforcement, truth, state, "
            "evidence_spans, cf_version, created_at, last_seen, last_touched_ts")
    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        conn.execute(
            f"INSERT INTO acus({cols}) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("stale | high | raw", "self", "self", "self", "L2", 10, "confirmed", "active",
             "[]", 1, stale, stale, stale))
        conn.execute(
            f"INSERT INTO acus({cols}) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("fresh | low | raw", "self", "self", "self", "L2", 4, "confirmed", "active",
             "[]", 1, fresh, fresh, fresh))
        conn.commit()
    finally:
        conn.close()

    # Flag off: raw reinforcement DESC → stale (10) first.
    monkeypatch.delenv("MONOLITH_ACU_DECAY_V1", raising=False)
    assert [r["canonical"] for r in store.retrieve(limit=2)][0] == "stale | high | raw"

    # Flag on: stale raw=10 decays over 400d (~0.02) below fresh raw=4 → fresh first.
    monkeypatch.setenv("MONOLITH_ACU_DECAY_V1", "1")
    cans = [r["canonical"] for r in store.retrieve(limit=2, now=now)]
    assert cans == ["fresh | low | raw", "stale | high | raw"]
