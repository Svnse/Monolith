"""Tests for the acatalepsy-acus adapter (spec §5, row `acatalepsy-acus`).

Two flavours:
  * pure mapping unit tests (hand-built dict rows + monkeypatched ACUStore) for
    _to_record / _recurrence_key / _provenance / locked-exclusion logic;
  * one REAL end-to-end against a seeded isolated acatalepsy DB (monkeypatch
    core.db_connect.DB_PATH), mirroring tests/monosearch/test_canonical_log_e2e.py
    and tests/test_acu_store.py — exercises the real ACUStore reads + the extended
    _READ_COLS projection (eqid + cid).
"""
from __future__ import annotations

from unittest.mock import patch

from core.monosearch.adapters.acus import AcuAdapter
from core.monosearch.record import EvidenceTier, Provenance


def _row(**over):
    """A full _READ_COLS-shaped dict row (post-extension: includes eqid + cid)."""
    base = {
        "id": 42,
        "canonical": "sky | is | blue",
        "veracity": 5.0,
        "reinforcement": 3,
        "source": "model",
        "provenance": "self",
        "l_level": "L1",
        "kind": "claim",
        "domain": None,
        "truth": None,
        "truth_confidence": None,
        "truth_checked_at": None,
        "state": "active",
        "created_at": "2026-06-02T18:00:00+00:00",
        "last_seen": "2026-06-02T18:00:00+00:00",
        "last_touched_ts": "2026-06-02T18:00:00+00:00",
        "confidentity": 0.0,
        "locked": 0,
        "lock_reason": None,
        "eqid": None,
        "cid": None,
    }
    base.update(over)
    return base


# ── pure mapping ──────────────────────────────────────────────────────


def test_to_record_shape():
    a = AcuAdapter()
    r = a._to_record(_row())
    assert r.namespaced_id == "acu:42"
    assert r.source == "acatalepsy-acus"
    assert r.evidence_tier == EvidenceTier.DERIVED
    assert r.provenance is Provenance.SELF
    assert r.text == "sky | is | blue"
    assert r.ts == 1780423200.0  # 2026-06-02T18:00:00Z -> epoch
    assert r.metadata["kind"] == "claim"
    assert r.metadata["l_level"] == "L1"
    assert r.metadata["state"] == "active"
    assert r.metadata["truth"] is None
    assert r.metadata["provenance"] == "self"
    assert r.metadata["eqid"] is None
    assert r.metadata["cid"] is None


def test_provenance_maps_from_row_column():
    a = AcuAdapter()
    assert a._provenance(_row(provenance="user")) is Provenance.USER
    assert a._provenance(_row(provenance="world")) is Provenance.WORLD
    assert a._provenance(_row(provenance="self")) is Provenance.SELF
    # Unknown / None -> SELF (the ACU default).
    assert a._provenance(_row(provenance=None)) is Provenance.SELF
    assert a._provenance(_row(provenance="garbage")) is Provenance.SELF


def test_recurrence_key_none_for_cold_corpus():
    # Day-1 reality: eqid + cid both empty -> lookup-only, not salience-eligible.
    a = AcuAdapter()
    assert a._recurrence_key(_row(eqid=None, cid=None)) is None
    assert a._recurrence_key(_row(eqid="", cid="")) is None


def test_recurrence_key_prefers_eqid_then_cid():
    a = AcuAdapter()
    assert a._recurrence_key(_row(eqid="EQ-1", cid="CID-9")) == "EQ-1"  # eqid wins
    assert a._recurrence_key(_row(eqid=None, cid="CID-9")) == "CID-9"   # fallback to cid
    assert a._recurrence_key(_row(eqid="", cid="CID-9")) == "CID-9"     # empty eqid -> cid


def test_recurrence_key_sealed_for_self_self():
    # SEAL: identity memory (kind=self AND provenance=self) is NOT salience-eligible,
    # even after it mints cid/eqid -> recurrence_key None (lookup-only). Otherwise a
    # self-repeated belief would surface on the live `recurring()` tool at full parity.
    a = AcuAdapter()
    assert a._recurrence_key(_row(kind="self", provenance="self", eqid="EQ-1", cid="CID-9")) is None
    # user/world-sourced self-facts keep their key (externally grounded, allowed).
    assert a._recurrence_key(_row(kind="self", provenance="user", eqid="EQ-1")) == "EQ-1"
    # a non-self claim with self provenance keeps its key (only kind=self is identity).
    assert a._recurrence_key(_row(kind="world-fact", provenance="self", eqid="EQ-2")) == "EQ-2"


def test_get_parses_namespaced_id():
    a = AcuAdapter()
    with patch("core.monosearch.adapters.acus.ACUStore") as Store:
        Store.return_value.get_by_id.return_value = _row(id=7)
        r = a.get("acu:7")
    assert r is not None and r.namespaced_id == "acu:7"
    assert a.get("fault:7") is None  # wrong namespace


def test_get_excludes_locked_origin0_row():
    # The adapter (not the store) must drop locked=1 Origin-0 rows; identity owns them.
    a = AcuAdapter()
    with patch("core.monosearch.adapters.acus.ACUStore") as Store:
        Store.return_value.get_by_id.return_value = _row(id=7, locked=1)
        assert a.get("acu:7") is None


def test_search_filters_locked_rows():
    a = AcuAdapter()
    rows = [_row(id=1, locked=0), _row(id=2, locked=1), _row(id=3, locked=0)]
    with patch("core.monosearch.adapters.acus.ACUStore") as Store:
        Store.return_value.search.return_value = rows
        recs = a.search("sky", {}, 10)
    ids = {r.namespaced_id for r in recs}
    assert ids == {"acu:1", "acu:3"}  # acu:2 (locked) excluded


def test_list_uses_retrieve_and_filters_locked():
    a = AcuAdapter()
    rows = [_row(id=1, locked=0), _row(id=2, locked=1)]
    with patch("core.monosearch.adapters.acus.ACUStore") as Store:
        Store.return_value.retrieve.return_value = rows
        recs = a.list({}, 50)
    assert {r.namespaced_id for r in recs} == {"acu:1"}


# ── REAL end-to-end against a seeded isolated acatalepsy DB ────────────


def test_real_end_to_end_seeded_store(tmp_path, monkeypatch):
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
    # One ordinary self ACU + one locked Origin-0 ACU.
    normal_id = store.ingest("python | is | language", source="model")
    locked_id = store.ingest_locked("origin | values | precision")
    assert normal_id >= 1 and locked_id >= 1

    a = AcuAdapter()

    # get() round-trips the normal row, through the REAL extended _READ_COLS
    # projection (proves eqid + cid select without error).
    r = a.get(f"acu:{normal_id}")
    assert r is not None
    assert r.namespaced_id == f"acu:{normal_id}"
    assert r.source == "acatalepsy-acus"
    assert r.evidence_tier == EvidenceTier.DERIVED
    assert r.text == "python | is | language"
    # Cold corpus: no eqid/cid minted -> not salience-eligible.
    assert r.recurrence_key is None
    assert "eqid" in r.metadata and "cid" in r.metadata

    # The locked Origin-0 row is owned by identity; the adapter must NOT serve it.
    assert a.get(f"acu:{locked_id}") is None

    # list() (the salience.rebuild iteration path) excludes the locked row too.
    listed = {rec.namespaced_id for rec in a.list({}, 100)}
    assert f"acu:{normal_id}" in listed
    assert f"acu:{locked_id}" not in listed


def test_real_recurrence_key_reads_extended_read_cols(tmp_path, monkeypatch):
    """Makes edit #1 (the _READ_COLS eqid+cid extension) LOAD-BEARING.

    Seeds a row with a non-null `cid` via direct SQL, then asserts the adapter
    surfaces it as recurrence_key through the REAL ACUStore read. If the
    _READ_COLS projection ever drops `cid`/`eqid`, `dict(row).get('cid')` returns
    None and this assertion fails — which the all-None cold-corpus tests would
    NOT catch (they pass for the wrong reason). eqid wins over cid when both set.
    """
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

    cols = ("canonical, source, provenance, kind, l_level, reinforcement, state, "
            "evidence_spans, cf_version, created_at, last_seen, last_touched_ts, "
            "cid, eqid, locked")
    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        cur = conn.execute(
            f"INSERT INTO acus({cols}) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("alpha | beta | gamma", "self", "self", "claim", "L3", 5, "active",
             "[]", 1, "2026-06-02T18:00:00+00:00", "2026-06-02T18:00:00+00:00",
             "2026-06-02T18:00:00+00:00", "CID-XYZ", "EQ-7", 0))
        conn.commit()
        seeded_id = int(cur.lastrowid)
    finally:
        conn.close()

    a = AcuAdapter()
    r = a.get(f"acu:{seeded_id}")
    assert r is not None
    # eqid present -> it wins; proves BOTH columns flow through _READ_COLS.
    assert r.recurrence_key == "EQ-7"
    assert r.metadata["cid"] == "CID-XYZ"
    assert r.metadata["eqid"] == "EQ-7"
