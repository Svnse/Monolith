"""Tests for the Acatalepsy v9 spine migration (schema reproducibility = A1).

A1 acceptance gate: a fresh boot must reproduce the full spine schema from
code alone (the live DB's rich columns came from legacy migrators not present
here — drift). These tests pin that migrate() is self-sufficient on a fresh
DB and idempotent on re-run, renames times_seen->reinforcement, and flags
pre-existing rows pre_cid_legacy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Columns the spine requires on `acus` for a fresh DB to be functional.
_SPINE_COLS = {
    "cid", "cf_version", "provenance", "valid_from", "valid_to",
    "last_confirmed_at", "parent_cid", "generation", "source_event",
    "state", "evidence_spans", "pre_cid_legacy", "l_level",
    "canonical_triple", "promoted_to_l2_ts", "promoted_to_l3_ts",
    "last_touched_ts", "merged_into", "eqid",
    # v10: full reproduction (E wants domain/subdomain now for validation).
    "domain", "subdomain", "kind",
}

_TARGET = 14

_TRUTH_COLS = {"truth", "truth_confidence", "truth_method", "truth_checked_at",
               "evidence_url", "evidence_json"}


def _point_db_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "schema_v9.sqlite3"
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", db_path, raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    return db_path


def _acus_cols(dbc) -> set[str]:
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        return {r[1] for r in conn.execute("PRAGMA table_info(acus)")}
    finally:
        conn.close()


def _tables(dbc) -> set[str]:
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def test_fresh_db_reaches_v9_with_tables_and_spine_columns(tmp_path, monkeypatch):
    _point_db_at(tmp_path, monkeypatch)
    from core import db_connect as _dbc
    from core.acatalepsy import schema

    result = schema.migrate()
    assert result["ended_version"] == _TARGET

    tables = _tables(_dbc)
    for t in ("acus", "canonical_log", "acu_relations", "acu_candidates",
              "acu_decisions", "affect_readings"):
        assert t in tables, f"missing table {t}"

    cols = _acus_cols(_dbc)
    missing = _SPINE_COLS - cols
    assert not missing, f"missing spine columns: {missing}"
    assert not (_TRUTH_COLS - cols), f"missing truth columns: {_TRUTH_COLS - cols}"
    assert "reinforcement" in cols
    assert "times_seen" not in cols


def test_migrate_is_idempotent(tmp_path, monkeypatch):
    _point_db_at(tmp_path, monkeypatch)
    from core.acatalepsy import schema
    schema.migrate()
    second = schema.migrate()
    assert second["skipped"] is True
    assert second["ended_version"] == _TARGET


def test_eqid_column_and_index_present(tmp_path, monkeypatch):
    _point_db_at(tmp_path, monkeypatch)
    from core import db_connect as _dbc
    from core.acatalepsy import schema
    schema.migrate()
    cols = _acus_cols(_dbc)
    assert "eqid" in cols
    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        idx = {r[1] for r in conn.execute("PRAGMA index_list(acus)")}
        assert "idx_acus_eqid" in idx
    finally:
        conn.close()


def test_rename_preserves_value_and_flags_legacy_rows(tmp_path, monkeypatch):
    _point_db_at(tmp_path, monkeypatch)
    from core import db_connect as _dbc
    from core.acatalepsy import schema

    # Simulate a pre-v9 (v8) live DB: old acus shape with times_seen + a row.
    conn = _dbc.connect_acatalepsy(role="migration")
    conn.executescript(
        """
        CREATE TABLE acus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical TEXT NOT NULL,
            veracity REAL NOT NULL DEFAULT 5.0,
            times_seen INTEGER NOT NULL DEFAULT 1,
            source TEXT NOT NULL DEFAULT 'model',
            created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            candidate_id INTEGER,
            decision_id INTEGER
        );
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY, applied_at REAL NOT NULL, note TEXT
        );
        INSERT INTO schema_version(version, applied_at, note) VALUES (8, 0, 'seed');
        INSERT INTO acus(canonical, times_seen, source, created_at, last_seen)
            VALUES ('a | r | b', 3, 'model', 'now', 'now');
        """
    )
    conn.commit()
    conn.close()

    schema.migrate()

    cols = _acus_cols(_dbc)
    assert "reinforcement" in cols and "times_seen" not in cols

    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        row = conn.execute(
            "SELECT reinforcement, pre_cid_legacy FROM acus WHERE canonical='a | r | b'"
        ).fetchone()
    finally:
        conn.close()
    assert row["reinforcement"] == 3        # value preserved through rename
    assert row["pre_cid_legacy"] == 1       # pre-existing row flagged legacy


def test_cid_partial_unique_index(tmp_path, monkeypatch):
    _point_db_at(tmp_path, monkeypatch)
    import sqlite3
    from core import db_connect as _dbc
    from core.acatalepsy import schema

    schema.migrate()
    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        # Many NULL cids allowed (L1 stubs).
        conn.execute(
            "INSERT INTO acus(canonical, source, created_at, last_seen) "
            "VALUES ('x | r | y','model','now','now')")
        conn.execute(
            "INSERT INTO acus(canonical, source, created_at, last_seen) "
            "VALUES ('p | r | q','model','now','now')")
        # First non-NULL cid OK.
        conn.execute(
            "INSERT INTO acus(canonical, cid, source, created_at, last_seen) "
            "VALUES ('m | r | n','cid:sha256:abc','model','now','now')")
        conn.commit()
        # Duplicate non-NULL cid rejected by the partial-unique index.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO acus(canonical, cid, source, created_at, last_seen) "
                "VALUES ('m2 | r | n2','cid:sha256:abc','model','now','now')")
            conn.commit()
    finally:
        conn.close()


def _seed_v8(dbc, inserts: str):
    conn = dbc.connect_acatalepsy(role="migration")
    conn.executescript(
        """
        CREATE TABLE acus (
            id INTEGER PRIMARY KEY AUTOINCREMENT, canonical TEXT NOT NULL,
            veracity REAL NOT NULL DEFAULT 5.0, times_seen INTEGER NOT NULL DEFAULT 1,
            source TEXT NOT NULL DEFAULT 'model', created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL, candidate_id INTEGER, decision_id INTEGER
        );
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL, note TEXT);
        INSERT INTO schema_version(version, applied_at, note) VALUES (8, 0, 'seed');
        """
        + inserts
    )
    conn.commit()
    conn.close()


def test_legacy_canonicals_normalized(tmp_path, monkeypatch):
    _point_db_at(tmp_path, monkeypatch)
    from core import db_connect as _dbc
    from core.acatalepsy import schema
    _seed_v8(
        _dbc,
        "INSERT INTO acus(canonical, times_seen, source, created_at, last_seen) "
        "VALUES ('Trump | President_Of | USA.', 1, 'model', 'now', 'now');",
    )
    schema.migrate()
    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        row = conn.execute("SELECT canonical, canonical_triple FROM acus").fetchone()
    finally:
        conn.close()
    assert row["canonical"] == "trump | president_of | usa"  # normalized in place
    assert row["canonical_triple"] is not None


def test_legacy_collision_collapses_and_folds_signal(tmp_path, monkeypatch):
    _point_db_at(tmp_path, monkeypatch)
    from core import db_connect as _dbc
    from core.acatalepsy import schema
    _seed_v8(
        _dbc,
        "INSERT INTO acus(canonical, times_seen, source, created_at, last_seen) "
        "VALUES ('Foo | r | Bar', 2, 'model', 'now', 'now');"
        "INSERT INTO acus(canonical, times_seen, source, created_at, last_seen) "
        "VALUES ('foo | r | bar.', 3, 'model', 'now', 'now');",
    )
    schema.migrate()
    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        active = conn.execute(
            "SELECT id, reinforcement FROM acus "
            "WHERE merged_into IS NULL AND canonical='foo | r | bar'").fetchall()
        archived = conn.execute(
            "SELECT COUNT(*) FROM acus WHERE merged_into IS NOT NULL").fetchone()[0]
    finally:
        conn.close()
    assert len(active) == 1
    assert active[0]["reinforcement"] == 5  # 2 + 3 folded, no signal lost
    assert archived == 1
