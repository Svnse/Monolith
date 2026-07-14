"""Idempotent schema migrations for Acatalepsy v1 producer loop.

Three changes vs current acatalepsy.sqlite3 (schema_version=7 as of
port date):

  1. CREATE TABLE acu_candidates — the auditor's pending-decision
     buffer. One row per atomic claim the auditor proposes.
  2. CREATE TABLE acu_decisions — append-only event log of accept/
     reject/edit/defer on candidates. Linked back to candidates by FK.
  3. ALTER TABLE acus — add candidate_id + decision_id columns so any
     ACU leaf can be traced back through the provenance chain.

All operations use ``IF NOT EXISTS`` / column-presence checks. Re-runs
are no-ops. Migration uses the ``migration`` role of db_connect
(authorizer bypassed for one-shot ceremony).

Schema version after migration: 8 — recorded in the existing
``schema_version`` table with note "Acatalepsy v1 producer loop:
acu_candidates + acu_decisions + acus provenance pointers".
"""
from __future__ import annotations

import sqlite3
from typing import Any

from core.db_connect import connect_acatalepsy


_TARGET_VERSION = 14
_VERSION_NOTE = (
    "Acatalepsy substrate spine + full reproduction + Truth branch (v11) + L3 "
    "trust (v13): reproducible base tables, CID identity + provenance + temporal + "
    "lineage + state + structured evidence_spans + pre_cid_legacy, "
    "times_seen->reinforcement, the full legacy acus column set, normalization of "
    "pre-CID legacy canonicals, the Truth-branch columns (truth verdict + confidence "
    "+ method + evidence + checked_at), and promoted_to_l3_ts (L2->L3 TRUSTED "
    "promotion). v14 adds eqid (deterministic inverse-relation equivalence grouping; "
    "re-derivable overlay, identity-safe — never touches CID). NOTE: `veracity` is "
    "dormant — reserved for affect-scope, not read."
)


# ── DDL statements ────────────────────────────────────────────────────


_CREATE_ACU_CANDIDATES = """
CREATE TABLE IF NOT EXISTS acu_candidates (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_form      TEXT NOT NULL,
    evidence_log_id     INTEGER NOT NULL,
    evidence_char_start INTEGER NOT NULL,
    evidence_char_end   INTEGER NOT NULL,
    evidence_span       TEXT NOT NULL,
    source              TEXT NOT NULL,
    reason              TEXT NOT NULL,
    reinforcement_count INTEGER NOT NULL DEFAULT 1,
    contradicts_acu_id  INTEGER,
    state               TEXT NOT NULL DEFAULT 'pending',
    created_at          TEXT NOT NULL,
    auditor_run_id      INTEGER,
    FOREIGN KEY (evidence_log_id) REFERENCES canonical_log(event_id),
    FOREIGN KEY (contradicts_acu_id) REFERENCES acus(id),
    CHECK (state IN ('pending', 'accepted', 'rejected', 'edited', 'deferred'))
)
"""

_CREATE_ACU_DECISIONS = """
CREATE TABLE IF NOT EXISTS acu_decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id     INTEGER NOT NULL,
    decision         TEXT NOT NULL,
    decided_by       TEXT NOT NULL,
    decided_at       TEXT NOT NULL,
    reject_reason    TEXT,
    edited_form      TEXT,
    note             TEXT,
    resulting_acu_id INTEGER,
    FOREIGN KEY (candidate_id) REFERENCES acu_candidates(id),
    FOREIGN KEY (resulting_acu_id) REFERENCES acus(id),
    CHECK (decision IN ('accept', 'reject', 'edit', 'defer')),
    CHECK ((decision != 'reject') OR (reject_reason IS NOT NULL AND length(reject_reason) > 0)),
    CHECK ((decision != 'edit') OR (edited_form IS NOT NULL AND length(edited_form) > 0))
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_candidates_state ON acu_candidates(state)",
    "CREATE INDEX IF NOT EXISTS idx_candidates_canonical ON acu_candidates(canonical_form)",
    "CREATE INDEX IF NOT EXISTS idx_candidates_auditor_run ON acu_candidates(auditor_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_candidates_evidence_log ON acu_candidates(evidence_log_id)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_candidate ON acu_decisions(candidate_id)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_decided_by ON acu_decisions(decided_by)",
)


# ── v9 base tables (make a FRESH DB reproduce the schema from code) ────
# The live DB's rich schema came from legacy migrators not present in this
# tree, so a fresh boot would otherwise lack `acus`. These ensure the spine's
# base tables idempotently. Note `reinforcement` (not `times_seen`) — a fresh
# DB is born with the new name; an existing DB is renamed below.

_CREATE_CANONICAL_LOG = """
CREATE TABLE IF NOT EXISTS canonical_log (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    kind       TEXT NOT NULL,
    session_id TEXT,
    acu_id     INTEGER,
    payload    TEXT
)
"""

_CREATE_ACUS_BASE = """
CREATE TABLE IF NOT EXISTS acus (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical     TEXT    NOT NULL,
    veracity      REAL    NOT NULL DEFAULT 5.0,
    reinforcement INTEGER NOT NULL DEFAULT 1,
    source        TEXT    NOT NULL DEFAULT 'model',
    created_at    TEXT    NOT NULL,
    last_seen     TEXT    NOT NULL,
    confidentity  REAL    NOT NULL DEFAULT 0.0,
    locked        INTEGER NOT NULL DEFAULT 0,
    lock_reason   TEXT
)
"""

_CREATE_ACU_RELATIONS = """
CREATE TABLE IF NOT EXISTS acu_relations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id  INTEGER NOT NULL,
    target_id  INTEGER NOT NULL,
    relation   TEXT    NOT NULL,
    score      REAL    NOT NULL DEFAULT 1.0,
    created_at TEXT    NOT NULL,
    updated_at TEXT,
    axis_tags  TEXT
)
"""

# v12 Affect branch: an immutable time-series of affect readings about a person.
# The live affect model is a recency-weighted query-view over these (NOT a stored
# fact that gets overwritten); individual readings never become false, only old.
_CREATE_AFFECT_READINGS = """
CREATE TABLE IF NOT EXISTS affect_readings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    acu_id    INTEGER,
    subject   TEXT NOT NULL,
    valence   REAL NOT NULL,
    arousal   REAL NOT NULL,
    intensity REAL NOT NULL DEFAULT 0,
    target    TEXT,
    source    TEXT,
    ts        TEXT NOT NULL
)
"""

_V12_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_affect_subject ON affect_readings(subject)",
)

# v9 spine columns on acus. (candidate_id/decision_id come from v8.)
_SPINE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cid", "TEXT"),
    ("cf_version", "INTEGER NOT NULL DEFAULT 1"),
    ("provenance", "TEXT"),
    ("l_level", "TEXT NOT NULL DEFAULT 'L1'"),
    ("canonical_triple", "TEXT"),
    ("evidence_spans", "TEXT"),
    ("valid_from", "TEXT"),
    ("valid_to", "TEXT"),
    ("last_confirmed_at", "TEXT"),
    ("last_touched_ts", "TEXT"),
    ("promoted_to_l2_ts", "TEXT"),
    ("promoted_to_l3_ts", "TEXT"),   # L2->L3 TRUSTED promotion (v13)
    ("eqid", "TEXT"),                # v14: deterministic equivalence-group id (overlay)
    ("parent_cid", "TEXT"),
    ("generation", "INTEGER NOT NULL DEFAULT 0"),
    ("source_event", "INTEGER"),
    ("state", "TEXT NOT NULL DEFAULT 'active'"),
    ("merged_into", "INTEGER"),
    ("pre_cid_legacy", "INTEGER NOT NULL DEFAULT 0"),
    # Legacy base columns also ensured here so migrate() fully upgrades any
    # acus table (incl. minimal hand-rolled ones) to the complete schema.
    ("confidentity", "REAL NOT NULL DEFAULT 0.0"),
    ("locked", "INTEGER NOT NULL DEFAULT 0"),
    ("lock_reason", "TEXT"),
)

_V9_INDEXES = (
    # Partial-unique: many NULL cids (L1 stubs) allowed; CID uniqueness
    # enforced only once a claim is crystallized (cold).
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_acus_cid ON acus(cid) WHERE cid IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_acus_state ON acus(state)",
    "CREATE INDEX IF NOT EXISTS idx_acus_l_level ON acus(l_level)",
    "CREATE INDEX IF NOT EXISTS idx_acus_merged_into ON acus(merged_into)",
)

# v10: the remaining legacy acus columns, so a FRESH DB reproduces the full live
# schema (the live DB has these from legacy migrators; a fresh boot otherwise
# lacks them). domain/subdomain are wanted now for validation; kind feeds the
# Kind branch; the rest stay dormant until their phase.
_LEGACY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("domain", "TEXT"),
    ("subdomain", "TEXT"),
    ("subject", "TEXT"),
    ("kind", "TEXT"),
    ("qualifiers_json", "TEXT"),
    ("manual_importance", "REAL"),
    ("cluster_id", "INTEGER"),
    ("axis_tags", "TEXT"),
    ("first_seen_ts", "TEXT"),
)

_V10_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_acus_domain ON acus(domain)",
    "CREATE INDEX IF NOT EXISTS idx_acus_kind ON acus(kind)",
)

# v11 Truth branch: a claim's truth verdict is separate from its `kept` signal
# (reinforcement) and its identity (cid). `state='-inf'` marks confirmed
# falsehoods (kept for audit, excluded from recall).
_TRUTH_COLUMNS: tuple[tuple[str, str], ...] = (
    ("truth", "TEXT"),               # confirmed | contradicted | unverifiable | contested
    ("truth_confidence", "REAL"),    # 0..1
    ("truth_method", "TEXT"),        # internal | tavily
    ("truth_checked_at", "TEXT"),
    ("evidence_url", "TEXT"),
    ("evidence_json", "TEXT"),        # JSON: list of {url, snippet, source}
)

_V11_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_acus_truth ON acus(truth)",
)

_V14_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_acus_eqid ON acus(eqid)",
)


# ── ALTER TABLE helpers (idempotent) ──────────────────────────────────


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _add_acus_provenance_pointers(conn: sqlite3.Connection) -> list[str]:
    """Add candidate_id + decision_id columns to acus if missing.

    Returns list of column names actually added (for migration logging).
    """
    added: list[str] = []
    if not _column_exists(conn, "acus", "candidate_id"):
        conn.execute("ALTER TABLE acus ADD COLUMN candidate_id INTEGER")
        added.append("candidate_id")
    if not _column_exists(conn, "acus", "decision_id"):
        conn.execute("ALTER TABLE acus ADD COLUMN decision_id INTEGER")
        added.append("decision_id")
    return added


def _add_spine_columns(conn: sqlite3.Connection) -> list[str]:
    """Add the v9 spine columns to acus if missing. Idempotent.

    Returns the column names actually added (for migration logging).
    """
    added: list[str] = []
    for col, ddl in _SPINE_COLUMNS:
        if not _column_exists(conn, "acus", col):
            conn.execute(f"ALTER TABLE acus ADD COLUMN {col} {ddl}")
            added.append(col)
    return added


def _rename_reinforcement(conn: sqlite3.Connection) -> bool:
    """Rename the legacy ``times_seen`` column to ``reinforcement``.

    After provenance-weighted reinforcement the field is a weighted signal,
    not a raw count, so the count-name reintroduced the same value-overloading
    that made ``veracity`` hard to reason about. Idempotent: a fresh DB is
    already born with ``reinforcement`` (base DDL), so this no-ops there.
    Returns True iff a rename occurred.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(acus)")}
    if "times_seen" in cols and "reinforcement" not in cols:
        conn.execute("ALTER TABLE acus RENAME COLUMN times_seen TO reinforcement")
        return True
    return False


def _add_legacy_columns(conn: sqlite3.Connection) -> list[str]:
    """Ensure the full legacy acus column set (v10) so a fresh DB == live."""
    added: list[str] = []
    for col, ddl in _LEGACY_COLUMNS:
        if not _column_exists(conn, "acus", col):
            conn.execute(f"ALTER TABLE acus ADD COLUMN {col} {ddl}")
            added.append(col)
    return added


def _add_truth_columns(conn: sqlite3.Connection) -> list[str]:
    """Ensure the v11 Truth-branch columns on acus. Idempotent."""
    added: list[str] = []
    for col, ddl in _TRUTH_COLUMNS:
        if not _column_exists(conn, "acus", col):
            conn.execute(f"ALTER TABLE acus ADD COLUMN {col} {ddl}")
            added.append(col)
    return added


def _normalize_legacy_canonicals(conn: sqlite3.Connection) -> int:
    """Normalize pre-CID (cid IS NULL) rows' `canonical` to the CF_VERSION normal
    form so legacy rows dedup with normalized intake. Collisions collapse via
    `merged_into` (folding reinforcement + evidence). Populates `canonical_triple`.

    Idempotent: already-normalized survivors re-run as no-ops; archived rows are
    excluded. Returns the number of rows folded (archived).
    """
    import json
    from core.acatalepsy.normalize import normalize_canonical, parse_triple

    # Exclude locked rows: Origin-0 identity prose keeps its original casing and
    # is not part of the normalized-dedup flow (intake never matches non-atomic
    # prose anyway).
    rows = conn.execute(
        "SELECT id, canonical, reinforcement, evidence_spans, locked "
        "FROM acus WHERE cid IS NULL AND merged_into IS NULL "
        "AND (locked IS NULL OR locked = 0) ORDER BY id"
    ).fetchall()

    groups: dict[str, list[dict]] = {}
    for r in rows:
        norm = normalize_canonical(r["canonical"])
        groups.setdefault(norm, []).append({
            "id": int(r["id"]),
            "reinforcement": int(r["reinforcement"] or 0),
            "spans": r["evidence_spans"],
            "locked": int(r["locked"] or 0),
        })

    def _load(raw):
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return []
        return [s for s in data if isinstance(s, dict)] if isinstance(data, list) else []

    folded = 0
    for norm, members in groups.items():
        triple = parse_triple(norm)
        ct = json.dumps({
            "entity_a": triple.entity_a, "relation": triple.relation,
            "entity_b": triple.entity_b, "qualifiers": triple.qualifiers,
        }) if triple else None
        # Survivor: a locked member if any (never archive a locked row), else lowest id.
        locked = [m for m in members if m["locked"]]
        survivor = min(locked or members, key=lambda m: m["id"])
        conn.execute(
            "UPDATE acus SET canonical=?, canonical_triple=? WHERE id=?",
            (norm, ct, survivor["id"]),
        )
        if len(members) > 1:
            total = survivor["reinforcement"]
            spans = _load(survivor["spans"])
            seen = {(s.get("text"), s.get("provenance")) for s in spans}
            for m in members:
                if m["id"] == survivor["id"]:
                    continue
                total += m["reinforcement"]
                for s in _load(m["spans"]):
                    key = (s.get("text"), s.get("provenance"))
                    if key not in seen:
                        spans.append(s)
                        seen.add(key)
                conn.execute(
                    "UPDATE acus SET merged_into=?, state='archived' WHERE id=?",
                    (survivor["id"], m["id"]),
                )
                folded += 1
            conn.execute(
                "UPDATE acus SET reinforcement=?, evidence_spans=? WHERE id=?",
                (total, json.dumps(spans), survivor["id"]),
            )
    return folded


# ── Version tracking ──────────────────────────────────────────────────


def _current_schema_version(conn: sqlite3.Connection) -> int:
    """Read the highest schema_version row. Returns 0 if table missing
    or empty (fresh DB)."""
    try:
        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.OperationalError:
        # schema_version table doesn't exist — fresh DB
        return 0


def _ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version    INTEGER PRIMARY KEY,
            applied_at REAL NOT NULL,
            note       TEXT
        )
        """
    )


def _record_version(conn: sqlite3.Connection, version: int, note: str) -> None:
    import time
    conn.execute(
        "INSERT OR IGNORE INTO schema_version(version, applied_at, note) VALUES (?, ?, ?)",
        (version, time.time(), note),
    )


# ── Public API ────────────────────────────────────────────────────────


def migrate() -> dict[str, Any]:
    """Run the Acatalepsy v1 migration. Idempotent — safe to call on
    every boot. Returns a summary dict.

    Uses the migration role of db_connect (authorizer bypassed). All
    DDL operations are wrapped in a transaction so failures roll back.
    """
    conn = connect_acatalepsy(role="migration")
    summary: dict[str, Any] = {
        "started_version": None,
        "target_version": _TARGET_VERSION,
        "ended_version": None,
        "tables_created": [],
        "columns_added": [],
        "indexes_created": [],
        "skipped": False,
    }
    try:
        _ensure_schema_version_table(conn)
        started = _current_schema_version(conn)
        summary["started_version"] = started

        if started >= _TARGET_VERSION:
            summary["skipped"] = True
            summary["ended_version"] = started
            return summary

        conn.execute("BEGIN")
        try:
            # 0. Ensure base tables exist so a FRESH DB reproduces the full
            #    schema from code alone (closes the schema-drift liability).
            #    canonical_log first — acu_candidates FK-references it.
            for ddl in (_CREATE_CANONICAL_LOG, _CREATE_ACUS_BASE, _CREATE_ACU_RELATIONS,
                        _CREATE_AFFECT_READINGS):
                conn.execute(ddl)

            # 1. acu_candidates
            existed_before = _table_exists(conn, "acu_candidates")
            conn.execute(_CREATE_ACU_CANDIDATES)
            if not existed_before:
                summary["tables_created"].append("acu_candidates")

            # 2. acu_decisions
            existed_before = _table_exists(conn, "acu_decisions")
            conn.execute(_CREATE_ACU_DECISIONS)
            if not existed_before:
                summary["tables_created"].append("acu_decisions")

            # 3. ALTER acus — v8 pointers + v9 spine + v10 legacy + v11 truth columns
            cols_added = _add_acus_provenance_pointers(conn)
            cols_added += _add_spine_columns(conn)
            cols_added += _add_legacy_columns(conn)
            cols_added += _add_truth_columns(conn)
            summary["columns_added"] = [f"acus.{c}" for c in cols_added]

            # 3b. Rename times_seen -> reinforcement (weighted signal, not a count).
            summary["renamed_times_seen"] = _rename_reinforcement(conn)

            # 3c. Flag pre-existing rows pre-CID legacy. No CID backfill: legacy
            #     rows re-crystallize through the L1 pass on next encounter so the
            #     normalizer owns every CID it mints. (Empty on a fresh DB.)
            conn.execute("UPDATE acus SET pre_cid_legacy = 1")

            # 3d. Normalize legacy canonicals (v10) so they dedup with intake's
            #     normalized forms; collapse collisions via merged_into.
            summary["legacy_folded"] = _normalize_legacy_canonicals(conn)

            # 4. Indexes (v8 + v9 + v10 + v11)
            all_indexes = (*_INDEXES, *_V9_INDEXES, *_V10_INDEXES, *_V11_INDEXES,
                           *_V12_INDEXES, *_V14_INDEXES)
            for ddl in all_indexes:
                conn.execute(ddl)
            summary["indexes_created"] = list(all_indexes)

            # 5. Record version
            _record_version(conn, _TARGET_VERSION, _VERSION_NOTE)

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        summary["ended_version"] = _current_schema_version(conn)
    finally:
        conn.close()

    return summary


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


if __name__ == "__main__":
    import json
    result = migrate()
    print(json.dumps(result, indent=2, default=str))
