"""friction_store — Layer F (`prediction_traces`) of the turn_trace store.

The Predict→Settle organ's persistence. Co-located in the SAME SQLite file as
the other turn_trace layers (LOG_DIR/turn_trace.sqlite3 — single-store
discipline) but in its OWN table and module so it cannot destabilize
turn_trace.py paths.

Flag: MONOLITH_FRICTION_V1 (default OFF). When OFF, every WRITE is a no-op and
the table is NEVER created — so a flag-off run touches the DB zero times
(byte-identical guarantee). Reads tolerate a missing table (return None / []).

A prediction is `open` when recorded (predict beat), `settled` once the settler
scores the next outer turn's friction (settle beat), or `abandoned` when a newer
prediction supersedes it within the same outer turn (tool-loop followups).
`confidence` is the read's DISPOSABLE predict-time confidence — never accumulated
into a growing certainty about the user (spec §5).

SCHEMA v2 (2026-06-22, Path B): adds `prediction_set_json` — the FROZEN,
enumerated, code-checkable commitment {directions:[{move,referent}], referents:[...]}
that the membership settler grades. v1 rows (no set) settle `unresolved`.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from core.paths import LOG_DIR

_FLAG_ENV = "MONOLITH_FRICTION_V1"
_TRUTHY = {"1", "true", "yes", "on"}
_DB_PATH = LOG_DIR / "turn_trace.sqlite3"
_TABLE = "prediction_traces"
_SCHEMA_VERSION = 2

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version  INTEGER NOT NULL DEFAULT {_SCHEMA_VERSION},
    turn_id         TEXT,
    turn_n          INTEGER,
    created_at      TEXT,
    kind            TEXT,
    claim           TEXT,
    falsifier       TEXT,
    confidence      REAL,
    horizon         TEXT,
    status          TEXT DEFAULT 'open',
    prediction_set_json TEXT,
    friction_score  REAL,
    friction_type   TEXT,
    channel_json    TEXT,
    settled_at      TEXT,
    settled_turn_id TEXT,
    observation     TEXT,
    surfaced        INTEGER DEFAULT 0
);
"""


def flag_enabled() -> bool:
    return os.environ.get(_FLAG_ENV, "0").strip().lower() in _TRUTHY


def _path(db_path: Any = None):
    return str(db_path) if db_path is not None else str(_DB_PATH)


def _connect(db_path: Any = None) -> sqlite3.Connection:
    conn = sqlite3.connect(_path(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _column_names(conn: sqlite3.Connection) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({_TABLE})").fetchall()
        return {str(r["name"]) for r in rows}
    except Exception:
        return set()


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create the table if absent, then run guarded ADD COLUMN migrations so an
    existing v1 DB (no prediction_set_json) upgrades in place. Idempotent."""
    conn.execute(_DDL)
    cols = _column_names(conn)
    if "prediction_set_json" not in cols:
        # v1 -> v2 migration: add the frozen-set column to an existing table.
        try:
            conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN prediction_set_json TEXT")
        except Exception:
            pass


def _table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_TABLE,)
    ).fetchone()
    return row is not None


# ── writes (flag-gated: no-op + no table creation when OFF) ───────────


def abandon_open(db_path: Any = None) -> int:
    """Mark all currently-open predictions 'abandoned'. Returns rows affected.

    Called by the producer BEFORE inserting a new open prediction so that at
    most ONE prediction is ever open — a tool-loop followup's prediction
    supersedes the initial generation's, and an unreplied prediction can't
    accumulate as an orphan that a later turn mis-settles. No-op when flag OFF."""
    if not flag_enabled():
        return 0
    conn = _connect(db_path)
    try:
        _ensure_table(conn)
        cur = conn.execute(
            f"UPDATE {_TABLE} SET status='abandoned' WHERE status='open'"
        )
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def record_prediction(
    turn_id: str,
    turn_n: int,
    kind: str,
    claim: str,
    falsifier: str,
    confidence: float,
    horizon: str,
    *,
    now_iso: str,
    prediction_set_json: dict | None = None,
    db_path: Any = None,
) -> int:
    """Insert an open prediction; return its row id. No-op (-1) when flag OFF.

    `prediction_set_json` is the frozen, code-checkable commitment the settler
    grades by membership (Path B). Stored as JSON text."""
    if not flag_enabled():
        return -1
    pset = json.dumps(prediction_set_json) if prediction_set_json is not None else None
    conn = _connect(db_path)
    try:
        _ensure_table(conn)
        cur = conn.execute(
            f"""INSERT INTO {_TABLE}
                (schema_version, turn_id, turn_n, created_at, kind, claim,
                 falsifier, confidence, horizon, status, prediction_set_json, surfaced)
                VALUES (?,?,?,?,?,?,?,?,?, 'open', ?, 0)""",
            (_SCHEMA_VERSION, str(turn_id), int(turn_n), str(now_iso), str(kind),
             str(claim), str(falsifier), float(confidence), str(horizon), pset),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def settle_prediction(
    pred_id: int,
    friction_score: float,
    friction_type: str,
    channel_json: dict,
    observation: str,
    settled_turn_id: str,
    *,
    now_iso: str,
    db_path: Any = None,
) -> None:
    """Move an open prediction to settled with its friction reading. No-op when OFF."""
    if not flag_enabled():
        return
    conn = _connect(db_path)
    try:
        _ensure_table(conn)
        conn.execute(
            f"""UPDATE {_TABLE}
                SET status='settled', friction_score=?, friction_type=?,
                    channel_json=?, observation=?, settled_turn_id=?, settled_at=?
                WHERE id=? AND status='open'""",
            (float(friction_score), str(friction_type), json.dumps(channel_json),
             str(observation), str(settled_turn_id), str(now_iso), int(pred_id)),
        )
        conn.commit()
    finally:
        conn.close()


def mark_surfaced(pred_id: int, db_path: Any = None) -> None:
    if not flag_enabled():
        return
    conn = _connect(db_path)
    try:
        _ensure_table(conn)
        conn.execute(f"UPDATE {_TABLE} SET surfaced=1 WHERE id=?", (int(pred_id),))
        conn.commit()
    finally:
        conn.close()


# ── reads (tolerate a missing table; safe to call any time) ───────────


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for jcol in ("channel_json", "prediction_set_json"):
        if d.get(jcol):
            try:
                d[jcol] = json.loads(d[jcol])
            except Exception:
                pass
    return d


def latest_open(kind: str | None = None, db_path: Any = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        if not _table_exists(conn):
            return None
        if kind is None:
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE status='open' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE status='open' AND kind=? ORDER BY id DESC LIMIT 1",
                (str(kind),),
            ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def recent_settled(limit: int = 12, db_path: Any = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        if not _table_exists(conn):
            return []
        rows = conn.execute(
            f"SELECT * FROM {_TABLE} WHERE status='settled' ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()
