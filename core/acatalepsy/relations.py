"""Read primitives over ``acu_relations`` — the reader half of the contradiction/
overlap graph (the ``contradicts`` edges ``verifier.py`` writes and the ``overlaps``
edges ``intake.py`` writes). Until now nothing read these on a live turn; the
MonoSearch ``acatalepsy-relations`` adapter wraps these functions (the same way
``fault_response``'s read functions back ``FaultAdapter``).

Each read does a SINGLE query with two ``LEFT JOIN acus`` so both endpoints are
resolved in one shot — no N+1, no per-edge ``ACUStore``. A missing endpoint (ACU
deleted/never-existed) surfaces faithfully as ``NULL`` canonical; the SKIP decision
is the adapter's, not ours. There is intentionally NO ``keyword`` parameter:
``acu_relations`` has no text column of its own (its text is the resolved
endpoints), so keyword matching must happen in the adapter after resolution.

Connections are opened fresh per call (``connect_acatalepsy(role="reader")``),
mirroring ``acu_store._reader`` — no thread-local cache to reset in tests.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from core.db_connect import connect_acatalepsy

# The endpoint-resolving projection shared by all three reads.
_SELECT = """
    SELECT r.id, r.source_id, r.target_id, r.relation, r.score,
           r.created_at, r.updated_at,
           s.canonical AS source_canonical, s.locked AS source_locked,
           s.state AS source_state,
           t.canonical AS target_canonical, t.locked AS target_locked,
           t.state AS target_state
    FROM acu_relations r
    LEFT JOIN acus s ON s.id = r.source_id
    LEFT JOIN acus t ON t.id = r.target_id
"""


def _reader() -> sqlite3.Connection:
    conn = connect_acatalepsy(role="reader")
    conn.row_factory = sqlite3.Row
    return conn


def read_recent(
    limit: int = 20, *, since: str | None = None, relation: str | None = None
) -> list[dict[str, Any]]:
    """Most-recent edges first (by id). Filter by ISO ``since`` (created_at >=) and/or
    ``relation`` ('contradicts'|'overlaps'). Both filters push to SQL — keyword does
    not (no text column; the adapter post-filters)."""
    where: list[str] = []
    params: list[Any] = []
    if since:
        where.append("r.created_at >= ?")
        params.append(since)
    if relation:
        where.append("r.relation = ?")
        params.append(relation)
    sql = _SELECT
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY r.id DESC LIMIT ?"
    params.append(int(limit))
    conn = _reader()
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def read_one(edge_id: int) -> dict[str, Any] | None:
    """A single edge by its ``acu_relations`` primary key, endpoints resolved."""
    conn = _reader()
    try:
        row = conn.execute(_SELECT + " WHERE r.id = ?", (int(edge_id),)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def read_since_id(after: int, limit: int = 500) -> list[dict[str, Any]]:
    """Ascending iteration by id (the path ``salience.rebuild`` uses to page)."""
    conn = _reader()
    try:
        rows = conn.execute(
            _SELECT + " WHERE r.id > ? ORDER BY r.id ASC LIMIT ?",
            (int(after), int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
