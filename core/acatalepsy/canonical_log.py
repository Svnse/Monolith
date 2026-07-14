"""canonical_log — immutable audit floor of Acatalepsy v1.

Every chat dispatch, auditor run, candidate emission, and decision
event writes one row here. The auditor consumes from here. The
provenance chain begins here.

Public API:
  - append(kind, payload, *, session_id, acu_id) -> event_id
  - read_since(last_event_id, limit) -> list[Event]
  - read_one(event_id) -> Event | None
  - read_session(session_id, limit) -> list[Event]
  - latest_event_id() -> int

Writes use the `memory_writer` role + `authorized_write()` sentinel —
the SQLite authorizer denies inserts outside an authorized context.
The append() function enters the sentinel internally; callers don't
need to wrap it themselves (but they MAY wrap a multi-write batch in
a single sentinel context).

Kind validation: every kind is checked against KNOWN_KINDS at write
time. Unknown kinds raise UnknownKind. See canonical_log_kinds.py for
the declarative enum.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

from core.db_connect import connect_acatalepsy, authorized_write
from core.acatalepsy.canonical_log_kinds import assert_valid_kind


__all__ = (
    "Event",
    "append",
    "append_on",
    "read_since",
    "read_one",
    "read_recent",
    "search",
    "read_session",
    "latest_event_id",
)


@dataclass(frozen=True)
class Event:
    """Immutable view of a canonical_log row."""
    event_id: int
    ts: float
    kind: str
    session_id: str | None
    acu_id: int | None
    payload: dict[str, Any] | None


# ── Connection pool ───────────────────────────────────────────────────


# Thread-local connections — sqlite3 connections aren't shareable across
# threads without check_same_thread=False, and we want each thread to
# get its own writer with the authorizer attached.
_tl = threading.local()


def _get_writer_conn() -> sqlite3.Connection:
    conn = getattr(_tl, "writer_conn", None)
    if conn is None:
        conn = connect_acatalepsy(role="memory_writer")
        _tl.writer_conn = conn
    return conn


def _get_reader_conn() -> sqlite3.Connection:
    conn = getattr(_tl, "reader_conn", None)
    if conn is None:
        conn = connect_acatalepsy(role="reader")
        _tl.reader_conn = conn
    return conn


# ── Public write API ──────────────────────────────────────────────────


def append(
    kind: str,
    payload: dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
    acu_id: int | None = None,
    sentinel_reason: str | None = None,
) -> int:
    """Append one event to canonical_log. Returns the new event_id.

    The writer enters an ``authorized_write`` sentinel internally if
    not already inside one — callers can rely on the write succeeding
    without manually wrapping. The sentinel reason defaults to
    ``f"canonical_log:{kind}"`` if not supplied.

    Raises:
        UnknownKind — kind not in declarative enum (see canonical_log_kinds)
        sqlite3.DatabaseError / MemoryWriteForbidden — if the authorizer
            denies the write (e.g., a config bug bypassed the sentinel)
    """
    assert_valid_kind(kind)
    payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
    reason = sentinel_reason or f"canonical_log:{kind}"
    conn = _get_writer_conn()
    with authorized_write(reason):
        cur = conn.execute(
            "INSERT INTO canonical_log(ts, kind, session_id, acu_id, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), kind, session_id, acu_id, payload_json),
        )
        conn.commit()
        return int(cur.lastrowid)


def append_on(
    conn: sqlite3.Connection,
    kind: str,
    payload: dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
    acu_id: int | None = None,
) -> int:
    """Append an event on a CALLER-supplied connection WITHOUT committing.

    For atomic multi-write transactions: the caller owns the connection, the
    ``authorized_write`` sentinel, and the commit. This lets a state mutation
    and its event land in ONE transaction, so the event is never lost relative
    to the mutation (the replay/audit invariant). Validates kind like append().
    """
    assert_valid_kind(kind)
    payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
    cur = conn.execute(
        "INSERT INTO canonical_log(ts, kind, session_id, acu_id, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        (time.time(), kind, session_id, acu_id, payload_json),
    )
    return int(cur.lastrowid)


# ── Public read API ───────────────────────────────────────────────────


def _row_to_event(row: sqlite3.Row) -> Event:
    raw = row["payload"]
    payload: dict[str, Any] | None = None
    if raw:
        try:
            parsed = json.loads(raw)
            payload = parsed if isinstance(parsed, dict) else {"_raw": parsed}
        except json.JSONDecodeError:
            payload = {"_raw": raw}
    return Event(
        event_id=int(row["event_id"]),
        ts=float(row["ts"]),
        kind=str(row["kind"]),
        session_id=row["session_id"],
        acu_id=int(row["acu_id"]) if row["acu_id"] is not None else None,
        payload=payload,
    )


def read_since(last_event_id: int, limit: int = 1000) -> list[Event]:
    """Return events with event_id > last_event_id, ordered ascending.

    The auditor's cursor uses this: ``read_since(cursor, limit)``
    returns the unprocessed slice. Pass 0 to read from the beginning.
    """
    if limit < 1:
        return []
    conn = _get_reader_conn()
    cur = conn.execute(
        "SELECT event_id, ts, kind, session_id, acu_id, payload "
        "FROM canonical_log WHERE event_id > ? "
        "ORDER BY event_id ASC LIMIT ?",
        (int(last_event_id), int(limit)),
    )
    return [_row_to_event(row) for row in cur.fetchall()]


def read_one(event_id: int) -> Event | None:
    """Read a specific event by id, or None if not found."""
    conn = _get_reader_conn()
    cur = conn.execute(
        "SELECT event_id, ts, kind, session_id, acu_id, payload "
        "FROM canonical_log WHERE event_id = ?",
        (int(event_id),),
    )
    row = cur.fetchone()
    return _row_to_event(row) if row else None


def read_recent(limit: int = 20) -> list[Event]:
    """Most-recent events first (event_id DESC). Read-only."""
    limit = max(1, min(int(limit or 20), 500))
    conn = _get_reader_conn()
    rows = conn.execute(
        "SELECT event_id, ts, kind, session_id, acu_id, payload "
        "FROM canonical_log ORDER BY event_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_event(r) for r in rows]


def search(query: str, limit: int = 20) -> list[Event]:
    """Substring match over the payload text (un-indexed full scan — acceptable
    at slice-1 scale; canonical_log has no payload index)."""
    limit = max(1, min(int(limit or 20), 500))
    conn = _get_reader_conn()
    rows = conn.execute(
        "SELECT event_id, ts, kind, session_id, acu_id, payload "
        "FROM canonical_log WHERE payload LIKE ? ORDER BY event_id DESC LIMIT ?",
        (f"%{query}%", limit),
    ).fetchall()
    return [_row_to_event(r) for r in rows]


def read_session(session_id: str, limit: int = 1000) -> list[Event]:
    """Return events for a given session_id, ordered by event_id ascending."""
    if not session_id:
        return []
    conn = _get_reader_conn()
    cur = conn.execute(
        "SELECT event_id, ts, kind, session_id, acu_id, payload "
        "FROM canonical_log WHERE session_id = ? "
        "ORDER BY event_id ASC LIMIT ?",
        (str(session_id), int(limit)),
    )
    return [_row_to_event(row) for row in cur.fetchall()]


def latest_event_id() -> int:
    """Return the highest event_id, or 0 if the log is empty.

    Used to seed the auditor cursor on first run.
    """
    conn = _get_reader_conn()
    cur = conn.execute("SELECT MAX(event_id) FROM canonical_log")
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0
