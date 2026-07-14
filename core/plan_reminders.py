from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.paths import LOG_DIR


_DB_PATH = LOG_DIR / "turn_trace.sqlite3"
_DDL = (
    """
    CREATE TABLE IF NOT EXISTS plan_reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reminder_uid TEXT NOT NULL UNIQUE,
        plan_uid TEXT,
        message TEXT NOT NULL,
        due_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        seen_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_plan_reminders_due ON plan_reminders(status, due_at)",
)

_db_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_db_path_override: Path | None = None


def _get_db_path() -> Path:
    return _db_path_override if _db_path_override is not None else _DB_PATH


def set_db_path(path: Path | None) -> None:
    global _conn, _db_path_override
    with _db_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
        _db_path_override = path


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    path = _get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=4000")
    for stmt in _DDL:
        conn.execute(stmt)
    _conn = conn
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_due_at(value: str | datetime) -> str:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    text = str(value or "").strip()
    if not text:
        raise ValueError("due_at is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def create_reminder(message: str, due_at: str | datetime, *, plan_uid: str | None = None) -> str:
    msg = str(message or "").strip()
    if not msg:
        raise ValueError("reminder requires a message")
    uid = uuid.uuid4().hex
    due = _normalize_due_at(due_at)
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO plan_reminders(reminder_uid, plan_uid, message, due_at, status, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (uid, plan_uid, msg, due, _now_iso()),
        )
    return uid


def _row_to_reminder(row: sqlite3.Row) -> dict:
    return {
        "reminder_uid": row["reminder_uid"],
        "plan_uid": row["plan_uid"],
        "message": row["message"],
        "due_at": row["due_at"],
        "status": row["status"],
        "created_at": row["created_at"],
        "seen_at": row["seen_at"],
    }


def list_due_reminders(*, now: str | datetime | None = None, limit: int = 20) -> list[dict]:
    cutoff = _normalize_due_at(now or datetime.now(timezone.utc))
    limit = max(1, min(int(limit or 20), 200))
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM plan_reminders WHERE status='pending' AND due_at <= ? "
            "ORDER BY due_at ASC, id ASC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    return [_row_to_reminder(row) for row in rows]


def list_reminders(limit: int = 50) -> list[dict]:
    limit = max(1, min(int(limit or 50), 500))
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM plan_reminders ORDER BY due_at ASC, id ASC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_reminder(row) for row in rows]


def mark_reminder_seen(reminder_uid: str, *, seen_at: str | datetime | None = None) -> None:
    seen = _normalize_due_at(seen_at or datetime.now(timezone.utc))
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE plan_reminders SET status='seen', seen_at=? WHERE reminder_uid=?",
            (seen, str(reminder_uid)),
        )
