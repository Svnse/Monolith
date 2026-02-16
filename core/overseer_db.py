from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from core.paths import LOG_DIR


class OverseerDB:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(LOG_DIR / "overseer.sqlite3", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _get_conn(self) -> sqlite3.Connection | None:
        return self._conn

    def _create_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    engine_key TEXT NOT NULL,
                    event TEXT NOT NULL,
                    payload TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    engine_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ts TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_event(self, engine_key: str, event: str, payload: Any) -> int:
        payload_text = json.dumps(payload)
        with self._lock:
            conn = self._get_conn()
            if conn is None:
                return -1
            cur = conn.execute(
                "INSERT INTO events(ts, engine_key, event, payload) VALUES(?, ?, ?, ?)",
                (self._now(), engine_key, event, payload_text),
            )
            conn.commit()
            return int(cur.lastrowid)

    def log_task(self, task_id: str, engine_key: str, status: str) -> int:
        with self._lock:
            conn = self._get_conn()
            if conn is None:
                return -1
            cur = conn.execute(
                "INSERT INTO tasks(task_id, engine_key, status, ts) VALUES(?, ?, ?, ?)",
                (task_id, engine_key, status, self._now()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_recent_events(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            if conn is None:
                return []
            cur = conn.execute(
                "SELECT id, ts, engine_key, event, payload FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = [self._row_to_event_dict(row) for row in cur.fetchall()]
            rows.reverse()
            return rows

    def get_recent_tasks(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            if conn is None:
                return []
            cur = conn.execute(
                "SELECT id, task_id, engine_key, status, ts FROM tasks ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = [dict(row) for row in cur.fetchall()]
            rows.reverse()
            return rows

    def query_events(
        self,
        engine_key: str | None = None,
        event: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if engine_key is not None:
            clauses.append("engine_key = ?")
            params.append(engine_key)
        if event is not None:
            clauses.append("event = ?")
            params.append(event)
        if after is not None:
            clauses.append("ts >= ?")
            params.append(after)
        if before is not None:
            clauses.append("ts <= ?")
            params.append(before)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._lock:
            conn = self._get_conn()
            if conn is None:
                return []
            cur = conn.execute(
                f"SELECT id, ts, engine_key, event, payload FROM events {where} ORDER BY id DESC LIMIT ?",
                params,
            )
            rows = [self._row_to_event_dict(row) for row in cur.fetchall()]
            rows.reverse()
            return rows

    def _row_to_event_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        payload_raw = row["payload"]
        payload: Any = payload_raw
        if payload_raw is not None:
            try:
                payload = json.loads(payload_raw)
            except Exception:
                payload = payload_raw
        return {
            "id": row["id"],
            "ts": row["ts"],
            "engine_key": row["engine_key"],
            "event": row["event"],
            "payload": payload,
        }

    def close(self) -> None:
        with self._lock:
            if self._conn is None:
                return
            self._conn.close()
            self._conn = None
