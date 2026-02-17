from __future__ import annotations

import base64
import hashlib
import json
import queue
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal


@dataclass
class LedgerEvent:
    trace_id: str
    sequence_id: int
    timestamp: str
    actor: Literal["user", "assistant", "tool", "system"]
    event_type: Literal["input", "inference", "tool_invocation", "tool_result", "error", "yield"]
    reasoning_hash: str | None
    execution_hash: str | None
    payload: dict[str, Any]


class AppendOnlyLedger:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or str(uuid.uuid4())
        self._events: list[LedgerEvent] = []
        self._sequence = 0

    def append(
        self,
        *,
        actor: Literal["user", "assistant", "tool", "system"],
        event_type: Literal["input", "inference", "tool_invocation", "tool_result", "error", "yield"],
        payload: dict[str, Any],
        reasoning: str | None = None,
        execution: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        self._sequence += 1
        event = LedgerEvent(
            trace_id=self.trace_id,
            sequence_id=self._sequence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            event_type=event_type,
            reasoning_hash=self.reasoning_hash(reasoning),
            execution_hash=self.execution_hash(execution),
            payload=payload,
        )
        self._events.append(event)
        return event

    def snapshot(self) -> list[LedgerEvent]:
        return list(self._events)

    @staticmethod
    def reasoning_hash(reasoning: str | None) -> str | None:
        if not reasoning:
            return None
        return hashlib.sha256(reasoning.encode("utf-8")).hexdigest()

    @staticmethod
    def execution_hash(execution: dict[str, Any] | None) -> str | None:
        if not execution:
            return None
        tool_name = str(execution.get("tool_name", ""))
        arguments = execution.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        serialized_args = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{tool_name}:{serialized_args}".encode("utf-8")).hexdigest()


class EventLedger:
    def __init__(self, db: Any, app_version: str = "") -> None:
        self._db = db
        self._app_version = app_version
        self._session_id = str(uuid.uuid4())
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()

        self._ensure_schema()

        self._writer_thread = threading.Thread(target=self._writer_loop, name="event-ledger-writer", daemon=True)
        self._writer_thread.start()

        self.record(
            "app",
            "lifecycle",
            "session_start",
            payload={"app_version": app_version, "session_id": self._session_id},
        )

    def _ensure_schema(self) -> None:
        conn = getattr(self._db, "_conn", None)
        if conn is None:
            return

        statements = (
            "ALTER TABLE events ADD COLUMN session_id TEXT DEFAULT ''",
            "ALTER TABLE events ADD COLUMN seq INTEGER DEFAULT 0",
            "ALTER TABLE events ADD COLUMN source TEXT DEFAULT ''",
            "ALTER TABLE events ADD COLUMN kind TEXT DEFAULT ''",
            "ALTER TABLE events ADD COLUMN severity INTEGER DEFAULT 1",
            "ALTER TABLE events ADD COLUMN correlation_id TEXT",
            "ALTER TABLE events ADD COLUMN parent_id INTEGER",
        )

        with self._db._lock:
            for statement in statements:
                try:
                    conn.execute(statement)
                except Exception:
                    continue
            try:
                conn.commit()
            except Exception:
                pass

    def ingest(
        self,
        *,
        source: str,
        kind: str,
        name: str,
        engine_key: str = "",
        payload: Any = None,
        severity: int = 1,
        correlation_id: str | None = None,
        parent_id: int | None = None,
    ) -> None:
        try:
            payload_text = self._safe_payload(payload)
            with self._seq_lock:
                self._seq += 1
                seq = self._seq

            item = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "engine_key": engine_key,
                "event": name,
                "payload": payload_text,
                "session_id": self._session_id,
                "seq": seq,
                "source": source,
                "kind": kind,
                "severity": severity,
                "correlation_id": correlation_id,
                "parent_id": parent_id,
            }
            self._queue.put_nowait(item)
        except queue.Full:
            return
        except Exception:
            return

    def record(
        self,
        source: str,
        kind: str,
        name: str,
        engine_key: str = "",
        payload: Any = None,
        severity: int = 1,
        correlation_id: str | None = None,
        parent_id: int | None = None,
    ) -> None:
        self.ingest(
            source=source,
            kind=kind,
            name=name,
            engine_key=engine_key,
            payload=payload,
            severity=severity,
            correlation_id=correlation_id,
            parent_id=parent_id,
        )

    def _writer_loop(self) -> None:
        batch: list[dict[str, Any]] = []
        batch_start: float | None = None

        while True:
            if self._stop_event.is_set() and self._queue.empty() and not batch:
                break

            timeout = 0.1
            if batch_start is not None:
                timeout = max(0.0, 0.1 - (time.monotonic() - batch_start))

            try:
                item = self._queue.get(timeout=timeout)
                batch.append(item)
                if batch_start is None:
                    batch_start = time.monotonic()
            except queue.Empty:
                pass

            if not batch:
                continue

            if len(batch) >= 50 or self._stop_event.is_set() or (batch_start is not None and time.monotonic() - batch_start >= 0.1):
                self._write_batch(batch)
                batch = []
                batch_start = None

        while True:
            try:
                batch.append(self._queue.get_nowait())
                if len(batch) >= 50:
                    self._write_batch(batch)
                    batch = []
            except queue.Empty:
                break

        if batch:
            self._write_batch(batch)

    def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return

        rows = [
            (
                item["ts"],
                item["engine_key"],
                item["event"],
                item["payload"],
                item["session_id"],
                item["seq"],
                item["source"],
                item["kind"],
                item["severity"],
                item["correlation_id"],
                item["parent_id"],
            )
            for item in batch
        ]

        try:
            with self._db._lock:
                conn = self._db._get_conn()
                if conn is None:
                    return
                conn.executemany(
                    """
                    INSERT INTO events(
                        ts, engine_key, event, payload, session_id, seq,
                        source, kind, severity, correlation_id, parent_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
        except Exception as exc:
            print(f"[EventLedger] write failed: {exc}", file=sys.stderr)

    def _safe_payload(self, payload: Any) -> str:
        try:
            return json.dumps(self._serialize(payload), ensure_ascii=False)
        except Exception:
            fallback = {"_error": "serialization_failed", "repr": repr(payload)[:500]}
            try:
                return json.dumps(fallback, ensure_ascii=False)
            except Exception:
                return '{"_error":"serialization_failed"}'

    def _serialize(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return self._serialize(value.value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        if is_dataclass(value):
            return self._serialize(asdict(value))
        if isinstance(value, dict):
            return {str(k): self._serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize(v) for v in value]
        if value.__class__.__name__ == "QImage":
            return {"_type": "QImage", "repr": repr(value)}
        if hasattr(value, "__dict__"):
            data = {k: v for k, v in vars(value).items() if not k.startswith("_")}
            return self._serialize(data)
        return str(value)

    def shutdown(self) -> None:
        if self._stop_event.is_set():
            return

        self.record(
            "app",
            "lifecycle",
            "session_end",
            payload={"app_version": self._app_version, "session_id": self._session_id},
        )
        self._stop_event.set()
        self._writer_thread.join(timeout=2.0)

        batch: list[dict[str, Any]] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
                if len(batch) >= 50:
                    self._write_batch(batch)
                    batch = []
            except queue.Empty:
                break
        if batch:
            self._write_batch(batch)
