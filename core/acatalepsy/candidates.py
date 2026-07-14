"""acu_candidates — the auditor's pending-decision buffer.

Public API:
  - insert_candidate(...)                 -> candidate_id
  - read_pending(limit)                   -> list[Candidate]
  - read_by_state(state, limit)           -> list[Candidate]
  - read_by_auditor_run(auditor_run_id)   -> list[Candidate]
  - read_one(candidate_id)                -> Candidate | None
  - update_state(candidate_id, state)     -> None  (used by decisions module)
  - count_by_state()                      -> dict[state, count]

All writes go through ``authorized_write()`` — SQLite authorizer denies
unsentinel'd writes to acu_candidates per db_connect.GUARDED_TABLES.

The atomicity gate runs at insert time. Non-atomic candidates raise
CandidateAtomicityError; callers MAY catch and log a
``auditor_atomicity_reject`` event to canonical_log instead of letting
the exception propagate.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.db_connect import connect_acatalepsy, authorized_write
from core.acatalepsy.atomicity import is_atomic


__all__ = (
    "Candidate",
    "CandidateAtomicityError",
    "VALID_STATES",
    "count_by_state",
    "insert_candidate",
    "read_by_auditor_run",
    "read_by_state",
    "read_one",
    "read_pending",
    "update_state",
)


VALID_STATES: frozenset[str] = frozenset({
    "pending", "accepted", "rejected", "edited", "deferred",
})


class CandidateAtomicityError(ValueError):
    """Raised when a candidate's canonical_form fails the atomicity gate.

    Carries the rejection reason from is_atomic() so callers can log it
    to canonical_log without re-running the gate.
    """
    def __init__(self, canonical_form: str, reason: str) -> None:
        super().__init__(
            f"non-atomic canonical_form rejected: {reason} "
            f"(form={canonical_form!r})"
        )
        self.canonical_form = canonical_form
        self.reason = reason


@dataclass(frozen=True)
class Candidate:
    id: int
    canonical_form: str
    evidence_log_id: int
    evidence_char_start: int
    evidence_char_end: int
    evidence_span: str
    source: str
    reason: str
    reinforcement_count: int
    contradicts_acu_id: int | None
    state: str
    created_at: str
    auditor_run_id: int | None


# ── Connection (thread-local writer + reader) ────────────────────────


_tl = threading.local()


def _writer_conn() -> sqlite3.Connection:
    conn = getattr(_tl, "writer", None)
    if conn is None:
        conn = connect_acatalepsy(role="memory_writer")
        _tl.writer = conn
    return conn


def _reader_conn() -> sqlite3.Connection:
    conn = getattr(_tl, "reader", None)
    if conn is None:
        conn = connect_acatalepsy(role="reader")
        _tl.reader = conn
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Write ─────────────────────────────────────────────────────────────


def insert_candidate(
    *,
    canonical_form: str,
    evidence_log_id: int,
    evidence_char_start: int,
    evidence_char_end: int,
    evidence_span: str,
    source: str,
    reason: str,
    reinforcement_count: int = 1,
    contradicts_acu_id: int | None = None,
    auditor_run_id: int | None = None,
    sentinel_reason: str | None = None,
) -> int:
    """Insert one candidate. Atomicity gate enforced. Returns row id.

    Raises:
        CandidateAtomicityError — canonical_form failed the gate
        ValueError — invalid char offsets, negative reinforcement, etc.
    """
    gate = is_atomic(canonical_form)
    if not gate.ok:
        raise CandidateAtomicityError(canonical_form, gate.reason or "unknown")

    if evidence_char_start < 0:
        raise ValueError(f"evidence_char_start must be >= 0, got {evidence_char_start}")
    if evidence_char_end < evidence_char_start:
        raise ValueError(
            f"evidence_char_end ({evidence_char_end}) < evidence_char_start ({evidence_char_start})"
        )
    if reinforcement_count < 1:
        raise ValueError(f"reinforcement_count must be >= 1, got {reinforcement_count}")
    if not source.strip():
        raise ValueError("source must be non-empty")
    if not reason.strip():
        raise ValueError("reason must be non-empty")

    s_reason = sentinel_reason or f"candidate:{source}"
    conn = _writer_conn()
    with authorized_write(s_reason):
        cur = conn.execute(
            "INSERT INTO acu_candidates("
            "canonical_form, evidence_log_id, evidence_char_start, "
            "evidence_char_end, evidence_span, source, reason, "
            "reinforcement_count, contradicts_acu_id, state, "
            "created_at, auditor_run_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (
                canonical_form.strip(),
                int(evidence_log_id),
                int(evidence_char_start),
                int(evidence_char_end),
                evidence_span,
                source.strip(),
                reason.strip(),
                int(reinforcement_count),
                contradicts_acu_id,
                _now_iso(),
                auditor_run_id,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_state(candidate_id: int, state: str, *, sentinel_reason: str | None = None) -> None:
    """Set a candidate's state. Used by decisions module to mark
    accept/reject/edit/defer after a decision lands.

    Raises ValueError if state is invalid.
    """
    if state not in VALID_STATES:
        raise ValueError(f"invalid state {state!r}; valid: {sorted(VALID_STATES)}")
    s_reason = sentinel_reason or f"candidate_state:{state}"
    conn = _writer_conn()
    with authorized_write(s_reason):
        conn.execute(
            "UPDATE acu_candidates SET state = ? WHERE id = ?",
            (state, int(candidate_id)),
        )
        conn.commit()


# ── Read ──────────────────────────────────────────────────────────────


def _row_to_candidate(row: sqlite3.Row) -> Candidate:
    return Candidate(
        id=int(row["id"]),
        canonical_form=row["canonical_form"],
        evidence_log_id=int(row["evidence_log_id"]),
        evidence_char_start=int(row["evidence_char_start"]),
        evidence_char_end=int(row["evidence_char_end"]),
        evidence_span=row["evidence_span"],
        source=row["source"],
        reason=row["reason"],
        reinforcement_count=int(row["reinforcement_count"]),
        contradicts_acu_id=(
            int(row["contradicts_acu_id"]) if row["contradicts_acu_id"] is not None else None
        ),
        state=row["state"],
        created_at=row["created_at"],
        auditor_run_id=(
            int(row["auditor_run_id"]) if row["auditor_run_id"] is not None else None
        ),
    )


_SELECT_BASE = (
    "SELECT id, canonical_form, evidence_log_id, evidence_char_start, "
    "evidence_char_end, evidence_span, source, reason, reinforcement_count, "
    "contradicts_acu_id, state, created_at, auditor_run_id "
    "FROM acu_candidates"
)


def read_pending(limit: int = 50) -> list[Candidate]:
    """Pending candidates, oldest first (triage queue order)."""
    return read_by_state("pending", limit=limit)


def read_by_state(state: str, limit: int = 50) -> list[Candidate]:
    if state not in VALID_STATES:
        raise ValueError(f"invalid state {state!r}; valid: {sorted(VALID_STATES)}")
    conn = _reader_conn()
    cur = conn.execute(
        f"{_SELECT_BASE} WHERE state = ? ORDER BY id ASC LIMIT ?",
        (state, int(limit)),
    )
    return [_row_to_candidate(row) for row in cur.fetchall()]


def read_by_auditor_run(auditor_run_id: int) -> list[Candidate]:
    conn = _reader_conn()
    cur = conn.execute(
        f"{_SELECT_BASE} WHERE auditor_run_id = ? ORDER BY id ASC",
        (int(auditor_run_id),),
    )
    return [_row_to_candidate(row) for row in cur.fetchall()]


def read_one(candidate_id: int) -> Candidate | None:
    conn = _reader_conn()
    cur = conn.execute(f"{_SELECT_BASE} WHERE id = ?", (int(candidate_id),))
    row = cur.fetchone()
    return _row_to_candidate(row) if row else None


def count_by_state() -> dict[str, int]:
    """Return {state: count} for all states present in the table."""
    conn = _reader_conn()
    cur = conn.execute(
        "SELECT state, COUNT(*) FROM acu_candidates GROUP BY state"
    )
    return {str(row[0]): int(row[1]) for row in cur.fetchall()}
