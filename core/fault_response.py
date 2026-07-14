"""Fault response — centralized detection, emission, and read-back for four
named runtime failure modes.

Storage: uses the existing fault_traces table in turn_trace.sqlite3 (Layer E).
No schema changes — existing columns are remapped:
  detector_name  -> source_name
  evidence+meta  -> payload_json dict keys "evidence" and "meta"
  source_kind    = "policy"   (enforced by FaultTraceRecord.__post_init__)
  severity       = "warn"     (required by FaultTraceRecord when fault_kind set)
  event_kind     = "FaultDetectedEvent"

Kill switch: inherits MONOLITH_TURN_TRACE_V1 from turn_trace. Reads return
empty lists gracefully when the flag is off.

Public surface:
  KNOWN_KINDS         frozenset of valid fault_kind strings
  FaultRecord         dataclass view returned by read functions
  emit_fault(...)     append to fault_traces; returns new row id or -1
  read_recent(limit)  newest-first list
  read_by_kind(...)   filtered list
  run_all_detectors   run all four detectors; returns list of FaultRecord
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core import turn_trace as _tt

# ── constants ──────────────────────────────────────────────────────────────

KNOWN_KINDS: frozenset[str] = frozenset({
    "markdown_corruption",
    "tool_no_fire",
    "think_leak",
    "regen_mismatch",
    # Bearing addon escalation kinds — emitted via emit_fault but originate
    # from addons/system/bearing, not from kernel detectors. Documented in
    # the Bearing V0 plan §7 (escalation policy).
    "bearing_structural_unrecoverable",
    "bearing_grounding_failed",
    # MonoExplore expedition-runner escalation (halt on RED coherence / 3 faults).
    "expedition_halted",
    # Self-maintenance daemon (engine/self_maint_runner) halt on 3 consecutive wake
    # faults — the autonomous review-queue triage loop. Sibling of expedition_halted.
    "self_maint_halted",
    # Grounded-verdict V1: the answer cited a recall handle that resolves to
    # nothing (a ground never shown this turn) — the laundering seam firing live.
    # The ONE grounded-verdict bucket that is unambiguously a fault.
    "fabricated_cite",
})

_SEQ_LOCK = threading.Lock()


# ── dataclass view ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FaultRecord:
    """Read-back view of a single fault row.

    Maps from the fault_traces Layer E schema:
      id            row id from SQLite
      turn_id       turn this fault belongs to
      fault_kind    one of KNOWN_KINDS
      detected_at   ISO timestamp (= emitted_at column)
      detector_name which detector function fired (= source_name column)
      evidence      regex match or line slice (stored in payload_json)
      metadata      detector-specific extra context (stored in payload_json)
    """
    id: int
    turn_id: str
    fault_kind: str
    detected_at: str
    detector_name: str
    evidence: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    severity: str | None = None


# ── write path ─────────────────────────────────────────────────────────────


def _next_seq_for_turn(turn_id: str) -> int:
    """Return max(seq)+1 for the turn in fault_traces, thread-safe."""
    try:
        with _tt._db_lock:
            conn = _tt._get_conn()
            if conn is None:
                return 0
            row = conn.execute(
                "SELECT MAX(seq) FROM fault_traces WHERE turn_id = ?",
                (str(turn_id),),
            ).fetchone()
            if row is None or row[0] is None:
                return 0
            return int(row[0]) + 1
    except Exception:
        return 0


def emit_fault(
    turn_id: str,
    fault_kind: str,
    detector_name: str,
    evidence: str | None,
    metadata: dict | None = None,
) -> int:
    """Append a fault row to fault_traces.

    Returns the new row id, or -1 on write failure or unknown kind.
    Never raises — callers are generation-path code.
    """
    if fault_kind not in KNOWN_KINDS:
        _tt._trace_failure(
            f"emit_fault: unknown fault_kind {fault_kind!r}; expected one of {sorted(KNOWN_KINDS)}"
        )
        return -1
    try:
        payload: dict[str, Any] = {}
        if evidence is not None:
            payload["evidence"] = str(evidence)
        if isinstance(metadata, dict) and metadata:
            payload["meta"] = metadata
        seq = _next_seq_for_turn(turn_id)
        record = _tt.FaultTraceRecord(
            turn_id=str(turn_id),
            parent_turn_id=None,
            seq=seq,
            emitted_at=datetime.now(timezone.utc).isoformat(),
            event_kind="FaultDetectedEvent",
            source_kind="policy",
            source_name=str(detector_name),
            authority_tier="observation",
            fault_kind=str(fault_kind),
            severity="warn",
            payload=payload,
        )
        _tt.record_fault(record)
        # Retrieve inserted id.
        try:
            with _tt._db_lock:
                conn = _tt._get_conn()
                if conn is None:
                    return -1
                row = conn.execute(
                    "SELECT id FROM fault_traces WHERE turn_id=? AND source_name=? "
                    "ORDER BY id DESC LIMIT 1",
                    (str(turn_id), str(detector_name)),
                ).fetchone()
            if row is not None:
                return int(row[0])
        except Exception:
            pass
        return -1
    except Exception as exc:
        _tt._trace_failure(f"emit_fault failed: {exc}")
        return -1


# ── read paths ─────────────────────────────────────────────────────────────


def _row_to_fault_record(row: sqlite3.Row) -> FaultRecord:
    try:
        payload = json.loads(row["payload_json"] or "{}")
    except (TypeError, ValueError):
        payload = {}
    keys = row.keys()
    severity = str(row["severity"]) if "severity" in keys and row["severity"] is not None else None
    return FaultRecord(
        id=int(row["id"]),
        turn_id=str(row["turn_id"]),
        fault_kind=str(row["fault_kind"]),
        detected_at=str(row["emitted_at"]),
        detector_name=str(row["source_name"]),
        evidence=payload.get("evidence"),
        metadata=payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
        severity=severity,
    )


_BASE_COLS = (
    "SELECT id, turn_id, emitted_at, event_kind, source_name, severity, "
    "fault_kind, payload_json FROM fault_traces"
)


def _post_filter(records: list[FaultRecord], since: str | None, keyword: str | None) -> list[FaultRecord]:
    out = records
    if since:
        out = [r for r in out if r.detected_at >= since]  # ISO strings sort lexically
    if keyword:
        kw = keyword.lower()
        out = [
            r for r in out
            if kw in (r.fault_kind or "").lower()
            or kw in (r.evidence or "").lower()
            or kw in json.dumps(r.metadata).lower()
        ]
    return out


def read_recent(limit: int = 20, *, since: str | None = None, keyword: str | None = None) -> list[FaultRecord]:
    """Recent fault records, newest first. Optional ISO `since` + substring `keyword`."""
    limit = max(1, min(int(limit or 20), 200))
    try:
        with _tt._db_lock:
            conn = _tt._get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(
                _BASE_COLS + " WHERE fault_kind IS NOT NULL ORDER BY id DESC LIMIT ?",
                (limit,),
            ))
    except Exception as exc:
        _tt._trace_failure(f"read_recent failed: {exc}")
        return []
    return _post_filter([_row_to_fault_record(r) for r in rows], since, keyword)


def read_by_kind(fault_kind: str, limit: int = 20, *, since: str | None = None, keyword: str | None = None) -> list[FaultRecord]:
    """Fault records filtered by fault_kind, newest first."""
    limit = max(1, min(int(limit or 20), 200))
    try:
        with _tt._db_lock:
            conn = _tt._get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(
                _BASE_COLS + " WHERE fault_kind = ? ORDER BY id DESC LIMIT ?",
                (str(fault_kind), limit),
            ))
    except Exception as exc:
        _tt._trace_failure(f"read_by_kind failed: {exc}")
        return []
    return _post_filter([_row_to_fault_record(r) for r in rows], since, keyword)


def read_one(fault_id: int) -> FaultRecord | None:
    """Single fault by row id (for adapter get('fault:<id>'))."""
    try:
        with _tt._db_lock:
            conn = _tt._get_conn()
            if conn is None:
                return None
            row = conn.execute(_BASE_COLS + " WHERE id = ?", (int(fault_id),)).fetchone()
    except Exception as exc:
        _tt._trace_failure(f"read_one failed: {exc}")
        return None
    return _row_to_fault_record(row) if row is not None else None


def read_since_id(after_id: int, limit: int = 500) -> list[FaultRecord]:
    """Paged ascending read by id (for salience.rebuild). Returns id > after_id."""
    limit = max(1, min(int(limit or 500), 2000))
    try:
        with _tt._db_lock:
            conn = _tt._get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(
                _BASE_COLS + " WHERE fault_kind IS NOT NULL AND id > ? ORDER BY id ASC LIMIT ?",
                (int(after_id), limit),
            ))
    except Exception as exc:
        _tt._trace_failure(f"read_since_id failed: {exc}")
        return []
    return [_row_to_fault_record(r) for r in rows]


# ── detector runner ────────────────────────────────────────────────────────


def run_all_detectors(
    response_text: str,
    turn_id: str,
    context: dict,
) -> list[FaultRecord]:
    """Run all four detectors against a response; return any faults found.

    Each detector returns a FaultRecord (not yet emitted) or None.
    Caller is responsible for calling emit_fault on each returned record.
    Detector failures are swallowed — detection must never break generation.
    """
    from core.fault_detectors import (
        detect_markdown_corruption,
        detect_regen_mismatch,
        detect_think_leak,
        detect_tool_no_fire,
    )
    found: list[FaultRecord] = []
    for fn in (
        detect_markdown_corruption,
        detect_tool_no_fire,
        detect_think_leak,
        detect_regen_mismatch,
    ):
        try:
            record = fn(response_text, turn_id, context)
            if record is not None:
                found.append(record)
        except Exception:
            pass
    return found
