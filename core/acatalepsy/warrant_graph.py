"""Read-only warrant graph projection over the existing Acatalepsy spine.

This module does not introduce a new store. It joins the tables Acatalepsy
already writes:

    canonical_log -> acu_candidates -> acu_decisions -> acus -> acu_relations

The projection is intentionally model-readable rather than schema-authoritative:
it exposes claim roots with their evidence span, warrant/reason, decision,
truth state, and local ACU relation edges. Pending/rejected candidates that have
not materialized as ACUs are exposed as candidate roots.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from core.db_connect import connect_acatalepsy


_ACU_SELECT = """
    SELECT
        'acu' AS node_kind,
        a.id AS acu_id,
        a.canonical AS claim_text,
        a.provenance AS acu_provenance,
        a.source AS acu_source,
        a.kind AS acu_kind,
        a.l_level AS l_level,
        a.state AS acu_state,
        a.truth AS truth,
        a.truth_confidence AS truth_confidence,
        a.truth_method AS truth_method,
        a.truth_checked_at AS truth_checked_at,
        a.evidence_url AS evidence_url,
        a.evidence_json AS evidence_json,
        a.evidence_spans AS acu_evidence_spans,
        a.source_event AS acu_source_event,
        a.reinforcement AS reinforcement,
        a.cid AS cid,
        a.eqid AS eqid,
        a.locked AS locked,
        a.lock_reason AS lock_reason,
        a.created_at AS acu_created_at,
        a.last_seen AS acu_last_seen,

        c.id AS candidate_id,
        c.canonical_form AS candidate_claim,
        c.evidence_log_id AS evidence_log_id,
        c.evidence_char_start AS evidence_char_start,
        c.evidence_char_end AS evidence_char_end,
        c.evidence_span AS evidence_span,
        c.source AS candidate_source,
        c.reason AS warrant_text,
        c.reinforcement_count AS candidate_reinforcement_count,
        c.contradicts_acu_id AS contradicts_acu_id,
        c.state AS candidate_state,
        c.created_at AS candidate_created_at,
        c.auditor_run_id AS auditor_run_id,

        d.id AS decision_id,
        d.decision AS decision,
        d.decided_by AS decided_by,
        d.decided_at AS decided_at,
        d.reject_reason AS reject_reason,
        d.edited_form AS edited_form,
        d.note AS decision_note,
        d.resulting_acu_id AS resulting_acu_id,

        cl.event_id AS evidence_event_id,
        cl.kind AS evidence_event_kind,
        cl.ts AS evidence_event_ts,
        cl.session_id AS evidence_session_id,

        (
            SELECT COUNT(*)
            FROM acu_relations r
            WHERE r.source_id = a.id OR r.target_id = a.id
        ) AS relation_count,
        (
            SELECT GROUP_CONCAT(
                CASE
                    WHEN r.source_id = a.id THEN
                        'out:' || r.relation || ':acu:' || r.target_id || ':' || COALESCE(t.canonical, '<missing>')
                    ELSE
                        'in:' || r.relation || ':acu:' || r.source_id || ':' || COALESCE(s.canonical, '<missing>')
                END,
                X'0A'
            )
            FROM acu_relations r
            LEFT JOIN acus s ON s.id = r.source_id
            LEFT JOIN acus t ON t.id = r.target_id
            WHERE r.source_id = a.id OR r.target_id = a.id
        ) AS relation_summary
    FROM acus a
    LEFT JOIN acu_candidates c ON c.id = a.candidate_id
    LEFT JOIN acu_decisions d ON d.id = a.decision_id
    LEFT JOIN canonical_log cl ON cl.event_id = COALESCE(c.evidence_log_id, a.source_event)
"""


_CANDIDATE_SELECT = """
    SELECT
        'candidate' AS node_kind,
        NULL AS acu_id,
        c.canonical_form AS claim_text,
        NULL AS acu_provenance,
        NULL AS acu_source,
        NULL AS acu_kind,
        NULL AS l_level,
        NULL AS acu_state,
        NULL AS truth,
        NULL AS truth_confidence,
        NULL AS truth_method,
        NULL AS truth_checked_at,
        NULL AS evidence_url,
        NULL AS evidence_json,
        NULL AS acu_evidence_spans,
        NULL AS acu_source_event,
        NULL AS reinforcement,
        NULL AS cid,
        NULL AS eqid,
        0 AS locked,
        NULL AS lock_reason,
        NULL AS acu_created_at,
        NULL AS acu_last_seen,

        c.id AS candidate_id,
        c.canonical_form AS candidate_claim,
        c.evidence_log_id AS evidence_log_id,
        c.evidence_char_start AS evidence_char_start,
        c.evidence_char_end AS evidence_char_end,
        c.evidence_span AS evidence_span,
        c.source AS candidate_source,
        c.reason AS warrant_text,
        c.reinforcement_count AS candidate_reinforcement_count,
        c.contradicts_acu_id AS contradicts_acu_id,
        c.state AS candidate_state,
        c.created_at AS candidate_created_at,
        c.auditor_run_id AS auditor_run_id,

        d.id AS decision_id,
        d.decision AS decision,
        d.decided_by AS decided_by,
        d.decided_at AS decided_at,
        d.reject_reason AS reject_reason,
        d.edited_form AS edited_form,
        d.note AS decision_note,
        d.resulting_acu_id AS resulting_acu_id,

        cl.event_id AS evidence_event_id,
        cl.kind AS evidence_event_kind,
        cl.ts AS evidence_event_ts,
        cl.session_id AS evidence_session_id,

        0 AS relation_count,
        NULL AS relation_summary
    FROM acu_candidates c
    LEFT JOIN acu_decisions d ON d.id = (
        SELECT d2.id
        FROM acu_decisions d2
        WHERE d2.candidate_id = c.id
        ORDER BY d2.id DESC
        LIMIT 1
    )
    LEFT JOIN canonical_log cl ON cl.event_id = c.evidence_log_id
    WHERE NOT EXISTS (
        SELECT 1 FROM acus a WHERE a.candidate_id = c.id
    )
"""


def _reader() -> sqlite3.Connection:
    conn = connect_acatalepsy(role="reader")
    conn.row_factory = sqlite3.Row
    return conn


def _to_epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _sort_ts(row: dict[str, Any]) -> float:
    for key in ("decided_at", "candidate_created_at", "acu_last_seen", "acu_created_at", "evidence_event_ts"):
        ts = _to_epoch(row.get(key))
        if ts is not None:
            return ts
    return 0.0


def _has_defeater(row: dict[str, Any]) -> bool:
    if row.get("contradicts_acu_id") is not None:
        return True
    if str(row.get("acu_state") or "").strip() == "-inf":
        return True
    if str(row.get("truth") or "").strip().lower() == "contradicted":
        return True
    summary = str(row.get("relation_summary") or "").lower()
    return "contradicts" in summary


def _matches_filters(
    row: dict[str, Any],
    *,
    since: str | None = None,
    state: str | None = None,
    truth: str | None = None,
    decision: str | None = None,
    relation: str | None = None,
    has_defeater: bool | None = None,
) -> bool:
    if since:
        cutoff = _to_epoch(since)
        if cutoff is not None and _sort_ts(row) < cutoff:
            return False
    if state:
        wanted = str(state).strip().lower()
        values = {
            str(row.get("acu_state") or "").strip().lower(),
            str(row.get("candidate_state") or "").strip().lower(),
        }
        if wanted not in values:
            return False
    if truth and str(row.get("truth") or "").strip().lower() != str(truth).strip().lower():
        return False
    if decision and str(row.get("decision") or "").strip().lower() != str(decision).strip().lower():
        return False
    if relation:
        needle = str(relation).strip().lower()
        if needle not in str(row.get("relation_summary") or "").lower():
            return False
    if has_defeater is not None and _has_defeater(row) is not bool(has_defeater):
        return False
    return True


def read_recent(
    limit: int = 20,
    *,
    since: str | None = None,
    node_kind: str | None = None,
    state: str | None = None,
    truth: str | None = None,
    decision: str | None = None,
    relation: str | None = None,
    has_defeater: bool | None = None,
) -> list[dict[str, Any]]:
    """Return recent warrant roots.

    ``node_kind`` may be ``"acu"`` or ``"candidate"``. When omitted, accepted
    ACU roots and non-materialized candidates are combined and sorted by the
    latest relevant timestamp.
    """
    limit = max(1, min(int(limit or 20), 500))
    kind = str(node_kind or "").strip().lower()
    rows: list[dict[str, Any]] = []
    conn = _reader()
    try:
        if kind in {"", "acu"}:
            rows.extend(dict(row) for row in conn.execute(_ACU_SELECT).fetchall())
        if kind in {"", "candidate"}:
            rows.extend(dict(row) for row in conn.execute(_CANDIDATE_SELECT).fetchall())
    finally:
        conn.close()

    rows = [
        row for row in rows
        if _matches_filters(
            row,
            since=since,
            state=state,
            truth=truth,
            decision=decision,
            relation=relation,
            has_defeater=has_defeater,
        )
    ]
    rows.sort(key=_sort_ts, reverse=True)
    return rows[:limit]


def read_one(node_kind: str, node_id: int) -> dict[str, Any] | None:
    kind = str(node_kind or "").strip().lower()
    conn = _reader()
    try:
        if kind == "acu":
            row = conn.execute(_ACU_SELECT + " WHERE a.id = ?", (int(node_id),)).fetchone()
            return dict(row) if row else None
        if kind == "candidate":
            row = conn.execute(_CANDIDATE_SELECT + " AND c.id = ?", (int(node_id),)).fetchone()
            return dict(row) if row else None
    finally:
        conn.close()
    return None
