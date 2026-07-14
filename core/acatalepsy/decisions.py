"""acu_decisions — accept/reject/edit/defer events on candidates.

Public API:
  - insert_decision(...)                  -> decision_id
  - read_by_candidate(candidate_id)       -> list[Decision]
  - read_by_decider(decided_by, limit)    -> list[Decision]
  - read_one(decision_id)                 -> Decision | None

Authorization rule (enforced at insert):
  - decided_by = "user_e" → may decide any candidate
  - decided_by = "agent_<name>" → may decide ONLY candidates whose
    source is "auditor_<name>" (same suffix). Keeps E's authorship
    asymmetric: peer agents curate their own contributions, never
    user-stated or other-agent claims.

Insert side-effects:
  - Sets the candidate's state to accepted/rejected/edited/deferred
    via candidates.update_state().
  - Appends a corresponding event to canonical_log
    (candidate_accepted/rejected/edited/deferred).

All writes go through ``authorized_write`` — SQLite authorizer denies
unsentinel'd writes to acu_decisions per db_connect.GUARDED_TABLES.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from core.db_connect import connect_acatalepsy, authorized_write
from core.acatalepsy import candidates as _candidates
from core.acatalepsy import canonical_log as _canonical_log
from core.acatalepsy import intake as _intake


__all__ = (
    "Decision",
    "DecisionAuthorizationError",
    "VALID_DECISIONS",
    "insert_decision",
    "read_by_candidate",
    "read_by_decider",
    "read_one",
)


VALID_DECISIONS: frozenset[str] = frozenset({
    "accept", "reject", "edit", "defer",
})


# Map decision → resulting candidate state and canonical_log event kind.
_DECISION_TO_CANDIDATE_STATE: dict[str, str] = {
    "accept": "accepted",
    "reject": "rejected",
    "edit": "edited",
    "defer": "deferred",
}
_DECISION_TO_LOG_KIND: dict[str, str] = {
    "accept": "candidate_accepted",
    "reject": "candidate_rejected",
    "edit": "candidate_edited",
    "defer": "candidate_deferred",
}


class DecisionAuthorizationError(PermissionError):
    """Raised when an agent attempts to decide on a candidate it isn't
    authorized to curate (cross-agent decisions on user-stated or
    other-agent-sourced candidates)."""


@dataclass(frozen=True)
class Decision:
    id: int
    candidate_id: int
    decision: str
    decided_by: str
    decided_at: str
    reject_reason: str | None
    edited_form: str | None
    note: str | None
    resulting_acu_id: int | None


# ── Authorization ─────────────────────────────────────────────────────


def _is_authorized_for(decided_by: str, candidate_source: str) -> bool:
    """Return True if ``decided_by`` is allowed to decide a candidate
    with source ``candidate_source``. See module docstring rule.
    """
    db = decided_by.strip()
    src = candidate_source.strip()
    if db == "user_e":
        return True
    if not db.startswith("agent_"):
        # Unknown decider role — deny by default. v1 conservative.
        return False
    if not src.startswith("auditor_"):
        # Non-agent-sourced candidate (e.g., user_stated) → agents may
        # not curate.
        return False
    # Compare the suffix after the prefix.
    agent_suffix = db[len("agent_"):]
    source_suffix = src[len("auditor_"):]
    return agent_suffix == source_suffix


def _from_source(candidate_source: str) -> str:
    """Map a candidate's free-text source to a self|user|world provenance."""
    s = (candidate_source or "").strip().lower()
    if s in {"user_stated", "mononote_note"}:
        return "user"
    if s == "tool" or s.startswith("world"):
        return "world"
    return "self"  # auditor_*, model, etc.


def _provenance_for(decision: str, decided_by: str, candidate_source: str) -> str:
    """Provenance the materialized ACU should carry.

    accept: user_e vouches → 'user'; an agent accepting retains the claim's
    extraction provenance ('self' for auditor_*), never 'world'.
    edit: inherits the ORIGINAL claim's provenance, never the editor's — this
    prevents reverse truth-laundering (a user fact downgraded to self by an
    auditor edit, losing its Mad-Cow truth-standing).
    """
    if decision == "edit":
        return _from_source(candidate_source)
    if decided_by.strip() == "user_e":
        return "user"
    return _from_source(candidate_source)


# ── Connections ───────────────────────────────────────────────────────


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


def insert_decision(
    *,
    candidate_id: int,
    decision: str,
    decided_by: str,
    reject_reason: str | None = None,
    edited_form: str | None = None,
    note: str | None = None,
    resulting_acu_id: int | None = None,
    sentinel_reason: str | None = None,
) -> int:
    """Insert one decision. Enforces:
      - decision in VALID_DECISIONS
      - authorization: decided_by must be allowed to curate this candidate
      - reject_reason required when decision='reject'
      - edited_form required when decision='edit'

    Side effects:
      - Updates the candidate's state via candidates.update_state()
      - Appends a candidate_<decision> event to canonical_log

    Returns the new decision_id.

    Raises:
        DecisionAuthorizationError — decided_by may not curate this candidate
        ValueError — invalid decision, missing required fields, missing candidate
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"invalid decision {decision!r}; valid: {sorted(VALID_DECISIONS)}"
        )
    if decision == "reject" and not (reject_reason and reject_reason.strip()):
        raise ValueError("reject decision requires non-empty reject_reason")
    if decision == "edit" and not (edited_form and edited_form.strip()):
        raise ValueError("edit decision requires non-empty edited_form")
    if not decided_by.strip():
        raise ValueError("decided_by must be non-empty")

    # Load candidate to check authorization + existence.
    candidate = _candidates.read_one(candidate_id)
    if candidate is None:
        raise ValueError(f"candidate_id={candidate_id} does not exist")

    if not _is_authorized_for(decided_by, candidate.source):
        raise DecisionAuthorizationError(
            f"decided_by={decided_by!r} not authorized to decide on candidate "
            f"id={candidate_id} (source={candidate.source!r})"
        )

    # ONE atomic transaction on ONE connection: materialize/reinforce the leaf
    # ACU via the one-writer L1 intake (passing our conn so it does NOT commit),
    # write the decision row, update candidate state, and emit the decision
    # event — so the ACU + its intake event + the decision + the decision event
    # all commit together. No orphan ACU, no event lost relative to its mutation.
    s_reason = sentinel_reason or f"decision:{decision}:{decided_by}"
    conn = _writer_conn()
    materialized_acu_id: int | None = resulting_acu_id
    intake_outcome: str | None = None
    with authorized_write(s_reason):
        if decision in ("accept", "edit") and resulting_acu_id is None:
            canonical = (
                edited_form.strip() if decision == "edit" and edited_form
                else candidate.canonical_form
            )
            prov = _provenance_for(decision, decided_by, candidate.source)
            intake_res = _intake.ingest_l1(
                raw_form=canonical, provenance=prov, conn=conn,
                source_event=int(candidate_id),
            )
            intake_outcome = intake_res.outcome
            materialized_acu_id = intake_res.acu_id if intake_res.acu_id > 0 else None

        cur = conn.execute(
            "INSERT INTO acu_decisions("
            "candidate_id, decision, decided_by, decided_at, "
            "reject_reason, edited_form, note, resulting_acu_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(candidate_id),
                decision,
                decided_by.strip(),
                _now_iso(),
                reject_reason.strip() if reject_reason else None,
                edited_form.strip() if edited_form else None,
                note.strip() if note else None,
                materialized_acu_id,
            ),
        )
        decision_id = int(cur.lastrowid)

        # Link provenance pointers only when intake created a NEW row
        # (novel/partial). For a dedup MATCH the existing row keeps its
        # first-decision pointers; this decision is captured by the
        # acu_decisions row + the intake's canonical_log event.
        if materialized_acu_id is not None and intake_outcome in ("novel", "partial"):
            conn.execute(
                "UPDATE acus SET candidate_id = ?, decision_id = ? "
                "WHERE id = ? AND merged_into IS NULL",
                (int(candidate_id), decision_id, materialized_acu_id),
            )

        # Update candidate state
        new_state = _DECISION_TO_CANDIDATE_STATE[decision]
        conn.execute(
            "UPDATE acu_candidates SET state = ? WHERE id = ?",
            (new_state, int(candidate_id)),
        )

        # Emit the decision event on the SAME connection/transaction.
        log_kind = _DECISION_TO_LOG_KIND[decision]
        _canonical_log.append_on(
            conn,
            log_kind,
            payload={
                "candidate_id": int(candidate_id),
                "decision_id": decision_id,
                "decided_by": decided_by.strip(),
                "reject_reason": reject_reason.strip() if reject_reason else None,
                "edited_form": edited_form.strip() if edited_form else None,
                "resulting_acu_id": materialized_acu_id,
            },
            acu_id=materialized_acu_id,
        )
        conn.commit()

    return decision_id


# ── Read ──────────────────────────────────────────────────────────────


def _row_to_decision(row: sqlite3.Row) -> Decision:
    return Decision(
        id=int(row["id"]),
        candidate_id=int(row["candidate_id"]),
        decision=row["decision"],
        decided_by=row["decided_by"],
        decided_at=row["decided_at"],
        reject_reason=row["reject_reason"],
        edited_form=row["edited_form"],
        note=row["note"],
        resulting_acu_id=(
            int(row["resulting_acu_id"]) if row["resulting_acu_id"] is not None else None
        ),
    )


_SELECT_BASE = (
    "SELECT id, candidate_id, decision, decided_by, decided_at, "
    "reject_reason, edited_form, note, resulting_acu_id "
    "FROM acu_decisions"
)


def read_by_candidate(candidate_id: int) -> list[Decision]:
    """All decisions on a given candidate, ordered ascending by id.

    A candidate may have multiple decisions (deferred then accepted,
    edit-then-accept, etc.) — the chain is preserved.
    """
    conn = _reader_conn()
    cur = conn.execute(
        f"{_SELECT_BASE} WHERE candidate_id = ? ORDER BY id ASC",
        (int(candidate_id),),
    )
    return [_row_to_decision(row) for row in cur.fetchall()]


def read_by_decider(decided_by: str, limit: int = 100) -> list[Decision]:
    conn = _reader_conn()
    cur = conn.execute(
        f"{_SELECT_BASE} WHERE decided_by = ? ORDER BY id DESC LIMIT ?",
        (str(decided_by), int(limit)),
    )
    return [_row_to_decision(row) for row in cur.fetchall()]


def read_one(decision_id: int) -> Decision | None:
    conn = _reader_conn()
    cur = conn.execute(f"{_SELECT_BASE} WHERE id = ?", (int(decision_id),))
    row = cur.fetchone()
    return _row_to_decision(row) if row else None
