"""Tests for core/acatalepsy/decisions.py — authorization + decision flow.

Uses an in-memory SQLite database mounted at the connector's DB_PATH
for isolation. Doesn't touch the live acatalepsy.sqlite3.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest


def _setup_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point db_connect at a temp file DB and seed the schema."""
    db_path = tmp_path / "test_acatalepsy.sqlite3"
    # Override the DB path constant for this test session
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", db_path, raising=True)
    # Strict auth on for these tests
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    # Seed schema using a migration connection
    conn = _dbc.connect_acatalepsy(role="migration")
    # canonical_log + acus skeletons (minimal — just what FKs need)
    conn.executescript("""
        CREATE TABLE canonical_log (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            kind TEXT NOT NULL,
            session_id TEXT,
            acu_id INTEGER,
            payload TEXT
        );
        CREATE TABLE acus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical TEXT NOT NULL,
            veracity REAL NOT NULL DEFAULT 5.0,
            times_seen INTEGER NOT NULL DEFAULT 1,
            source TEXT NOT NULL DEFAULT 'model',
            created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            candidate_id INTEGER,
            decision_id INTEGER
        );
        CREATE TABLE acu_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL,
            evidence_log_id INTEGER NOT NULL,
            evidence_char_start INTEGER NOT NULL,
            evidence_char_end INTEGER NOT NULL,
            evidence_span TEXT NOT NULL,
            source TEXT NOT NULL,
            reason TEXT NOT NULL,
            reinforcement_count INTEGER NOT NULL DEFAULT 1,
            contradicts_acu_id INTEGER,
            state TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            auditor_run_id INTEGER
        );
        CREATE TABLE acu_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER NOT NULL,
            decision TEXT NOT NULL,
            decided_by TEXT NOT NULL,
            decided_at TEXT NOT NULL,
            reject_reason TEXT,
            edited_form TEXT,
            note TEXT,
            resulting_acu_id INTEGER
        );
    """)
    conn.commit()
    conn.close()
    # Upgrade the hand-rolled skeleton to the full reproducible v9 schema
    # (adds spine columns, renames times_seen->reinforcement) so the L1 intake
    # the decision layer now delegates to has the columns it writes.
    from core.acatalepsy import schema
    schema.migrate()
    # Clear thread-local connections in the modules so they reconnect to the new DB
    from core.acatalepsy import canonical_log, candidates, decisions, intake
    for mod in (canonical_log, candidates, decisions, intake):
        if hasattr(mod, "_tl"):
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(mod._tl, attr):
                    delattr(mod._tl, attr)
    return db_path


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _setup_isolated_db(tmp_path, monkeypatch)


@pytest.fixture
def sample_candidate(db: Path) -> int:
    """Insert a sample candidate from auditor_claude. Returns its id."""
    from core.acatalepsy import canonical_log, candidates
    log_id = canonical_log.append(
        "assistant_message",
        payload={"text": "Monolith uses seven effort tiers."},
    )
    return candidates.insert_candidate(
        canonical_form="Monolith | uses | seven effort tiers",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=44,
        evidence_span="Monolith uses seven effort tiers.",
        source="auditor_claude",
        reason="load-bearing scaffold count",
    )


# ── Authorization rule ───────────────────────────────────────────────


def test_user_e_can_decide_any(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import decisions
    decision_id = decisions.insert_decision(
        candidate_id=sample_candidate,
        decision="accept",
        decided_by="user_e",
    )
    assert decision_id > 0


def test_agent_can_decide_own_source(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import decisions
    decision_id = decisions.insert_decision(
        candidate_id=sample_candidate,
        decision="accept",
        decided_by="agent_claude",
    )
    assert decision_id > 0


def test_agent_cannot_decide_other_agent_source(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import decisions
    with pytest.raises(decisions.DecisionAuthorizationError):
        decisions.insert_decision(
            candidate_id=sample_candidate,
            decision="accept",
            decided_by="agent_gpt",
        )


def test_agent_cannot_decide_user_stated_source(db: Path) -> None:
    """A candidate with source='user_stated' should be decidable only by user_e."""
    from core.acatalepsy import canonical_log, candidates, decisions
    log_id = canonical_log.append("user_message", payload={"text": "test"})
    candidate_id = candidates.insert_candidate(
        canonical_form="user | likes | python",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=20,
        evidence_span="I like python",
        source="user_stated",
        reason="explicit preference",
    )
    with pytest.raises(decisions.DecisionAuthorizationError):
        decisions.insert_decision(
            candidate_id=candidate_id,
            decision="accept",
            decided_by="agent_claude",
        )
    # But user_e should be allowed
    decision_id = decisions.insert_decision(
        candidate_id=candidate_id,
        decision="accept",
        decided_by="user_e",
    )
    assert decision_id > 0


def test_unknown_role_denied_by_default(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import decisions
    with pytest.raises(decisions.DecisionAuthorizationError):
        decisions.insert_decision(
            candidate_id=sample_candidate,
            decision="accept",
            decided_by="rando_caller",
        )


# ── Required fields per decision type ────────────────────────────────


def test_reject_requires_reason(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import decisions
    with pytest.raises(ValueError, match="reject_reason"):
        decisions.insert_decision(
            candidate_id=sample_candidate,
            decision="reject",
            decided_by="user_e",
        )


def test_edit_requires_form(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import decisions
    with pytest.raises(ValueError, match="edited_form"):
        decisions.insert_decision(
            candidate_id=sample_candidate,
            decision="edit",
            decided_by="user_e",
        )


def test_invalid_decision_rejected(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import decisions
    with pytest.raises(ValueError, match="invalid decision"):
        decisions.insert_decision(
            candidate_id=sample_candidate,
            decision="annihilate",
            decided_by="user_e",
        )


# ── Candidate state mutation on decision ─────────────────────────────


def test_accept_marks_candidate_accepted(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import candidates, decisions
    decisions.insert_decision(
        candidate_id=sample_candidate,
        decision="accept",
        decided_by="user_e",
    )
    cand = candidates.read_one(sample_candidate)
    assert cand is not None
    assert cand.state == "accepted"


def test_reject_marks_candidate_rejected_and_preserves(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import candidates, decisions
    decisions.insert_decision(
        candidate_id=sample_candidate,
        decision="reject",
        decided_by="user_e",
        reject_reason="too generic",
    )
    cand = candidates.read_one(sample_candidate)
    assert cand is not None
    assert cand.state == "rejected"
    # Candidate row preserved — not deleted
    assert cand.canonical_form == "Monolith | uses | seven effort tiers"


def test_defer_marks_candidate_deferred(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import candidates, decisions
    decisions.insert_decision(
        candidate_id=sample_candidate,
        decision="defer",
        decided_by="user_e",
        note="come back to this",
    )
    cand = candidates.read_one(sample_candidate)
    assert cand is not None
    assert cand.state == "deferred"


# ── Multiple decisions on same candidate ─────────────────────────────


def test_defer_then_accept_chain(db: Path, sample_candidate: int) -> None:
    """A candidate can be deferred and then later accepted; both decisions log."""
    from core.acatalepsy import candidates, decisions
    d1 = decisions.insert_decision(
        candidate_id=sample_candidate,
        decision="defer",
        decided_by="user_e",
    )
    d2 = decisions.insert_decision(
        candidate_id=sample_candidate,
        decision="accept",
        decided_by="user_e",
    )
    chain = decisions.read_by_candidate(sample_candidate)
    assert [d.id for d in chain] == [d1, d2]
    assert [d.decision for d in chain] == ["defer", "accept"]

    cand = candidates.read_one(sample_candidate)
    assert cand is not None
    assert cand.state == "accepted"  # final state


# ── canonical_log linkage ────────────────────────────────────────────


def test_decision_emits_canonical_log_event(db: Path, sample_candidate: int) -> None:
    from core.acatalepsy import canonical_log, decisions
    before = canonical_log.latest_event_id()
    decisions.insert_decision(
        candidate_id=sample_candidate,
        decision="accept",
        decided_by="user_e",
    )
    after = canonical_log.latest_event_id()
    assert after > before
    ev = canonical_log.read_one(after)
    assert ev is not None
    assert ev.kind == "candidate_accepted"
    assert ev.payload["candidate_id"] == sample_candidate
    assert ev.payload["decided_by"] == "user_e"


# ── Nonexistent candidate ────────────────────────────────────────────


def test_decision_on_missing_candidate(db: Path) -> None:
    from core.acatalepsy import decisions
    with pytest.raises(ValueError, match="does not exist"):
        decisions.insert_decision(
            candidate_id=99999,
            decision="accept",
            decided_by="user_e",
        )
