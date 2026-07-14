"""End-to-end provenance chain test — A1 acceptance criterion #6.

Verifies that for any accepted ACU, the full lineage
    canonical_log → acu_candidates → acu_decisions → acus
is SQL-queryable as a single join, with every link populated.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "test_acatalepsy.sqlite3"
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", db_path, raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    conn = _dbc.connect_acatalepsy(role="migration")
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
    from core.acatalepsy import schema
    schema.migrate()
    from core.acatalepsy import canonical_log, candidates, decisions, intake
    for mod in (canonical_log, candidates, decisions, intake):
        if hasattr(mod, "_tl"):
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(mod._tl, attr):
                    delattr(mod._tl, attr)
    return db_path


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _setup(tmp_path, monkeypatch)


def test_accept_creates_acu_with_full_provenance_chain(db: Path) -> None:
    """Walk the full chain: log event → candidate → accept → ACU.
    Verify SQL join returns every link populated."""
    from core.acatalepsy import canonical_log, candidates, decisions
    from core import db_connect

    # 1. User message lands in canonical_log
    log_id = canonical_log.append(
        "user_message",
        payload={"text": "Monolith uses seven effort tiers.", "agent": "user_e"},
        session_id="connect:user_e",
    )

    # 2. Auditor proposes a candidate referencing that log entry
    candidate_id = candidates.insert_candidate(
        canonical_form="Monolith | uses | seven effort tiers",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=33,
        evidence_span="Monolith uses seven effort tiers.",
        source="auditor_claude",
        reason="load-bearing scaffold count",
    )

    # 3. User accepts
    decision_id = decisions.insert_decision(
        candidate_id=candidate_id,
        decision="accept",
        decided_by="user_e",
    )

    # 4. SQL JOIN — the full provenance chain should resolve in one query
    conn = db_connect.connect_acatalepsy(role="reader")
    row = conn.execute(
        """
        SELECT
            cl.event_id        AS log_event_id,
            cl.kind            AS log_kind,
            cl.payload         AS log_payload,
            c.id               AS candidate_id,
            c.canonical_form   AS candidate_form,
            c.source           AS candidate_source,
            c.state            AS candidate_state,
            d.id               AS decision_id,
            d.decision         AS decision_kind,
            d.decided_by       AS decision_decider,
            d.resulting_acu_id AS decision_acu,
            a.id               AS acu_id,
            a.canonical        AS acu_form,
            a.candidate_id     AS acu_candidate_ptr,
            a.decision_id      AS acu_decision_ptr
        FROM acu_decisions d
        JOIN acu_candidates c ON c.id = d.candidate_id
        JOIN canonical_log cl ON cl.event_id = c.evidence_log_id
        JOIN acus a ON a.id = d.resulting_acu_id
        WHERE d.id = ?
        """,
        (decision_id,),
    ).fetchone()

    assert row is not None, "provenance chain join should return a row"
    # Every link populated end-to-end
    assert row["log_event_id"] == log_id
    assert row["log_kind"] == "user_message"
    assert row["candidate_id"] == candidate_id
    assert row["candidate_form"] == "Monolith | uses | seven effort tiers"
    assert row["candidate_state"] == "accepted"
    assert row["decision_id"] == decision_id
    assert row["decision_kind"] == "accept"
    assert row["decision_decider"] == "user_e"
    assert row["acu_candidate_ptr"] == candidate_id
    assert row["acu_decision_ptr"] == decision_id
    assert row["decision_acu"] == row["acu_id"]
    # Stored canonical is the NORMALIZED form (intake normalizes on ingest).
    assert row["acu_form"] == "monolith | uses | seven effort tiers"


def test_edit_creates_acu_with_edited_form(db: Path) -> None:
    """Edit decision: the resulting ACU's canonical reflects the edited form,
    not the candidate's original form. Both are queryable."""
    from core.acatalepsy import canonical_log, candidates, decisions
    from core import db_connect

    log_id = canonical_log.append("user_message", payload={"text": "test"})
    candidate_id = candidates.insert_candidate(
        canonical_form="rough | claim | with_typo",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_claude",
        reason="needs refining",
    )

    decisions.insert_decision(
        candidate_id=candidate_id,
        decision="edit",
        decided_by="user_e",
        edited_form="refined | claim | no_typo",
    )

    conn = db_connect.connect_acatalepsy(role="reader")
    # The ACU should hold the EDITED form; the candidate keeps the original.
    row = conn.execute(
        """
        SELECT c.canonical_form AS candidate_form, a.canonical AS acu_form
        FROM acus a JOIN acu_candidates c ON c.id = a.candidate_id
        WHERE c.id = ?
        """,
        (candidate_id,),
    ).fetchone()
    assert row is not None
    assert row["candidate_form"] == "rough | claim | with_typo"  # preserved
    assert row["acu_form"] == "refined | claim | no_typo"  # edited form


def test_reject_does_not_create_acu(db: Path) -> None:
    """Rejected candidates do NOT create acus rows."""
    from core.acatalepsy import canonical_log, candidates, decisions
    from core import db_connect

    log_id = canonical_log.append("user_message", payload={"text": "test"})
    candidate_id = candidates.insert_candidate(
        canonical_form="trivial | filler | claim",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_claude",
        reason="surfaced",
    )

    decisions.insert_decision(
        candidate_id=candidate_id,
        decision="reject",
        decided_by="user_e",
        reject_reason="theater — meets no criteria",
    )

    conn = db_connect.connect_acatalepsy(role="reader")
    count = conn.execute(
        "SELECT COUNT(*) FROM acus WHERE candidate_id = ?", (candidate_id,)
    ).fetchone()[0]
    assert count == 0, "reject should not create an acus row"


def _acu_provenance(db_connect, candidate_id: int) -> str:
    conn = db_connect.connect_acatalepsy(role="reader")
    try:
        return conn.execute(
            "SELECT provenance FROM acus WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()["provenance"]
    finally:
        conn.close()


def test_edit_inherits_original_provenance_not_editor(db: Path) -> None:
    """An EDIT inherits the original claim's provenance, never the editor's.
    user_e editing an auditor (self) claim keeps 'self' — no reverse
    truth-laundering and no edit-as-upgrade."""
    from core.acatalepsy import canonical_log, candidates, decisions
    from core import db_connect

    log_id = canonical_log.append("assistant_message", payload={"text": "x"})
    cid = candidates.insert_candidate(
        canonical_form="model | inferred | thing",
        evidence_log_id=log_id, evidence_char_start=0, evidence_char_end=4,
        evidence_span="x", source="auditor_claude", reason="surfaced",
    )
    decisions.insert_decision(
        candidate_id=cid, decision="edit", decided_by="user_e",
        edited_form="model | inferred | refined_thing",
    )
    assert _acu_provenance(db_connect, cid) == "self"  # inherited, not upgraded


def test_user_e_accept_confers_user_provenance(db: Path) -> None:
    """An ACCEPT by user_e DOES confer 'user' standing (E is vouching) — the
    asymmetry vs edit is intentional."""
    from core.acatalepsy import canonical_log, candidates, decisions
    from core import db_connect

    log_id = canonical_log.append("assistant_message", payload={"text": "y"})
    cid = candidates.insert_candidate(
        canonical_form="model | claimed | other_thing",
        evidence_log_id=log_id, evidence_char_start=0, evidence_char_end=4,
        evidence_span="y", source="auditor_claude", reason="surfaced",
    )
    decisions.insert_decision(candidate_id=cid, decision="accept", decided_by="user_e")
    assert _acu_provenance(db_connect, cid) == "user"
