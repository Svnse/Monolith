"""Tests for core/acatalepsy/auditor.py — LLM extraction loop.

Uses a stub LLMCallable so tests run without any actual LLM. Verifies:
  - empty slice short-circuits with status='empty_slice'
  - successful extraction inserts atomic candidates
  - non-atomic proposals are rejected and logged
  - parse errors land in status='failed' without advancing cursor
  - cursor advances on success
  - idempotent re-runs over the same slice produce same proposals (stub),
    but only successful runs advance cursor
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _setup_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    # Force modules to reconnect against the new DB
    from core.acatalepsy import canonical_log, candidates, decisions, auditor
    for mod in (canonical_log, candidates, decisions):
        if hasattr(mod, "_tl"):
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(mod._tl, attr):
                    delattr(mod._tl, attr)
    return db_path


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _setup_isolated_db(tmp_path, monkeypatch)


# ── Stub LLMs ─────────────────────────────────────────────────────────


def make_stub_llm(payload: dict | str) -> "callable":
    """Build an LLMCallable that returns a canned response."""
    if isinstance(payload, dict):
        text = json.dumps(payload)
    else:
        text = payload

    def stub(*, system_prompt: str, user_content: str) -> str:
        # Sanity: prompt has the expected structure
        assert "Acatalepsy auditor" in system_prompt
        assert "Atomicity rule" in system_prompt
        return text

    return stub


# ── empty slice path ─────────────────────────────────────────────────


def test_empty_slice_short_circuits(db: Path) -> None:
    from core.acatalepsy import auditor

    # No log events yet → empty slice
    stub = make_stub_llm({"candidates": [{"canonical_form": "should | not | be inserted",
                                          "evidence_log_id": 0, "evidence_char_start": 0,
                                          "evidence_char_end": 1, "evidence_span": "x",
                                          "reason": "n/a"}]})
    result = auditor.run_audit(stub, source="auditor_test")
    assert result.status == "empty_slice"
    assert result.events_processed == 0
    assert result.candidates_inserted == 0


# ── success path ─────────────────────────────────────────────────────


def test_successful_audit_inserts_candidates(db: Path) -> None:
    from core.acatalepsy import auditor, canonical_log, candidates

    # Seed two user messages
    e1 = canonical_log.append("user_message", payload={"text": "Monolith uses seven effort tiers."})
    e2 = canonical_log.append("user_message", payload={"text": "core/acu_store.py defines a 7-column ACU table."})

    stub = make_stub_llm({
        "candidates": [
            {
                "canonical_form": "Monolith | uses | seven effort tiers",
                "evidence_log_id": e1,
                "evidence_char_start": 0,
                "evidence_char_end": 33,
                "evidence_span": "Monolith uses seven effort tiers.",
                "reason": "load-bearing scaffold count",
            },
            {
                "canonical_form": "core/acu_store.py | defines | 7-column ACU table",
                "evidence_log_id": e2,
                "evidence_char_start": 0,
                "evidence_char_end": 47,
                "evidence_span": "core/acu_store.py defines a 7-column ACU table.",
                "reason": "schema reference",
            },
        ]
    })

    result = auditor.run_audit(stub, source="auditor_test")
    assert result.status == "success"
    assert result.events_processed == 2
    assert result.proposals_returned == 2
    assert result.candidates_inserted == 2
    assert result.candidates_rejected == 0

    pending = candidates.read_pending()
    assert len(pending) == 2
    assert all(c.source == "auditor_test" for c in pending)
    assert all(c.auditor_run_id == result.run_id for c in pending)


# ── non-atomic proposals rejected ─────────────────────────────────────


def test_non_atomic_proposals_rejected(db: Path) -> None:
    from core.acatalepsy import auditor, canonical_log, candidates

    e1 = canonical_log.append("user_message", payload={"text": "anything"})

    stub = make_stub_llm({
        "candidates": [
            # Compound — should be rejected
            {
                "canonical_form": "Monolith | has | tiers and addons",
                "evidence_log_id": e1, "evidence_char_start": 0, "evidence_char_end": 8,
                "evidence_span": "anything", "reason": "test",
            },
            # Valid atomic
            {
                "canonical_form": "valid | atomic | claim",
                "evidence_log_id": e1, "evidence_char_start": 0, "evidence_char_end": 8,
                "evidence_span": "anything", "reason": "test",
            },
        ]
    })
    result = auditor.run_audit(stub, source="auditor_test")
    assert result.status == "success"
    assert result.proposals_returned == 2
    assert result.candidates_inserted == 1
    assert result.candidates_rejected == 1
    assert any("compound_marker" in r for r in result.rejection_reasons)

    pending = candidates.read_pending()
    assert len(pending) == 1
    assert pending[0].canonical_form == "valid | atomic | claim"


# ── parse errors don't advance cursor ────────────────────────────────


def test_parse_error_failed_status_cursor_advances_past_poison(db: Path) -> None:
    from core.acatalepsy import auditor, canonical_log

    e1 = canonical_log.append("user_message", payload={"text": "anything"})

    stub = make_stub_llm("not valid json at all")
    result = auditor.run_audit(stub, source="auditor_test")
    assert result.status == "failed"
    assert result.candidates_inserted == 0
    assert result.error is not None and "parse" in result.error

    # Cursor advances past the poisoned slice so the next run doesn't
    # retry the same events forever. The skipped range is recorded in
    # the auditor_run_failed payload for human inspection.
    assert auditor.last_processed_event_id() >= e1


# ── cursor advances on success ────────────────────────────────────────


def test_cursor_advances_on_success(db: Path) -> None:
    from core.acatalepsy import auditor, canonical_log

    e1 = canonical_log.append("user_message", payload={"text": "first"})
    e2 = canonical_log.append("user_message", payload={"text": "second"})
    e2_id = e2

    stub = make_stub_llm({"candidates": []})  # empty but valid
    result = auditor.run_audit(stub, source="auditor_test")
    assert result.status == "success"
    assert auditor.last_processed_event_id() >= e2_id

    # Second run sees nothing new
    e3 = canonical_log.append("user_message", payload={"text": "third"})
    result2 = auditor.run_audit(stub, source="auditor_test")
    assert result2.slice_start_event_id == result.slice_end_event_id
    # e3 should be in the new slice
    assert result2.slice_end_event_id >= e3


# ── LLM exception handling ────────────────────────────────────────────


def test_llm_exception_yields_failed(db: Path) -> None:
    from core.acatalepsy import auditor, canonical_log

    canonical_log.append("user_message", payload={"text": "test"})

    def broken_llm(*, system_prompt, user_content):
        raise RuntimeError("backend down")

    result = auditor.run_audit(broken_llm, source="auditor_test")
    assert result.status == "failed"
    assert "RuntimeError" in (result.error or "")
    assert auditor.last_processed_event_id() == 0  # cursor not advanced


# ── orphaned-run staleness vs. backward clock ─────────────────────────


def test_close_orphaned_runs_closes_future_dated_run_after_backward_clock(db, monkeypatch):
    """When-plane fix: if the wall clock steps backward after a run started, the
    stored started-ts is now in the FUTURE relative to now. close_orphaned_runs
    must still close such an orphan instead of judging ts>=threshold as 'too
    fresh' forever — otherwise a dead run shows as RUNNING with no worker alive."""
    import time as _time
    from core.acatalepsy import auditor, canonical_log

    # An orphaned run: started, never terminated. ts stamped at the real clock.
    before = _time.time()
    canonical_log.append("auditor_run_started", payload={"trigger": "test"})

    # Clock steps backward well past the started ts (NTP correction / VM resume).
    monkeypatch.setattr(auditor.time, "time", lambda: before - 1000.0)

    closed = auditor.close_orphaned_runs(stale_after_secs=0.0)
    assert closed == 1
