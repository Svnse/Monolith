"""Tests for the MonoBase status read model."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _setup_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "test_acatalepsy.sqlite3"
    from core import db_connect as _dbc

    monkeypatch.setattr(_dbc, "DB_PATH", db_path, raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")

    conn = _dbc.connect_acatalepsy(role="migration")
    conn.executescript(
        """
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
            reinforcement INTEGER NOT NULL DEFAULT 1,
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
        """
    )
    conn.commit()
    conn.close()

    from core.acatalepsy import schema
    schema.migrate()

    from core.acatalepsy import canonical_log, candidates, decisions, intake, runtime

    runtime.deregister_worker()
    for mod in (canonical_log, candidates, decisions, intake):
        if hasattr(mod, "_tl"):
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(mod._tl, attr):
                    delattr(mod._tl, attr)
    return db_path


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _setup_isolated_db(tmp_path, monkeypatch)


def _stub_llm(payload: dict) -> object:
    text = json.dumps(payload)

    def stub(*, system_prompt: str, user_content: str) -> str:
        return text

    return stub


def test_audit_candidates_are_not_reported_as_acu_writes(db: Path) -> None:
    from core.acatalepsy import auditor, canonical_log, candidates, decisions
    from core.acatalepsy.monobase_status import build_monobase_snapshot, read_acu, read_recent_acus

    evidence_id = canonical_log.append(
        "user_message",
        payload={"text": "Monolith uses seven effort tiers."},
    )
    auditor.run_audit(
        _stub_llm(
            {
                "candidates": [
                    {
                        "canonical_form": "Monolith | uses | seven effort tiers",
                        "evidence_log_id": evidence_id,
                        "evidence_char_start": 0,
                        "evidence_char_end": 33,
                        "evidence_span": "Monolith uses seven effort tiers.",
                        "reason": "load-bearing scaffold count",
                    }
                ]
            }
        ),
        source="auditor_test",
    )

    pending = candidates.read_pending()
    assert len(pending) == 1
    snapshot = build_monobase_snapshot()
    assert snapshot.pending_candidate_count == 1
    assert snapshot.recent_acu_writes == ()

    decision_id = decisions.insert_decision(
        candidate_id=pending[0].id,
        decision="accept",
        decided_by="user_e",
    )
    decision = decisions.read_one(decision_id)
    assert decision is not None
    assert decision.resulting_acu_id is not None

    snapshot = build_monobase_snapshot()
    assert snapshot.pending_candidate_count == 0
    assert snapshot.candidate_counts["accepted"] == 1
    assert len(snapshot.recent_acu_writes) == 1
    assert snapshot.recent_acu_writes[0].acu_id == decision.resulting_acu_id
    assert snapshot.recent_acu_writes[0].candidate_id == pending[0].id
    assert snapshot.recent_acu_writes[0].canonical_form == pending[0].canonical_form
    acus = read_recent_acus()
    assert len(acus) == 1
    assert acus[0].id == decision.resulting_acu_id
    assert read_acu(decision.resulting_acu_id).id == decision.resulting_acu_id
    # The stored ACU canonical is now the NORMALIZED form (intake normalizes).
    from core.acatalepsy.normalize import normalize_canonical
    assert acus[0].canonical == normalize_canonical(pending[0].canonical_form)


def test_snapshot_reports_llm_call_in_progress(db: Path) -> None:
    from core.acatalepsy import canonical_log
    from core.acatalepsy.monobase_status import build_monobase_snapshot

    run_id = canonical_log.append(
        "auditor_run_started",
        payload={
            "source": "auditor_test",
            "slice_start_event_id": 10,
            "slice_end_event_id": 20,
            "prompt_version": 1,
        },
    )
    canonical_log.append(
        "auditor_llm_call_started",
        payload={
            "run_id": run_id,
            "prompt_chars": 100,
            "user_chars": 200,
            "events_in_slice": 2,
        },
    )

    snapshot = build_monobase_snapshot(now=9999999999.0)
    assert snapshot.phase == "calling_llm"
    assert snapshot.phase_title == "Calling auditor LLM"
    assert snapshot.llm_elapsed_secs is not None
    assert "LLM elapsed" in snapshot.phase_detail


def test_snapshot_reports_candidate_update_after_llm_returns(db: Path) -> None:
    from core.acatalepsy import canonical_log
    from core.acatalepsy.monobase_status import build_monobase_snapshot

    run_id = canonical_log.append(
        "auditor_run_started",
        payload={
            "source": "auditor_test",
            "slice_start_event_id": 0,
            "slice_end_event_id": 4,
            "prompt_version": 1,
        },
    )
    canonical_log.append(
        "auditor_llm_call_started",
        payload={"run_id": run_id, "prompt_chars": 100, "user_chars": 200},
    )
    canonical_log.append(
        "auditor_llm_call_returned",
        payload={"run_id": run_id, "elapsed_secs": 12.4, "status": "ok"},
    )

    snapshot = build_monobase_snapshot()
    assert snapshot.phase == "updating_candidates"
    assert "pending candidates" in snapshot.phase_detail
    assert "ACUs are written only on Accept/Edit" in snapshot.phase_detail


def test_latest_llm_output_reads_returned_preview(db: Path) -> None:
    from core.acatalepsy import auditor, canonical_log
    from core.acatalepsy.monobase_status import read_latest_llm_output

    canonical_log.append("user_message", payload={"text": "Monolith uses seven effort tiers."})
    response = {
        "candidates": [
            {
                "canonical_form": "Monolith | uses | seven effort tiers",
                "evidence_log_id": 1,
                "evidence_char_start": 0,
                "evidence_char_end": 33,
                "evidence_span": "Monolith uses seven effort tiers.",
                "reason": "load-bearing scaffold count",
            }
        ]
    }
    auditor.run_audit(_stub_llm(response), source="auditor_test")

    output = read_latest_llm_output()
    assert output is not None
    assert output.status == "ok"
    assert output.response_chars is not None and output.response_chars > 0
    assert "Monolith | uses | seven effort tiers" in output.response_preview


def test_worker_status_detects_stopping_thread() -> None:
    from core.acatalepsy.monobase_status import read_worker_status

    class _Thread:
        def is_alive(self) -> bool:
            return True

    class _StopEvent:
        def is_set(self) -> bool:
            return True

    class _Queue:
        def size(self) -> int:
            return 3

    class _Worker:
        _thread = _Thread()
        _stop_event = _StopEvent()
        _size_threshold = 50
        _max_events = 25
        queue_handle = _Queue()

    status = read_worker_status(_Worker())
    assert status.registered is True
    assert status.thread_alive is True
    assert status.stop_requested is True
    assert status.queue_size == 3
    assert status.size_threshold == 50
    assert status.max_events_per_run == 25


def test_status_strip_separates_run_and_llm_clocks() -> None:
    _app()
    from core.acatalepsy.monobase_status import MonobaseSnapshot, WorkerStatus
    from ui.addons.monobase_widgets import MonoBaseStatusStrip

    strip = MonoBaseStatusStrip()
    try:
        snapshot = MonobaseSnapshot(
            now=100.0,
            phase="calling_llm",
            phase_title="Calling auditor LLM",
            phase_detail="Run #7 slice 1-2; LLM elapsed 9s.",
            phase_tone="active",
            cursor=1,
            latest_event_id=2,
            pending_log_events=1,
            candidate_counts={"pending": 1, "accepted": 2, "rejected": 3},
            pending_candidate_count=1,
            worker=WorkerStatus(
                registered=True,
                thread_alive=True,
                stop_requested=False,
                queue_size=0,
                size_threshold=50,
                max_events_per_run=25,
            ),
            in_flight_run={"event_id": 7},
            run_elapsed_secs=12,
            llm_elapsed_secs=9,
            recent_runs=(),
            recent_acu_writes=(),
        )

        strip.apply_snapshot(snapshot)

        assert strip._run.text() == "Run clock\n12s"
        assert strip._llm.text() == "LLM request\n9s live"
        assert strip._stage_line.text() == "LOG -> [LLM] -> CANDIDATES -> ACU"
    finally:
        strip.close()
