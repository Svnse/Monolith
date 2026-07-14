from __future__ import annotations

import json
from pathlib import Path

import pytest


def _clear_acatalepsy_tls() -> None:
    from core.acatalepsy import canonical_log, candidates, decisions, intake

    for mod in (canonical_log, candidates, decisions, intake):
        tl = getattr(mod, "_tl", None)
        if tl is None:
            continue
        for attr in ("writer_conn", "reader_conn", "writer", "reader"):
            if hasattr(tl, attr):
                delattr(tl, attr)


def _setup_acatalepsy_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from core import db_connect as dbc

    db_path = tmp_path / "acatalepsy.sqlite3"
    monkeypatch.setattr(dbc, "DB_PATH", db_path, raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    _clear_acatalepsy_tls()
    from core.acatalepsy import schema

    schema.migrate()
    _clear_acatalepsy_tls()
    return db_path


def _stub_llm(payload: dict):
    text = json.dumps(payload)

    def stub(*, system_prompt: str, user_content: str) -> str:
        assert "Acatalepsy auditor" in system_prompt
        assert "mononote_note_read" in user_content
        return text

    return stub


def test_note_read_evidence_becomes_user_curated_candidate_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_acatalepsy_db(tmp_path, monkeypatch)

    from core.acatalepsy import auditor, canonical_log, candidates, decisions

    note_event_id = canonical_log.append(
        "mononote_note_read",
        payload={
            "schema": "MONONOTE_NOTE_READ_V1",
            "note_id": "note-1",
            "title": "Project Ideas",
            "sha256": "abc",
            "selection_start": 0,
            "selection_end": 10,
        },
    )

    stub = _stub_llm({
        "candidates": [
            {
                "canonical_form": "MonoNote | records | explicit project fact",
                "evidence_log_id": note_event_id,
                "evidence_char_start": 0,
                "evidence_char_end": 10,
                "evidence_span": "project fact",
                "reason": "note-read evidence",
            }
        ]
    })

    result = auditor.run_audit(stub, source="auditor_monolith", start_event_id=0)
    assert result.status == "success"
    assert result.candidates_inserted == 1

    pending = candidates.read_pending()
    assert len(pending) == 1
    candidate = pending[0]
    assert candidate.source == "mononote_note"
    assert "audit_source=auditor_monolith" in candidate.reason

    with pytest.raises(decisions.DecisionAuthorizationError):
        decisions.insert_decision(
            candidate_id=candidate.id,
            decision="accept",
            decided_by="agent_monolith",
        )

    decision_id = decisions.insert_decision(
        candidate_id=candidate.id,
        decision="accept",
        decided_by="user_e",
    )
    assert decision_id > 0
