from __future__ import annotations

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


def test_read_note_skill_returns_metatag_and_logs_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_acatalepsy_db(tmp_path, monkeypatch)

    from core import paths, skill_runtime
    from core.acatalepsy import canonical_log
    from core.mononote import store
    from core.skill_registry import clear_skill_cache

    monkeypatch.setattr(paths, "NOTES_DIR", tmp_path / "notes", raising=False)
    store.write_note("Project Ideas", "abcdef")
    skill_runtime._DYNAMIC_EXECUTOR_CACHE.clear()
    clear_skill_cache()

    ctx = skill_runtime.ToolExecutionContext(archive_dir=tmp_path, parent_turn_id="turn-1")
    result = skill_runtime.execute_tool_call_enveloped(
        {"tool": "read_note", "title": "Project Ideas", "max_chars": 3},
        ctx,
    )

    assert result.ok is True
    assert "[NOTE_READ event_id=" in result.text
    assert 'title="Project Ideas"' in result.text
    assert "range=0:3" in result.text
    assert "truncated=true" in result.text
    assert result.text.endswith("\nabc")

    event = canonical_log.read_one(canonical_log.latest_event_id())
    assert event is not None
    assert event.kind == "mononote_note_read"
    assert event.session_id == "turn-1"


def test_read_note_skill_denies_when_prompt_does_not_authorize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_acatalepsy_db(tmp_path, monkeypatch)

    from core import paths, skill_runtime
    from core.acatalepsy import canonical_log
    from core.mononote import store
    from core.skill_registry import clear_skill_cache

    monkeypatch.setattr(paths, "NOTES_DIR", tmp_path / "notes", raising=False)
    store.write_note("Secret Plan", "do not read")
    skill_runtime._DYNAMIC_EXECUTOR_CACHE.clear()
    clear_skill_cache()

    class World:
        def snapshot(self) -> dict:
            return {"session": {"last_user_prompt": "tell me a joke"}}

    ctx = skill_runtime.ToolExecutionContext(archive_dir=tmp_path, world_state=World())
    result = skill_runtime.execute_tool_call_enveloped(
        {"tool": "read_note", "title": "Secret Plan"},
        ctx,
    )

    assert result.ok is False
    assert "denied" in result.text
    assert canonical_log.latest_event_id() == 0
