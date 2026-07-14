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


def test_record_note_read_appends_canonical_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_acatalepsy_db(tmp_path, monkeypatch)

    from core import paths
    from core.acatalepsy import canonical_log
    from core.mononote import provenance, store

    monkeypatch.setattr(paths, "NOTES_DIR", tmp_path / "notes", raising=False)
    note = store.write_note("Project Ideas", "abcdef")
    _loaded, body = store.load_note("Project Ideas")
    window = store.slice_note_text(body, max_chars=3)

    event_id = provenance.record_note_read(note, window, read_mode="skill")
    event = canonical_log.read_one(event_id)

    assert event is not None
    assert event.kind == "mononote_note_read"
    assert event.payload is not None
    assert event.payload["schema"] == "MONONOTE_NOTE_READ_V1"
    assert event.payload["note_id"] == note.note_id
    assert event.payload["sha256"] == note.sha256
    assert event.payload["selection_start"] == 0
    assert event.payload["selection_end"] == 3
    assert event.payload["truncated"] is True


def test_skill_note_read_auth_uses_prompt_when_available() -> None:
    from core.mononote import provenance

    class World:
        def snapshot(self) -> dict:
            return {"session": {"last_user_prompt": "tell me a joke"}}

    class Ctx:
        level = 1
        world_state = World()

    assert provenance.is_skill_note_read_authorized("secret-plan", Ctx()) is False

    class NoteWorld:
        def snapshot(self) -> dict:
            return {"session": {"last_user_prompt": "read the secret plan note"}}

    class NoteCtx:
        level = 1
        world_state = NoteWorld()

    assert provenance.is_skill_note_read_authorized("secret-plan", NoteCtx()) is True

    class DenyWorld:
        def snapshot(self) -> dict:
            return {"session": {"last_user_prompt": "do not read notes this turn"}}

    class DenyCtx:
        level = 1
        world_state = DenyWorld()

    assert provenance.is_skill_note_read_authorized("secret-plan", DenyCtx()) is False
