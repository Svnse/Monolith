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


def _setup_acatalepsy_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import db_connect as dbc

    monkeypatch.setattr(dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    _clear_acatalepsy_tls()
    from core.acatalepsy import schema

    schema.migrate()
    _clear_acatalepsy_tls()


def test_mononote_adapter_search_and_get_note_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import paths
    from core.mononote import store
    from core.monosearch import registry, service
    from core.monosearch.adapters.mononote import MonoNoteAdapter

    monkeypatch.setattr(paths, "NOTES_DIR", tmp_path / "notes", raising=False)
    note = store.write_note("Project Ideas", "Alpha plan\nBeta link")
    registry.clear()
    registry.register(MonoNoteAdapter())

    recs = service.search("beta", {"source": "notes"}, limit=5)

    assert len(recs) == 1
    rec = recs[0]
    assert rec.namespaced_id == f"mononote:{note.note_id}"
    assert rec.source == "mononote"
    assert rec.metadata["acu_evidence"] is False
    assert "not ACU evidence" in rec.text
    assert "Beta link" in rec.text

    detail = service.get(rec.namespaced_id)
    assert detail is not None
    assert "content:" in detail.text
    assert "Alpha plan" in detail.text


def test_mononote_search_does_not_append_note_read_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_acatalepsy_db(tmp_path, monkeypatch)

    from core import paths
    from core.acatalepsy import canonical_log
    from core.mononote import store
    from core.monosearch import registry, service
    from core.monosearch.adapters.mononote import MonoNoteAdapter

    monkeypatch.setattr(paths, "NOTES_DIR", tmp_path / "notes", raising=False)
    store.write_note("Secret", "A searchable note")
    registry.clear()
    registry.register(MonoNoteAdapter())

    before = canonical_log.latest_event_id()
    recs = service.search("searchable", {"source": "mononote"}, limit=5)
    after = canonical_log.latest_event_id()

    assert recs
    assert after == before


def test_mononote_router_aliases() -> None:
    from core.monosearch import router

    assert router.resolve_source("notes") == "mononote"
    assert router.resolve_source("mononote") == "mononote"
    assert router.resolve_source("vault") == "mononote"
    assert "source='notes'" in (router.source_usage_hint("markdown note vault") or "")
