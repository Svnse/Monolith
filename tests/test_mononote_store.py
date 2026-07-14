from __future__ import annotations

from pathlib import Path


def test_mononote_store_writes_reads_and_hashes_notes(tmp_path: Path, monkeypatch) -> None:
    from core import paths
    from core.mononote import store

    vault = tmp_path / "notes"
    monkeypatch.setattr(paths, "NOTES_DIR", vault, raising=False)

    note = store.write_note("Project Ideas", "# Project Ideas\n\nBody")
    assert note.safe_title == "Project-Ideas"
    assert note.path == vault / "Project-Ideas.md"
    assert note.sha256
    assert note.note_id

    loaded, body = store.load_note("Project Ideas")
    assert loaded.note_id == note.note_id
    assert body == "# Project Ideas\n\nBody"

    records = store.list_notes("project")
    assert [r.safe_title for r in records] == ["Project-Ideas"]


def test_mononote_store_rejects_path_escape(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from core import paths
    from core.mononote import store

    vault = tmp_path / "notes"
    monkeypatch.setattr(paths, "NOTES_DIR", vault, raising=False)

    with pytest.raises(ValueError):
        store.ensure_note_path("../outside.md", create=True)


def test_mononote_text_window_tracks_range_and_truncation() -> None:
    from core.mononote import store

    window = store.slice_note_text("abcdef", offset=2, max_chars=3)
    assert window.text == "cde"
    assert window.start == 2
    assert window.end == 5
    assert window.truncated is True

    selected = store.slice_note_text("abcdef", selection_start=1, selection_end=4, max_chars=10)
    assert selected.text == "bcd"
    assert selected.start == 1
    assert selected.end == 4
    assert selected.truncated is False
