from __future__ import annotations

import json
import os
from pathlib import Path

from ui.pages.chat_archive import ChatArchiveManager
from ui.pages.chat_session import ChatSessionManager


def _write_archive(path: Path, updated_at: str, title: str) -> None:
    payload = {
        "meta": {
            "title": title,
            "created_at": updated_at,
            "updated_at": updated_at,
            "message_count": 1,
            "assistant_tokens": 0,
            "summary": [title],
        },
        "messages": [{"i": 1, "time": updated_at, "role": "user", "text": "hi"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_list_archives_respects_default_cap(tmp_path: Path) -> None:
    sessions = ChatSessionManager("")
    manager = ChatArchiveManager(tmp_path, sessions, max_list_items=2)

    for idx in range(5):
        p = tmp_path / f"archive_{idx}.json"
        stamp = f"2026-01-0{idx + 1}T00:00:00+00:00"
        _write_archive(p, stamp, f"title-{idx}")
        os.utime(p, (1_700_000_000 + idx, 1_700_000_000 + idx))

    items = manager.list_archives()

    assert len(items) == 2
    assert items[0].title == "title-4"
    assert items[1].title == "title-3"


def test_list_archives_limit_overrides_default_cap(tmp_path: Path) -> None:
    sessions = ChatSessionManager("")
    manager = ChatArchiveManager(tmp_path, sessions, max_list_items=2)

    for idx in range(4):
        p = tmp_path / f"archive_{idx}.json"
        stamp = f"2026-02-0{idx + 1}T00:00:00+00:00"
        _write_archive(p, stamp, f"entry-{idx}")
        os.utime(p, (1_700_100_000 + idx, 1_700_100_000 + idx))

    items = manager.list_archives(limit=4)

    assert len(items) == 4
    assert [item.title for item in items] == ["entry-3", "entry-2", "entry-1", "entry-0"]


def test_delete_archive_returns_false_when_missing(tmp_path: Path) -> None:
    sessions = ChatSessionManager("")
    manager = ChatArchiveManager(tmp_path, sessions)
    missing = tmp_path / "missing.json"

    assert manager.delete_archive(missing) is False


def test_delete_archive_returns_true_when_deleted(tmp_path: Path) -> None:
    sessions = ChatSessionManager("")
    manager = ChatArchiveManager(tmp_path, sessions)
    target = tmp_path / "exists.json"
    target.write_text("{}", encoding="utf-8")

    assert manager.delete_archive(target) is True
    assert not target.exists()
