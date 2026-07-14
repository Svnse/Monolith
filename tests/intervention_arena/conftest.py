"""Pytest fixtures for intervention_arena tests."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture
def tmp_arena_db(tmp_path, monkeypatch) -> Iterator[Path]:
    """Redirect intervention_arena writes to a per-test temp DB.

    `core.intervention_arena.storage._DB_PATH` is resolved once at module
    import time, so patching the constant is the load-bearing step (the
    env var would only matter for sub-processes that re-import the module).
    """
    db_path = tmp_path / "intervention_arena.sqlite3"
    from core.intervention_arena import storage
    monkeypatch.setattr(storage, "_DB_PATH", db_path)
    yield db_path
