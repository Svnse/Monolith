from __future__ import annotations

from pathlib import Path

import pytest

from core import turn_trace as tt
from core.skill_runtime import ToolExecutionContext, execute_tool_call_enveloped


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    # Ensure the stats skill is visible after SKILL.md/executor.py are on disk
    from core.skill_registry import clear_skill_cache
    from core.skill_runtime import clear_dynamic_executor_cache
    clear_skill_cache()
    clear_dynamic_executor_cache()
    yield db
    tt.set_db_path(None)
    clear_skill_cache()
    clear_dynamic_executor_cache()


def test_stats_tool_lifetime_verb(fresh_db):
    """`stats verb=lifetime` returns a JSON-shaped envelope with the lifetime
    summary."""
    ctx = ToolExecutionContext(archive_dir=Path("/tmp"))
    result = execute_tool_call_enveloped(
        {"tool": "stats", "verb": "lifetime"}, ctx
    )
    assert result.ok, f"stats lifetime errored: {result.text}"
    assert "turns" in result.text or "turns" in str(result.data)


def test_stats_tool_unknown_verb_returns_error(fresh_db):
    ctx = ToolExecutionContext(archive_dir=Path("/tmp"))
    result = execute_tool_call_enveloped(
        {"tool": "stats", "verb": "definitely_not_a_verb"}, ctx
    )
    assert not result.ok
    assert "unknown" in result.text.lower() or "verb" in result.text.lower()
