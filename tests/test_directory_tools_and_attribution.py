"""Tests for the directory-visibility fix (BUG-TOOL-018/019) and the
CONTINUITY envelope fix (BUG-CONT-002)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import core.skill_runtime as skill_runtime
import core.continuity as continuity


def _ctx(tmp_path: Path) -> skill_runtime.ToolExecutionContext:
    return skill_runtime.ToolExecutionContext(archive_dir=tmp_path)


# ── list_files ──────────────────────────────────────────────────────


def test_list_files_renders_subdirectories_with_trailing_slash(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "another").mkdir()

    result = skill_runtime.execute_list_files({"path": str(tmp_path)}, _ctx(tmp_path))

    assert "1 file" in result and "2 dirs" in result
    assert "another/" in result
    assert "subdir/" in result
    assert "a.py (" in result


def test_list_files_empty_but_dirs_present_is_honest(tmp_path: Path) -> None:
    (tmp_path / "only_a_dir").mkdir()

    result = skill_runtime.execute_list_files({"path": str(tmp_path)}, _ctx(tmp_path))

    assert "no files matching" not in result
    assert "0 files" in result
    assert "1 dir" in result
    assert "only_a_dir/" in result


def test_list_files_completely_empty_returns_honest_zero(tmp_path: Path) -> None:
    result = skill_runtime.execute_list_files({"path": str(tmp_path)}, _ctx(tmp_path))

    assert "0 files" in result and "0 dirs" in result


# ── find_files ──────────────────────────────────────────────────────


def test_find_files_default_excludes_directories(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text("x", encoding="utf-8")

    result = skill_runtime.execute_find_files(
        {"path": str(tmp_path), "pattern": "prompts"}, _ctx(tmp_path)
    )
    assert "no files matching 'prompts'" in result


def test_find_files_include_dirs_finds_directories(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text("x", encoding="utf-8")

    result = skill_runtime.execute_find_files(
        {"path": str(tmp_path), "pattern": "prompts", "include_dirs": True},
        _ctx(tmp_path),
    )
    assert "include_dirs=true" in result
    assert "prompts/" in result and "prompts/'" not in result


# ── CONTINUITY envelope ────────────────────────────────────────────


def test_continuity_block_wraps_ambient_state_envelope() -> None:
    fake_store = {
        "active": [
            {"category": "pending", "id": 1, "text": "iterate effort tier drafts"},
            {"category": "lesson", "id": 7, "text": "turn-trace shipped"},
        ],
        "retired": [],
    }
    with patch.object(continuity, "_load", return_value=fake_store):
        block = continuity.render_continuity_block()

    assert block is not None
    first_line = block.splitlines()[0]
    assert "[CONTINUITY]" in first_line
    assert "ambient state from prior sessions" in first_line
    assert "NOT this turn's request" in first_line
    assert block.rstrip().endswith("[/CONTINUITY]")


def test_continuity_block_returns_none_when_no_pins() -> None:
    fake_store = {"active": [], "retired": []}
    with patch.object(continuity, "_load", return_value=fake_store):
        assert continuity.render_continuity_block() is None
