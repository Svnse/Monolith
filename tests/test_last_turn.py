"""Tests for core/last_turn.py — prior-turn action history contributor.

Data model verified against production turn_trace.sqlite3:
- Outer turns (parent_turn_id IS NULL) have NO role=tool messages — the frame
  snapshot is captured BEFORE tool execution.
- Child turns (parent_turn_id set) carry tool results as role=user messages
  whose content_preview starts with "[TOOL RESULT:tool_name] ...".
- Tool names are extracted from the [TOOL RESULT:name] prefix in child frames.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from core import last_turn
from core import turn_trace as tt


# ── shared fixtures ─────────────────────────────────────────────────


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect last_turn._DB_PATH to a temp file AND seed the schema via
    turn_trace.set_db_path() so record_frame() works in helper functions.

    Yields the Path of the db file (not yet created until schema is seeded).
    """
    db_path = tmp_path / "turn_trace.sqlite3"
    monkeypatch.setattr(last_turn, "_DB_PATH", db_path)
    tt.set_db_path(db_path)
    yield db_path
    tt.set_db_path(None)


def _make_frame(
    turn_id: str,
    *,
    parent_turn_id: str | None = None,
    captured_at: str = "2026-05-15T10:00:00Z",
    extra_user_msgs: list[str] | None = None,
) -> tt.FrameTraceRecord:
    """Build a FrameTraceRecord with optional extra user messages.

    `extra_user_msgs` lets tests inject messages like "[TOOL RESULT:read_file] ok"
    as user-role messages — matching what Monolith writes in child turns.
    """
    msgs: list[tt.FrameMessage] = [
        tt.FrameMessage.from_message({"role": "system", "content": "sys"}),
        tt.FrameMessage.from_message({"role": "user", "content": "hi"}),
    ]
    if extra_user_msgs:
        for text in extra_user_msgs:
            msgs.append(tt.FrameMessage.from_message({"role": "user", "content": text}))
    total = sum(m.content_chars for m in msgs)
    return tt.FrameTraceRecord(
        turn_id=turn_id,
        parent_turn_id=parent_turn_id,
        captured_at=captured_at,
        backend="gguf_api",
        engine_key="llm",
        gen_id=1,
        final_messages=tuple(msgs),
        system_prompt_chars=3,
        user_prompt_chars=2,
        total_chars=total,
    )


# ── db-missing test ─────────────────────────────────────────────────


def test_block_is_none_when_db_missing(monkeypatch, tmp_path) -> None:
    """No DB on disk → return None (don't crash on fresh install)."""
    missing = tmp_path / "does_not_exist.sqlite3"
    monkeypatch.setattr(last_turn, "_DB_PATH", missing)
    assert not missing.exists()
    assert last_turn.render_last_turn_block() is None


# ── schema-seeded tests ─────────────────────────────────────────────


def test_block_is_none_when_no_prior_turn(tmp_db) -> None:
    """DB exists but no recorded turns → return None."""
    tt.list_recent_turns(limit=1)   # triggers schema creation
    assert tmp_db.exists()
    assert last_turn.render_last_turn_block() is None


def test_block_skips_injection_when_prior_turn_had_no_tools(tmp_db) -> None:
    """Prior outer turn recorded but no child turns → None (no tool activity)."""
    tt.record_frame(_make_frame("outer-1"))
    assert last_turn.render_last_turn_block() is None


def test_block_surfaces_tool_summary_from_child_turns(tmp_db) -> None:
    """Child turn carries [TOOL RESULT:read_file] → block names the tool."""
    tt.record_frame(_make_frame("outer-2", captured_at="2026-05-15T10:01:00Z"))
    tt.record_frame(_make_frame(
        "child-2a",
        parent_turn_id="outer-2",
        captured_at="2026-05-15T10:01:01Z",
        extra_user_msgs=["[TOOL RESULT:read_file] content of the file"],
    ))
    block = last_turn.render_last_turn_block()
    assert block is not None
    assert block.startswith("[LAST TURN]")
    assert "read_file" in block


def test_block_counts_repeated_tools(tmp_db) -> None:
    """Multiple child turns with same tool → count shown (name x2 format)."""
    tt.record_frame(_make_frame("outer-3", captured_at="2026-05-15T10:02:00Z"))
    for i, suffix in enumerate(["a", "b"], start=1):
        tt.record_frame(_make_frame(
            f"child-3{suffix}",
            parent_turn_id="outer-3",
            captured_at=f"2026-05-15T10:02:0{i}Z",
            extra_user_msgs=[f"[TOOL RESULT:read_file] file{i} content"],
        ))
    block = last_turn.render_last_turn_block()
    assert block is not None
    assert "read_file x2" in block


def test_block_uses_most_recent_outer_turn(tmp_db) -> None:
    """When multiple outer turns exist, only the latest is used."""
    # Earlier outer turn — has a child with tools.
    tt.record_frame(_make_frame("outer-old", captured_at="2026-05-15T09:00:00Z"))
    tt.record_frame(_make_frame(
        "child-old",
        parent_turn_id="outer-old",
        captured_at="2026-05-15T09:00:01Z",
        extra_user_msgs=["[TOOL RESULT:scratchpad] old result"],
    ))
    # Newer outer turn — no child turns (no tools).
    tt.record_frame(_make_frame("outer-new", captured_at="2026-05-15T10:00:00Z"))
    # Must be None because the NEWEST outer turn had no tool activity.
    assert last_turn.render_last_turn_block() is None


def test_block_multiple_distinct_tools(tmp_db) -> None:
    """Prior turn used multiple distinct tools — all named in the block."""
    tt.record_frame(_make_frame("outer-m", captured_at="2026-05-15T10:03:00Z"))
    tt.record_frame(_make_frame(
        "child-m1",
        parent_turn_id="outer-m",
        captured_at="2026-05-15T10:03:01Z",
        extra_user_msgs=["[TOOL RESULT:read_file] content"],
    ))
    tt.record_frame(_make_frame(
        "child-m2",
        parent_turn_id="outer-m",
        captured_at="2026-05-15T10:03:02Z",
        extra_user_msgs=["[TOOL RESULT:run_command] exit 0"],
    ))
    block = last_turn.render_last_turn_block()
    assert block is not None
    assert "read_file" in block
    assert "run_command" in block


def test_block_is_under_char_cap(tmp_db) -> None:
    """Block text stays within the 290-char cap even with many tool names."""
    tt.record_frame(_make_frame("outer-big", captured_at="2026-05-15T10:04:00Z"))
    for i in range(20):
        tt.record_frame(_make_frame(
            f"child-big-{i}",
            parent_turn_id="outer-big",
            captured_at=f"2026-05-15T10:04:{i:02d}Z",
            extra_user_msgs=[f"[TOOL RESULT:very_long_tool_name_{i}] result data here"],
        ))
    block = last_turn.render_last_turn_block()
    assert block is not None
    assert len(block) <= last_turn._BLOCK_CHAR_CAP


def test_contribute_section_respects_flag(tmp_db, monkeypatch) -> None:
    """MONOLITH_LAST_TURN_V1=0 → contribute_section returns None regardless."""
    tt.record_frame(_make_frame("outer-flag", captured_at="2026-05-15T10:05:00Z"))
    tt.record_frame(_make_frame(
        "child-flag",
        parent_turn_id="outer-flag",
        captured_at="2026-05-15T10:05:01Z",
        extra_user_msgs=["[TOOL RESULT:scratchpad] result"],
    ))
    monkeypatch.setenv("MONOLITH_LAST_TURN_V1", "0")
    assert last_turn.contribute_section([], {}) is None


def test_contribute_section_returns_section_result_when_enabled(tmp_db, monkeypatch) -> None:
    """MONOLITH_LAST_TURN_V1=1 with tool activity → SectionResult returned."""
    from core.ephemeral_coalescer import SectionResult
    tt.record_frame(_make_frame("outer-en", captured_at="2026-05-15T10:06:00Z"))
    tt.record_frame(_make_frame(
        "child-en",
        parent_turn_id="outer-en",
        captured_at="2026-05-15T10:06:01Z",
        extra_user_msgs=["[TOOL RESULT:scratchpad] pinned anchor(5)"],
    ))
    monkeypatch.setenv("MONOLITH_LAST_TURN_V1", "1")
    result = last_turn.contribute_section([], {})
    assert isinstance(result, SectionResult)
    assert result.name == "last_turn"
    assert result.text.startswith("[LAST TURN]")
    assert "scratchpad" in result.text


# ── enriched-metadata tests ─────────────────────────────────────────


def test_block_surfaces_write_file_path(tmp_db) -> None:
    """write_file in prior turn → block lists the path in 'edits:' line."""
    tt.record_frame(_make_frame("outer-wf", captured_at="2026-05-15T10:07:00Z"))
    tt.record_frame(_make_frame(
        "child-wf",
        parent_turn_id="outer-wf",
        captured_at="2026-05-15T10:07:01Z",
        extra_user_msgs=[
            "[TOOL RESULT:write_file] [write_file: written 42 chars to C:/proj/foo.py]"
        ],
    ))
    block = last_turn.render_last_turn_block()
    assert block is not None
    assert "write_file" in block
    assert "edits: C:/proj/foo.py" in block


def test_block_surfaces_run_command_exit_code(tmp_db) -> None:
    """run_command in prior turn → block lists exit code in 'run:' line."""
    tt.record_frame(_make_frame("outer-rc", captured_at="2026-05-15T10:08:00Z"))
    tt.record_frame(_make_frame(
        "child-rc",
        parent_turn_id="outer-rc",
        captured_at="2026-05-15T10:08:01Z",
        extra_user_msgs=[
            "[TOOL RESULT:run_command] [run_command: exit_code=0]\nsome output"
        ],
    ))
    block = last_turn.render_last_turn_block()
    assert block is not None
    assert "run_command" in block
    assert "run: exit 0" in block


def test_block_dedupes_repeated_edits_to_same_file(tmp_db) -> None:
    """Two write_file calls to the same path → one entry in the 'edits:' line.

    Uses write_file format (the tool that actually embeds an absolute path via
    'written N chars to /path'). edit_file only includes path.name in its output
    so path extraction silently skips it per spec.
    """
    tt.record_frame(_make_frame("outer-ded", captured_at="2026-05-15T10:09:00Z"))
    for i, suffix in enumerate(["a", "b"], start=1):
        tt.record_frame(_make_frame(
            f"child-ded-{suffix}",
            parent_turn_id="outer-ded",
            captured_at=f"2026-05-15T10:09:0{i}Z",
            extra_user_msgs=[
                "[TOOL RESULT:write_file] [write_file: written 20 chars to C:/proj/foo.py]"
            ],
        ))
    block = last_turn.render_last_turn_block()
    assert block is not None
    # Path appears exactly once in the edits line (deduped).
    assert block.count("C:/proj/foo.py") == 1
    assert "edits: C:/proj/foo.py" in block


def test_block_omits_edits_line_when_no_writes(tmp_db) -> None:
    """No write/edit tools in prior turn → no 'edits:' line in block."""
    tt.record_frame(_make_frame("outer-nw", captured_at="2026-05-15T10:10:00Z"))
    tt.record_frame(_make_frame(
        "child-nw",
        parent_turn_id="outer-nw",
        captured_at="2026-05-15T10:10:01Z",
        extra_user_msgs=["[TOOL RESULT:read_file] file contents here"],
    ))
    block = last_turn.render_last_turn_block()
    assert block is not None
    assert "read_file" in block
    assert "edits:" not in block
