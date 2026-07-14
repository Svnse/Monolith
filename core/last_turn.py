"""Last-turn action history — surfaces prior turn's tool calls to the model.

Monolith named the gap itself: "I don't know what I touched last turn — files
I wrote, commands I ran, decisions I locked." The data lives in turn_trace.sqlite3
already; this contributor projects a one-paragraph summary into each prompt
so Monolith has recent action history without scrolling or re-deriving.

Drop priority: inside the coalescer below runtime_state/review_loop/observer
and above confidence/rating/context reminders — useful but not load-bearing
under budget pressure.

Query strategy: reads frame_traces for the most recent OUTER turn (parent_turn_id
IS NULL), then finds all CHILD frame records (tool-followup chains) whose
parent_turn_id matches that turn_id. Tool calls are identified by user-role
messages whose content_preview starts with "[TOOL RESULT:tool_name]" — the format
injected by skill_runtime.execute_tool_call_enveloped().

Outer frame snapshots do NOT carry role=tool messages in production (the frame is
captured before tool execution). Child frames carry the tool result context as
role=user messages with the [TOOL RESULT:...] prefix.

Flag: MONOLITH_LAST_TURN_V1 (default ON). Set =0 to disable.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from core.paths import LOG_DIR


_BLOCK_TAG = "[LAST TURN]"
_FLAG_ENV = "MONOLITH_LAST_TURN_V1"
_DB_PATH = LOG_DIR / "turn_trace.sqlite3"

# Max chars for the emitted block — keeps KV-cache pressure low.
_BLOCK_CHAR_CAP = 290

# Pattern: "[TOOL RESULT:tool_name] ..." — extracts tool_name from result messages.
_TOOL_RESULT_RE = re.compile(r"^\[TOOL RESULT:([^\]]+)\]")

# Patterns for enriched metadata — copied from core/skill_runtime.py:378-383
# (kept local to preserve interceptor self-containment; do not import from skill_runtime).
# NOTE: content_preview collapses \n → space, so $ matches end-of-string here.
_WRITE_PATH_RE = re.compile(r" to (.+?)\]?$", re.MULTILINE)
_EXIT_CODE_RE = re.compile(r"exit_code=(\d+)")

# Tools that carry a written-file path in their result body.
_WRITE_TOOLS = frozenset({"write_file", "edit_file"})
# Tools that carry an exit code in their result body.
_SHELL_TOOLS = frozenset({"run_command", "run_tests"})


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _open_db_ro(db_path: Path) -> sqlite3.Connection | None:
    """Open the DB read-only. Returns None if the file doesn't exist."""
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def _extract_tool_calls_from_child_frames(
    conn: sqlite3.Connection, outer_turn_id: str
) -> list[tuple[str, str | None, int | None]]:
    """Find all tool calls from child-turn frame records hanging off outer_turn_id.

    Child turns are ephemeral tool-followup generations whose parent_turn_id
    points at the outer turn. Their final_messages_json contains user-role
    messages whose content_preview starts with "[TOOL RESULT:tool_name]".

    Returns a list of (tool_name, path_or_None, exit_code_or_None) tuples,
    one per tool call (may have duplicates for count aggregation). Path is
    extracted only for write_file/edit_file tools; exit code only for
    run_command/run_tests tools.
    """
    try:
        rows = conn.execute(
            """
            SELECT final_messages_json
            FROM frame_traces
            WHERE parent_turn_id = ?
            ORDER BY id ASC
            """,
            (outer_turn_id,),
        ).fetchall()
    except Exception:
        return []

    calls: list[tuple[str, str | None, int | None]] = []
    for row in rows:
        try:
            msgs = json.loads(row["final_messages_json"] or "[]")
        except (TypeError, ValueError):
            continue
        if not isinstance(msgs, list):
            continue
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if str(m.get("role", "")).lower() != "user":
                continue
            preview = str(m.get("content_preview", ""))
            match = _TOOL_RESULT_RE.match(preview)
            if not match:
                continue
            name = match.group(1).strip()
            if not name:
                continue

            path: str | None = None
            exit_code: int | None = None

            # Extract path for write/edit tools — body follows the envelope prefix.
            if name in _WRITE_TOOLS:
                pm = _WRITE_PATH_RE.search(preview)
                if pm:
                    path = pm.group(1).strip()

            # Extract exit code for shell tools.
            elif name in _SHELL_TOOLS:
                em = _EXIT_CODE_RE.search(preview)
                if em:
                    exit_code = int(em.group(1))

            calls.append((name, path, exit_code))
    return calls


# Keep the old name as an alias so any external caller (ephemeral_coalescer) isn't broken.
# render_last_turn_block() is the only real consumer; tests go through that.
def _extract_tool_names_from_child_frames(conn: sqlite3.Connection, outer_turn_id: str) -> list[str]:
    return [name for name, _, __ in _extract_tool_calls_from_child_frames(conn, outer_turn_id)]


def render_last_turn_block() -> str | None:
    """Build the [LAST TURN] block from the most recent outer turn.

    Returns None if:
    - The DB doesn't exist (fresh install)
    - No prior outer turn is recorded
    - The prior turn had no tool activity (no child turns with tool results)

    Otherwise returns a compact summary under ~300 chars including optional
    "edits:" (file paths written/edited) and "run:" (exit codes) lines.
    """
    conn = _open_db_ro(_DB_PATH)
    if conn is None:
        return None

    try:
        # Most recent outer turn (parent_turn_id IS NULL = not a tool-followup).
        # Use the id column (autoincrement) for ORDER BY — faster and monotonic.
        row = conn.execute(
            """
            SELECT turn_id
            FROM frame_traces
            WHERE parent_turn_id IS NULL
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None

        outer_turn_id = row["turn_id"]
        tool_calls = _extract_tool_calls_from_child_frames(conn, outer_turn_id)

    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # No tool activity — suppress the block (don't inject noise).
    if not tool_calls:
        return None

    tool_names = [name for name, _, __ in tool_calls]

    # --- tools: line ---
    counts = Counter(tool_names)
    parts = [f"{name} x{c}" if c > 1 else name for name, c in counts.most_common()]
    tools_line = "tools: " + ", ".join(parts)

    # --- edits: line — dedupe paths, preserve first-seen order ---
    seen_paths: dict[str, None] = {}
    for name, path, _ in tool_calls:
        if name in _WRITE_TOOLS and path:
            seen_paths[path] = None
    edits_line: str | None = None
    if seen_paths:
        edits_line = "edits: " + ", ".join(seen_paths)

    # --- run: line — list all exit codes in call order ---
    exit_codes = [ec for name, _, ec in tool_calls if name in _SHELL_TOOLS and ec is not None]
    run_line: str | None = None
    if exit_codes:
        run_line = "run: " + ", ".join(f"exit {ec}" for ec in exit_codes)

    # Assemble block.
    lines = [_BLOCK_TAG, tools_line]
    if edits_line:
        lines.append(edits_line)
    if run_line:
        lines.append(run_line)
    block = "\n".join(lines)

    # Hard cap — truncate the edits list intentionally if needed, then fall back
    # to raw slice only as a last resort so mid-path cuts don't happen on short blocks.
    if len(block) > _BLOCK_CHAR_CAP and edits_line and seen_paths:
        path_list = list(seen_paths)
        # Drop paths from the end until we fit or run out.
        while len(path_list) > 1:
            path_list.pop()
            edits_line = "edits: " + ", ".join(path_list) + ", …"
            lines_try = [_BLOCK_TAG, tools_line, edits_line]
            if run_line:
                lines_try.append(run_line)
            candidate = "\n".join(lines_try)
            if len(candidate) <= _BLOCK_CHAR_CAP:
                block = candidate
                break
        else:
            # Even one path is too long — drop edits entirely.
            lines_try = [_BLOCK_TAG, tools_line]
            if run_line:
                lines_try.append(run_line)
            block = "\n".join(lines_try)

    # Final safety truncation.
    if len(block) > _BLOCK_CHAR_CAP:
        block = block[: _BLOCK_CHAR_CAP - 1].rstrip() + "…"

    return block


def contribute_section(messages: list[dict], config: dict):
    """Section-contributor for the ephemeral_coalescer."""
    from core.ephemeral_coalescer import SectionResult
    if not _flag_enabled():
        return None
    block = render_last_turn_block()
    if block is None:
        return None
    return SectionResult(name="last_turn", text=block)
