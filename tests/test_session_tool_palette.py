from __future__ import annotations

import json

from core.session_tool_palette import render_session_tool_palette
from ui.pages.chat_session import ChatSessionManager


def test_session_tool_palette_collects_called_and_discovered_tools() -> None:
    messages = [
        {
            "role": "tool_result",
            "text": json.dumps(
                {
                    "tool": "monosearch",
                    "result": "[monosearch:find meta=tools count=1]\n  tool:edit_file [tools/telemetry] ...",
                }
            ),
        },
        {
            "role": "tool_result",
            "text": json.dumps({"tool": "run_tests", "result": "[run_tests: exit_code=0]"}),
        },
    ]

    block = render_session_tool_palette(messages)

    assert block is not None
    assert "[SESSION TOOL PALETTE]" in block
    assert "monosearch" in block
    assert "edit_file" in block
    assert "run_tests" in block
    assert "High-probability next moves:" in block


def test_session_palette_discovery_fallback_teaches_source_form() -> None:
    """The always-on palette primed meta=/get id= but never source=, so the model's
    first guess on a store-specific query was an invalid meta (2026-06-15/-16).
    The discovery fallback must also teach the source= form."""
    messages = [
        {
            "role": "tool_result",
            "text": json.dumps({"tool": "monosearch", "result": "ok"}),
        }
    ]

    block = render_session_tool_palette(messages)

    assert block is not None
    assert "source=" in block


def test_chat_history_injects_palette_before_latest_user_only_from_prior_results() -> None:
    sessions = ChatSessionManager("MASTER")
    sessions.set_current(
        sessions.create_session(
            messages=[
                {"role": "user", "text": "First", "time": "2026-01-01T00:00:00"},
                {
                    "role": "tool_result",
                    "text": json.dumps({"tool": "monosearch", "result": "tool:edit_file"}),
                    "time": "2026-01-01T00:00:01",
                },
                {"role": "assistant", "text": "Done", "time": "2026-01-01T00:00:02"},
                {"role": "user", "text": "Now edit it", "time": "2026-01-01T00:00:03"},
            ]
        )
    )

    history = sessions.build_engine_history()

    palette_idx = next(i for i, item in enumerate(history) if item.get("source") == "session_tool_palette")
    latest_user_idx = next(i for i, item in enumerate(history) if "Now edit it" in item.get("content", ""))
    assert palette_idx == latest_user_idx - 1
    assert history[palette_idx]["ephemeral"] is True
    assert "edit_file" in history[palette_idx]["content"]


def test_chat_history_does_not_palette_future_tool_result_before_same_user() -> None:
    sessions = ChatSessionManager("MASTER")
    sessions.set_current(
        sessions.create_session(
            messages=[
                {"role": "user", "text": "Read it", "time": "2026-01-01T00:00:00"},
                {
                    "role": "tool_result",
                    "text": json.dumps({"tool": "read_file", "result": "[read_file: a.txt]\nhello"}),
                    "time": "2026-01-01T00:00:01",
                },
            ]
        )
    )

    history = sessions.build_engine_history()

    assert not any(item.get("source") == "session_tool_palette" for item in history)
