"""Tests for core.channel_tag — runtime-injected [CHANNEL: ...] header.

Tests the new /prompt-based channel tag format:
  [CHANNEL: USER, prompts=falsify+descent, monothink=on]

Also tests strip_leading_channel_tag for idempotency and embedded-tag safety.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from core.channel_tag import (
    build_channel_tag,
    strip_channel_tags_for_display,
    strip_leading_channel_tag,
)


def test_build_tag_role_only_no_modes() -> None:
    tag = build_channel_tag("USER")
    assert tag == "[CHANNEL: USER]"
    assert build_channel_tag("ASSISTANT") == "[CHANNEL: ASSISTANT]"


def test_build_tag_peer_with_transport() -> None:
    tag = build_channel_tag(
        "connect/Codex", transport="mcp send_message", include_modes=True
    )
    assert tag.startswith("[CHANNEL: connect/Codex, mcp send_message")


def test_build_tag_include_modes_with_prompts() -> None:
    with patch("core.channel_tag._peek_prompts", return_value=["falsify", "descent"]):
        with patch("core.channel_tag._peek_monothink", return_value=None):
            tag = build_channel_tag("USER", include_modes=True)
    assert tag == "[CHANNEL: USER, prompts=falsify+descent]"


def test_build_tag_include_modes_with_monothink() -> None:
    with patch("core.channel_tag._peek_prompts", return_value=["orient"]):
        with patch("core.channel_tag._peek_monothink", return_value=True):
            tag = build_channel_tag("USER", include_modes=True)
    assert tag == "[CHANNEL: USER, prompts=orient, monothink=on]"


def test_build_tag_no_prompts_no_monothink() -> None:
    with patch("core.channel_tag._peek_prompts", return_value=[]):
        with patch("core.channel_tag._peek_monothink", return_value=None):
            tag = build_channel_tag("USER", include_modes=True)
    assert tag == "[CHANNEL: USER]"


# ── strip_leading_channel_tag ────────────────────────────────────────


def test_strip_leading_channel_tag_removes_one_prefix() -> None:
    body, removed = strip_leading_channel_tag(
        "[CHANNEL: connect/X, mcp send_message]\n\nhello world"
    )
    assert body == "hello world"
    assert removed == "[CHANNEL: connect/X, mcp send_message]"


def test_strip_leading_channel_tag_idempotent() -> None:
    body, removed = strip_leading_channel_tag("plain message")
    assert body == "plain message"
    assert removed is None


def test_strip_leading_channel_tag_leaves_embedded_tags_alone() -> None:
    body, removed = strip_leading_channel_tag(
        "[CHANNEL: USER]\n\nI saw [CHANNEL: ASSISTANT] in the logs"
    )
    assert body == "I saw [CHANNEL: ASSISTANT] in the logs"
    assert removed == "[CHANNEL: USER]"


def test_strip_channel_tags_for_display_clears_leading_tag() -> None:
    assert strip_channel_tags_for_display("[CHANNEL: USER]\n\nhello") == "hello"


def test_strip_channel_tags_for_display_clears_tag_after_agent_prefix() -> None:
    # The exact shape a stored connect/peer turn has: "[Codex] [CHANNEL: ...]\n\nbody"
    out = strip_channel_tags_for_display(
        "[Codex] [CHANNEL: connect/Codex, mcp send_message, prompts=falsify]\n\nHello there"
    )
    assert out == "[Codex] Hello there"
    assert "CHANNEL" not in out


def test_strip_channel_tags_for_display_clears_assistant_tag() -> None:
    assert strip_channel_tags_for_display("[CHANNEL: ASSISTANT]\n\nanswer") == "answer"


def test_strip_channel_tags_for_display_noop_without_tag() -> None:
    assert strip_channel_tags_for_display("just a normal message") == "just a normal message"
