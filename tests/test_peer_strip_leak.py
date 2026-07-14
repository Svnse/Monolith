"""Peer-channel leak fix: clean lane must be free of orphan think tags AND
embedded [CHANNEL]/[AGENT] brackets. (Observed live: ".</think>I'll update my
bearing [CHANNEL: connect/Claude, /chat blocking]" leaked into a peer reply.)
The thinking itself stays available to the peer via raw_response — not stripped
from existence, just kept out of the clean answer lane."""
from __future__ import annotations

from core.internal_tags import strip_tag_blocks


def test_strip_removes_orphan_closing_tag() -> None:
    assert "</think>" not in strip_tag_blocks("answer.</think> more", ("think",))


def test_strip_removes_orphan_opening_tag() -> None:
    assert "<think>" not in strip_tag_blocks("<think> leftover text", ("think",))


def test_strip_keeps_balanced_removal_and_surrounding_text() -> None:
    out = strip_tag_blocks("before <think>secret</think> after", ("think",))
    assert "secret" not in out
    assert "before" in out and "after" in out


def test_strip_empty_tags_is_noop() -> None:
    assert strip_tag_blocks("text </think>", ()) == "text </think>"


def test_clean_agent_response_strips_orphan_and_brackets() -> None:
    from engine.agent_server import _clean_agent_response
    leaked = ".</think>I'll update my bearing [CHANNEL: connect/Claude, /chat blocking] now [AGENT:Claude]"
    clean = _clean_agent_response(leaked)
    assert "</think>" not in clean
    assert "[CHANNEL" not in clean
    assert "[AGENT" not in clean
    assert "update my bearing" in clean  # the real answer text survives
