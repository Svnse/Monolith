"""Tests for core.fault_detectors — per-detector positive/negative/edge cases."""
from __future__ import annotations

import pytest

from core.fault_detectors import (
    detect_markdown_corruption,
    detect_regen_mismatch,
    detect_think_leak,
    detect_tool_no_fire,
)


_TURN = "turn-test"
_CTX: dict = {}


# ── detect_markdown_corruption ───────────────────────────────────────────────


def test_markdown_corruption_fence_imbalance_fires():
    text = "```python\nsome code\n```\n\n```python\nunclosed block"
    r = detect_markdown_corruption(text, _TURN, _CTX)
    assert r is not None
    assert r.fault_kind == "markdown_corruption"
    assert "fence_imbalance" in (r.evidence or "")


def test_markdown_corruption_clean_fences_no_fault():
    text = "```python\ncode here\n```\nSome prose."
    r = detect_markdown_corruption(text, _TURN, _CTX)
    assert r is None


def test_markdown_corruption_single_backtick_imbalance():
    # Odd number of single backticks outside fences
    text = "Use `foo and bar without close."
    r = detect_markdown_corruption(text, _TURN, _CTX)
    assert r is not None
    assert "single_backtick_imbalance" in (r.evidence or "")


def test_markdown_corruption_backticks_inside_closed_fence_ignored():
    # Single backtick inside a closed fence should NOT trigger
    text = "```\n`inner backtick`\n```"
    r = detect_markdown_corruption(text, _TURN, _CTX)
    # fence count: 2 (even), backticks after scrub: 0
    assert r is None


def test_markdown_corruption_empty_response_no_fault():
    r = detect_markdown_corruption("", _TURN, _CTX)
    assert r is None


# ── detect_tool_no_fire ──────────────────────────────────────────────────────


def test_tool_no_fire_intent_without_call_fires():
    text = "I'll check the file contents now."
    r = detect_tool_no_fire(text, _TURN, _CTX)
    assert r is not None
    assert r.fault_kind == "tool_no_fire"


def test_tool_no_fire_intent_with_tool_call_no_fault():
    text = "I'll check the file.\n<tool_call>{\"tool\": \"read_file\"}</tool_call>"
    r = detect_tool_no_fire(text, _TURN, _CTX)
    assert r is None


def test_tool_no_fire_no_intent_no_fault():
    text = "Here is the answer to your question."
    r = detect_tool_no_fire(text, _TURN, _CTX)
    assert r is None


def test_tool_no_fire_search_intent_fires():
    text = "I will search for that information."
    r = detect_tool_no_fire(text, _TURN, _CTX)
    assert r is not None


def test_tool_no_fire_evidence_contains_snippet():
    text = "I will verify that claim for you."
    r = detect_tool_no_fire(text, _TURN, _CTX)
    assert r is not None
    assert r.evidence is not None
    assert len(r.evidence) > 0


# ── detect_think_leak ────────────────────────────────────────────────────────


def test_think_leak_missing_close_tag_fires():
    text = "<think>reasoning that never closes"
    r = detect_think_leak(text, _TURN, _CTX)
    assert r is not None
    assert r.fault_kind == "think_leak"
    assert "think_tag_imbalance" in (r.evidence or "")


def test_think_leak_balanced_tags_no_fault():
    text = "<think>reasoning here</think>\nThe answer is 42."
    r = detect_think_leak(text, _TURN, _CTX)
    assert r is None


def test_think_leak_acatalepsy_marker_fires():
    text = "Answer is here. <acatalepsy>internal content</acatalepsy>"
    r = detect_think_leak(text, _TURN, _CTX)
    assert r is not None
    assert "acatalepsy_marker" in (r.evidence or "")


def test_think_leak_empty_text_no_fault():
    r = detect_think_leak("", _TURN, _CTX)
    assert r is None


def test_think_leak_multiple_balanced_blocks_no_fault():
    # Two balanced think blocks (shouldn't happen normally but not a fault)
    text = "<think>block 1</think>\n<think>block 2</think>\nFinal answer."
    r = detect_think_leak(text, _TURN, _CTX)
    assert r is None


def test_think_leak_inline_code_mention_no_fault():
    # The model *talks about* the tag in backticks — not a real leaked tag.
    # Reproduces the 2026-06-03 curiosity incident: 1 real balanced pair +
    # a backticked `<think>` mention was miscounted as open=2/close=1.
    text = (
        "<think>plan the answer</think>\n"
        "I structure the thinking within `<think>` blocks. "
        "Final answer: outside `<think>` — here it is: 42."
    )
    r = detect_think_leak(text, _TURN, _CTX)
    assert r is None


def test_think_leak_fenced_block_mention_no_fault():
    # A tag shown as a fenced code example is documentation, not a leak — even
    # when the example itself is an *unclosed* tag (would otherwise imbalance).
    text = (
        "<think>reason</think>\nAn unclosed tag looks like:\n"
        "```\n<think>\n```\nThat is the failure shape."
    )
    r = detect_think_leak(text, _TURN, _CTX)
    assert r is None


def test_think_leak_real_imbalance_still_fires_despite_code_mention():
    # A genuine missing </think> must STILL fire even when the text also
    # mentions the tag in backticks — stripping code must not mask real leaks.
    text = "<think>reasoning that never closes. I also mention `<think>` here."
    r = detect_think_leak(text, _TURN, _CTX)
    assert r is not None
    assert "think_tag_imbalance" in (r.evidence or "")


# ── detect_regen_mismatch ────────────────────────────────────────────────────


def test_regen_mismatch_ref_exceeds_count_fires():
    context = {"frame_traces": []}  # 0 tool results
    text = "See tool_result_0 for the answer."
    r = detect_regen_mismatch(text, _TURN, context)
    assert r is not None
    assert r.fault_kind == "regen_mismatch"
    assert 0 in r.metadata.get("bad_refs", [])


def test_regen_mismatch_ref_within_count_no_fault():
    # Two tool results in frame, reference is tool_result_0 and tool_result_1
    frame = [
        {"role": "tool", "content": "result A"},
        {"role": "tool", "content": "result B"},
    ]
    context = {"frame_traces": frame}
    text = "tool_result_0 and tool_result_1 are both present."
    r = detect_regen_mismatch(text, _TURN, context)
    assert r is None


def test_regen_mismatch_no_refs_no_fault():
    context = {"frame_traces": [{"role": "tool", "content": "x"}]}
    text = "Here is a normal response with no tool result references."
    r = detect_regen_mismatch(text, _TURN, context)
    assert r is None


def test_regen_mismatch_missing_context_skips():
    # No frame_traces key → detector skips, returns None
    text = "See tool_result_5 for details."
    r = detect_regen_mismatch(text, _TURN, {})
    assert r is None


def test_regen_mismatch_metadata_populated():
    context = {"frame_traces": []}
    text = "tool_result_3 and tool_result_7 are referenced."
    r = detect_regen_mismatch(text, _TURN, context)
    assert r is not None
    assert sorted(r.metadata["bad_refs"]) == [3, 7]
    assert r.metadata["tool_result_count"] == 0
