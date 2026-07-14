"""Unit tests for output_sanitizer.compute_terminal_correction.

The M1 terminal-mutation organ. Pure function: given the public answer text,
return a corrected string with internal-tag leaks stripped, or None when no
correction is needed. No pipeline / Qt required — this is the testable core
of the execution-plane consumption seam (see
docs/superpowers/specs/2026-06-02-execution-plane-sync-design.md).
"""
from __future__ import annotations

from core.pipeline_policies.output_sanitizer import compute_terminal_correction


def test_strips_paired_internal_tag_leak():
    leaked = "Here is the answer. <think>secret reasoning</think> Done."
    corrected = compute_terminal_correction(leaked)
    assert corrected is not None
    assert "<think>" not in corrected
    assert "</think>" not in corrected
    assert "secret reasoning" not in corrected
    assert "Here is the answer." in corrected
    assert "Done." in corrected


def test_truncates_unclosed_internal_open_tag():
    # Normalizer regression: a <think> leaked and never closed — everything
    # after it is leaked reasoning that ran to the end of the stream.
    leaked = "Real answer here. <think> and then it just kept reasoning forever"
    corrected = compute_terminal_correction(leaked)
    assert corrected is not None
    assert "<think>" not in corrected
    assert "kept reasoning forever" not in corrected
    assert "Real answer here." in corrected


def test_strips_orphan_close_tag():
    # A stray closing tag with no matching open — remove the bare marker,
    # keep the answer text on both sides.
    leaked = "Answer text.</think> trailing prose"
    corrected = compute_terminal_correction(leaked)
    assert corrected is not None
    assert "</think>" not in corrected
    assert "Answer text." in corrected
    assert "trailing prose" in corrected


def test_clean_answer_returns_none():
    assert compute_terminal_correction("A perfectly normal answer.") is None


def test_empty_returns_none():
    assert compute_terminal_correction("") is None


def test_legit_thinking_word_is_not_a_false_positive():
    # The leak detector uses \\b so a real word like <thinking> (or prose
    # mentioning thinking) must NOT trigger a correction.
    text = "I was thinking about <thinking> as an HTML-ish token, no leak here."
    assert compute_terminal_correction(text) is None


def test_correction_is_idempotent():
    leaked = "Answer. <think>noise</think> tail"
    once = compute_terminal_correction(leaked)
    assert once is not None
    # Re-running on already-corrected text must be a no-op (stable).
    assert compute_terminal_correction(once) is None
