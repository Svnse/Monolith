"""The normalizer must track <think> nesting depth.

Live failure (2026-06-02): the bound model nested <think> inside <think> (and
re-deliberated after closing). The normalizer tracked thinking with a single
boolean, so the FIRST </think> exited and all subsequent reasoning leaked into
the answer lane — leaving no clean answer and defeating non-convergence
detection ("catch the true last think").
"""
from __future__ import annotations

from ui.pages.assistant_turn_box import AssistantStreamNormalizer


def test_nested_think_keeps_inner_reasoning_out_of_the_answer():
    raw = (
        "<think>I should reason carefully. "
        "<think>a nested aside</think> "
        "back to the outer reasoning.</think>"
        "Here is the actual answer."
    )
    norm = AssistantStreamNormalizer.from_text(raw)
    assert norm.answer_text.strip() == "Here is the actual answer."
    assert "a nested aside" in norm.thinking_text
    assert "back to the outer reasoning" in norm.thinking_text
    assert "a nested aside" not in norm.answer_text
    assert "back to the outer reasoning" not in norm.answer_text
    assert "</think>" not in norm.answer_text


def test_deeply_nested_think_three_levels():
    raw = "<think>a <think>b <think>c</think> d</think> e</think>ANSWER"
    norm = AssistantStreamNormalizer.from_text(raw)
    assert norm.answer_text.strip() == "ANSWER"
    for frag in ("a", "b", "c", "d", "e"):
        assert frag in norm.thinking_text


def test_single_think_still_separates_cleanly():
    # Regression: the common (non-nested) case must be unchanged.
    norm = AssistantStreamNormalizer.from_text("<think>reasoning here</think>the answer")
    assert norm.answer_text.strip() == "the answer"
    assert "reasoning here" in norm.thinking_text
    assert norm.thinking_seen is True
    assert norm.thinking_active is False  # cleanly closed
