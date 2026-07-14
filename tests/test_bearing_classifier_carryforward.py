from __future__ import annotations

import pytest

from core import turn_classifier


class _StubProvider:
    """Test double for BearingProvider — controls get_active_goal()."""

    def __init__(self, goal: str = "") -> None:
        self._goal = goal

    def get_active_goal(self) -> str:
        return self._goal


@pytest.fixture(autouse=True)
def restore_provider():
    """Always restore the classifier's bearing_provider to None after each test."""
    yield
    turn_classifier.set_bearing_provider(None)


def _msg(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ── default (no provider) — classifier behaves as before ─────────────


def test_no_provider_no_carry_forward() -> None:
    shape = turn_classifier.classify(_msg("ok"), {})
    # Default behavior: "ok" → conversation (no intent match → "chat" → conversation)
    assert shape.task_type == "conversation"


def test_set_provider_with_empty_goal_no_carry_forward() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal=""))
    shape = turn_classifier.classify(_msg("ok"), {})
    assert shape.task_type == "conversation"


# ── active_goal + weak signal → carry-forward overrides ─────────────


def test_active_goal_weak_signal_overrides_to_analysis() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal="ongoing work"))
    shape = turn_classifier.classify(_msg("ok"), {})
    # Carry-forward bumps task_type from "conversation" to "analysis"
    assert shape.task_type == "analysis"


def test_active_goal_continue_msg_overrides() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal="ongoing work"))
    shape = turn_classifier.classify(_msg("continue"), {})
    assert shape.task_type == "analysis"


def test_prior_task_type_in_config_wins_over_default() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal="ongoing work"))
    shape = turn_classifier.classify(_msg("ok"), {"_prior_task_type": "action"})
    assert shape.task_type == "action"


def test_invalid_prior_task_type_falls_back_to_analysis() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal="ongoing work"))
    shape = turn_classifier.classify(_msg("ok"), {"_prior_task_type": "not-a-task-type"})
    assert shape.task_type == "analysis"


# ── strong signals — carry-forward should NOT fire ──────────────────


def test_greeting_keeps_conversation_even_with_active_goal() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal="ongoing work"))
    shape = turn_classifier.classify(_msg("hello"), {})
    assert shape.task_type == "conversation"


def test_vent_keeps_conversation_even_with_active_goal() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal="ongoing work"))
    shape = turn_classifier.classify(_msg("I'm frustrated with this"), {})
    assert shape.task_type == "conversation"


def test_code_intent_keeps_action_with_active_goal() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal="ongoing work"))
    shape = turn_classifier.classify(_msg("refactor this function"), {})
    assert shape.task_type == "action"


def test_debug_intent_keeps_action_with_active_goal() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal="ongoing work"))
    shape = turn_classifier.classify(_msg("there's a bug in the parser"), {})
    assert shape.task_type == "action"


def test_plan_intent_keeps_analysis_with_active_goal() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal="ongoing work"))
    shape = turn_classifier.classify(_msg("plan the next phase"), {})
    assert shape.task_type == "analysis"


# ── error containment ───────────────────────────────────────────────


def test_classifier_swallows_provider_failures() -> None:
    """If get_active_goal() raises, classifier must not crash — falls back
    to no-carry-forward path."""
    class _BrokenProvider:
        def get_active_goal(self) -> str:
            raise RuntimeError("broken")

    turn_classifier.set_bearing_provider(_BrokenProvider())
    shape = turn_classifier.classify(_msg("ok"), {})
    assert shape.task_type == "conversation"


# ── empty/whitespace user text ──────────────────────────────────────


def test_empty_text_no_carry_forward() -> None:
    turn_classifier.set_bearing_provider(_StubProvider(goal="goal"))
    shape = turn_classifier.classify(_msg("   "), {})
    # Empty text → "chat" default → conversation. But carry-forward needs
    # non-empty text to fire (no signal to read).
    assert shape.task_type == "conversation"
