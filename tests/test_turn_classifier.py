"""Golden tests for core.turn_classifier.

Property: classification is a pure function of (messages, config). Same input
always produces the same TurnShape. No LLM round-trip, no state read.
"""
from __future__ import annotations

import pytest

from core.turn_classifier import classify
from core.turn_shape import TurnShape


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ── Determinism ──────────────────────────────────────────────────────


def test_classify_is_deterministic():
    msgs = _user("Hey")
    shapes = [classify(msgs, {}) for _ in range(5)]
    assert all(s == shapes[0] for s in shapes), "classify must be pure"


def test_empty_messages_returns_default_shape():
    shape = classify([], {})
    assert shape.effort_tier in {"low", "med"}
    assert "chat" in shape.intent_tags
    assert shape.task_type == "conversation"


# ── Greeting ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("greeting", ["Hey", "hi", "Hello!", "yo", "Good morning"])
def test_greeting_is_low_tier_chat(greeting: str):
    shape = classify(_user(greeting), {})
    assert shape.effort_tier == "low"
    assert shape.task_type == "conversation"
    assert "chat" in shape.intent_tags
    assert shape.complexity_score < 25
    assert shape.confidence >= 0.7


# ── Action intents ───────────────────────────────────────────────────


def test_debug_request_routed_to_action():
    shape = classify(_user("There's a bug in auth.py — login() crashes with a NoneType traceback"), {})
    assert shape.task_type == "action"
    assert "debug" in shape.intent_tags or "code" in shape.intent_tags
    assert shape.effort_tier in {"med", "high", "xhigh"}


def test_code_request_tagged_as_code():
    shape = classify(_user("Refactor the authenticate() function in auth.py to use async"), {})
    assert "code" in shape.intent_tags
    assert shape.task_type == "action"


def test_retrieval_request_tagged():
    shape = classify(_user("Find all callers of the login function"), {})
    assert "retrieval" in shape.intent_tags


# ── Analysis intents ─────────────────────────────────────────────────


def test_analysis_request():
    shape = classify(_user("Why is this approach better than the alternatives? Compare the trade-offs."), {})
    assert shape.task_type == "analysis"
    assert "analysis" in shape.intent_tags


def test_plan_request():
    shape = classify(_user("Plan out the migration from monolith to microservices"), {})
    assert "plan" in shape.intent_tags
    assert shape.task_type == "analysis"


def test_learn_request():
    shape = classify(_user("Explain how the prompt compiler works"), {})
    assert "learn" in shape.intent_tags
    assert shape.task_type == "analysis"


# ── Conversation / vent ──────────────────────────────────────────────


def test_vent_recognized():
    shape = classify(_user("I'm so frustrated with this. Nothing is working."), {})
    assert "vent" in shape.intent_tags
    assert shape.task_type == "conversation"


def test_creative_request_tagged():
    shape = classify(_user("Write me a short story about a lighthouse keeper"), {})
    assert "creative" in shape.intent_tags


# ── Score → tier mapping ─────────────────────────────────────────────


def test_score_to_tier_mapping_monotonic():
    """Tiers must monotonically increase with complexity score."""
    from core.turn_classifier import _score_to_tier
    tiers_in_order = ["low", "med", "high", "xhigh", "ultimate"]
    seen: list[str] = []
    for score in range(0, 101, 5):
        tier = _score_to_tier(score)
        if not seen or tier != seen[-1]:
            seen.append(tier)
    # Should be a subsequence of tiers_in_order
    idx = 0
    for tier in seen:
        while idx < len(tiers_in_order) and tiers_in_order[idx] != tier:
            idx += 1
        assert idx < len(tiers_in_order), f"unexpected tier ordering: {seen}"
        idx += 1


# ── TurnShape serialization ──────────────────────────────────────────


def test_turn_shape_round_trips_through_dict():
    shape = classify(_user("Hello there"), {})
    raw = shape.to_dict()
    rebuilt = TurnShape.from_dict(raw)
    assert rebuilt == shape


def test_turn_shape_from_dict_handles_garbage():
    assert TurnShape.from_dict(None) is None
    assert TurnShape.from_dict("not a dict") is None
    # Empty dict yields a default shape, not None
    shape = TurnShape.from_dict({})
    assert shape is not None
    assert shape.effort_tier == "med"


# ── Frame ephemeral filtering ────────────────────────────────────────


def test_ephemeral_messages_ignored_for_classification():
    """Ephemeral context blocks (BUDGET GUIDANCE, RUNTIME STATE, etc.) must
    not shape classification — they're system-injected, not user input."""
    msgs = [
        {"role": "user", "content": "[BUDGET GUIDANCE] Respond concisely", "ephemeral": True},
        {"role": "user", "content": "[RUNTIME STATE] Current execution facts: backend=openai", "ephemeral": True},
        {"role": "user", "content": "Hey"},
    ]
    shape_a = classify(msgs, {})
    shape_b = classify(_user("Hey"), {})
    # Tier + intent should match — ephemerals shouldn't perturb the classification
    assert shape_a.effort_tier == shape_b.effort_tier
    assert shape_a.intent_tags == shape_b.intent_tags
