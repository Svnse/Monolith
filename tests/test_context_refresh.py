from __future__ import annotations

import core.context_refresh as cr


def _msgs_with_depth(depth: int) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "You are Monolith."}]
    for i in range(max(0, depth - 1)):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"m{i}"})
    if messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "final user prompt"})
    return messages


def _reset_refresh_state() -> None:
    cr._last_context_refresh = {}


def test_context_refresh_inserts_tagged_reminder_before_latest_user():
    _reset_refresh_state()
    msgs = _msgs_with_depth(cr.REFRESH_THRESHOLD + 1)
    original_last_user = msgs[-1]["content"]

    result = cr.context_refresh_interceptor(msgs, {})

    assert result is not None
    assert len(result) == len(msgs) + 1
    assert result[-2]["role"] == "user"
    assert result[-2].get("source") == "context_refresh"
    assert "[SYSTEM REMINDER]" in result[-2]["content"]
    assert result[-1]["role"] == "user"
    assert result[-1]["content"] == original_last_user


def test_context_refresh_does_not_reinject_within_interval():
    _reset_refresh_state()
    msgs = _msgs_with_depth(cr.REFRESH_THRESHOLD + 2)
    first = cr.context_refresh_interceptor(msgs, {})
    assert first is not None

    # One-message growth should still be within REFRESH_INTERVAL.
    next_msgs = _msgs_with_depth(cr.REFRESH_THRESHOLD + 3)
    second = cr.context_refresh_interceptor(next_msgs, {})
    assert second is None


def test_context_refresh_skips_when_tag_already_present():
    _reset_refresh_state()
    msgs = _msgs_with_depth(cr.REFRESH_THRESHOLD + 1)
    msgs.insert(-1, {"role": "user", "content": "[SYSTEM REMINDER] existing"})

    result = cr.context_refresh_interceptor(msgs, {})

    assert result is None


def test_context_refresh_snapshot_tracks_insert_indices():
    _reset_refresh_state()
    msgs = _msgs_with_depth(cr.REFRESH_THRESHOLD + 1)

    result = cr.context_refresh_interceptor(msgs, {})
    assert result is not None

    snapshot = cr.get_last_context_refresh()
    assert snapshot.get("triggered") is True
    assert isinstance(snapshot.get("insert_index"), int)
    assert isinstance(snapshot.get("target_user_index"), int)
    assert snapshot.get("target_user_index") == snapshot.get("insert_index") + 1


def test_condensed_reminder_excludes_full_skills_catalog():
    """Fix #5 (2026-05-14 audit): {skills_catalog} in prompts/system.md is the
    canonical render of the tool list. context_refresh's [SYSTEM REMINDER] must
    not duplicate it — otherwise the catalog ships twice per long conversation
    (once at cold start via the system prompt, again every REFRESH_INTERVAL).
    """
    from core.skill_registry import build_tool_catalog

    catalog = build_tool_catalog()
    reminder = cr._build_condensed_reminder()

    assert catalog not in reminder, (
        "build_tool_catalog() output must not be embedded in the SYSTEM REMINDER "
        "— it duplicates prompts/system.md's {skills_catalog} slot."
    )


def test_condensed_reminder_is_minimal():
    """Reminder should be the identity + tool-envelope + plaintext-only nudge,
    nothing more. Keeps long-conversation overhead small."""
    reminder = cr._build_condensed_reminder()
    # Identity + nudge fits in ~200 chars; with catalog it's 1000+.
    assert len(reminder) < 500, (
        f"SYSTEM REMINDER is {len(reminder)} chars — catalog likely still embedded."
    )


def test_reset_refresh_state_clears_marker():
    """When-plane fix #6: a new conversation must not inherit the prior
    conversation's message high-water mark. reset_refresh_state() clears the
    module global so the count-gate starts fresh (last_refresh_count == 0).
    """
    cr._last_context_refresh = {"message_count": 80, "triggered": True}
    cr.reset_refresh_state()
    assert cr.get_last_context_refresh() == {}
