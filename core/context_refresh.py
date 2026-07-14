"""Context refresh interceptor.

When conversation history exceeds a threshold, inject a condensed tagged
reminder so the LLM doesn't lose its identity or tool awareness in long
conversations. Kept minimal for 9B model context budgets.
"""
from __future__ import annotations

import time

REFRESH_THRESHOLD = 20   # first injection after this many messages
REFRESH_INTERVAL = 15    # re-inject every N messages after that

_REMINDER_TAG = "[SYSTEM REMINDER]"
_last_context_refresh: dict[str, object] = {}


def _build_condensed_reminder() -> str:
    # Behavioral nudge only. Identity ("You are Monolith") is already in the
    # cached system prefix; restating it here costs per-turn budget without
    # changing model behavior. Keep the rules that actually decay in long
    # conversations: output channel, envelope shape, fabrication discipline.
    return (
        f"{_REMINDER_TAG} Plaintext + <tool_call> envelopes. "
        "Say Unknown when unknown; do not fabricate state."
    )


def context_refresh_interceptor(
    messages: list[dict], config: dict
) -> list[dict] | None:
    global _last_context_refresh
    if len(messages) < REFRESH_THRESHOLD:
        return None

    for msg in messages:
        if _REMINDER_TAG in str(msg.get("content", "")):
            return None

    last_refresh_count = int(_last_context_refresh.get("message_count", 0) or 0)
    messages_since = len(messages) - last_refresh_count if last_refresh_count else len(messages)
    if last_refresh_count > 0 and messages_since < REFRESH_INTERVAL:
        return None

    result = list(messages)
    last_user_idx = -1
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx < 0:
        return None

    reminder = _build_condensed_reminder()
    result.insert(
        last_user_idx,
        {
            "role": "user",
            "content": reminder,
            "ephemeral": True,
            "source": "context_refresh",
        },
    )

    _last_context_refresh = {
        "triggered": True,
        "message_count": len(messages),
        "messages_since_last": messages_since,
        "insert_index": last_user_idx,
        "target_user_index": last_user_idx + 1,
        "reminder": reminder,
        "timestamp": time.time(),
    }
    return result


def get_last_context_refresh() -> dict[str, object]:
    return dict(_last_context_refresh)


def reset_refresh_state() -> None:
    """Clear the process-global refresh marker.

    Called from the engine's reset_conversation() so a new conversation starts
    with last_refresh_count == 0. Without this, the prior conversation's message
    high-water mark survives in the module global and suppresses refresh in the
    new conversation until its own message count exceeds that mark.
    """
    global _last_context_refresh
    _last_context_refresh = {}


def contribute_section(messages: list[dict], config: dict):
    """Section-contributor variant for the ephemeral_coalescer.

    Defers ``_last_context_refresh`` state update to ``on_commit`` — it fires
    only when the coalescer keeps this section under budget AND actually
    inserts the coalesced block. Otherwise a dropped section would still
    advance the refresh gate, skipping the next REFRESH_INTERVAL turns of
    refresh attempts.
    """
    from core.ephemeral_coalescer import SectionResult
    if len(messages) < REFRESH_THRESHOLD:
        return None
    for msg in messages:
        if _REMINDER_TAG in str(msg.get("content", "")):
            return None
    last_refresh_count = int(_last_context_refresh.get("message_count", 0) or 0)
    messages_since = len(messages) - last_refresh_count if last_refresh_count else len(messages)
    if last_refresh_count > 0 and messages_since < REFRESH_INTERVAL:
        return None
    reminder = _build_condensed_reminder()
    msg_count = len(messages)
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user" and not messages[i].get("ephemeral"):
            last_user_idx = i
            break

    def _commit() -> None:
        global _last_context_refresh
        _last_context_refresh = {
            "triggered": True,
            "message_count": msg_count,
            "messages_since_last": messages_since,
            "insert_index": last_user_idx,
            "target_user_index": last_user_idx + 1,
            "reminder": reminder,
            "timestamp": time.time(),
        }

    return SectionResult(name="context_refresh", text=reminder, on_commit=_commit)
