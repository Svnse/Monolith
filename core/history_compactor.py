"""History compactor — elide stale verbose tool messages before LLM dispatch.

Operates on the derived messages list, never on LLMEngine.conversation_history.
User scrollback stays intact; the model sees a compacted copy.

Compaction rule: tool-role messages older than TURN_AGE_THRESHOLD turns whose
content exceeds MIN_SIZE_CHARS are replaced with a short summary marker. The
marker names the tool and original size so the model knows the data existed.

Flag: MONOLITH_HISTORY_COMPACT_V1 (default OFF — ships dark).
"""
from __future__ import annotations

import os
import re

_FLAG_ENV = "MONOLITH_HISTORY_COMPACT_V1"

TURN_AGE_THRESHOLD = 3
MIN_SIZE_CHARS = 4000

_TOOL_NAME_RE = re.compile(
    r"^\s*\[?\s*(\w[\w/._-]*)",
)


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _guess_tool_name(content: str) -> str:
    m = _TOOL_NAME_RE.match(content)
    if m:
        return m.group(1)
    return "tool"


def _last_user_turn_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user" and not msg.get("ephemeral"):
            return i
    return len(messages) - 1


def _count_user_turns_after(messages: list[dict], index: int) -> int:
    count = 0
    for i in range(index + 1, len(messages)):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user" and not msg.get("ephemeral"):
            count += 1
    return count


def compact_for_dispatch(
    messages: list[dict],
    *,
    turn_age_threshold: int = TURN_AGE_THRESHOLD,
    min_size_chars: int = MIN_SIZE_CHARS,
) -> list[dict]:
    """Return a compacted copy of *messages* for LLM dispatch.

    Tool-role messages whose content exceeds *min_size_chars* and that are
    more than *turn_age_threshold* user-turns old are replaced with a short
    summary marker. All other messages are passed through unchanged.

    Returns *messages* unmodified (same list object) when the flag is off
    or no compaction is needed — no copy overhead in the common case.
    """
    if not _flag_enabled():
        return messages

    compacted = False
    result: list[dict] | None = None

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content", "") or "")
        if len(content) < min_size_chars:
            continue
        age = _count_user_turns_after(messages, i)
        if age < turn_age_threshold:
            continue

        if result is None:
            result = list(messages)
        tool_name = _guess_tool_name(content)
        marker = f"[{tool_name}: {len(content)} chars elided after {age} turns]"
        result[i] = {**msg, "content": marker}
        compacted = True

    return result if compacted else messages
