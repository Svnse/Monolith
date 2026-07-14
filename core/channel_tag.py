"""[CHANNEL: ...] header authoring.

Single source of truth for the runtime-injected channel header that the model
sees at the start of every turn (and at the start of every replayed past turn
in context). Reads active prompts and monothink state from the prompt library
and world_state.

Contract documented in prompts/system.md (CHANNEL AWARENESS section). Output
shapes:

  Generating turn (local UI):        [CHANNEL: USER, prompts=falsify+descent]
  Generating turn (peer):            [CHANNEL: connect/Codex, mcp send_message, prompts=falsify]
  Generating turn (monothink on):    [CHANNEL: USER, prompts=orient, monothink=on]
  Past USER turn (in history):       [CHANNEL: USER]
  Past assistant turn:               [CHANNEL: ASSISTANT]
  Past agent/tool turn:              [CHANNEL: AGENT]

Mode fields appear only on the *generating* turn (the one the model is about
to answer). Replayed history shows the role token only — modes describe the
scaffold for *generating*, not for replay.
"""
from __future__ import annotations

import re


# Strip one leading [CHANNEL: ...] header. Tags don't contain "]" or newlines
# by construction, so the bracket-and-newline excludes are safe.
_LEADING_CHANNEL_RE = re.compile(r"\A\s*\[CHANNEL:[^\]\n]*\]\s*\n*")

# Strip ANY [CHANNEL: ...] header (leading or embedded) plus the blank line it
# introduced. For DISPLAY only — peer/connect turns are stored with the tag
# embedded after an "[agent] " prefix so the model still receives its channel
# context; this clears it from the rendered text without touching storage.
_ANY_CHANNEL_RE = re.compile(r"\[CHANNEL:[^\]\n]*\]\n*")


def _peek_prompts() -> list[str]:
    """Read active prompts from the prompt library without consuming."""
    try:
        from core.prompt_library import peek
        return peek()
    except Exception:
        return []


def _peek_monothink() -> bool | None:
    """Read monothink state from world_state without consuming."""
    try:
        from core.monothink import _monothink_world_state
        if _monothink_world_state is None:
            return None
        snap = _monothink_world_state.get_monothink_state()
        if snap.get("once") is not None:
            return snap["once"]
        if snap.get("enabled"):
            return True
        return None
    except Exception:
        return None


def build_channel_tag(
    role: str,
    *,
    transport: str | None = None,
    include_modes: bool = False,
) -> str:
    """Compose a [CHANNEL: ...] header.

    Args:
        role: The sender role token. One of "USER", "ASSISTANT", "AGENT", or
            "connect/<peer_name>" for peer-LLM turns.
        transport: For peer turns only — names the inbound transport
            ("/chat blocking", "/chat/stream", "mcp send_message"). Ignored
            for local roles.
        include_modes: When True, appends active prompts and monothink state
            read live from world_state. Defaults to False — only the generating
            turn's tag should carry mode fields.

    Returns:
        A single-line `[CHANNEL: ...]` string with no trailing newline.
    """
    parts = [f"[CHANNEL: {role}"]
    if transport:
        parts.append(f", {transport}")
    if include_modes:
        prompts = _peek_prompts()
        if prompts:
            parts.append(f", prompts={'+'.join(prompts)}")
        monothink = _peek_monothink()
        if monothink is not None:
            parts.append(f", monothink={'on' if monothink else 'off'}")
    parts.append("]")
    return "".join(parts)


def strip_leading_channel_tag(message: str) -> tuple[str, str | None]:
    """Remove one leading [CHANNEL: ...] header from *message*.

    Returns (body_without_tag, removed_tag_or_None). Embedded tags later in
    the body are left alone — only the leading prefix is stripped. Idempotent:
    safe to call on already-stripped messages.
    """
    if not isinstance(message, str):
        return ("", None)
    m = _LEADING_CHANNEL_RE.match(message)
    if not m:
        return (message, None)
    return (message[m.end():], m.group(0).strip())


def strip_channel_tags_for_display(message: str) -> str:
    """Remove every [CHANNEL: ...] header from *message* for rendering.

    Unlike strip_leading_channel_tag, this also clears tags that sit after a
    prefix (e.g. the "[Codex] " agent-name prefix on a connect turn) and any
    echoed mid-text tags. DISPLAY-ONLY: callers must keep the stored message
    intact so the model and bearing still see the channel context.
    """
    if not isinstance(message, str):
        return ""
    return _ANY_CHANNEL_RE.sub("", message)


__all__ = [
    "build_channel_tag",
    "strip_leading_channel_tag",
    "strip_channel_tags_for_display",
]
