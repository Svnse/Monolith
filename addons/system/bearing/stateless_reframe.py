"""MonoFrame v2 — ask extraction + the stateless re-derivation prompt.

The two pure pieces MonoFrame v2 still needs from the original producer:
  - ``session_asks`` — pull the real recent asks out of an agent-server/connect
    session (role=agent, text-field, name/channel prefixes), which the GUI-shaped
    drift.recent_asks misses. The /frame handler uses this to give the stateless
    CONTROL call real context.
  - ``build_reframe_messages`` — construct the bearing-stripped (clean) / bearing-
    injected (control) side-call messages, call-shape fixed.

The v1 numeric second-opinion loop (run_second_opinion/dispatch_async/observe_turn)
was superseded by the CorrectionCard pipeline (correction_runner) and removed.
"""
from __future__ import annotations

import re

_PREFIX_RE = re.compile(r"^\s*\[[^\]]*\]\s*")


def _msg_text_any(m: dict) -> str:
    """Text of a session message across shapes: the agent-server `text` field
    (content is None) OR the GUI `content` (str or list-of-blocks)."""
    t = m.get("text")
    if isinstance(t, str) and t.strip():
        return t
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(
            b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _strip_prefixes(text: str) -> str:
    """Strip leading bracket prefixes — agent name ([Claude (examiner)]) and/or
    [CHANNEL: ...] — that wrap the actual ask in the server session."""
    prev = None
    while prev != text:
        prev = text
        text = _PREFIX_RE.sub("", text)
    return text.strip()


def session_asks(messages: list[dict], k: int = 3) -> list[str]:
    """The last *k* real asks from a server/connect session (newest last in,
    oldest-first out). Treats role in (user, agent) as the ask side, reads the
    `text`-or-`content` body, strips name/channel prefixes, skips empties.

    drift.recent_asks is GUI-shaped (role==user, content); this is the
    agent-server analogue and is what the /frame handler must use."""
    out: list[str] = []
    for m in reversed(messages or ()):
        if not isinstance(m, dict):
            continue
        if m.get("role") not in ("user", "agent"):
            continue
        if m.get("ephemeral"):
            continue
        text = _strip_prefixes(_msg_text_any(m))
        if not text:
            continue
        out.append(text)
        if len(out) >= k:
            break
    return list(reversed(out))


_REFRAME_INSTRUCTION = (
    "You are re-deriving a one-sentence situational frame for an ongoing "
    "conversation, from scratch. Read the recent exchange and output ONE plain "
    "present-tense sentence naming what is being worked on RIGHT NOW — the live "
    "request, not anything already finished. Output only the sentence, no preamble."
)


def build_reframe_messages(
    recent_asks: list[str], *, bearing_block: str | None = None
) -> list[dict[str, str]]:
    """Construct the side-call messages. Call-shape is FIXED — the instruction
    (messages[0]) is identical whether or not ``bearing_block`` is present; only
    a bearing message is inserted for the control. Pure.
    """
    msgs: list[dict[str, str]] = [{"role": "system", "content": _REFRAME_INSTRUCTION}]
    if bearing_block:
        msgs.append({"role": "system", "content": str(bearing_block)})
    context = "\n".join(str(a) for a in (recent_asks or ()) if a)
    msgs.append({"role": "user", "content": f"Recent exchange:\n{context}"})
    return msgs
