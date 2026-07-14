"""Rating Telemetry — per-turn observation feed of recent rating outcomes.

Reads the Layer D `outcome_traces` store (kind=rating) at turn-start and
injects a compact [RATING TELEMETRY] ephemeral user message before the
latest non-ephemeral user turn. The model sees rolling avg + recent
values + worst/best with reasons.

This is the consumer side of the rating loop; the producer is
`record_outcome()` writes from chat.py (`/rating` slash command, thumbs
buttons). One signal, one surface.

Discipline: telemetry is *observation*, not rule. The block is plain
data; the model decides what to do. Pendings get obligation framing
because they're explicit promises; telemetry does NOT — overfitting the
signal (e.g. "user rates terse high → always be terse") is the risk this
framing guards against.

Same shape as the other standalone-compatible prompt contributors:
  * gated on a flag
  * read state (here: turn_trace store)
  * project to a string
  * insert as ephemeral user message before the latest non-ephemeral user

Flag: MONOLITH_RATING_TELEMETRY_V1 (default ON). Set =0 to disable.
"""
from __future__ import annotations

import os
from typing import Any

from core import turn_trace as _tt

_FLAG_ENV = "MONOLITH_RATING_TELEMETRY_V1"
_TAG = "[RATING TELEMETRY]"
_DEFAULT_WINDOW = 10
_MAX_RECENT_SHOWN = 5
_MAX_REASON_CHARS = 80


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _trim_reason(reason: Any) -> str | None:
    if not reason:
        return None
    text = str(reason).strip()
    if not text:
        return None
    if len(text) <= _MAX_REASON_CHARS:
        return text
    return text[: _MAX_REASON_CHARS - 1].rstrip() + "…"


def render_telemetry_block(*, window: int = _DEFAULT_WINDOW) -> str | None:
    """Build the [RATING TELEMETRY] block string. Returns None when no data."""
    snap = _tt.recent_ratings_summary(window=window)
    count = int(snap.get("count", 0) or 0)
    if count == 0:
        return None
    rolling = float(snap.get("rolling_avg", 0.0) or 0.0)
    recent = list(snap.get("recent", []) or [])
    worst = snap.get("worst") if isinstance(snap.get("worst"), dict) else None
    best = snap.get("best") if isinstance(snap.get("best"), dict) else None

    if len(recent) > _MAX_RECENT_SHOWN:
        recent = recent[-_MAX_RECENT_SHOWN:]

    lines = [_TAG]
    lines.append(f"- rolling avg: {rolling:.1f} (n={count})")
    if recent:
        recent_str = ", ".join(str(int(v)) for v in recent)
        lines.append(f"- recent: {recent_str} (oldest→newest)")
    if worst is not None:
        reason = _trim_reason(worst.get("reason"))
        line = f"- worst recent: {int(worst.get('value', 0))}"
        if reason:
            line += f' — "{reason}"'
        lines.append(line)
    # Skip best line when it's the same turn as worst (only one rating, or all tied).
    show_best = (
        best is not None
        and (worst is None or best.get("turn_id") != worst.get("turn_id"))
    )
    if show_best:
        reason = _trim_reason(best.get("reason"))
        line = f"- best recent:  {int(best.get('value', 0))}"
        if reason:
            line += f' — "{reason}"'
        lines.append(line)
    return "\n".join(lines)


def _is_peer_turn(messages: list[dict]) -> bool:
    """True when this turn arrived on a CONNECT peer channel (an examiner /
    trainer LLM), identified by the ``[CHANNEL: connect/...]`` tag the
    agent-server prepends to the user message. Rating telemetry is suppressed on
    these turns so the peer doesn't see Monolith's own grades and perform to them
    — the observer effect that warms a 'cold' training session. UI (E) turns,
    which carry ``[CHANNEL: USER...]``, keep telemetry."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            return "[CHANNEL: connect/" in str(msg.get("content") or "")
    return False


def rating_telemetry_interceptor(
    messages: list[dict], config: dict
) -> list[dict] | None:
    """Inject [RATING TELEMETRY] before the latest non-ephemeral user turn.

    Returns None when:
      - flag MONOLITH_RATING_TELEMETRY_V1 is off
      - the turn is a CONNECT peer turn (observer-effect guard)
      - no rating outcomes recorded yet
      - block is already present (defense vs double-fire on tool followups)
      - no non-ephemeral user message exists in the history
    """
    if not _flag_enabled() or _is_peer_turn(messages):
        return None
    block = render_telemetry_block()
    if block is None:
        return None
    for msg in messages:
        if _TAG in str(msg.get("content", "")):
            return None
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            last_user_idx = i
            break
    if last_user_idx < 0:
        return None
    result = list(messages)
    result.insert(
        last_user_idx,
        {
            "role": "user",
            "content": block,
            "ephemeral": True,
            "source": "rating_telemetry",
        },
    )
    return result


def contribute_section(messages: list[dict], config: dict):
    """Section-contributor variant for the ephemeral_coalescer.

    Suppressed on CONNECT peer turns (observer-effect guard) — see
    ``_is_peer_turn``; this is the live path via the ephemeral_coalescer."""
    from core.ephemeral_coalescer import SectionResult
    if not _flag_enabled() or _is_peer_turn(messages):
        return None
    block = render_telemetry_block()
    if block is None:
        return None
    return SectionResult(name="rating_telemetry", text=block)
