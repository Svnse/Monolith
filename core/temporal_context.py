"""Temporal context — current local time/date for runtime grounding.

Monolith confabulates time when not grounded. This contributor injects the
local wall-clock time so the model can answer "what time is it?" / "what day
is it?" without inventing a value.

Live prompt delivery now flows through the temporal_context lane inside
core.runtime_state_projection's [RUNTIME STATE] block. The legacy
render_temporal_block()/contribute_section() surface remains as a standalone
helper for tests and compatibility, but it is no longer registered directly in
the ephemeral coalescer.

Flag: MONOLITH_TEMPORAL_CONTEXT_V1 (default ON). Set =0 to disable.
"""
from __future__ import annotations

import os
from datetime import datetime


_BLOCK_TAG = "[TEMPORAL CONTEXT]"
_FLAG_ENV = "MONOLITH_TEMPORAL_CONTEXT_V1"


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_temporal_enabled() -> bool:
    """Public flag check for coalesced runtime-state projection."""
    return _flag_enabled()


def format_temporal_value(now: datetime | None = None) -> str:
    """Return the local wall-clock value without the block wrapper."""
    when = now if now is not None else datetime.now().astimezone()
    return when.strftime("%Y-%m-%d %H:%M %Z (%A)")


def render_temporal_block(now: datetime | None = None) -> str:
    """Build the [TEMPORAL CONTEXT] block.

    Format: ``current_time: 2026-05-14 14:32 EDT (Thursday)``

    Uses local timezone from the OS. The `now` parameter accepts an injected
    datetime for tests.
    """
    return f"{_BLOCK_TAG}\ncurrent_time: {format_temporal_value(now)}"


def contribute_section(messages: list[dict], config: dict):
    """Section-contributor for the ephemeral_coalescer."""
    from core.ephemeral_coalescer import SectionResult
    if not _flag_enabled():
        return None
    return SectionResult(name="temporal_context", text=render_temporal_block())
