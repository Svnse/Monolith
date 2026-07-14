"""Observer V0 prompt compiler."""
from __future__ import annotations

from . import runtime, store


def contribute_section(messages: list[dict], config: dict):
    """Return the latest turn-boundary Observer block for coalescer injection."""

    from core.ephemeral_coalescer import SectionResult

    if not runtime.is_enabled():
        return None
    latest = store.read_latest()
    if not latest:
        return None
    block = str(latest.get("block") or "").strip()
    if not block:
        return None
    return SectionResult(name="observer", text=block)
