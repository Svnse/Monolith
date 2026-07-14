"""Tests for core/llm_config.build_system_prompt temporal grounding.

When-plane fix #5: the cacheable system prefix must carry a coarse, DATE-only
"now" so the model has a stable temporal anchor even when the minute-resolution
temporal_context lane in [RUNTIME STATE] is dropped under budget. Date-only keeps the
prefix stable within a day (one cache miss/day, not one per minute).
"""
from __future__ import annotations

from datetime import datetime

from core.llm_config import build_system_prompt
from core.runtime_state_lanes import CONTRACT_PLACEHOLDER, LANES


def test_system_prompt_includes_coarse_current_date():
    now = datetime(2026, 6, 2, 12, 0)  # naive → strftime formats verbatim
    prompt = build_system_prompt({"system_prompt": "BASE PROMPT"}, now=now)
    assert "Current date: 2026-06-02 (" in prompt


def test_system_prompt_current_date_is_date_only_not_minute():
    """The prefix anchor must be date-resolution so it stays cacheable within a
    day; minute-resolution time belongs to the ephemeral temporal_context lane."""
    now = datetime(2026, 6, 2, 14, 37)
    prompt = build_system_prompt({"system_prompt": "BASE PROMPT"}, now=now)
    assert "14:37" not in prompt


def test_runtime_state_lane_contract_generated_from_registry():
    prompt = build_system_prompt(
        {"system_prompt": f"BASE PROMPT\n{CONTRACT_PLACEHOLDER}"},
        now=datetime(2026, 6, 2, 12, 0),
    )
    assert CONTRACT_PLACEHOLDER not in prompt
    for lane in LANES:
        assert lane.lead_phrase in prompt
    assert "Use temporal_relative only for elapsed-time or session-gap orientation" in prompt
