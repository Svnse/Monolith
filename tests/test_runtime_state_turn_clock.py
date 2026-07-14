"""TurnClock convergence: both temporal RENDER lanes derive from ONE injected
instant (config['_now_iso']) instead of each reading the OS clock independently.

This is the live-divergence the build targets — verified pre-change that
render_temporal_lane (local tz) and render_relative_time_lane (UTC) each called
datetime.now() in the same [RUNTIME STATE] block. After the fix, contribute_section
passes the captured instant into render_runtime_state's existing now= seam.
"""
from __future__ import annotations

import pytest

from core import continuity
from core import runtime_state_projection as rsp
from core import temporal_context
from core import turn_clock
from core.runtime_state_lanes import lead_phrase


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(continuity, "_STORE_PATH", tmp_path / "continuity.json")
    yield


def _user(text: str = "tell me about restore monolith runtime state please"):
    return [{"role": "user", "content": text}]


def _multi_turn():
    return [
        {"role": "user", "content": "first question about restore monolith runtime"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second question about restore monolith runtime"},
    ]


def _temporal_line(text: str) -> str | None:
    lead = lead_phrase("temporal_context")
    for line in text.splitlines():
        if line.startswith(lead):
            return line
    return None


FIXED = "2026-06-07T15:30:00+00:00"


def test_absolute_lane_renders_the_injected_instant(tmp_store, monkeypatch):
    monkeypatch.setenv("MONOLITH_TURN_CLOCK_V1", "1")
    monkeypatch.setenv("MONOLITH_TEMPORAL_CONTEXT_V1", "1")
    section = rsp.contribute_section(_user(), {"_now_iso": FIXED})
    assert section is not None
    expected = temporal_context.format_temporal_value(turn_clock.parse_local(FIXED))
    assert expected in section.text


def test_temporal_lane_is_deterministic_under_injected_now(tmp_store, monkeypatch):
    monkeypatch.setenv("MONOLITH_TURN_CLOCK_V1", "1")
    monkeypatch.setenv("MONOLITH_TEMPORAL_CONTEXT_V1", "1")
    a = rsp.contribute_section(_user(), {"_now_iso": FIXED}).text
    b = rsp.contribute_section(_user(), {"_now_iso": FIXED}).text
    line = _temporal_line(a)
    assert line is not None
    assert line == _temporal_line(b)  # no OS-clock drift between calls
    assert line == _temporal_line(
        f"{lead_phrase('temporal_context')} "
        + temporal_context.format_temporal_value(turn_clock.parse_local(FIXED))
    )


def test_relative_lane_uses_the_same_injected_instant(tmp_store, monkeypatch):
    # If the injected now did NOT reach the relative lane, elapsed would be ~now-OS
    # (huge), never "5m ago". Proves both lanes share the one instant.
    monkeypatch.setenv("MONOLITH_TURN_CLOCK_V1", "1")
    monkeypatch.setenv("MONOLITH_TEMPORAL_CONTEXT_V1", "1")
    monkeypatch.setenv("MONOLITH_RELATIVE_TIME_V1", "1")
    continuity.set_last_turn_at("2026-06-07T15:25:00+00:00")  # 5 min before FIXED
    section = rsp.contribute_section(_multi_turn(), {"_now_iso": FIXED})
    assert section is not None
    assert "last turn" in section.text and "5m ago" in section.text


def test_no_now_iso_key_falls_through_to_os_clock(tmp_store, monkeypatch):
    # Flag-off byte-identical path: no _now_iso key → parse_local(None) → None →
    # render_runtime_state(now=None) → each lane reads the OS clock, as before.
    monkeypatch.delenv("MONOLITH_TURN_CLOCK_V1", raising=False)
    monkeypatch.setenv("MONOLITH_TEMPORAL_CONTEXT_V1", "1")
    section = rsp.contribute_section(_user(), {})
    assert section is not None
    assert _temporal_line(section.text) is not None  # still renders, no crash


def test_corrupt_now_iso_falls_through_safely(tmp_store, monkeypatch):
    monkeypatch.setenv("MONOLITH_TEMPORAL_CONTEXT_V1", "1")
    section = rsp.contribute_section(_user(), {"_now_iso": "garbage"})
    assert section is not None
    assert _temporal_line(section.text) is not None  # parse_local→None→OS clock
