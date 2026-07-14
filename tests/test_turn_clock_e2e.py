"""End-to-end: engine glue stamps config['_now_iso'] → temporal lane uses it.

Covers the full production chain MINUS the Qt-bound glue in engine/llm.py that
sets config['_now_iso'] (the same standard test_turn_counter_bearing_e2e.py
holds — it covers everything but the 4 Qt-bound engine lines). `engine_glue`
below is a FAITHFUL mirror of those lines:

    self._last_now_iso = turn_clock.resolve_turn_now(self._last_now_iso, is_outer)
    if self._last_now_iso:
        config["_now_iso"] = self._last_now_iso

The clock is default-OFF, so the live suite can never exercise the stamp; this
e2e drives the real resolver + stamp logic + real consumer (contribute_section),
asserting presence-on-outer, reuse-on-inner (frozen across tool-followups), and
flag-off-no-key. The residual literal engine lines are read-verified, matching the
counter's accepted standard.
"""
from __future__ import annotations

import pytest

from core import continuity
from core import runtime_state_projection as rsp
from core import temporal_context
from core import turn_clock as tk
from core.runtime_state_lanes import lead_phrase


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(continuity, "_STORE_PATH", tmp_path / "continuity.json")
    yield


def _user(text: str = "tell me about restore monolith runtime state please"):
    return [{"role": "user", "content": text}]


def _temporal_line(text: str) -> str | None:
    lead = lead_phrase("temporal_context")
    for line in text.splitlines():
        if line.startswith(lead):
            return line
    return None


def engine_glue(prev_iso: str, is_outer: bool, turn_id: str):
    """Faithful mirror of the engine/llm.py clock glue lines (~:1427)."""
    new_iso = tk.resolve_turn_now(prev_iso, is_outer)
    cfg = {"_turn_id": turn_id}
    if new_iso:
        cfg["_now_iso"] = new_iso
    return new_iso, cfg


def test_outer_turn_stamps_now_iso_and_temporal_lane_uses_it(tmp_store, monkeypatch):
    monkeypatch.setenv("MONOLITH_TURN_CLOCK_V1", "1")
    monkeypatch.setenv("MONOLITH_TEMPORAL_CONTEXT_V1", "1")
    last_iso, cfg = engine_glue("", is_outer=True, turn_id="u1")
    assert cfg.get("_now_iso")  # (a) presence on outer
    section = rsp.contribute_section(_user(), cfg)
    assert section is not None
    # the SAME instant the engine stamped reaches the lane
    expected = temporal_context.format_temporal_value(tk.parse_local(cfg["_now_iso"]))
    assert expected in section.text


def test_inner_turn_reuses_outer_instant_and_lane_is_frozen(tmp_store, monkeypatch):
    monkeypatch.setenv("MONOLITH_TURN_CLOCK_V1", "1")
    monkeypatch.setenv("MONOLITH_TEMPORAL_CONTEXT_V1", "1")
    outer_iso, cfg_outer = engine_glue("", is_outer=True, turn_id="u1")
    inner_iso, cfg_inner = engine_glue(outer_iso, is_outer=False, turn_id="u1")
    assert inner_iso == outer_iso  # (b) frozen across the tool-followup
    assert cfg_inner["_now_iso"] == cfg_outer["_now_iso"]
    s_outer = rsp.contribute_section(_user(), cfg_outer)
    s_inner = rsp.contribute_section(_user(), cfg_inner)
    # KV-stable: the temporal lane renders identically across the followup
    assert _temporal_line(s_outer.text) == _temporal_line(s_inner.text)


def test_flag_off_stamps_no_now_iso_key(monkeypatch):
    monkeypatch.delenv("MONOLITH_TURN_CLOCK_V1", raising=False)
    new_iso, cfg = engine_glue("", is_outer=True, turn_id="u1")
    assert new_iso == ""
    assert "_now_iso" not in cfg  # byte-identical: key never written


def test_mid_session_toggle_off_stops_stamping(monkeypatch):
    monkeypatch.setenv("MONOLITH_TURN_CLOCK_V1", "1")
    last, cfg_on = engine_glue("", is_outer=True, turn_id="u1")
    assert cfg_on.get("_now_iso")
    # toggled OFF mid-session: next outer turn forgets it → no key written
    monkeypatch.delenv("MONOLITH_TURN_CLOCK_V1", raising=False)
    last, cfg_off = engine_glue(last, is_outer=True, turn_id="u2")
    assert "_now_iso" not in cfg_off
