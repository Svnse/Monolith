"""Tests for core.turn_clock — the canonical per-turn wall-clock instant.

Mirrors core/turn_counter.py: one timezone-aware UTC instant captured once per
outer turn, frozen across tool-followups, stamped on config['_now_iso']. The flag
gates the stamp; flag off → "" → key never written → prompt byte-identical. All
paths are best-effort and never raise into generation.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core import turn_clock as tk


# ── feature flag ─────────────────────────────────────────────────────


def test_enabled_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("MONOLITH_TURN_CLOCK_V1", raising=False)
    assert tk.enabled() is False


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("", False),
])
def test_enabled_reflects_env(monkeypatch, val, expected) -> None:
    monkeypatch.setenv("MONOLITH_TURN_CLOCK_V1", val)
    assert tk.enabled() is expected


# ── capture_now_iso ──────────────────────────────────────────────────


def test_capture_now_iso_is_aware_utc_and_roundtrips() -> None:
    iso = tk.capture_now_iso()
    dt = datetime.fromisoformat(iso)  # must not raise
    assert dt.tzinfo is not None
    # captured in UTC
    assert dt.utcoffset() == timezone.utc.utcoffset(None)


# ── resolve_turn_now (engine glue decision, unit-testable) ───────────


def test_resolve_turn_now_disabled_returns_empty(monkeypatch) -> None:
    """Flag off (incl. mid-session ON->OFF): never stamps — flag-off byte-identical."""
    monkeypatch.delenv("MONOLITH_TURN_CLOCK_V1", raising=False)
    assert tk.resolve_turn_now("", is_outer=True) == ""
    assert tk.resolve_turn_now("2026-01-01T00:00:00+00:00", is_outer=False) == ""


def test_resolve_turn_now_outer_enabled_captures_fresh(monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_TURN_CLOCK_V1", "1")
    out = tk.resolve_turn_now("", is_outer=True)
    assert out  # non-empty
    assert datetime.fromisoformat(out).tzinfo is not None
    # a second outer turn captures a NEW instant (does not reuse)
    out2 = tk.resolve_turn_now(out, is_outer=True)
    assert datetime.fromisoformat(out2) >= datetime.fromisoformat(out)


def test_resolve_turn_now_inner_reuses_prev(monkeypatch) -> None:
    """Inner/tool-followup reuses the outer instant verbatim — frozen within a turn."""
    monkeypatch.setenv("MONOLITH_TURN_CLOCK_V1", "1")
    prev = "2026-06-07T15:30:00+00:00"
    assert tk.resolve_turn_now(prev, is_outer=False) == prev


def test_resolve_turn_now_inner_without_prev_captures_fresh(monkeypatch) -> None:
    """Mid-turn flag toggle: inner turn with no prior capture mints one rather
    than emit "" (which would mislabel the turn as flag-off)."""
    monkeypatch.setenv("MONOLITH_TURN_CLOCK_V1", "1")
    out = tk.resolve_turn_now("", is_outer=False)
    assert out
    assert datetime.fromisoformat(out).tzinfo is not None


# ── parse_local ──────────────────────────────────────────────────────


def test_parse_local_none_and_empty_return_none() -> None:
    assert tk.parse_local(None) is None
    assert tk.parse_local("") is None


def test_parse_local_corrupt_returns_none() -> None:
    assert tk.parse_local("not-a-timestamp") is None
    assert tk.parse_local("{bogus") is None


def test_parse_local_valid_returns_local_aware_same_instant() -> None:
    iso = "2026-06-07T15:30:00+00:00"
    dt = tk.parse_local(iso)
    assert dt is not None
    assert dt.tzinfo is not None  # aware
    # same instant, just rendered in local zone
    assert dt.astimezone(timezone.utc) == datetime(2026, 6, 7, 15, 30, tzinfo=timezone.utc)


def test_parse_local_naive_input_assumed_utc() -> None:
    dt = tk.parse_local("2026-06-07T15:30:00")  # no offset
    assert dt is not None
    assert dt.astimezone(timezone.utc) == datetime(2026, 6, 7, 15, 30, tzinfo=timezone.utc)
