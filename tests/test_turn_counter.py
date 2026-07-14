"""Tests for core.turn_counter — the persisted monotonic outer-turn counter.

The counter must be monotonic ACROSS restarts (persisted), because bearing.json
outlives a session and a session-scoped counter would compute a negative age for
a bearing carried from a prior session. All IO is best-effort: corruption or an
unwritable path degrades to a safe value and never raises.
"""
from __future__ import annotations

import pytest

from core import turn_counter as tc


# ── increment / read ─────────────────────────────────────────────────


def test_next_turn_starts_at_one_on_fresh_file(tmp_path) -> None:
    p = tmp_path / "turn_counter.json"
    assert tc.next_turn(p) == 1
    assert tc.next_turn(p) == 2
    assert tc.next_turn(p) == 3


def test_current_turn_reads_without_incrementing(tmp_path) -> None:
    p = tmp_path / "turn_counter.json"
    tc.next_turn(p)
    tc.next_turn(p)  # -> 2
    assert tc.current_turn(p) == 2
    assert tc.current_turn(p) == 2  # idempotent read


def test_current_turn_zero_when_file_missing(tmp_path) -> None:
    assert tc.current_turn(tmp_path / "nope.json") == 0


# ── persistence across simulated restart ─────────────────────────────


def test_counter_is_monotonic_across_simulated_restart(tmp_path) -> None:
    """No in-memory state: a fresh call on the same path continues the count,
    which IS the restart contract (process dies, file persists)."""
    p = tmp_path / "turn_counter.json"
    assert tc.next_turn(p) == 1
    assert tc.next_turn(p) == 2
    # "restart": nothing in memory; only the file knows.
    assert tc.next_turn(p) == 3
    assert tc.current_turn(p) == 3


# ── resilience (best-effort, never raises) ───────────────────────────


def test_recovers_from_malformed_json(tmp_path) -> None:
    p = tmp_path / "turn_counter.json"
    p.write_text("{not valid", encoding="utf-8")
    assert tc.current_turn(p) == 0
    assert tc.next_turn(p) == 1  # recovers, does not raise


def test_negative_stored_value_clamped_to_zero(tmp_path) -> None:
    p = tmp_path / "turn_counter.json"
    p.write_text('{"n": -5}', encoding="utf-8")
    assert tc.current_turn(p) == 0


def test_next_turn_never_raises_on_unwritable_path(tmp_path) -> None:
    """Path points at a directory → open('w') fails. Must still return an int."""
    d = tmp_path / "a_dir"
    d.mkdir()
    # current_turn on a dir reads as 0; next_turn returns 1 without raising.
    assert tc.next_turn(d) == 1


# ── feature flag ─────────────────────────────────────────────────────


def test_enabled_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("MONOLITH_TURN_COUNTER_V1", raising=False)
    assert tc.enabled() is False


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("", False),
])
def test_enabled_reflects_env(monkeypatch, val, expected) -> None:
    monkeypatch.setenv("MONOLITH_TURN_COUNTER_V1", val)
    assert tc.enabled() is expected


# ── resolve_turn_n (engine glue decision, unit-testable) ─────────────


def test_resolve_turn_n_outer_enabled_increments(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_TURN_COUNTER_V1", "1")
    monkeypatch.setattr(tc, "_counter_path", lambda: tmp_path / "c.json")
    assert tc.resolve_turn_n(0, is_outer=True) == 1
    assert tc.resolve_turn_n(1, is_outer=True) == 2


def test_resolve_turn_n_inner_reuses_prev_without_writing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_TURN_COUNTER_V1", "1")
    monkeypatch.setattr(tc, "_counter_path", lambda: tmp_path / "c.json")
    assert tc.resolve_turn_n(5, is_outer=False) == 5  # reuse, no increment
    assert tc.current_turn(tmp_path / "c.json") == 0  # nothing persisted


def test_resolve_turn_n_disabled_outer_returns_zero(monkeypatch) -> None:
    """Flag off (incl. mid-session ON->OFF toggle): an outer turn forgets the
    count, so config['_turn_n'] is never written — flag-off byte-identical."""
    monkeypatch.delenv("MONOLITH_TURN_COUNTER_V1", raising=False)
    assert tc.resolve_turn_n(7, is_outer=True) == 0


def test_resolve_turn_n_disabled_inner_returns_zero(monkeypatch) -> None:
    """Even a within-turn toggle: a disabled inner turn drops to 0."""
    monkeypatch.delenv("MONOLITH_TURN_COUNTER_V1", raising=False)
    assert tc.resolve_turn_n(7, is_outer=False) == 0


def test_next_turn_atomic_no_tmp_leftover(tmp_path) -> None:
    """Atomic write (temp + os.replace), matching store.py — no .tmp left behind."""
    p = tmp_path / "turn_counter.json"
    tc.next_turn(p)
    assert p.exists()
    assert not p.with_name(p.name + ".tmp").exists()
