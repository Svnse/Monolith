"""Tests for core/acatalepsy/decay.py — the when-plane decay primitive.

Decay weights reinforcement by time-since-last-touch so un-reinforced old claims
lose ranking weight WITHOUT being deleted (record stays; model-reach fades).
Pure compute-on-read, mirroring authority.compute_authority. Reuses one
half-life formula shared with affect.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from core.acatalepsy import decay


NOW = datetime(2026, 6, 2, tzinfo=timezone.utc)


def _iso_days_ago(d: float) -> str:
    return (NOW - timedelta(days=d)).isoformat()


# ── decay_factor (the shared half-life curve) ─────────────────────────


def test_decay_factor_at_zero_age_is_one():
    assert decay.decay_factor(0.0, 45.0) == 1.0


def test_decay_factor_at_one_half_life_is_half():
    assert decay.decay_factor(45.0, 45.0) == pytest.approx(0.5)


def test_decay_factor_at_two_half_lives_is_quarter():
    assert decay.decay_factor(90.0, 45.0) == pytest.approx(0.25)


def test_decay_factor_negative_age_clamped_to_one():
    # A future-dated anchor (backward clock) must not amplify weight above raw.
    assert decay.decay_factor(-10.0, 45.0) == 1.0


# ── effective_reinforcement ───────────────────────────────────────────


def test_effective_equals_raw_at_zero_age():
    row = {"reinforcement": 8, "last_touched_ts": NOW.isoformat(), "provenance": "self"}
    assert decay.effective_reinforcement(row, now=NOW) == pytest.approx(8.0)


def test_effective_halves_after_one_half_life_for_self():
    row = {"reinforcement": 8, "last_touched_ts": _iso_days_ago(45), "provenance": "self"}
    assert decay.effective_reinforcement(row, now=NOW) == pytest.approx(4.0, abs=0.05)


def test_user_provenance_decays_slower_than_self():
    self_row = {"reinforcement": 8, "last_touched_ts": _iso_days_ago(45), "provenance": "self"}
    user_row = {"reinforcement": 8, "last_touched_ts": _iso_days_ago(45), "provenance": "user"}
    assert (decay.effective_reinforcement(user_row, now=NOW)
            > decay.effective_reinforcement(self_row, now=NOW))


def test_locked_acu_is_exempt_from_decay():
    row = {"reinforcement": 8, "last_touched_ts": _iso_days_ago(1000),
           "provenance": "self", "locked": 1}
    assert decay.effective_reinforcement(row, now=NOW) == pytest.approx(8.0)


def test_missing_anchor_returns_raw():
    row = {"reinforcement": 8, "provenance": "self"}  # no last_touched_ts/last_seen
    assert decay.effective_reinforcement(row, now=NOW) == pytest.approx(8.0)


def test_corrupt_anchor_returns_raw():
    row = {"reinforcement": 8, "last_touched_ts": "not-a-date", "provenance": "self"}
    assert decay.effective_reinforcement(row, now=NOW) == pytest.approx(8.0)


def test_effective_never_exceeds_raw():
    # Even a future-dated anchor (backward clock) caps at raw, never amplifies.
    row = {"reinforcement": 5, "last_touched_ts": (NOW + timedelta(days=10)).isoformat(),
           "provenance": "self"}
    assert decay.effective_reinforcement(row, now=NOW) <= 5.0
