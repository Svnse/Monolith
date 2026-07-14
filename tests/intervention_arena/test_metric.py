"""Tests for core.intervention_arena.metric math primitives."""
from __future__ import annotations

import math

from core.intervention_arena.metric import (
    spearman_rho,
    wilson_ci,
)


def test_wilson_ci_zero_n_returns_zeros():
    center, low, high = wilson_ci(0, 0)
    assert center == 0.0
    assert low == 0.0
    assert high == 0.0


def test_wilson_ci_half_at_n_100():
    center, low, high = wilson_ci(50, 100)
    assert abs(center - 0.5) < 0.01
    assert low < center < high
    assert 0.39 < low < 0.41  # approx 0.40
    assert 0.59 < high < 0.61  # approx 0.60


def test_wilson_ci_extreme_high():
    center, low, high = wilson_ci(95, 100)
    assert center > 0.9
    assert low > 0.87
    assert high < 1.0  # never crosses 1


def test_wilson_ci_extreme_low():
    center, low, high = wilson_ci(5, 100)
    assert center < 0.1
    assert low >= 0.0  # never crosses 0
    assert high < 0.13


def test_wilson_ci_clamps_to_unit_interval():
    _, low, high = wilson_ci(100, 100)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    _, low, high = wilson_ci(0, 100)
    assert 0.0 <= low <= 1.0


def test_spearman_perfect_positive():
    rho = spearman_rho([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert abs(rho - 1.0) < 1e-9


def test_spearman_perfect_negative():
    rho = spearman_rho([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert abs(rho - (-1.0)) < 1e-9


def test_spearman_zero_when_independent():
    # Carefully constructed non-monotonic pairing.
    rho = spearman_rho([1, 2, 3, 4], [3, 1, 4, 2])
    assert abs(rho) < 0.5


def test_spearman_handles_ties():
    rho = spearman_rho([1, 1, 2, 2], [1, 2, 1, 2])
    assert math.isfinite(rho)
    # With tied ranks the relationship is roughly null
    assert abs(rho) < 0.5


def test_spearman_returns_zero_on_tiny_input():
    assert spearman_rho([], []) == 0.0
    assert spearman_rho([1], [1]) == 0.0


def test_spearman_returns_zero_on_zero_variance():
    rho = spearman_rho([1, 1, 1, 1], [1, 2, 3, 4])
    assert rho == 0.0
