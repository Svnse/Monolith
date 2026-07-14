"""Hybrid C outcome metric math primitives.

Wilson CI, Spearman ρ, polarity mapping. Stdlib-only.

Composite signal and Layer D polarity readers live further down (T4).
"""
from __future__ import annotations

import math
from typing import Sequence


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (center, low, high), each clamped to [0.0, 1.0].
    n=0 returns (0.0, 0.0, 0.0).
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)
    if confidence != 0.95:
        # Future: extend with other z-scores. For Phase 0 only 0.95 is required.
        raise NotImplementedError(f"confidence={confidence} not supported")
    z = 1.96
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (center, low, high)


def _rank(values: Sequence[float]) -> list[float]:
    """Compute average ranks with tie-handling. Stdlib-only."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation. Stdlib-only.

    Returns 0.0 on empty/single-element inputs and on zero variance.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    n = len(xs)
    if n < 2:
        return 0.0
    rx = _rank(xs)
    ry = _rank(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry)) / n
    var_x = sum((a - mean_x) ** 2 for a in rx) / n
    var_y = sum((b - mean_y) ** 2 for b in ry) / n
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)
