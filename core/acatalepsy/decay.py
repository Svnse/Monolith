"""Decay — the when-plane temporal weighting for ACU reinforcement.

Reinforcement is a monotonic assertion count: it only ever grows. Without decay,
a claim asserted 5x two years ago permanently outranks one asserted 2x yesterday.
This module supplies a pure compute-on-read decay — exactly the stance
``authority.compute_authority`` takes — so the stored ``reinforcement`` column
stays the raw record while *model-reach* (the weight used for ranking) fades with
time-since-last-touch. The record is preserved; only the model forgets.

Composition with the other temporal mechanisms:
  - ``authority`` keys off ``truth_checked_at`` (VERIFICATION recency) and gates
    the behavioural TIER (can this steer at all). Decay keys off
    ``last_touched_ts`` (ASSERTION recency) and weights RANKING within a tier.
    Different timestamps, different jobs → they compose, never double-count.
  - ``affect`` already decays its readings with an exponential half-life. The
    shared ``decay_factor`` below is that same curve, extracted so there is one
    half-life formula in the tree, not two.

Flag: MONOLITH_ACU_DECAY_V1 (default OFF — ships dark for first observation).
Callers gate on ``decay_enabled()``; this module's functions are always pure.

Known limitation (deliberate, observe-first): decay measures ASSERTION recency
(``last_touched_ts``), not USAGE recency. Recalling a claim does not refresh its
anchor — adding write-on-read would couple the read path to a mutation. If
observation shows useful-but-unasserted claims fading, that is the first knob to
revisit.

Clock: ``now`` defaults to ``datetime.now(timezone.utc)`` (matching authority).
The ``now=`` seam is kept so the when-plane TurnClock can thread a single
per-turn clock through here later for deterministic replay.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from core.acatalepsy.authority import compute_authority, AU_LOCKED


_FLAG_ENV = "MONOLITH_ACU_DECAY_V1"
_HALF_LIFE_ENV = "MONOLITH_ACU_DECAY_HALF_LIFE_DAYS"

# Self-derived claims (creative overhang) fade at the base rate; user/world
# facts are higher-trust and resist decay (longer half-life) but are NOT immune.
_DEFAULT_HALF_LIFE_DAYS = 45.0
_PROVENANCE_HALF_LIFE_MULT = {"user": 4.0, "world": 4.0, "self": 1.0}


def decay_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def base_half_life_days() -> float:
    try:
        return max(1e-3, float(os.environ.get(_HALF_LIFE_ENV, _DEFAULT_HALF_LIFE_DAYS)))
    except (TypeError, ValueError):
        return _DEFAULT_HALF_LIFE_DAYS


def decay_factor(age_days: float, half_life_days: float) -> float:
    """Exponential half-life factor in (0, 1].

    ``2 ** (-age / half_life)`` — identical curve to affect's
    ``exp(-ln2/hl * age)``. Negative age (a future-dated anchor from a backward
    clock step) clamps to 0 so the factor never exceeds 1.0 (never amplifies).
    """
    age = max(0.0, float(age_days))
    hl = max(1e-3, float(half_life_days))
    return 2.0 ** (-age / hl)


def _get(row, key, default=None):
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if v is None else v


def _parse_ts(value):
    try:
        ts = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def effective_reinforcement(row, *, now=None, half_life_days: float | None = None) -> float:
    """Reinforcement weighted by time-since-last-touch (model-reach, not record).

    Pure function of ``(row, now)``. Exemptions and fallbacks (all conservative —
    they return the RAW value, never a fabricated decayed one):
      - AU4 / locked contract rules: never decay (resolved via ``authority``).
      - Missing or unparseable anchor: return raw (cannot date it → don't fade it,
        mirroring authority._is_stale's absent-timestamp stance).
      - Future-dated anchor (backward clock): ``decay_factor`` clamps to 1.0.
    """
    raw = float(_get(row, "reinforcement", 0) or 0)

    # AU4 / locked rules are contract-level — exempt from forgetting.
    if compute_authority(row) == AU_LOCKED:
        return raw

    anchor = _get(row, "last_touched_ts") or _get(row, "last_seen") or _get(row, "created_at")
    ts = _parse_ts(anchor)
    if ts is None:
        return raw  # no datable anchor → conservative: do not decay

    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    age_days = (now_dt - ts).total_seconds() / 86400.0

    base = base_half_life_days() if half_life_days is None else float(half_life_days)
    provenance = str(_get(row, "provenance", "self") or "self").lower()
    hl = base * _PROVENANCE_HALF_LIFE_MULT.get(provenance, 1.0)

    return raw * decay_factor(age_days, hl)
