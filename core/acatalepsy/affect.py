"""Affect branch (B3) — valence/arousal readings about a person.

The dead-``veracity`` successor. Affect does NOT rank and has no truth value: it
accumulates into a **recency-weighted profile** (the live model is a query-view
over immutable readings — "angry then sad" is a trajectory, not a contradiction).
It is the one branch with a genuine continuous decay (fast: affect is volatile).
Provenance inverts here: **user-sourced is the gold standard** for someone's own
emotional state.
"""
from __future__ import annotations

from datetime import datetime, timezone

from core.acatalepsy.normalize import CanonicalTriple
from core.acatalepsy.decay import decay_factor

__all__ = ("extract_affect", "append_reading", "affect_profile")

# Affect verb -> (valence in -1..1, arousal in 0..1). A K1/K2 lexicon floor.
_AFFECT_MAP: dict[str, tuple[float, float]] = {
    "loves": (0.9, 0.8), "love": (0.9, 0.8),
    "likes": (0.6, 0.4), "like": (0.6, 0.4),
    "prefers": (0.5, 0.3), "prefer": (0.5, 0.3),
    "enjoys": (0.7, 0.5), "enjoy": (0.7, 0.5),
    "wants": (0.4, 0.5), "want": (0.4, 0.5),
    "happy": (0.8, 0.6),
    "hates": (-0.9, 0.85), "hate": (-0.9, 0.85),
    "dislikes": (-0.6, 0.5), "dislike": (-0.6, 0.5),
    "fears": (-0.7, 0.75), "fear": (-0.7, 0.75), "afraid": (-0.7, 0.75),
    "resents": (-0.7, 0.6),
    "frustrated": (-0.6, 0.7),
    "sad": (-0.6, 0.3),
    "angry": (-0.8, 0.9),
}

# Provenance weighting (inverted vs Truth): the person is the authority on their
# own state.
_SOURCE_WEIGHT = {"user": 1.0, "world": 0.5, "self": 0.3}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(ts: object) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except (TypeError, ValueError):
        return None


def extract_affect(triple: CanonicalTriple | None) -> tuple[float, float, float] | None:
    """Return (valence, arousal, intensity) for an affective claim, else None."""
    if triple is None:
        return None
    for tok in str(triple.relation or "").lower().replace("_", " ").split():
        if tok in _AFFECT_MAP:
            v, a = _AFFECT_MAP[tok]
            return (v, a, abs(v))
    return None


def append_reading(conn, *, subject, valence, arousal, intensity=None, target=None,
                   source=None, acu_id=None, ts=None) -> int:
    """Append one immutable affect reading. Returns the reading id."""
    ts = ts or _now_iso()
    intensity = abs(float(valence)) if intensity is None else float(intensity)
    cur = conn.execute(
        "INSERT INTO affect_readings(acu_id, subject, valence, arousal, intensity, "
        "target, source, ts) VALUES(?,?,?,?,?,?,?,?)",
        (acu_id, str(subject), float(valence), float(arousal), intensity,
         target, source, ts),
    )
    return int(cur.lastrowid)


def affect_profile(subject: str, conn, *, now=None, half_life_days: float = 7.0,
                   scan_limit: int = 2000) -> dict | None:
    """The live affect model for ``subject``: a recency-weighted (and provenance-
    weighted) view over the immutable readings. Returns None if no readings.

    Recent readings dominate (exponential decay, fast half-life — affect is
    volatile); old readings fade from the *model* without being deleted from the
    *record*.
    """
    # Bounded scan: old readings have ~0 weight anyway; cap cost as the
    # time-series grows under reinforcement loops. Order by the monotonic
    # autoincrement id (insertion order), NOT the wall-clock ts — a backward
    # clock step would otherwise let a stale-but-later-stamped reading evict the
    # genuinely-newest one from the scan window (when-plane clock fix).
    rows = conn.execute(
        "SELECT valence, arousal, intensity, source, ts FROM affect_readings "
        "WHERE subject=? ORDER BY id DESC LIMIT ?",
        (str(subject), int(scan_limit)),
    ).fetchall()
    if not rows:
        return None
    now_dt = _parse(now) or datetime.now(timezone.utc)
    sv = sa = si = sw = 0.0
    for r in rows:
        dt = _parse(r["ts"])
        age_days = max(0.0, (now_dt - dt).total_seconds() / 86400.0) if dt else 0.0
        # Unknown/None source -> untrusted floor (0.3), never above `self`.
        src_w = _SOURCE_WEIGHT.get((r["source"] or "self").lower(), 0.3)
        w = decay_factor(age_days, half_life_days) * src_w
        sv += w * float(r["valence"])
        sa += w * float(r["arousal"])
        si += w * float(r["intensity"])
        sw += w
    if sw <= 0:
        return None
    # `eff_n` (sum of weights) = effective sample size, so consumers can tell a
    # fresh reading from decayed noise; `n` is the raw count.
    return {"valence": sv / sw, "arousal": sa / sw, "intensity": si / sw,
            "n": len(rows), "eff_n": sw}
