"""Authority branch (B4) — behavioral reach, gated by Kind / Truth / Affect.

    AU1 stored-only  -> preserved for audit, cannot affect behavior
    AU2 recall-eligible -> may be retrieved as context
    AU3 behavior-shaping -> may influence routing / response framing / assembly
    AU4 locked-rule  -> contract-level authority

This is where recall finally ranks by branch-resolution — the real form of the
deferred A2 acceptance criterion — instead of the reinforcement stopgap.
Compute-on-read: a pure function of the ACU's resolved branches + provenance +
maturity. (A stored column + governance arbitration is Phase 6 / a downstream
seam; intrinsic-computed is the v1 stance.)
"""
from __future__ import annotations

from datetime import datetime, timezone

__all__ = ("compute_authority", "AU_STORED", "AU_RECALL", "AU_BEHAVIOR", "AU_LOCKED")

AU_STORED = 1
AU_RECALL = 2
AU_BEHAVIOR = 3
AU_LOCKED = 4

# A confirmed fact not re-checked within this window decays out of behavior-
# shaping reach (down to recall-eligible). The Temporal axis of the spec: truth
# is grounded at a point in time, and time erodes that grounding. We never bury
# it (it stays recall-eligible and is never auto-demoted to -inf) — staleness
# only revokes the right to silently steer behavior until re-confirmed.
_STALENESS_DAYS = 90.0


def _get(row, key, default=None):
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if v is None else v


def _is_stale(checked_at, now, staleness_days) -> bool:
    if not checked_at:
        return False   # never re-checked timestamp recorded -> treat as fresh
    try:
        ts = datetime.fromisoformat(str(checked_at))
    except (ValueError, TypeError):
        # A corrupt/unparseable timestamp cannot certify freshness. Treat it as
        # stale so the confirmed fact drops to recall-only until re-confirmed,
        # rather than silently steering behavior forever on an invalid date.
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() > float(staleness_days) * 86400.0


def compute_authority(row, *, now=None, staleness_days: float = _STALENESS_DAYS) -> int:
    """Resolve an ACU's Authority level (1-4) from its row (dict or sqlite3.Row).

    ``now``/``staleness_days`` parameterize the Temporal axis: a ``confirmed`` fact
    whose ``truth_checked_at`` is older than the window is treated as recall-only
    (not behavior-shaping) until re-confirmed.
    """
    if int(_get(row, "locked", 0) or 0):
        return AU_LOCKED
    if str(_get(row, "state", "active") or "active") != "active":
        return AU_STORED   # -inf falsehood / archived: stored for audit only

    truth = _get(row, "truth")
    if truth == "confirmed":
        if _is_stale(_get(row, "truth_checked_at"), now, staleness_days):
            return AU_RECALL   # confirmed but stale -> recall-eligible, not steering
        return AU_BEHAVIOR
    if truth == "contradicted":
        return AU_STORED
    if truth == "contested":
        return AU_RECALL

    # Not truth-checked (kept kinds: self/meta/emotional) or unverifiable:
    # maturity + provenance drive reach. Self stays on a leash (<= AU2).
    l_level = str(_get(row, "l_level", "L1") or "L1")
    provenance = str(_get(row, "provenance", "self") or "self").lower()
    reinforcement = int(_get(row, "reinforcement", 0) or 0)
    kind = str(_get(row, "kind", "") or "").strip().lower()
    if l_level == "L2":
        # SEAL: a kind=self claim (about Monolith itself) never reaches behavior-
        # shaping authority via the maturity path, even when user/world-sourced and
        # reinforced. Identity belongs to the identity projection channel, not to
        # AU3 routing/assembly. (Closes the latent bug where compute_authority never
        # read `kind`; self-identity-memory L2 ACUs must stay <= AU_RECALL.)
        if provenance in ("user", "world") and reinforcement >= 2 and kind != "self":
            return AU_BEHAVIOR
        return AU_RECALL
    return AU_STORED   # L1 stub (creative overhang) is not recall-grade
