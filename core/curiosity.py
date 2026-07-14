"""Curiosity detector (M3 V0) — the fresh disposition of the identity signal.

One identity signal, two dispositions split by stability (see
core/identity_alignment.stability_score):

  identity-aligned + STABLE      -> emergence  (consolidate into identity, M2)
  identity-aligned + NOT stable  -> CURIOSITY  (a fresh pull to explore, here)

So a "pull" = a fresh, identity-aligned, not-yet-integrated claim — something the
system is drawn toward *because it relates to who it is*, but hasn't settled into
identity yet. Deterministic, no LLM. Propose-only: it forms and ranks what it is
curious about (its own judgment) and surfaces it; it NEVER acts, promotes, or
spends compute pursuing (pursuit is the planner's job, M1).

Retirement (advisor): a pull resurfaces at most ``surface_cap`` times (tracked in
the milestone ledger's seen-set) so the heartbeat doesn't loop the same items
forever. A claim that reinforces into stability also leaves curiosity (it becomes
an emergence candidate instead).

Flag: MONOLITH_CURIOSITY_V1 (default OFF — ships dark like the emergence
heartbeat; the curiosity skill bypasses via force=True for manual inspection).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from core import identity_emergence as _em
from core import identity_milestones as _milestones
from core.identity_alignment import score_confidentity, stability_score, STABILITY_THRESHOLD

_FLAG_ENV = "MONOLITH_CURIOSITY_V1"
_DEFAULT_ALIGN = 0.20      # identity-aligned bar (mirrors emergence's threshold)
_SURFACE_CAP = 3           # retire a pull after it has resurfaced this many times
_LIMIT = 10


@dataclass(frozen=True)
class CuriosityReport:
    fired: bool
    pulls: tuple[dict, ...]
    message: str


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_pulls(
    *,
    align_threshold: float = _DEFAULT_ALIGN,
    limit: int = _LIMIT,
    backend: str | None = None,
    force: bool = False,
    surface_cap: int = _SURFACE_CAP,
) -> CuriosityReport:
    """Surface fresh, identity-aligned claims as ranked curiosity pulls.

    ``force=True`` bypasses the dark flag (used by the curiosity skill)."""
    if not force and not _flag_enabled():
        return CuriosityReport(False, (), "disabled")

    corpus = _em.compute_identity_corpus()
    rows = _em._read_self_derived()
    surfaced = _milestones.get_curiosity_surfaced()
    killed = _milestones.get_curiosity_killed()  # M3.1: retired pulls stay retired

    pulls: list[dict] = []
    for r in rows:
        conf = score_confidentity(r, corpus, backend=backend)
        if conf < align_threshold:
            continue
        stab = stability_score(r)
        if stab >= STABILITY_THRESHOLD:   # stable -> emergence's job, not curiosity
            continue
        key = str(r.get("canonical", "")).strip()
        if not key or key in corpus:   # empty, or already reflected in identity
            continue
        if key in killed:   # M3.1: retired as noise by the kill-actuator
            continue
        if surfaced.get(key, 0) >= surface_cap:   # retired from the surfaced list
            continue
        pulls.append({
            "id": r.get("id"),
            "canonical": key,
            "pull_strength": round(conf * (1.0 - stab), 4),  # fresher (lower stability) pulls harder
            "confidentity": round(conf, 4),
            "stability": round(stab, 4),
            "provenance": r.get("provenance"),
        })

    pulls.sort(key=lambda p: p["pull_strength"], reverse=True)
    pulls = pulls[:limit]
    fired = len(pulls) > 0
    message = (
        f"{len(pulls)} curiosity pull(s) — fresh, identity-aligned claims drawing "
        f"exploration (not yet integrated into identity)."
    )
    report = CuriosityReport(fired, tuple(pulls), message)

    if fired:
        _milestones.set_latest_curiosity_signal({
            "detected_at": _now(),
            "pull_count": len(pulls),
            "top": pulls[:5],
            "message": message,
        })
        _milestones.bump_curiosity_surfaced([p["canonical"] for p in pulls])
        try:
            from core.acatalepsy import canonical_log
            canonical_log.append("curiosity_pull_detected", payload={"pull_count": len(pulls)})
        except Exception:
            pass  # detection must never break the caller
    else:
        # A real detection (we got past the dark-flag early-return) that found
        # nothing means any stored signal is now stale — clear it so the
        # Observer can't surface a ghost that contradicts the live tool.
        # A transient empty (e.g. a momentary read miss) also clears; that is
        # intentional — the signal simply re-fires on the next detection.
        if _milestones.get_latest_curiosity_signal() is not None:
            _milestones.set_latest_curiosity_signal(None)

    return report


def latest_surfaceable_signal(*, surface_cap: int = _SURFACE_CAP) -> dict | None:
    """The stored curiosity signal, but only if it is still *live*.

    A signal is live if at least one of its pulls is still surfaceable — not
    retired (``surfaced < surface_cap``) and not killed. A signal whose pulls
    have all retired/been killed is a ghost: the live tool returns 0 while the
    Observer would still echo the old count (the 2026-06-03 contradiction).
    This is the read-site guard the Observer uses; it is a PURE read (no
    mutation), so it respects the Observer's no-authority contract. Ledger
    hygiene (actually clearing) is the write-site's job in ``detect_pulls``.
    """
    sig = _milestones.get_latest_curiosity_signal()
    if not isinstance(sig, dict):
        return None
    top = sig.get("top") or []
    if not top:
        return sig   # no per-pull detail to assess -> can't prove stale; surface it
    killed = _milestones.get_curiosity_killed()
    surfaced = _milestones.get_curiosity_surfaced()
    for p in top:
        key = str((p or {}).get("canonical", "")).strip()
        if not key or key in killed:
            continue
        if surfaced.get(key, 0) >= surface_cap:
            continue
        return sig   # at least one pull still surfaceable -> the signal is current
    return None      # every pull retired/killed -> ghost; do not surface


def kill_pull(canonical: str, reason: str = "") -> bool:
    """Retire a pull as noise — the SAFE half of closing the curiosity loop.

    Records the kill to the milestone ledger (so detect_pulls excludes it) and
    emits a canonical_log ``curiosity_pull_killed`` event so the action is
    auditable and reversible (via identity_milestones.unkill_curiosity_pull).
    KILL only — promotion-into-identity stays human-gated. Returns True on record.
    """
    key = str(canonical or "").strip()
    if not key:
        return False
    _milestones.kill_curiosity_pull(key, reason)
    try:
        from core.acatalepsy import canonical_log
        canonical_log.append(
            "curiosity_pull_killed",
            payload={"canonical": key, "reason": str(reason or "")},
        )
    except Exception:
        pass
    return True
