"""Identity emergence detector (M2 V0) — deterministic, no LLM.

Notices when high-confidentity self-derived claims have accumulated that the
operative identity does not yet reflect, and surfaces a read-only advisory
signal. It NEVER proposes or applies anything (propose-only): drafting an
amendment is the bidden ``identity_review`` skill's job; applying is E's.

Responsibilities:
  * score self-derived (non-locked) ACUs' confidentity against the current
    identity corpus and PERSIST the score onto those rows (Decision B: the
    identity machinery's own ``authorized_write`` — intake stays pristine, and
    E can see the weight in get_by_id / the monobase_dev triage UI);
  * gate on a watermark (only fire after enough new accrual) + a confidentity
    threshold;
  * on fire: update the milestone ledger's ``latest_emergence_signal`` (which
    the Observer surfaces, read-only) and emit ``identity_emergence_detected``.

M3 unify (behavior change): emergence is the STABLE disposition — a candidate
now also requires stability_score >= STABILITY_THRESHOLD. Fresh identity-aligned
claims (stability < threshold) are exclusively CURIOSITY (core/curiosity.py), not
consolidation candidates. On all-fresh data this detector correctly returns
fired=False (nothing stable to consolidate yet) while curiosity fires —
mutually exclusive by construction. This also closes a latent Mad-Cow gap (no
fresh self-claim can be proposed for consolidation).

Flag: MONOLITH_IDENTITY_EMERGENCE_V1 (default OFF — ships dark like
acu_retrieval; the auto-run heartbeat is a no-op until E enables it, so nothing
spends compute / writes the DB unbidden. The identity_review skill bypasses the
flag via force=True, so manual detect/draft always works).
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from core import identity_milestones as _milestones
from core import identity_regions as _regions
from core.identity_alignment import score_confidentity, stability_score, STABILITY_THRESHOLD

_FLAG_ENV = "MONOLITH_IDENTITY_EMERGENCE_V1"
# Calibrated against 68 REAL accumulated self-derived ACUs (2026-06-02): the
# meaningful identity-relevant cluster lands at 0.20–0.30 (real triples are
# multi-token, so the overlap-coefficient denominator is larger than the crafted
# fire-rate fixtures). At 0.30 only 1/68 surfaced; 0.20 surfaces ~10 genuinely
# identity-relevant claims. Self-provenance caps confidentity at 0.5, so 0.20 =
# alignment ≥0.40 (≈2 identity terms in a claim). The human gate filters noise.
_DEFAULT_THRESHOLD = 0.20
_DEFAULT_MIN_NEW = 8
_MAX_CANDIDATES = 200


@dataclass(frozen=True)
class EmergenceReport:
    fired: bool
    new_acu_count: int
    candidates: tuple[dict, ...]
    message: str


def _flag_enabled() -> bool:
    # Ships DARK (default OFF) like acu_retrieval — the auto-run heartbeat (the
    # record_outcome + bootstrap hooks) is a no-op until E enables it for first
    # observation. The explicitly-invoked identity_review skill bypasses this
    # via force=True, so manual detect/draft always works.
    raw = str(os.environ.get(_FLAG_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_database_locked(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "database is locked" in msg or "database table is locked" in msg


def detect_emergence_best_effort(
    *,
    retries: int = 2,
    backoff: float = 0.05,
    **kwargs,
) -> EmergenceReport:
    """Background-safe emergence detector.

    Explicit identity-review calls should use ``detect_emergence`` and fail
    loudly. Heartbeat hooks use this wrapper: if SQLite is briefly busy, retry;
    if it stays busy, skip this heartbeat instead of emitting scary trace noise.
    The next turn/rating/bootstrap can observe the same substrate again.
    """
    tries = max(0, int(retries)) + 1
    for attempt in range(tries):
        try:
            return detect_emergence(**kwargs)
        except sqlite3.OperationalError as exc:
            if not _is_database_locked(exc):
                raise
            if attempt < tries - 1:
                time.sleep(max(0.0, float(backoff)) * (attempt + 1))
                continue
            return EmergenceReport(False, 0, (), "database locked; skipped")

    return EmergenceReport(False, 0, (), "database locked; skipped")


def compute_identity_corpus() -> str:
    """Current identity prose (Origin-0 + accepted Emergent) as one string."""
    from core import identity
    origin0, emergent = _regions.split_regions(identity.load_identity())
    return (origin0 + "\n" + emergent).strip()


def _read_self_derived(limit: int = _MAX_CANDIDATES) -> list[dict]:
    from core.acu_store import ACUStore
    s = ACUStore()
    try:
        rows = s.retrieve(limit=limit)
    finally:
        s.close()
    return [r for r in rows if not int(r.get("locked", 0) or 0)]


def score_and_persist(rows: list[dict], corpus: str, *, backend: str | None = None,
                      persist: bool = True) -> list[dict]:
    """Score each row's confidentity vs the corpus; optionally persist onto
    self-derived (non-locked) rows via the identity machinery's own sentinel."""
    scored: list[dict] = []
    for r in rows:
        c = score_confidentity(r, corpus, backend=backend)
        r2 = dict(r)
        r2["confidentity"] = c
        scored.append(r2)
    if persist and scored:
        from core.db_connect import connect_acatalepsy, authorized_write
        conn = connect_acatalepsy(role="memory_writer")
        try:
            now = _now()
            with authorized_write("identity_scoring:emergence"):
                for r in scored:
                    conn.execute(
                        "UPDATE acus SET confidentity=?, last_touched_ts=? "
                        "WHERE id=? AND locked=0",
                        (float(r["confidentity"]), now, int(r["id"])),
                    )
                conn.commit()
        finally:
            conn.close()
    return scored


def identity_candidates(
    *,
    threshold_confidentity: float = _DEFAULT_THRESHOLD,
    backend: str | None = None,
    persist: bool = True,
    limit: int = _MAX_CANDIDATES,
    corpus: str | None = None,
) -> tuple[dict, ...]:
    """Return stable, novel, high-confidentity ACUs.

    This is the candidate-selection half of ``detect_emergence``. Runtime
    projection callers can pass ``persist=False`` to score in memory without
    touching the substrate or milestone ledger.
    """
    identity_corpus = compute_identity_corpus() if corpus is None else str(corpus or "").strip()
    scored = score_and_persist(
        _read_self_derived(limit),
        identity_corpus,
        backend=backend,
        persist=persist,
    )

    candidates: list[dict] = []
    for r in scored:
        if r["confidentity"] < threshold_confidentity:
            continue
        stability = stability_score(r)
        if stability < STABILITY_THRESHOLD:
            continue
        canon = str(r.get("canonical", "")).strip()
        if not canon or canon in identity_corpus:
            continue
        candidates.append({
            "id": r.get("id"),
            "canonical": canon,
            "confidentity": round(float(r["confidentity"]), 4),
            "stability": round(float(stability), 4),
            "provenance": r.get("provenance"),
            "reinforcement": int(r.get("reinforcement", 1) or 1),
            "l_level": str(r.get("l_level") or "L1"),
        })
    candidates.sort(key=lambda c: (c["confidentity"], c["reinforcement"]), reverse=True)
    return tuple(candidates)


def detect_emergence(
    *,
    threshold_confidentity: float = _DEFAULT_THRESHOLD,
    min_new_acus: int = _DEFAULT_MIN_NEW,
    backend: str | None = None,
    persist: bool = True,
    force: bool = False,
) -> EmergenceReport:
    """Cheap deterministic emergence check. Fires only when enough new ACUs have
    accrued AND ≥1 self-derived claim crosses the confidentity threshold.

    ``force=True`` bypasses the auto-detection flag — used by the explicitly
    invoked identity_review skill (the flag gates only background detection).
    """
    if not force and not _flag_enabled():
        return EmergenceReport(False, 0, (), "disabled")

    from core.acu_store import ACUStore
    s = ACUStore()
    try:
        count = s.count()
    finally:
        s.close()
    watermark = _milestones.get_watermark()
    new = max(0, count - watermark)

    corpus = compute_identity_corpus()
    scored = score_and_persist(_read_self_derived(), corpus, backend=backend, persist=persist)

    candidates: list[dict] = []
    for r in scored:
        if r["confidentity"] < threshold_confidentity:
            continue
        # M3 unify: emergence is the STABLE disposition — only consolidate claims
        # that have earned reinforcement/crystallization. Fresh identity-aligned
        # claims are CURIOSITY (core/curiosity.py), not consolidation candidates.
        # This is also the Mad-Cow gate (no fresh self-claim → identity).
        if stability_score(r) < STABILITY_THRESHOLD:
            continue
        canon = str(r.get("canonical", "")).strip()
        # NOTE (V0 placeholder): `canon in corpus` is a near-no-op — a triple is
        # almost never a verbatim substring of the prose corpus, so this barely
        # dedups. Acceptable for M2 (the watermark + the human gate prevent spam),
        # but M3 defines a "pull" as NOVEL × aligned and NOVEL *is* this predicate
        # — M3 must replace this with real dedup (against accepted Emergent claims
        # / a seen-set), not lean on this check.
        if not canon or canon in corpus:
            continue
        candidates.append({
            "id": r.get("id"),
            "canonical": canon,
            "confidentity": round(float(r["confidentity"]), 4),
            "provenance": r.get("provenance"),
            "reinforcement": int(r.get("reinforcement", 1) or 1),
        })
    # confidentity first, reinforcement as tiebreaker.
    candidates.sort(key=lambda c: (c["confidentity"], c["reinforcement"]), reverse=True)

    fired = new >= min_new_acus and len(candidates) > 0
    message = (
        f"{len(candidates)} self-derived claim(s) at or above {threshold_confidentity} "
        f"confidentity; {new} new ACU(s) since last milestone check."
    )
    report = EmergenceReport(fired, new, tuple(candidates), message)

    if fired:
        try:
            from core.identity_projection import sync_identity_file_from_acus
            sync_identity_file_from_acus(threshold_confidentity=threshold_confidentity)
        except Exception:
            pass
        signal = {
            "detected_at": _now(),
            "new_acu_count": new,
            "candidate_count": len(candidates),
            "threshold": threshold_confidentity,
            "top": candidates[:5],
            "message": message,
        }
        _milestones.set_latest_emergence_signal(signal)
        _milestones.set_watermark(count)
        try:
            from core.acatalepsy import canonical_log
            canonical_log.append(
                "identity_emergence_detected",
                payload={
                    "new_acu_count": new,
                    "candidate_count": len(candidates),
                    "threshold": threshold_confidentity,
                },
            )
        except Exception:
            pass  # detection must never break the caller (record_outcome / bootstrap)

    return report
