"""CID assignment + crystallization (L1 hot -> L2 cold) + the Mad Cow gate.

Crystallization is the substrate's phase boundary: a hot L1 stub (generative,
lineage-tracked, mutable) cools into a content-addressed L2 ACU whose identity
(CID) is frozen. CID = hash(canonical_form + cf_version) ONLY — provenance,
l_level, and temporal fields are coordinates, never identity, so the same
claim from different sources mints the same CID and cannot fork.

Mad Cow rule: a ``self``-provenance claim (model-generated) may not crystallize
unless at least one evidence span carries ``user`` or ``world`` provenance.
Self cannot promote self. Reinforcement *count* is never a crystallization
trigger — only span provenance is.

This module does NOT log to canonical_log or open its own connection: callers
(intake/backfill) pass a connection and own the WAL emission + transaction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from core.acatalepsy.normalize import CF_VERSION, normalize_canonical

__all__ = (
    "CrystallizeError",
    "MadCowError",
    "CrystallizeResult",
    "compute_cid",
    "can_crystallize",
    "maybe_promote_l3",
    "crystallize",
)

# Provenance values that confer truth-standing for promotion.
_EXTERNAL = frozenset({"user", "world"})

# Self-identity-memory expansion (flag-gated, default OFF). A kind=self claim may
# crystallize to L2 ("identity memory") when it has recurred across this many
# DISTINCT source_events — occasions, not a raw reinforcement count (which is
# gameable by verbatim repetition). This NEVER grants world-truth/authority/recall/
# citation/salience reach — those are sealed independently (see the design spec
# 2026-06-26-self-acu-l2-identity-memory). L2->L3 stays external-only (unchanged).
_SELF_L2_FLAG_ENV = "MONOLITH_SELF_IDENTITY_L2_V1"
_SELF_L2_MIN_DISTINCT_EVENTS = 3


def _self_l2_enabled() -> bool:
    import os
    raw = str(os.environ.get(_SELF_L2_FLAG_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _distinct_source_events(spans: list[dict]) -> int:
    """Count distinct, non-null source_event ids across evidence spans (the
    per-occasion recurrence signal). Legacy spans without source_event contribute
    nothing — so the gate fails CLOSED on un-attributed self-claims."""
    return len({s.get("source_event") for s in spans if s.get("source_event") is not None})


class CrystallizeError(RuntimeError):
    """Raised when an ACU cannot be crystallized (missing, wrong state)."""


class MadCowError(CrystallizeError):
    """Raised when a self-only claim is asked to crystallize (self cannot
    promote self — needs user/world confirmation)."""


@dataclass(frozen=True)
class CrystallizeResult:
    cid: str
    crystallized: bool          # True iff this row went L1 -> L2
    collided_with: int | None   # existing acu id if the CID already exists


def compute_cid(canonical_form: str, cf_version: int = CF_VERSION) -> str:
    """Content hash over the normalized form + cf_version ONLY.

    The 0x1f unit-separator prevents ambiguity between the version prefix
    and the form. Identity is independent of provenance/l_level/temporal.
    """
    digest = hashlib.sha256(
        f"{int(cf_version)}\x1f{canonical_form}".encode("utf-8")
    ).hexdigest()
    return f"cid:sha256:{digest}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_spans(raw: object) -> list[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [s for s in data if isinstance(s, dict)] if isinstance(data, list) else []


def _merge_spans(a_raw: object, b_raw: object) -> list[dict]:
    """Union of two evidence-span arrays, deduped by (text, provenance, source_event).

    The source_event component mirrors intake._append_span so distinct OCCASIONS of
    the same claim survive a CID-collision merge (preserving the distinct-source-event
    count self->L2 reads). None-safe: legacy spans without source_event dedup exactly
    as the prior (text, provenance) key did.
    """
    out = list(_load_spans(a_raw))
    seen = {(s.get("text"), s.get("provenance"), s.get("source_event")) for s in out}
    for s in _load_spans(b_raw):
        key = (s.get("text"), s.get("provenance"), s.get("source_event"))
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out


def can_crystallize(acu_id: int, conn) -> bool:
    """Mad Cow gate: True iff this ACU may be promoted L1 -> L2.

    user/world provenance qualifies directly. A self claim qualifies only if
    at least one evidence span is user/world sourced — UNLESS the self-identity-
    memory expansion is enabled, in which case a kind=self claim that has recurred
    across >= _SELF_L2_MIN_DISTINCT_EVENTS distinct source_events also qualifies.
    Raw reinforcement count is NEVER a trigger (it is gameable by repetition);
    distinct source_event recurrence is.
    """
    row = conn.execute(
        "SELECT provenance, kind, state, evidence_spans FROM acus WHERE id=?",
        (int(acu_id),),
    ).fetchone()
    if row is None:
        return False
    # Existing external Mad Cow fast-path (UNCHANGED): provenance or an evidence
    # span carries user/world standing.
    if str(row["provenance"] or "").strip().lower() in _EXTERNAL:
        return True
    spans = _load_spans(row["evidence_spans"])
    for span in spans:
        if str(span.get("provenance", "")).strip().lower() in _EXTERNAL:
            return True
    # Self-identity-memory path (NEW, flag-gated): a self-ABOUT claim that has been
    # observed on enough distinct occasions becomes L2 identity memory. Scoped to
    # kind=self + active so a world-fact never reaches L2 by self-repetition.
    if (
        _self_l2_enabled()
        and str(row["kind"] or "").strip().lower() == "self"
        and (row["state"] or "active") == "active"
        and _distinct_source_events(spans) >= _SELF_L2_MIN_DISTINCT_EVENTS
    ):
        return True
    return False


_L3_CONFIDENCE = 0.7
_L3_REINFORCEMENT = 2


def maybe_promote_l3(acu_id: int, conn) -> bool:
    """Promote a confirmed, reinforced L2 ACU to L3 (TRUSTED / retrieval-grade) —
    the load-bearing layer. Pressure-testing passed: externally `confirmed` at
    high confidence AND reinforced (seen more than once). CID is unchanged (trust
    is earned, identity is not re-minted). Returns True iff it promoted.

    L3 is a re-derivable maturity overlay (re-run from truth + reinforcement), so
    no separate WAL event is required.
    """
    row = conn.execute(
        "SELECT l_level, truth, truth_confidence, reinforcement, state "
        "FROM acus WHERE id=?",
        (int(acu_id),),
    ).fetchone()
    if row is None or (row["l_level"] or "") != "L2":
        return False
    if (row["state"] or "active") != "active":
        return False  # an archived/merged loser or -inf falsehood never earns trust
    if row["truth"] != "confirmed" or float(row["truth_confidence"] or 0) < _L3_CONFIDENCE:
        return False
    if int(row["reinforcement"] or 0) < _L3_REINFORCEMENT:
        return False
    conn.execute(
        "UPDATE acus SET l_level='L3', promoted_to_l3_ts=?, last_touched_ts=? WHERE id=?",
        (_now_iso(), _now_iso(), int(acu_id)),
    )
    return True


def is_crystallize_eligible(acu_id: int, conn) -> bool:
    """Whether an L1 stub is ready to cool into L2: still L1 + no CID yet + its
    Kind is resolved (the gate) + it passes Mad Cow. This is the crystallization
    trigger's predicate."""
    row = conn.execute(
        "SELECT canonical, l_level, cid, kind, state FROM acus WHERE id=?", (int(acu_id),)
    ).fetchone()
    if row is None or row["cid"] is not None:
        return False
    if (row["l_level"] or "L1") != "L1":
        return False
    if (row["state"] or "active") != "active":
        return False  # a known falsehood (-inf) / archived row never crystallizes
    if not str(row["kind"] or "").strip():
        return False
    # Defense: never crystallize into a CID another row already holds — that would
    # archive this row mid-intake. Such a row is a duplicate MATCH should have
    # caught; leave it L1 rather than silently archive it here.
    cid = compute_cid(normalize_canonical(row["canonical"]))
    if conn.execute(
        "SELECT 1 FROM acus WHERE cid=? AND id<>?", (cid, int(acu_id))
    ).fetchone():
        return False
    return can_crystallize(acu_id, conn)


def crystallize(acu_id: int, *, conn) -> CrystallizeResult:
    """Promote an L1 stub to a cold, content-addressed L2 ACU.

    Enforces the Mad Cow gate. If the computed CID already belongs to another
    ACU, this row does NOT fork — it is archived and pointed at the existing
    holder (``merged_into``), and the caller is told via ``collided_with`` so it
    can route reinforcement onto the survivor.

    The caller owns the transaction and any canonical_log emission.
    """
    row = conn.execute(
        "SELECT canonical, l_level, cid FROM acus WHERE id=?", (int(acu_id),)
    ).fetchone()
    if row is None:
        raise CrystallizeError(f"acu {acu_id} not found")
    if row["cid"] is not None:
        # Already cold — idempotent.
        return CrystallizeResult(cid=str(row["cid"]), crystallized=False, collided_with=None)
    if (row["l_level"] or "L1") != "L1":
        raise CrystallizeError(
            f"acu {acu_id} is {row['l_level']!r}, not L1 — cannot crystallize"
        )
    if not can_crystallize(acu_id, conn):
        raise MadCowError(
            f"acu {acu_id} is self-only; needs user/world confirmation to crystallize"
        )

    cf = normalize_canonical(row["canonical"])
    cid = compute_cid(cf, CF_VERSION)

    existing = conn.execute(
        "SELECT id FROM acus WHERE cid=? AND id<>?", (cid, int(acu_id))
    ).fetchone()
    if existing is not None:
        survivor = int(existing["id"])
        # Transfer the loser's reinforcement + evidence into the survivor so no
        # signal is lost when two phrasings collide on one CID.
        loser = conn.execute(
            "SELECT reinforcement, evidence_spans FROM acus WHERE id=?", (int(acu_id),)
        ).fetchone()
        surv = conn.execute(
            "SELECT reinforcement, evidence_spans FROM acus WHERE id=?", (survivor,)
        ).fetchone()
        merged_reinf = int(surv["reinforcement"] or 0) + int(loser["reinforcement"] or 0)
        merged_spans = _merge_spans(surv["evidence_spans"], loser["evidence_spans"])
        conn.execute(
            "UPDATE acus SET reinforcement=?, evidence_spans=?, last_touched_ts=? WHERE id=?",
            (merged_reinf, json.dumps(merged_spans), _now_iso(), survivor),
        )
        conn.execute(
            "UPDATE acus SET merged_into=?, state='archived', last_touched_ts=? WHERE id=?",
            (survivor, _now_iso(), int(acu_id)),
        )
        return CrystallizeResult(cid=cid, crystallized=False, collided_with=survivor)

    from core.acatalepsy import eqid as _eqid
    eqid_val = _eqid.compute_eqid_for_form(cf)   # one entry point; normalize is idempotent
    conn.execute(
        "UPDATE acus SET cid=?, cf_version=?, eqid=?, l_level='L2', promoted_to_l2_ts=?, "
        "last_touched_ts=? WHERE id=?",
        (cid, CF_VERSION, eqid_val, _now_iso(), _now_iso(), int(acu_id)),
    )
    return CrystallizeResult(cid=cid, crystallized=True, collided_with=None)
