"""friction_organ — the predict + crystallize beats of the Friction Organ.

Predict (involuntary): each turn, mechanically extract ONE intent prediction
(claim + falsifier) from the just-produced answer, reusing the surviving
turn_classifier signal as a floor — NOT a model-authored <predict> envelope.
Designed to be called from a TurnReadyEvent policy or a finalize hook.

Crystallize (slow clock): only a friction PATTERN that recurs on a recurring
referent writes an interaction belief via intake.ingest_l1 — never an
accumulating "I know the user" certainty, never a one-off.

All writes go through friction_store, which is flag-gated (MONOLITH_FRICTION_V1).
"""
from __future__ import annotations

from typing import Any

from core import friction_store as _fs


def extract(public_answer: str, last_user_msg: str, *, turn_classifier_signal: Any = None) -> dict:
    """Floor extractor: predict the single thing the user wants + a falsifier.

    Reuses turn_classifier.classify (the surviving CompiledIntent classification
    half) for task_type/tags/confidence. Coarse BY DESIGN — the differ, not the
    prediction, is the load-bearing piece. Pure-ish (reads classifier only).
    `confidence` is disposable predict-time confidence; never accumulated."""
    task_type = "conversation"
    tags: tuple[str, ...] = ()
    confidence = 0.4
    shape = turn_classifier_signal
    if shape is None:
        try:
            from core import turn_classifier
            shape = turn_classifier.classify([{"role": "user", "content": last_user_msg or ""}], {})
        except Exception:
            shape = None
    if shape is not None:
        task_type = getattr(shape, "task_type", task_type) or task_type
        tags = tuple(getattr(shape, "intent_tags", ()) or ())
        try:
            confidence = float(getattr(shape, "confidence", confidence))
        except (TypeError, ValueError):
            pass

    tag_str = f" ({', '.join(tags)})" if tags else ""
    claim = f"wants a {task_type} response{tag_str}"
    falsifier = "next message corrects the framing, reframes the ask, or pivots to a different topic"
    return {"claim": claim, "falsifier": falsifier, "confidence": round(confidence, 2),
            "task_type": task_type, "intent_tags": list(tags)}


def on_turn_ready(public_answer: str, last_user_msg: str, turn_id: str, turn_n: int,
                  now_iso: str, *, use_card: bool = True, turn_classifier_signal: Any = None) -> int:
    """Predict beat (Path B): freeze ONE open intent prediction whose
    `prediction_set_json` is the code-checkable commitment the settler grades by
    membership. No-op (-1) when flag OFF.

    The frozen set = floor (pure-code mine_staked, always-on) merged with the
    optional Monoline card enrichment (None on any card failure → floor stands
    alone). `abandon_open()` first so at most one prediction is ever open (a
    tool-loop followup supersedes the initial generation → no orphan rows).

    v1 records intent only. `trajectory` (multi_turn) is DEFERRED to v1.1: a real
    trajectory settler keys off the outer-turn boundary over a window — not built."""
    from core import intent_extract
    floor = intent_extract.mine_staked(public_answer, last_user_msg)
    card = None
    if use_card:
        try:
            from core import intent_card
            card = intent_card.read_intent(public_answer, last_user_msg)
        except Exception:
            card = None
    pset = intent_extract.merge_prediction_sets(floor, card)

    base = extract(public_answer, last_user_msg, turn_classifier_signal=turn_classifier_signal)
    claim = (str(pset.get("intent_read") or "").strip()) or base["claim"]
    falsifier = ("next message's salient focus falls outside the staked set "
                 "(on-topic redirect), or corrects/reframes the framing")

    _fs.abandon_open()
    return _fs.record_prediction(
        turn_id, turn_n, "intent", claim, falsifier,
        base["confidence"], "next_turn", now_iso=now_iso, prediction_set_json=pset,
    )


# ── crystallize ──────────────────────────────────────────────────────

_RECURRENCE_MIN = 3   # a friction pattern must recur >= this to crystallize


def should_crystallize(rows: list[dict], friction_type: str, min_count: int = _RECURRENCE_MIN) -> bool:
    """Pure recurrence gate: does `friction_type` recur >= min_count among rows?
    Calm types never crystallize."""
    if friction_type in ("uptake", "unresolved"):
        return False
    count = sum(1 for r in rows if str(r.get("friction_type") or "") == friction_type)
    return count >= min_count


_CRYSTALLIZE_FLAG = "MONOLITH_FRICTION_CRYSTALLIZE_V1"
_TRUTHY = {"1", "true", "yes", "on"}


def _crystallize_enabled() -> bool:
    """Crystallization gates on its OWN flag — NOT the artifact flag — so an
    unvalidated settler never writes ACUs during the observe phase (spec §16)."""
    import os
    return os.environ.get(_CRYSTALLIZE_FLAG, "0").strip().lower() in _TRUTHY


def maybe_crystallize(friction_type: str, referent: str = "interaction", *,
                      db_path: Any = None, min_count: int = _RECURRENCE_MIN) -> bool:
    """If `friction_type` recurs on the referent, crystallize an interaction
    belief via intake.ingest_l1. Returns True iff it crystallized. Cap: recurring
    patterns only — a one-off never writes an ACU. Gated on the dedicated
    crystallize flag (observe phase is read-only on the substrate)."""
    if not _crystallize_enabled():
        return False
    rows = _fs.recent_settled(limit=24, db_path=db_path)
    if not should_crystallize(rows, friction_type, min_count):
        return False
    canonical = f"interaction pattern: '{friction_type}' recurs on {referent} (friction signal)"
    # Per-occasion id = the most recent settled row of this friction_type. Distinct
    # recurrences carry distinct ids, so this self-claim accrues distinct
    # source_events (the signal a self ACU reads to crystallize to L2 identity memory).
    source_event = None
    matching_ids = [int(r["id"]) for r in rows
                    if str(r.get("friction_type") or "") == friction_type and r.get("id") is not None]
    if matching_ids:
        source_event = max(matching_ids)
    try:
        from core.acatalepsy import intake
        intake.ingest_l1(raw_form=canonical, provenance="self", source_event=source_event)
    except Exception:
        # never raise into the turn path; crystallization is best-effort
        return False
    return True
