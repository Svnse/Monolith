"""Automatic ACU candidate triage for MonoBase.

This module intentionally reuses ``decisions.insert_decision`` instead of
writing ACUs directly. Auto-accepted candidates still pass through the normal
candidate -> decision -> L1 intake chain, and auto-rejected candidates get the
same canonical-log decision event as manual rejects.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.acatalepsy import candidates as _candidates
from core.acatalepsy import decisions as _decisions
from core.acatalepsy.atomicity import is_atomic
from core.acatalepsy.extraction_quality import is_extraction_quality_acceptable


__all__ = (
    "AutoReviewItem",
    "AutoReviewSummary",
    "classify_candidate",
    "decider_for_candidate",
    "review_pending",
)


@dataclass(frozen=True)
class AutoReviewItem:
    candidate_id: int
    action: str
    reason: str
    decision_id: int | None = None
    resulting_acu_id: int | None = None


@dataclass(frozen=True)
class AutoReviewSummary:
    accepted: int
    rejected: int
    skipped: int
    items: tuple[AutoReviewItem, ...]


def decider_for_candidate(candidate: _candidates.Candidate) -> str | None:
    """Return the non-user decider authorized for this auditor candidate."""
    source = (candidate.source or "").strip()
    if not source.startswith("auditor_"):
        return None
    suffix = source[len("auditor_"):].strip()
    if not suffix:
        return None
    return f"agent_{suffix}"


def classify_candidate(candidate: _candidates.Candidate) -> tuple[str, str]:
    """Return ``(action, reason)`` for a pending candidate.

    Conservative contract:
      - accept well-formed, non-contradictory auditor candidates;
      - reject stale/invalid candidates that should not survive review;
      - skip anything authorization-sensitive or semantically conflict-heavy.
    """
    if candidate.state != "pending":
        return "skip", f"state:{candidate.state}"

    if decider_for_candidate(candidate) is None:
        return "skip", f"unauthorized_source:{candidate.source or 'none'}"

    quality = is_extraction_quality_acceptable(candidate.canonical_form)
    if not quality.ok:
        return "reject", f"auto_review:extraction_quality:{quality.reason or 'unknown'}"

    atomic = is_atomic(candidate.canonical_form)
    if not atomic.ok:
        return "reject", f"auto_review:atomicity:{atomic.reason or 'unknown'}"

    if not (candidate.evidence_span or "").strip():
        return "reject", "auto_review:missing_evidence"

    if candidate.evidence_char_start < 0 or candidate.evidence_char_end < candidate.evidence_char_start:
        return "reject", "auto_review:invalid_evidence_span"

    if candidate.contradicts_acu_id is not None:
        return "skip", f"contradiction_requires_human:{candidate.contradicts_acu_id}"

    return "accept", "auto_review:safe_atomic_candidate"


def review_pending(*, limit: int = 50) -> AutoReviewSummary:
    """Auto-decide pending candidates and return a compact audit summary."""
    items: list[AutoReviewItem] = []
    accepted = 0
    rejected = 0
    skipped = 0

    for candidate in _candidates.read_pending(limit=max(1, int(limit))):
        action, reason = classify_candidate(candidate)
        decider = decider_for_candidate(candidate)
        if action == "skip" or decider is None:
            skipped += 1
            items.append(AutoReviewItem(candidate.id, "skip", reason))
            continue

        try:
            if action == "accept":
                decision_id = _decisions.insert_decision(
                    candidate_id=candidate.id,
                    decision="accept",
                    decided_by=decider,
                    note=reason,
                )
                decision = _decisions.read_one(decision_id)
                accepted += 1
                items.append(
                    AutoReviewItem(
                        candidate.id,
                        "accept",
                        reason,
                        decision_id=decision_id,
                        resulting_acu_id=(
                            decision.resulting_acu_id if decision is not None else None
                        ),
                    )
                )
            elif action == "reject":
                decision_id = _decisions.insert_decision(
                    candidate_id=candidate.id,
                    decision="reject",
                    decided_by=decider,
                    reject_reason=reason,
                    note=reason,
                )
                rejected += 1
                items.append(
                    AutoReviewItem(
                        candidate.id,
                        "reject",
                        reason,
                        decision_id=decision_id,
                    )
                )
            else:
                skipped += 1
                items.append(AutoReviewItem(candidate.id, "skip", f"unknown_action:{action}"))
        except Exception as exc:
            skipped += 1
            items.append(
                AutoReviewItem(
                    candidate.id,
                    "skip",
                    f"decision_failed:{type(exc).__name__}:{exc}",
                )
            )

    return AutoReviewSummary(
        accepted=accepted,
        rejected=rejected,
        skipped=skipped,
        items=tuple(items),
    )
