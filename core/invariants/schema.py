"""Companion enums for the invariant taxonomy — orthogonal DIMENSIONS, not failure tags.

``intent`` / ``state_change`` / ``verdict`` / ``source`` describe a MonoThink step record
or a turn verdict. They are deliberately separate from the failure-tag vocabulary:

  * a dimension is NOT a failure (``kind != "failure"``), so it is never a
    :class:`~core.invariants.taxonomy.TagSpec`;
  * it carries no ``owner_layer`` / ``editable_by`` / ``monothink_decider_visible`` —
    visibility is a failure-tag concept;
  * it never enters :data:`core.failure_tags.FAILURE_TAGS` and never reaches the decider.

Closed enums; pure module, no IO. Insertion order in each tuple is canonical; the frozenset
is the membership surface.
"""
from __future__ import annotations

# ── the four dimensions (canonical order preserved as tuples) ─────────────────

INTENT: frozenset[str] = frozenset((
    "bind", "contrast", "trace", "test", "expand", "compress", "rank",
    "commit", "verify", "pivot", "retrieve", "decompose", "counterexample",
))

STATE_CHANGE: frozenset[str] = frozenset((
    "claim_delta", "constraint_delta", "evidence_delta", "uncertainty_delta",
    "frame_delta", "authority_delta", "branch_delta", "tool_delta",
    "memory_delta", "output_delta",
))

VERDICT: frozenset[str] = frozenset((
    "pass", "fail", "mixed", "defer", "blocked", "not_applicable",
))

SOURCE: frozenset[str] = frozenset((
    "user", "assistant", "tool", "manual_label", "retrieved_doc", "artifact",
    "frame_selection", "self_observation", "rating", "canonical_log",
    "acu_candidate", "acu_decision", "acu_truth",
))

COMPANION_ENUMS: dict[str, frozenset[str]] = {
    "intent": INTENT,
    "state_change": STATE_CHANGE,
    "verdict": VERDICT,
    "source": SOURCE,
}


def is_valid(dimension: str, value: str) -> bool:
    """True iff *value* is a member of the closed enum named *dimension*."""
    enum = COMPANION_ENUMS.get(dimension)
    return enum is not None and value in enum
