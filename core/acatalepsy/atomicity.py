"""Deterministic atomicity gate for ACU candidates.

Per Acatalepsy v1 spec §4.4 — hard-reject in v1; auto-split is v1.5+
work. The gate runs after the LLM emits a candidate, before the
candidate lands in the acu_candidates table. Rejected candidates are
logged to canonical_log as ``auditor_atomicity_reject`` so we can tune
the auditor's behavior over time.

Atomic = ONE subject-predicate assertion. The canonical_form syntax is:

    "subject | relation | object"
    "subject | relation | object | qualifiers"   (4 parts allowed)

Anything that splits into multiple predicates is non-atomic. Compound
markers ('and', 'or', 'because', 'therefore', 'while') anywhere in the
canonical_form are heuristic rejection signals.

Deterministic. No LLM. No I/O. Pure function.
"""
from __future__ import annotations

from dataclasses import dataclass


__all__ = ("AtomicityResult", "is_atomic")


_COMPOUND_MARKERS: tuple[str, ...] = (
    " and ",
    " or ",
    " because ",
    " therefore ",
    " while ",
)

# Min/max pipe-delimited parts of the canonical form.
# - 3 parts: subject | relation | object
# - 4 parts: subject | relation | object | qualifiers
_MIN_PARTS = 3
_MAX_PARTS = 4


@dataclass(frozen=True)
class AtomicityResult:
    ok: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def is_atomic(canonical_form: str) -> AtomicityResult:
    """Check whether canonical_form passes the atomicity gate.

    Returns AtomicityResult(ok=True, reason=None) on pass.
    Returns AtomicityResult(ok=False, reason=<why>) on reject.

    Rejection reasons (stable strings — safe to use in canonical_log
    payload for downstream analysis):
      - "empty"
      - "compound_marker:<marker>"    (e.g. "compound_marker:and")
      - "too_few_parts:<n>"
      - "too_many_parts:<n>"
      - "empty_part:<index>"
    """
    if not isinstance(canonical_form, str):
        return AtomicityResult(ok=False, reason="not_a_string")

    stripped = canonical_form.strip()
    if not stripped:
        return AtomicityResult(ok=False, reason="empty")

    # Compound markers — heuristic but catches the obvious cases.
    lowered = stripped.lower()
    for marker in _COMPOUND_MARKERS:
        if marker in lowered:
            return AtomicityResult(ok=False, reason=f"compound_marker:{marker.strip()}")

    # Pipe-delimited parts.
    parts = stripped.split("|")
    n = len(parts)
    if n < _MIN_PARTS:
        return AtomicityResult(ok=False, reason=f"too_few_parts:{n}")
    if n > _MAX_PARTS:
        return AtomicityResult(ok=False, reason=f"too_many_parts:{n}")

    for idx, part in enumerate(parts):
        if not part.strip():
            return AtomicityResult(ok=False, reason=f"empty_part:{idx}")

    return AtomicityResult(ok=True, reason=None)
