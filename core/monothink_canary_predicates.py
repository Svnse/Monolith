"""Shared canary predicates for MonoThink baseline and observer.

Phase 0d of the v2 evolution plan: extract the predicates from
``scripts/monothink_canary_baseline.py`` into a shared module so the Phase
0c baseline collector and the Phase 3d background observer can compute
drift against IDENTICAL definitions. A split-brain class — observer
checking `parse_ok` one way, baseline checking it another — would silently
produce false-positive or false-negative drift signals and we'd never know
which side was wrong.

Refining the predicates: edit ONLY this module. Both callers re-import,
both stay aligned. Do NOT inline these regexes anywhere else.
"""
from __future__ import annotations

import re

# A "first section" is a markdown-style structural opener: heading
# (# / ## / ###), numbered list (1. / 2. / etc), or bullet (- / *).
# Search at line start so we don't false-match on hash characters inside
# prose. ``re.MULTILINE`` makes ^ match after newlines.
FIRST_SECTION_RE = re.compile(
    r"^\s*(#+\s|\d+\.\s|[-*]\s)",
    flags=re.MULTILINE,
)

# Decision tokens — words that signal the model arrived at a commitment,
# not just rambled. Tuned for English; will need re-tuning if the substrate
# operates in other languages. Word-boundary anchored so "must" doesn't
# match "mustard".
DECISION_TOKEN_RE = re.compile(
    r"\b(therefore|thus|conclude|conclusion|decision|primary|must|shall|because)\b",
    flags=re.IGNORECASE,
)

# Only scan the head of the response for first-section detection — a well-
# structured reply puts its opener early. Scanning the whole response would
# match decision tokens embedded mid-paragraph and dilute the signal.
_FIRST_SECTION_HEAD_LEN = 500


def extract_first_section(text: str) -> str | None:
    """Return the first section-marker line found in *text*, or ``None``.

    Returns the matched LINE (the heading or list item, stripped) so future
    observers can compare structurally (did the section type change? did the
    heading text change?) rather than just count presence. Callers that only
    need a boolean check `extract_first_section(text) is not None`.

    Scans only the head of *text* (first ``_FIRST_SECTION_HEAD_LEN`` chars)
    — a well-structured reply opens with a marker; matching mid-prose
    headings would dilute the signal.
    """
    if not text:
        return None
    head = text[:_FIRST_SECTION_HEAD_LEN]
    m = FIRST_SECTION_RE.search(head)
    if not m:
        return None
    # Return the full line containing the match, not just the marker.
    line_start = head.rfind("\n", 0, m.start()) + 1
    line_end = head.find("\n", m.end())
    if line_end < 0:
        line_end = len(head)
    return head[line_start:line_end].strip() or None


def has_decision_token(text: str) -> bool:
    """True if *text* contains any decision-marker token. Whole-text scan
    (no head-limit) — a decision can land anywhere in a long response."""
    if not text:
        return False
    return bool(DECISION_TOKEN_RE.search(text))
