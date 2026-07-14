"""Pre-atomicity extraction-quality filter for ACU candidates.

The atomicity gate catches compound claims and wrong-shape strings, but
it accepts conversational sentences that are well-formed but aren't
claims at all — e.g. "Want me to trim any of those?" was ingested as
an ACU on 2026-05-14 because the auditor extracted it and the
atomicity gate had no signal that the canonical_form was a question
instead of a structured assertion.

This module runs BEFORE atomicity.is_atomic() and rejects conversational
fragments before they become persistent canonical_log noise.

Deterministic. No LLM. No I/O. Pure function.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


__all__ = ("ExtractionQualityResult", "is_extraction_quality_acceptable")


# Conversational opener patterns — start of canonical_form.
_CONVERSATIONAL_OPENERS = (
    "i'll ",
    "ill ",
    "sure,",
    "sure ",
    "let me ",
    "let's ",
    "hmm,",
    "hmm ",
    "well,",
    "ok,",
    "okay,",
    "great,",
    "got it",
    "thanks",
    "thank you",
)

# Conversational phrase fragments — anywhere in canonical_form.
_CONVERSATIONAL_PHRASES = (
    "want me to",
    "let me know",
    "shall i",
    "should i",
    "could go",
    "would you like",
    "do you want",
    "feel free",
    "let me check",
    "let me know if",
)

_QUESTION_PATTERN = re.compile(r".*\?\s*$")
_MIN_LENGTH_CHARS = 8


@dataclass(frozen=True)
class ExtractionQualityResult:
    ok: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def is_extraction_quality_acceptable(canonical_form: str) -> ExtractionQualityResult:
    """Check whether canonical_form passes the extraction-quality gate.

    Rejection reasons (stable strings — safe to record in canonical_log):
      - "not_a_string"
      - "empty"
      - "too_short:<n>"
      - "ends_with_question"
      - "conversational_opener:<opener>"
      - "conversational_phrase:<phrase>"
    """
    if not isinstance(canonical_form, str):
        return ExtractionQualityResult(ok=False, reason="not_a_string")

    stripped = canonical_form.strip()
    if not stripped:
        return ExtractionQualityResult(ok=False, reason="empty")

    if len(stripped) < _MIN_LENGTH_CHARS:
        return ExtractionQualityResult(ok=False, reason=f"too_short:{len(stripped)}")

    if _QUESTION_PATTERN.match(stripped):
        return ExtractionQualityResult(ok=False, reason="ends_with_question")

    lowered = stripped.lower()
    for opener in _CONVERSATIONAL_OPENERS:
        if lowered.startswith(opener):
            return ExtractionQualityResult(
                ok=False,
                reason=f"conversational_opener:{opener.strip().rstrip(',')}",
            )

    for phrase in _CONVERSATIONAL_PHRASES:
        if phrase in lowered:
            return ExtractionQualityResult(
                ok=False,
                reason=f"conversational_phrase:{phrase}",
            )

    return ExtractionQualityResult(ok=True, reason=None)
