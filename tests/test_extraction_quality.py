"""Tests for the pre-atomicity extraction-quality filter.

Regression for the 2026-05-14 ACU pollution incident: the auditor
extracted four chat fragments ("Want me to trim any of those?", etc.)
that passed the atomicity gate because they were well-formed strings.
The extraction-quality filter rejects them before they reach atomicity.
"""
from __future__ import annotations

from core.acatalepsy.extraction_quality import (
    is_extraction_quality_acceptable,
)


def test_well_formed_claim_passes() -> None:
    result = is_extraction_quality_acceptable(
        "Monolith continuity | persists across | session boundary"
    )
    assert result.ok
    assert result.reason is None


def test_rejects_empty() -> None:
    result = is_extraction_quality_acceptable("")
    assert not result.ok
    assert result.reason == "empty"


def test_rejects_whitespace_only() -> None:
    result = is_extraction_quality_acceptable("   \t\n  ")
    assert not result.ok
    assert result.reason == "empty"


def test_rejects_non_string() -> None:
    result = is_extraction_quality_acceptable(None)  # type: ignore[arg-type]
    assert not result.ok
    assert result.reason == "not_a_string"


def test_rejects_too_short() -> None:
    result = is_extraction_quality_acceptable("a|b|c")
    assert not result.ok
    assert result.reason and result.reason.startswith("too_short:")


def test_rejects_question_ending() -> None:
    # The literal 2026-05-14 incident text
    result = is_extraction_quality_acceptable(
        "Want me to go ahead and trim any of those?"
    )
    assert not result.ok
    # First-checked: phrase comes before question check, but either is acceptable
    assert result.reason in {
        "ends_with_question",
        "conversational_phrase:want me to",
    }


def test_rejects_explicit_question() -> None:
    result = is_extraction_quality_acceptable(
        "Did the auditor pick up this turn correctly?"
    )
    assert not result.ok
    assert result.reason == "ends_with_question"


def test_rejects_conversational_opener_ill() -> None:
    result = is_extraction_quality_acceptable(
        "I'll check the logs and report back"
    )
    assert not result.ok
    assert result.reason and result.reason.startswith("conversational_opener:")


def test_rejects_conversational_opener_let_me() -> None:
    result = is_extraction_quality_acceptable(
        "Let me check the kernel state"
    )
    assert not result.ok
    assert result.reason and result.reason.startswith("conversational_opener:let me")


def test_rejects_conversational_opener_sure() -> None:
    result = is_extraction_quality_acceptable(
        "Sure, here is the answer you asked for"
    )
    assert not result.ok
    assert result.reason and result.reason.startswith("conversational_opener:sure")


def test_rejects_conversational_phrase_let_me_know() -> None:
    result = is_extraction_quality_acceptable(
        "Run the migration and let me know how it goes"
    )
    assert not result.ok
    assert result.reason and result.reason.startswith("conversational_phrase:")


def test_rejects_2026_05_14_incident_acus() -> None:
    # The four ACUs that polluted the store on 2026-05-14, contradicted today.
    # Verify the LAYERED defense (extraction-quality + atomicity) catches all four,
    # even though no single gate rejects every fragment.
    from core.acatalepsy.atomicity import is_atomic

    fragments = [
        "` block in the Memory section** feels like an artifact from a template.",
        "- **Double-negative `arguments` vs `arguments` in the `recall` tool call snippet** — looks like a typo.",
        "The **scratchpad** integration is solid — exactly what I need for continuity across sessions.",
        "Want me to go ahead and trim any of those?",
    ]
    for fragment in fragments:
        extraction_ok = is_extraction_quality_acceptable(fragment).ok
        atomicity_ok = is_atomic(fragment).ok
        assert not (extraction_ok and atomicity_ok), (
            f"both gates accepted a 2026-05-14 fragment that should have been rejected: "
            f"{fragment!r}"
        )


def test_passes_real_audit_claim_with_pipe_structure() -> None:
    # Real-shape claims from the ACU store
    real_claims = [
        "Monolith identity | local workstation contradicted by | observed cloud runtime",
        "continuity pins | are not injected | despite runtime asserting continuity_maintained true",
        "ACU store | flattens | evidence types into persistence",
    ]
    for c in real_claims:
        result = is_extraction_quality_acceptable(c)
        assert result.ok, f"expected pass but got {result.reason!r} for: {c}"
