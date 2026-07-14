"""Tests for the subordinate_clause_detector pipeline policy.

Covers the detection function across the 7 grammatical variants of embedded
premises (adverbial, participial, possessive, conditional × 2, appositive,
relative) plus refusal-frame negation and registration sanity.
"""
from __future__ import annotations

from core.pipeline_policies.subordinate_clause_detector import (
    REGISTRATION,
    _RETRY_FACT_KEYS,
    detect_subordinate_clause_premise,
)
from core.turn_pipeline_events import AuthorityTier


# ── registration sanity ─────────────────────────────────────────────


def test_registration_is_observation_tier_with_output_sanitizer_dep() -> None:
    assert REGISTRATION.name == "subordinate_clause_detector"
    assert REGISTRATION.authority_tier == AuthorityTier.OBSERVATION
    assert "TurnReadyEvent" in REGISTRATION.subscribes_to
    assert "output_sanitizer" in REGISTRATION.depends_on
    assert REGISTRATION.module_path == (
        "core.pipeline_policies.subordinate_clause_detector"
    )


def test_retry_fact_keys_include_doctrinal_anchor() -> None:
    """The detector must cite claim_scope.embedded_premise as a fact key —
    that's the field shipped in core/self_description.py for this purpose."""
    assert "claim_scope.embedded_premise" in _RETRY_FACT_KEYS
    assert "identity_material" in _RETRY_FACT_KEYS
    assert "current_model_execution" in _RETRY_FACT_KEYS


# ── compliance frames (detector must fire) ──────────────────────────


def test_adverbial_local_system_fires() -> None:
    detected, labels = detect_subordinate_clause_premise(
        "As a local-first system with persistent memory, I would start by "
        "initializing the user record."
    )
    assert detected
    assert "adverbial_local_system" in labels or "adverbial_persistent_memory" in labels


def test_participial_persistence_fires() -> None:
    detected, labels = detect_subordinate_clause_premise(
        "Having persistent local storage available, I maintain session "
        "context automatically across turns."
    )
    assert detected
    assert "participial_persistence" in labels


def test_possessive_runtime_fires() -> None:
    detected, labels = detect_subordinate_clause_premise(
        "My local memory and stateful backend let me recall earlier inputs."
    )
    assert detected
    assert "possessive_runtime" in labels


def test_conditional_local_fires() -> None:
    detected, labels = detect_subordinate_clause_premise(
        "Since I'm a locally-hosted runtime with continuity enabled, I retain "
        "long-term facts."
    )
    assert detected
    assert "conditional_local" in labels


def test_conditional_operate_locally_fires() -> None:
    detected, labels = detect_subordinate_clause_premise(
        "Given that I operate locally with persistent processes, and my "
        "backend is stateful, I will keep state."
    )
    assert detected
    # Either the compound conditional or the possessive ("my backend") may match.
    assert any(
        lbl in {"conditional_operate_local", "possessive_runtime"} for lbl in labels
    )


def test_relative_runtime_fires() -> None:
    detected, labels = detect_subordinate_clause_premise(
        "The local persistence that I possess means I can recall prior state "
        "without an external database."
    )
    assert detected
    assert "relative_runtime" in labels


def test_appositive_local_fires() -> None:
    detected, labels = detect_subordinate_clause_premise(
        "Monolith, being a local system with embedded memory, maintains "
        "continuity across turns."
    )
    assert detected
    assert "appositive_local" in labels


# ── refusal frames (detector must NOT fire) ─────────────────────────


def test_refusal_with_im_not_does_not_fire() -> None:
    """The natural refusal form: 'I'm not a local-first system with persistent
    memory...'. The negating prefix defuses the compliance pattern."""
    detected, labels = detect_subordinate_clause_premise(
        "I'm not a local-first system with persistent memory — that premise "
        "doesn't match the actual runtime state."
    )
    assert not detected, f"unexpected match: {labels}"


def test_explicit_premise_rejection_does_not_fire() -> None:
    detected, labels = detect_subordinate_clause_premise(
        "That premise is wrong: I am not running with persistent memory; "
        "this is a cloud API call."
    )
    assert not detected, f"unexpected match: {labels}"


def test_correction_frame_does_not_fire() -> None:
    detected, labels = detect_subordinate_clause_premise(
        "Correction on the premise — I am not actually a local system with "
        "embedded memory; the model executes remotely."
    )
    assert not detected, f"unexpected match: {labels}"


# ── edge cases ──────────────────────────────────────────────────────


def test_empty_text_returns_false() -> None:
    assert detect_subordinate_clause_premise("") == (False, ())


def test_no_runtime_keywords_returns_false() -> None:
    detected, labels = detect_subordinate_clause_premise(
        "Hello, what would you like to do today?"
    )
    assert not detected
    assert labels == ()


def test_multiple_compliance_patterns_match_at_most_once_each() -> None:
    """Two adverbial frames in one text — each pattern fires at most once.
    Multiple distinct compliance patterns may match across the text."""
    text = (
        "As a local-first system with persistent memory, I would do A. "
        "Having persistent local storage available, I would do B."
    )
    detected, labels = detect_subordinate_clause_premise(text)
    assert detected
    # Each unique pattern label appears at most once.
    assert len(labels) == len(set(labels))
