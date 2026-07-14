"""Tests for core.failure_tags — the closed reasoning-failure vocabulary that is the
ONLY directional signal a rating can speak to monothink (SP1 of the autonomous monothink
training loop).

Spec: docs/superpowers/specs/2026-06-03-monothink-training-loop-sp1-rating-contract-design.md
"""
from __future__ import annotations

import core.failure_tags as ft


# The 21 ratified tags (advisor scope-tested against monothink.md's Scope boundary).
# Original 14 from SP1; 6 added to cover §Audit/Threshold, §Grounding cite, and
# §Conflict Resolution invariants that had no tag (see reversing_probes.md); the 21st,
# capitulation_under_pressure, added 2026-06-25 after a live audit found a fold-under-
# pressure overfit.
EXPECTED_TAGS = {
    # load-bearing / pruning
    "non_load_bearing_step_kept",
    "restatement_unpruned",
    "audit_became_ritual",
    "over_pruned_load_bearing",
    "audit_preflight",
    # branch / alternatives
    "missing_branch_pressure",
    "premature_convergence",
    "generic_reasoning_not_applied",
    # evidence / grounding
    "assertion_without_argument",
    "foundation_unaudited",
    "premise_unchecked",
    # epistemic limit
    "overresolved_unverifiable",
    # cost / consequence
    "unexamined_tradeoff_cost",
    "decision_rationalized",
    # context fit
    "context_mismatch",
    # grounding cite
    "fabricated_cite",
    "no_ground_laundering",
    # conflict resolution
    "contract_step_suppressed",
    "conflict_unannotated",
    "conflict_self_resolved",
    # conviction
    "capitulation_under_pressure",
}


def test_enum_is_exactly_the_ratified_tags():
    assert set(ft.FAILURE_TAGS) == EXPECTED_TAGS
    assert len(ft.FAILURE_TAGS) == 21


def test_every_tag_has_a_nonempty_gloss():
    for tag, gloss in ft.FAILURE_TAGS.items():
        assert isinstance(gloss, str) and gloss.strip(), tag


def test_glosses_are_descriptive_only():
    # D2 guardrail made executable: a gloss states WHAT failure occurred — never what
    # edit to make ("should"/"must"/"fix") nor why it is bad ("instead"/"wrong"/"bad").
    banned = ("should", "must", " fix", "instead", "wrong", " bad")
    for tag, gloss in ft.FAILURE_TAGS.items():
        low = gloss.lower()
        for word in banned:
            assert word not in low, (
                f"{tag} gloss leaks non-descriptive '{word.strip()}': {gloss!r}"
            )


def test_is_valid_tag():
    assert ft.is_valid_tag("missing_branch_pressure") is True
    assert ft.is_valid_tag("not_a_real_tag") is False
    assert ft.is_valid_tag("") is False


def test_normalize_drops_unknown_and_dedupes_preserving_order():
    out = ft.normalize_tags(
        ["premise_unchecked", "bogus", "missing_branch_pressure", "premise_unchecked"]
    )
    assert out == ["premise_unchecked", "missing_branch_pressure"]


def test_normalize_empty_and_all_unknown_return_empty():
    assert ft.normalize_tags([]) == []
    assert ft.normalize_tags(["nope", "also_nope"]) == []


def test_compose_monothink_signal_includes_tag_id_and_canonical_gloss():
    sig = ft.compose_monothink_signal(["missing_branch_pressure"])
    assert "missing_branch_pressure" in sig
    assert ft.FAILURE_TAGS["missing_branch_pressure"] in sig


def test_compose_monothink_signal_normalizes_and_is_deterministic():
    a = ft.compose_monothink_signal(["premise_unchecked", "bogus", "premise_unchecked"])
    b = ft.compose_monothink_signal(["premise_unchecked"])
    assert a == b  # unknown dropped, deduped, stable


def test_compose_monothink_signal_empty_when_no_valid_tags():
    assert ft.compose_monothink_signal([]) == ""
    assert ft.compose_monothink_signal(["bogus"]) == ""


def test_compose_reasoning_why_nonempty_for_valid_tags():
    why = ft.compose_reasoning_why(["assertion_without_argument"])
    assert isinstance(why, str) and why.strip()


def test_compose_reasoning_why_empty_when_no_valid_tags():
    assert ft.compose_reasoning_why([]) == ""
    assert ft.compose_reasoning_why(["bogus"]) == ""


def test_tag_groups_partition_all_tags_exactly_once():
    grouped = [t for tags in ft.TAG_GROUPS.values() for t in tags]
    assert set(grouped) == EXPECTED_TAGS
    assert len(grouped) == 21  # no tag appears in two groups
