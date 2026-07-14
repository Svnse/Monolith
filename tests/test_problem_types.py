"""Tests for the closed BRANCH problem-type vocabulary."""
from __future__ import annotations

from core import problem_types as pt


def test_enum_is_the_14_seeded_types():
    assert len(pt.PROBLEM_TYPES) == 14


def test_groups_partition_the_enum():
    grouped = [t for ids in pt.TYPE_GROUPS.values() for t in ids]
    assert sorted(grouped) == sorted(pt.PROBLEM_TYPES)
    assert len(grouped) == len(set(grouped))  # no type in two groups


def test_glosses_are_descriptive_only():
    # the D2 discipline from failure_tags: glosses describe what the type IS,
    # never prescribe action. Banned imperatives leak prescription.
    banned = ("should", "must", " fix", "instead", "wrong", "bad")
    for tid, gloss in pt.PROBLEM_TYPES.items():
        low = gloss.lower()
        for w in banned:
            assert w not in low, f"{tid} gloss contains banned word {w!r}"


def test_approaches_cover_every_type():
    assert set(pt.APPROACHES) == set(pt.PROBLEM_TYPES)
    assert all(pt.APPROACHES[t].strip() for t in pt.PROBLEM_TYPES)


def test_normalize_exact_id():
    assert pt.normalize_type("worst_case_bound") == "worst_case_bound"


def test_normalize_strips_type_label_and_punctuation():
    assert pt.normalize_type("TYPE: order_statistic_estimation.") == "order_statistic_estimation"
    assert pt.normalize_type("`eliminative_deduction`") == "eliminative_deduction"


def test_normalize_id_embedded_in_sentence():
    assert pt.normalize_type("This is aggregate_ratio_composition I think") == "aggregate_ratio_composition"


def test_normalize_unknown_becomes_other_not_a_cell():
    out = pt.normalize_type("bayesian magic")
    assert pt.is_other(out)
    assert not pt.is_valid_type(out)


def test_normalize_empty_is_none():
    assert pt.normalize_type("") is None
    assert pt.normalize_type(None) is None


def test_menu_lists_every_type():
    menu = pt.compose_type_menu()
    for tid in pt.PROBLEM_TYPES:
        assert tid in menu
