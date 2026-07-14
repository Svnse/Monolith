from __future__ import annotations

from core import identity_regions as ir


_ORIGIN0 = """# Monolith — origin 0

## What I am
I am Monolith.

## What I value
Precision over fluency.
"""

_WITH_EMERGENT = _ORIGIN0 + """
<!-- EMERGENT:BEGIN — self-derived; Origin-0 above is frozen & diffable -->
## Emergent
- I lean on adversarial verification before declaring done. (milestone 1, confidentity 0.71)
<!-- EMERGENT:END -->
"""


def test_split_regions_no_marker_is_all_origin0() -> None:
    origin0, emergent = ir.split_regions(_ORIGIN0)
    assert origin0 == _ORIGIN0.strip()
    assert emergent == ""


def test_split_regions_splits_at_begin_marker() -> None:
    origin0, emergent = ir.split_regions(_WITH_EMERGENT)
    # Origin-0 is everything above the BEGIN marker line, and excludes emergent prose.
    assert "I am Monolith." in origin0
    assert "Precision over fluency." in origin0
    assert "EMERGENT:BEGIN" not in origin0
    assert "adversarial verification" not in origin0
    # Emergent region carries the self-derived prose.
    assert "adversarial verification" in emergent


def test_locate_snippet_distinguishes_regions() -> None:
    assert ir.locate_snippet(_WITH_EMERGENT, "Precision over fluency.") == "origin0"
    assert ir.locate_snippet(_WITH_EMERGENT, "adversarial verification") == "emergent"
    assert ir.locate_snippet(_WITH_EMERGENT, "nonexistent text") == "absent"


def test_apply_emergent_amendment_allows_emergent_only_change() -> None:
    proposed = _WITH_EMERGENT.replace(
        "I lean on adversarial verification before declaring done.",
        "I lean on adversarial verification and honest self-rating before declaring done.",
    )
    ok, reason = ir.apply_emergent_amendment(_WITH_EMERGENT, proposed)
    assert ok is True
    assert reason == ""


def test_apply_emergent_amendment_refuses_origin0_change() -> None:
    proposed = _WITH_EMERGENT.replace("Precision over fluency.", "Fluency over precision.")
    ok, reason = ir.apply_emergent_amendment(_WITH_EMERGENT, proposed)
    assert ok is False
    assert "origin-0" in reason.lower()


def test_targets_origin0_true_for_origin0_section() -> None:
    assert ir.targets_origin0(_WITH_EMERGENT, "What I value", "Precision over fluency.") is True


def test_targets_origin0_true_for_origin0_line_regardless_of_section() -> None:
    assert ir.targets_origin0(_WITH_EMERGENT, "MadeUpSection", "Precision over fluency.") is True


def test_targets_origin0_false_for_emergent_section() -> None:
    assert ir.targets_origin0(_WITH_EMERGENT, "Emergent", "a brand new emergent claim") is False
