"""Tests for core/acatalepsy/atomicity.py — the deterministic gate."""
from __future__ import annotations

import pytest

from core.acatalepsy.atomicity import is_atomic


# ── happy path: valid atomic forms pass ─────────────────────────────


@pytest.mark.parametrize("form", [
    "Monolith | uses | seven effort tiers",
    "core/effort.py | defines | tier-resolution layered fallback",
    "canonical_log | has | 2 rows | as_of=2026-05-13",
    "auditor | rejects | non-atomic candidates",
    "  Monolith  |  uses  |  effort tiers  ",   # whitespace tolerated
])
def test_valid_forms_pass(form: str) -> None:
    result = is_atomic(form)
    assert result.ok is True
    assert result.reason is None
    assert bool(result) is True


# ── compound markers rejected ───────────────────────────────────────


@pytest.mark.parametrize("form, marker", [
    ("Monolith | uses | effort tiers and CONNECT addon", "and"),
    ("user | likes | python or javascript", "or"),
    ("decay fires | because | claims are stale", "because"),
    ("x | breaks | because of y", "because"),
    ("CCG induces edges | therefore | claims connect", "therefore"),
    ("auditor runs | while | session is active", "while"),
])
def test_compound_markers_rejected(form: str, marker: str) -> None:
    result = is_atomic(form)
    assert result.ok is False
    assert result.reason == f"compound_marker:{marker}"


def test_compound_marker_case_insensitive() -> None:
    """Compound detection should be case-insensitive."""
    result = is_atomic("Monolith | uses | tiers AND addon")
    assert result.ok is False
    assert result.reason == "compound_marker:and"


# ── shape violations rejected ───────────────────────────────────────


def test_empty_string_rejected() -> None:
    result = is_atomic("")
    assert result.ok is False
    assert result.reason == "empty"


def test_whitespace_only_rejected() -> None:
    result = is_atomic("   ")
    assert result.ok is False
    assert result.reason == "empty"


@pytest.mark.parametrize("form, n", [
    ("just a subject", 1),
    ("subject | predicate", 2),
])
def test_too_few_parts_rejected(form: str, n: int) -> None:
    result = is_atomic(form)
    assert result.ok is False
    assert result.reason == f"too_few_parts:{n}"


def test_too_many_parts_rejected() -> None:
    form = "a | b | c | d | e"
    result = is_atomic(form)
    assert result.ok is False
    assert result.reason == "too_many_parts:5"


def test_empty_part_rejected() -> None:
    result = is_atomic("a |  | c")
    assert result.ok is False
    assert result.reason.startswith("empty_part:")
    assert result.reason == "empty_part:1"


def test_first_part_empty_rejected() -> None:
    result = is_atomic(" | rel | obj")
    assert result.ok is False
    assert result.reason == "empty_part:0"


def test_last_part_empty_rejected() -> None:
    result = is_atomic("subj | rel | ")
    assert result.ok is False
    assert result.reason == "empty_part:2"


# ── type safety ──────────────────────────────────────────────────────


def test_non_string_rejected() -> None:
    # type: ignore[arg-type] — intentional bad input
    result = is_atomic(None)  # type: ignore[arg-type]
    assert result.ok is False
    assert result.reason == "not_a_string"


# ── boolean conversion ──────────────────────────────────────────────


def test_result_is_truthy_when_ok() -> None:
    assert bool(is_atomic("a | b | c")) is True
    assert bool(is_atomic("bad")) is False
