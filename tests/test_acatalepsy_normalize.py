"""Tests for the Acatalepsy canonical-form normalizer (spine tranche).

The normalizer is retroactively load-bearing for identity: CID =
hash(normalized_form + cf_version). These tests pin the R1-R7 rules so a
later rule change is caught (and forces a cf_version bump) rather than
silently forking identity.
"""
from core.acatalepsy.normalize import (
    CF_VERSION,
    CanonicalTriple,
    normalize_canonical,
    parse_triple,
)


# ── normalize_canonical: the R1-R7 rules ──────────────────────────────

def test_r2_strip_and_r3_casefold():
    assert normalize_canonical("  Trump | Is | President  ") == "trump | is | president"


def test_r4_collapse_internal_whitespace():
    assert normalize_canonical("a |   r |  b") == "a | r | b"


def test_r5_pipe_spacing_is_normalized():
    assert normalize_canonical("a|r|b") == "a | r | b"


def test_r6_strips_trailing_punctuation_per_part():
    assert normalize_canonical("a | r | b.") == "a | r | b"
    assert normalize_canonical("trump | president_of | 2005.") == "trump | president_of | 2005"


def test_r6_preserves_internal_punctuation():
    # A real claim form must survive: dots/slashes inside a token are content.
    assert normalize_canonical("core/effort.py | defines | 7 tiers") == "core/effort.py | defines | 7 tiers"


def test_r7_smart_quotes_and_dashes_become_ascii():
    assert normalize_canonical("e | said | “hi”") == 'e | said | "hi"'
    assert normalize_canonical("a | — | b") == "a | - | b"


def test_dedup_equivalence_two_phrasings_collapse():
    # The intake MATCH path relies on these producing one identical form.
    assert normalize_canonical("A | r | B") == normalize_canonical(" a |  r | b. ")


def test_idempotency_normalize_of_normalize_is_stable():
    for raw in ["  Trump | IS | President. ", "a|r|b", "X | Y | Z | q.", "core/x.py | has | 7 tiers"]:
        once = normalize_canonical(raw)
        assert normalize_canonical(once) == once


def test_none_and_empty_safe():
    assert normalize_canonical(None) == ""
    assert normalize_canonical("   ") == ""


def test_cf_version_is_frozen_int_one():
    assert isinstance(CF_VERSION, int) and CF_VERSION == 1


# ── parse_triple ──────────────────────────────────────────────────────

def test_parse_triple_three_parts():
    t = parse_triple("trump | is | president")
    assert isinstance(t, CanonicalTriple)
    assert (t.entity_a, t.relation, t.entity_b, t.qualifiers) == ("trump", "is", "president", None)


def test_parse_triple_four_parts_keeps_qualifiers():
    t = parse_triple("trump | president_of | usa | 2017-2021")
    assert (t.entity_a, t.relation, t.entity_b, t.qualifiers) == (
        "trump", "president_of", "usa", "2017-2021",
    )


def test_parse_triple_rejects_two_parts():
    assert parse_triple("trump | president") is None


def test_parse_triple_rejects_five_parts():
    assert parse_triple("a | b | c | d | e") is None


def test_parse_triple_rejects_empty_part():
    assert parse_triple("a |  | c") is None
