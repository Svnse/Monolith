from pathlib import Path


def test_skill_md_states_reground_rule_for_unverified_hits():
    text = Path("skills/monosearch/SKILL.md").read_text(encoding="utf-8")
    low = text.lower()
    # The model-facing contract must define the [unverified] marker as a
    # re-ground constraint (not a citable premise).
    assert "[unverified]" in text
    assert "re-ground" in low or "reground" in low or "not a citable premise" in low


def test_skill_md_does_not_leak_raw_tier_names():
    # Non-performative: the model's tool doc must never name the machine tiers.
    low = Path("skills/monosearch/SKILL.md").read_text(encoding="utf-8").lower()
    assert "source_tier" not in low
    assert "faithful-trace" not in low
    assert "generation" not in low
