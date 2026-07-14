"""Tests for the single-source-of-truth internal-tag taxonomy.

Locks two things in place:

  1. The detector regex shape is fixed (open-or-close, word-boundary,
     case-insensitive). Both leak detectors must behave identically.
  2. The two name sets are non-empty supersets of the load-bearing tags
     so adding a tag flows to all callsites.

When adding a new internal tag, prefer extending INTERNAL_LEAK_TAGS or
EXTERNAL_STRIP_TAGS in core/internal_tags.py — do NOT re-inline a regex
in a new callsite.
"""
from __future__ import annotations

import re

import pytest

from core.internal_tags import (
    EXTERNAL_STRIP_TAGS,
    INTERNAL_LEAK_TAGS,
    make_leak_detection_pattern,
    strip_tag_blocks,
)


# ── canonical set invariants ────────────────────────────────────────


def test_internal_leak_tags_contains_load_bearing_tags() -> None:
    """think/analysis/reasoning/monolith_cmd/tool_call/acatalepsy are the
    six tags both detectors have always checked. Regression alarm if any
    falls out of the set."""
    required = {"think", "analysis", "reasoning", "monolith_cmd", "tool_call", "acatalepsy"}
    assert required.issubset(set(INTERNAL_LEAK_TAGS))


def test_external_strip_tags_is_superset_of_internal_leak_tags() -> None:
    """Anything the in-band detectors call a leak must also be stripped
    from external-peer output. The reverse is not required (some tags —
    tool_evidence/axes/intent — are stripped externally but excluded
    from the internal leak detector because the verifier has tighter
    dedicated checks for them)."""
    assert set(INTERNAL_LEAK_TAGS).issubset(set(EXTERNAL_STRIP_TAGS))


def test_monolith_cmd_strips_for_external_peers() -> None:
    """Regression: the agent_server external-strip set used to be
    missing monolith_cmd while the internal detectors flagged it. A
    peer receiving raw <monolith_cmd>...</monolith_cmd> markup was the
    drift symptom."""
    out = strip_tag_blocks("hello <monolith_cmd>internal</monolith_cmd> world", EXTERNAL_STRIP_TAGS)
    assert "monolith_cmd" not in out
    assert out.strip() == "hello  world"


# ── detector regex shape ────────────────────────────────────────────


@pytest.mark.parametrize("tag", list(INTERNAL_LEAK_TAGS))
def test_detector_matches_open_tag(tag: str) -> None:
    pattern = make_leak_detection_pattern(INTERNAL_LEAK_TAGS)
    assert pattern.search(f"some text <{tag}>content")


@pytest.mark.parametrize("tag", list(INTERNAL_LEAK_TAGS))
def test_detector_matches_close_tag(tag: str) -> None:
    pattern = make_leak_detection_pattern(INTERNAL_LEAK_TAGS)
    assert pattern.search(f"content</{tag}> trailing")


@pytest.mark.parametrize("tag", list(INTERNAL_LEAK_TAGS))
def test_detector_is_case_insensitive(tag: str) -> None:
    pattern = make_leak_detection_pattern(INTERNAL_LEAK_TAGS)
    assert pattern.search(f"<{tag.upper()}>")


def test_detector_word_boundary_excludes_lookalikes() -> None:
    """<thinking> should not match the `think` entry (different word).
    Without the trailing \\b, the partial-prefix match would fire."""
    pattern = make_leak_detection_pattern(INTERNAL_LEAK_TAGS)
    assert pattern.search("<thinking>") is None
    # But the actual <think> tag still matches, including with attributes.
    assert pattern.search("<think>")
    assert pattern.search("<think class='foo'>")


def test_detector_ignores_normal_prose() -> None:
    pattern = make_leak_detection_pattern(INTERNAL_LEAK_TAGS)
    assert pattern.search("hello world, no tags here at all.") is None


def test_empty_tag_set_yields_never_matches_pattern() -> None:
    pattern = make_leak_detection_pattern(())
    assert pattern.search("<think>anything</think>") is None


# ── strip behavior ─────────────────────────────────────────────────


def test_strip_removes_full_block_for_each_tag() -> None:
    text = (
        "before "
        "<think>internal thinking</think> "
        "<tool_call>{...}</tool_call> "
        "after"
    )
    out = strip_tag_blocks(text, EXTERNAL_STRIP_TAGS)
    assert "think" not in out
    assert "tool_call" not in out
    assert "before" in out
    assert "after" in out


def test_strip_handles_multiline_dotall() -> None:
    text = "before <think>\nline1\nline2\n</think> after"
    out = strip_tag_blocks(text, ("think",))
    assert "line1" not in out
    assert "line2" not in out


def test_strip_is_case_insensitive() -> None:
    text = "x <Think>internal</Think> y <TOOL_CALL>...</TOOL_CALL> z"
    out = strip_tag_blocks(text, EXTERNAL_STRIP_TAGS)
    assert "internal" not in out
    assert "..." not in out


def test_strip_passes_through_text_with_no_internal_tags() -> None:
    text = "Hello world. This is a normal answer."
    out = strip_tag_blocks(text, EXTERNAL_STRIP_TAGS)
    assert out == text


def test_strip_handles_empty_input() -> None:
    assert strip_tag_blocks("", EXTERNAL_STRIP_TAGS) == ""


# ── parity between the two detector sites ──────────────────────────


def test_both_detectors_see_same_leaks() -> None:
    """The output_sanitizer and response_verifier detectors used to be
    near-identical hand-inlined regexes. After centralization, they
    must by construction see the same set of leaks. This test imports
    both and verifies parity over a representative input set."""
    from core.pipeline_policies.output_sanitizer import _INTERNAL_TAG_RE as _SAN
    from core.response_verifier import _RAW_INTERNAL_TAG_RE as _VER

    cases = [
        "<think>x</think>",
        "</analysis>",
        "<monolith_cmd>do thing</monolith_cmd>",
        "<TOOL_CALL>{}</TOOL_CALL>",
        "<acatalepsy>note</acatalepsy>",
        "<reasoning>...",
        "no tags here",
        "<thinking>",  # word-boundary exclusion
    ]
    for text in cases:
        assert bool(_SAN.search(text)) == bool(_VER.search(text)), (
            f"detector drift on {text!r}: "
            f"sanitizer={bool(_SAN.search(text))} verifier={bool(_VER.search(text))}"
        )
