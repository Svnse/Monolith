"""Tests for the tolerant <bearing_update> recoverer.

The bound model persistently emits its bearing_update envelope tangled inside
its <think> reasoning (dangling close tags, embedded </think>, markdown fences,
trailing commas/prose), so the strict extractor returns parse_error every time
and the bearing has NEVER committed an update (updated_at_turn=None observed in
prod). This recovers the JSON from those real shapes WITHOUT touching the
structural-verifier semantics — verify_structural still runs on the result.
"""
from __future__ import annotations

from addons.system.bearing.tolerant_extract import (
    looks_like_attempt,
    recover_bearing_json,
)


def test_looks_like_attempt_distinguishes_broken_envelope_from_prose():
    assert looks_like_attempt('<bearing_update>{"a": 1') is True   # broken but real
    assert looks_like_attempt("use <bearing_update> in your docs") is False
    assert looks_like_attempt("no tag at all") is False


def test_recovers_envelope_tangled_in_think():
    text = (
        "<think>let me fix the bearing</think>\n"
        '<bearing_update>\n{"active_goal": {"new": "create hello.py"}}\n</bearing_update>'
    )
    assert recover_bearing_json(text) == {"active_goal": {"new": "create hello.py"}}


def test_recovers_dangling_open_no_close_tag():
    # Truncation: open tag, valid JSON, no </bearing_update> — strict drops it.
    text = '<bearing_update>\n{"active_goal": "x"}'
    assert recover_bearing_json(text) == {"active_goal": "x"}


def test_recovers_with_embedded_think_close_after_json():
    text = '<bearing_update>{"active_goal": "hello.py"}</think>\nthen more reasoning'
    assert recover_bearing_json(text) == {"active_goal": "hello.py"}


def test_recovers_through_markdown_fences():
    text = '<bearing_update>\n```json\n{"active_goal": "x"}\n```\n</bearing_update>'
    assert recover_bearing_json(text) == {"active_goal": "x"}


def test_recovers_trailing_comma():
    text = '<bearing_update>{"active_goal": "x",}</bearing_update>'
    assert recover_bearing_json(text) == {"active_goal": "x"}


def test_braces_inside_strings_do_not_break_balance():
    text = '<bearing_update>{"active_goal": "make a {dict} literal"}</bearing_update>'
    assert recover_bearing_json(text) == {"active_goal": "make a {dict} literal"}


def test_truncated_mid_object_is_unrecoverable_returns_none():
    # No closing brace at all — genuinely truncated; must NOT fabricate a partial.
    assert recover_bearing_json('<bearing_update>{"active_goal": "x"') is None


def test_no_envelope_returns_none():
    assert recover_bearing_json("just some prose, no envelope") is None


def test_prose_mention_without_json_is_not_a_false_recover():
    assert recover_bearing_json("use <bearing_update> in your docs like this") is None
