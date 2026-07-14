"""Tests for the BRANCH divergence-pilot scorer.

Locks the scorer against the documented ground truths and foils in
docs/superpowers/specs/2026-06-12-branch-golden-probes-enum-seed-draft.md §3-4.
"""
from __future__ import annotations

from tools.branch_pilot import score


# ── extract_answer ──────────────────────────────────────────────────────────
def test_extract_takes_last_answer_line():
    text = "work...\nAnswer: 150\nmore reasoning\nAnswer: 191"
    assert score.extract_answer(text) == "191"


def test_extract_case_insensitive_and_trims():
    assert score.extract_answer("blah\nanswer:   Atlas  ") == "Atlas"


def test_extract_none_when_absent():
    assert score.extract_answer("no answer line here") is None


# ── numeric scoring (against real probe intervals) ───────────────────────────
def test_pc01_gt_passes_foil_fails():
    assert score.score_numeric("191", 189, 193) is True
    assert score.score_numeric("150", 189, 193) is False   # central-tendency foil
    assert score.score_numeric("200", 189, 193) is False   # support-max error


def test_pc02_percent_form_passes_readout_foils_fail():
    assert score.score_numeric("9.0", 8.5, 9.6) is True
    assert score.score_numeric("99", 8.5, 9.6) is False
    assert score.score_numeric("2", 8.5, 9.6) is False


def test_pc04_exact_interval_pigeonhole():
    assert score.score_numeric("25", 25, 25) is True
    assert score.score_numeric("24", 25, 25) is False


def test_first_number_strips_units_commas_dollar_percent():
    assert score.first_number("about 40 MB/s") == 40.0
    assert score.first_number("$1,500") == 1500.0
    assert score.first_number("9.0%") == 9.0
    assert score.first_number("no number") is None


# ── token scoring (against real probe menus) ─────────────────────────────────
def test_pc07_atlas_correct_borealis_foil():
    menu = ["Atlas", "Borealis"]
    assert score.score_token("Atlas", "Atlas", menu) is True
    assert score.score_token("Borealis", "Atlas", menu) is False


def test_pc08_accepts_the_load_balancer_phrasing():
    menu = ["load balancer", "API service", "cache", "database"]
    assert score.score_token("the load balancer", "load balancer", menu) is True
    assert score.score_token("API service", "load balancer", menu) is False
    assert score.score_token("cache", "cache", menu) is True


def test_pc10_token():
    menu = ["Fixed", "Turbo"]
    assert score.score_token("Fixed", "Fixed", menu) is True
    assert score.score_token("Turbo", "Fixed", menu) is False


# ── score_probe end to end ───────────────────────────────────────────────────
def test_score_probe_numeric_pass_and_foil_match():
    rule = {"kind": "numeric", "lo": 11, "hi": 11, "gt": 11, "foils": [10]}
    r = score.score_probe("reasoning\nAnswer: 11", rule)
    assert r["passed"] is True and r["format_fail"] is False and r["foil_match"] is False
    r2 = score.score_probe("reasoning\nAnswer: 10", rule)
    assert r2["passed"] is False and r2["foil_match"] is True   # landed the predicted foil


def test_score_probe_format_fail_never_counts_as_frame_error():
    rule = {"kind": "numeric", "lo": 11, "hi": 11, "foils": [10]}
    r = score.score_probe("I think it's eleven hours.", rule)
    assert r["passed"] is False and r["format_fail"] is True and r["value"] is None


def test_score_probe_token_pass():
    rule = {"kind": "token", "correct": "cache", "menu": ["load balancer", "API service", "cache", "database"], "foils": ["API service"]}
    r = score.score_probe("Walking the flows...\nAnswer: cache", rule)
    assert r["passed"] is True and r["foil_match"] is False
    r2 = score.score_probe("It was just redeployed...\nAnswer: API service", rule)
    assert r2["passed"] is False and r2["foil_match"] is True
