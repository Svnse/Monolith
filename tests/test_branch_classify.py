"""Tests for the BRANCH classify pass (parse logic, mocked model)."""
from __future__ import annotations

from core import branch_classify as bc


def test_parse_clean_type_line():
    assert bc.parse_type("TYPE: worst_case_bound") == "worst_case_bound"


def test_parse_takes_last_type_line_after_thinking():
    raw = "let me think...\nTYPE: central_tendency_estimation\n(reconsider)\nTYPE: order_statistic_estimation"
    assert bc.parse_type(raw) == "order_statistic_estimation"


def test_parse_other_when_unlisted():
    out = bc.parse_type("TYPE: other: spectral decomposition")
    assert bc.pt.is_other(out)


def test_parse_falls_back_to_whole_text():
    assert bc.parse_type("I'd call this eliminative_deduction") == "eliminative_deduction"


def test_parse_empty_is_none():
    assert bc.parse_type("") is None


def test_classify_uses_injected_call():
    captured = {}
    def fake(prompt):
        captured["prompt"] = prompt
        return "TYPE: aggregate_ratio_composition"
    out = bc.classify("some rate question", call=fake)
    assert out == "aggregate_ratio_composition"
    # prompt carries the closed menu at the top (the steer placement)
    assert "CLOSED LIST:" in captured["prompt"]
    assert "aggregate_ratio_composition" in captured["prompt"]
    assert captured["prompt"].index("CLOSED LIST:") < captured["prompt"].index("TASK:")


def test_classify_none_call_result_is_none():
    assert bc.classify("x", call=lambda p: None) is None
