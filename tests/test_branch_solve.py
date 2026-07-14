"""Tests for the BRANCH solve pass (frame construction + orchestration, mocked)."""
from __future__ import annotations

from core import branch_solve as bs


def test_frame_instruction_carries_gloss_approach_and_type():
    f = bs.frame_instruction("worst_case_bound")
    assert f is not None
    assert "worst_case_bound" in f
    assert "every admissible arrangement" in f          # the gloss (coordinate)
    assert "Approach:" in f and "adversarial" in f       # the approach (the steer)
    assert f.strip().startswith("REASONING FRAME")       # top-placed
    assert "Answer:" in f                                # answer-line contract


def test_frame_instruction_none_for_other_or_unknown():
    assert bs.frame_instruction(None) is None
    assert bs.frame_instruction("other:spectral magic") is None
    assert bs.frame_instruction("not_a_real_type") is None


def test_branch_turn_classifies_then_frames_then_solves():
    seen = {}
    def classify_call(prompt):
        return "TYPE: worst_case_bound"
    def solve_call(task, frame):
        seen["task"] = task
        seen["frame"] = frame
        return "pigeonhole...\nAnswer: 25"
    out = bs.branch_turn("a shard guarantee question",
                         classify_call=classify_call, solve_call=solve_call)
    assert out["type"] == "worst_case_bound"
    assert "worst_case_bound" in out["frame"]            # frame built from the classified type
    assert out["answer"].endswith("Answer: 25")
    assert seen["task"] == "a shard guarantee question"  # original task, unaltered, to the solver


def test_branch_turn_unplaceable_type_solves_unframed():
    def classify_call(prompt):
        return "TYPE: other: vibes"
    captured = {}
    def solve_call(task, frame):
        captured["frame"] = frame
        return "Answer: 42"
    out = bs.branch_turn("weird task", classify_call=classify_call, solve_call=solve_call)
    assert bs.bc.pt.is_other(out["type"])
    assert out["frame"] is None
    assert captured["frame"] is None                     # solver ran unframed (baseline)
