"""Tests for core.intervention_arena.buckets."""
from __future__ import annotations

from core.intervention_arena.buckets import (
    BUCKET_REGISTRY,
    BucketSignature,
    evaluate,
)
from core.turn_shape import TurnShape


def _shape(
    effort_tier="med",
    complexity_score=50,
    intent_tags=("chat",),
    task_type="conversation",
    confidence=0.8,
) -> TurnShape:
    return TurnShape(
        effort_tier=effort_tier,
        complexity_score=complexity_score,
        intent_tags=tuple(intent_tags),
        task_type=task_type,
        confidence=confidence,
    )


def test_analysis_high_plus_matches():
    sig = BUCKET_REGISTRY["analysis_high_plus"]
    s = _shape(effort_tier="ultimate", intent_tags=("plan", "review"), task_type="analysis")
    assert evaluate(s, sig) is True


def test_analysis_high_plus_rejects_wrong_task_type():
    sig = BUCKET_REGISTRY["analysis_high_plus"]
    s = _shape(effort_tier="ultimate", intent_tags=("plan",), task_type="action")
    assert evaluate(s, sig) is False


def test_analysis_high_plus_rejects_low_effort():
    sig = BUCKET_REGISTRY["analysis_high_plus"]
    s = _shape(effort_tier="med", intent_tags=("analysis",), task_type="analysis")
    assert evaluate(s, sig) is False


def test_action_with_tools_matches():
    sig = BUCKET_REGISTRY["action_with_tools"]
    s = _shape(intent_tags=("debug",), task_type="action")
    assert evaluate(s, sig) is True


def test_action_with_tools_rejects_no_overlap():
    sig = BUCKET_REGISTRY["action_with_tools"]
    s = _shape(intent_tags=("chat",), task_type="action")
    assert evaluate(s, sig) is False


def test_attribution_calibration_accepts_minimum_confidence():
    sig = BUCKET_REGISTRY["attribution_calibration"]
    s = _shape(confidence=0.5)
    assert evaluate(s, sig) is True


def test_attribution_calibration_rejects_low_confidence():
    sig = BUCKET_REGISTRY["attribution_calibration"]
    s = _shape(confidence=0.4)
    assert evaluate(s, sig) is False


def test_signature_serializes_to_json():
    sig = BUCKET_REGISTRY["analysis_high_plus"]
    payload = sig.to_json()
    restored = BucketSignature.from_json(payload)
    assert restored == sig


def test_complexity_min_accepts_at_boundary():
    sig = BucketSignature(name="t", complexity_min=50)
    s = _shape(complexity_score=50)
    assert evaluate(s, sig) is True


def test_complexity_min_rejects_below_boundary():
    sig = BucketSignature(name="t", complexity_min=50)
    s = _shape(complexity_score=49)
    assert evaluate(s, sig) is False


def test_complexity_max_accepts_at_boundary():
    sig = BucketSignature(name="t", complexity_max=50)
    s = _shape(complexity_score=50)
    assert evaluate(s, sig) is True


def test_complexity_max_rejects_above_boundary():
    sig = BucketSignature(name="t", complexity_max=50)
    s = _shape(complexity_score=51)
    assert evaluate(s, sig) is False
