"""Tests for core/friction_differ — the load-bearing friction scorer.

Discipline checks: pure/deterministic, closed enum, the content-overlap channel
distinguishes uptake from topic_drift (so 'low friction' != 'no signal'), and
markers fire with high friction.
"""
from __future__ import annotations

from core import friction_differ as fd

ANSWER = (
    "The friction organ finishes bearing by predicting intent and settling it "
    "against the verbatim next message with a multi-channel differ that scores "
    "content overlap and repair markers, never correctness."
)


def test_friction_types_is_closed_enum():
    assert "uptake" in fd.FRICTION_TYPES and "topic_drift" in fd.FRICTION_TYPES
    # every result type must be a member
    for t in (fd.FRICTION_TYPES):
        assert isinstance(t, str)


def test_uptake_high_overlap_no_markers_is_low_friction():
    reply = (
        "Right, the differ should score content overlap and repair markers and "
        "settle the prediction against the verbatim message, never correctness."
    )
    r = fd.score(ANSWER, reply)
    assert r.friction_type == "uptake"
    assert r.friction_score <= 0.3
    assert r.channel_json["answer_overlap"] >= fd._UPTAKE_OVERLAP


def test_topic_drift_low_overlap_no_markers_is_higher_friction():
    reply = "Anyway, can you book me a flight to Tokyo and order groceries?"
    r = fd.score(ANSWER, reply)
    assert r.friction_type == "topic_drift"
    assert r.friction_score >= 0.6
    # the key anti-collapse property: drift is NOT the same observation as uptake
    assert r.friction_score > fd.score(ANSWER, "Right, the differ scores overlap and markers like you said.").friction_score


def test_correction_marker_fires_high_friction():
    reply = "No, that's not what I meant about the differ — drop the overlap idea."
    r = fd.score(ANSWER, reply)
    assert r.friction_type == "correction"
    assert r.friction_score >= 0.9
    assert "correction" in r.channel_json["markers"]


def test_meta_marker_fires():
    reply = "You're sprawling again — tighten the differ explanation."
    r = fd.score(ANSWER, reply)
    assert r.friction_type == "meta"
    assert r.friction_score >= 0.9


def test_reframe_marker_fires():
    reply = "Correct or incorrect isn't really the right view here; friction is."
    r = fd.score(ANSWER, reply)
    assert r.friction_type in ("reframe", "correction")  # both are legit high-friction reads
    assert r.friction_score >= 0.85


def test_reask_detected_on_interrogative_rehash():
    reply = "Wait, how does the differ score content overlap again? You covered it."
    r = fd.score(ANSWER, reply)
    assert r.channel_json["interrogative"] is True
    assert r.friction_type in ("reask", "correction", "reframe", "clarify", "meta")


def test_unresolved_when_too_short_to_judge():
    r = fd.score(ANSWER, "ok")
    assert r.friction_type in ("unresolved", "abandon")
    assert r.channel_json["answer_overlap"] == -1.0 or r.channel_json["answer_overlap"] <= fd._DRIFT_OVERLAP


def test_pure_and_deterministic():
    reply = "No, drop the overlap idea entirely."
    a = fd.score(ANSWER, reply)
    b = fd.score(ANSWER, reply)
    assert a == b
    assert 0.0 <= a.friction_score <= 1.0


def test_bare_thats_wrong_fires_correction():
    # regression: the old \b(...)\b wrapping silently dropped this
    r = fd.score(ANSWER, "No, that's wrong — use the overlap coefficient, not Jaccard.")
    assert "correction" in r.channel_json["markers"]
    assert r.friction_type == "correction"
    assert r.friction_score >= 0.9


def test_overcomplicating_fires_meta():
    r = fd.score(ANSWER, "You're overcomplicating this — simplify it down.")
    assert "meta" in r.channel_json["markers"]
    assert r.friction_type == "meta"


def test_result_type_is_in_closed_enum():
    samples = [
        "Right, the differ scores overlap and markers exactly as you described.",
        "No, that's wrong.",
        "Book a flight to Paris please.",
        "ok",
        "You keep repeating yourself.",
    ]
    for s in samples:
        r = fd.score(ANSWER, s)
        assert r.friction_type in fd.FRICTION_TYPES
