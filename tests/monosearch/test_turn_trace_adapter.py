"""Tests for the turn_trace adapter (one Record per TURN).

Unit tests hand-build TurnTraceJoined objects to exercise pure mapping
(_to_record / _recurrence_key / _provenance / frame-None guard). One real
end-to-end seeds the REAL turn_trace store (record_frame / record_outcome
under the autouse-isolated DB from conftest) and asserts get/search/list hit
the real read path.
"""
from __future__ import annotations

from unittest.mock import patch

import core.turn_trace as tt
from core.monosearch.adapters.turn_trace import TurnTraceAdapter
from core.monosearch.record import EvidenceTier, Provenance


def _joined(
    turn_id: str = "abc-123",
    *,
    with_frame: bool = True,
    effort_tier: str | None = "deliberate",
    classification: dict | None = None,
    last_rating: int | None = None,
    outcomes: tuple = (),
):
    frame = None
    if with_frame:
        frame = tt.FrameTraceRecord(
            turn_id=turn_id,
            captured_at="2026-06-03T12:00:00+00:00",
            backend="test-backend",
            engine_key="ek",
            gen_id=1,
            final_messages=(),
            system_prompt_chars=10,
            user_prompt_chars=20,
            total_chars=30,
            effort_tier=effort_tier,
            classification=classification,
        )
    summary = {
        "stage_count": 0,
        "errored_stage_count": 0,
        "total_chars": frame.total_chars if frame is not None else 0,
        "frame_present": frame is not None,
        "outcome_count": len(outcomes),
        "last_rating": last_rating,
        "effort_tier": effort_tier if frame is not None else None,
    }
    return tt.TurnTraceJoined(
        turn_id=turn_id,
        parent_turn_id=None,
        stages=(),
        frame=frame,
        outcomes=outcomes,
        summary=summary,
    )


# ── pure mapping ────────────────────────────────────────────────────


def test_to_record_basic_shape():
    a = TurnTraceAdapter()
    r = a._to_record(_joined("turn-xyz", effort_tier="careful"))
    assert r.namespaced_id == "turn:turn-xyz"
    assert r.source == "turn_trace"
    assert r.provenance is Provenance.SELF
    assert r.evidence_tier == EvidenceTier.LITERAL
    # Each turn is unique — lookup-only source, not a recurrence source.
    assert r.recurrence_key is None
    assert r.metadata["effort_tier"] == "careful"
    # ts parsed from captured_at ISO -> epoch float
    assert isinstance(r.ts, float)


def test_classification_included_only_when_dict():
    a = TurnTraceAdapter()
    cls = {"conversation_mode": "build", "intent": "implement"}
    r = a._to_record(_joined(classification=cls))
    assert r.metadata["classification"] == cls

    r2 = a._to_record(_joined(classification=None))
    assert "classification" not in r2.metadata


def test_rating_from_summary_last_rating():
    a = TurnTraceAdapter()
    r = a._to_record(_joined(last_rating=72))
    assert r.metadata["rating"] == 72


def test_frame_none_does_not_crash_and_falls_back_for_ts():
    a = TurnTraceAdapter()
    oc = tt.OutcomeTraceRecord(
        turn_id="sub-only",
        recorded_at="2026-06-03T09:30:00+00:00",
        kind="thumbs_up",
    )
    joined = _joined("sub-only", with_frame=False, outcomes=(oc,))
    r = a._to_record(joined)  # must not AttributeError on frame=None
    assert r.namespaced_id == "turn:sub-only"
    assert "effort_tier" not in r.metadata
    # ts falls back to the outcome's recorded_at
    assert isinstance(r.ts, float)


# ── routing ─────────────────────────────────────────────────────────


def test_get_routes_through_get_turn_trace():
    a = TurnTraceAdapter()
    with patch.object(tt, "get_turn_trace", return_value=_joined("zz")) as m:
        r = a.get("turn:zz")
    m.assert_called_once_with("zz")
    assert r.namespaced_id == "turn:zz"
    assert a.get("fault:5") is None  # wrong-prefix is gated out


def test_get_returns_none_when_store_misses():
    a = TurnTraceAdapter()
    with patch.object(tt, "get_turn_trace", return_value=None):
        assert a.get("turn:nope") is None


def test_search_keyword_with_no_filters_returns_empty():
    # turn_trace has no free-text index — a non-empty query with no filters
    # returns [] rather than keyword-irrelevant recent turns. It must NOT touch
    # the underlying reads at all in that case.
    a = TurnTraceAdapter()
    with patch.object(tt, "list_recent_turns") as recent, \
         patch.object(tt, "search_turns") as searched:
        assert a.search("somekeyword", {}, 5) == []
        recent.assert_not_called()
        searched.assert_not_called()


# ── real end-to-end against the isolated turn_trace store ───────────


def test_real_end_to_end_get_search_list():
    """Seed the REAL store via record_frame/record_outcome; assert all three
    public methods hit the real read path (conftest isolates the DB)."""
    turn_id = "real-turn-0001"
    tt.record_frame(tt.FrameTraceRecord(
        turn_id=turn_id,
        captured_at="2026-06-03T15:00:00+00:00",
        backend="ollama",
        engine_key="ek1",
        gen_id=1,
        final_messages=(),
        system_prompt_chars=100,
        user_prompt_chars=50,
        total_chars=150,
        effort_tier="deliberate",
        classification={"conversation_mode": "chat"},
    ))
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id=turn_id,
        recorded_at="2026-06-03T15:01:00+00:00",
        kind="rating",
        rating_value=88,
    ))

    a = TurnTraceAdapter()

    # get()
    g = a.get(f"turn:{turn_id}")
    assert g is not None
    assert g.namespaced_id == f"turn:{turn_id}"
    assert g.metadata["effort_tier"] == "deliberate"
    assert g.metadata["classification"] == {"conversation_mode": "chat"}
    assert g.metadata["rating"] == 88
    assert g.metadata["backend"] == "ollama"
    assert isinstance(g.ts, float)

    # search() (no filters -> list_recent_turns under the hood)
    hits = a.search("", {}, 10)
    assert any(r.namespaced_id == f"turn:{turn_id}" for r in hits)
    hit = next(r for r in hits if r.namespaced_id == f"turn:{turn_id}")
    assert hit.metadata["effort_tier"] == "deliberate"
    assert hit.metadata["rating"] == 88

    # search() WITH an ISO `since=` filter -> exercises the real search_turns
    # branch (the router-integration path, §6). Cutoff is before captured_at.
    since_hits = a.search("", {"since": "2026-06-03T14:00:00+00:00"}, 10)
    assert any(r.namespaced_id == f"turn:{turn_id}" for r in since_hits)
    since_hit = next(r for r in since_hits if r.namespaced_id == f"turn:{turn_id}")
    assert since_hit.metadata["effort_tier"] == "deliberate"
    assert since_hit.metadata["rating"] == 88
    # A cutoff AFTER captured_at filters the turn out.
    assert not a.search("", {"since": "2026-06-03T16:00:00+00:00"}, 10)

    # list() (the salience.rebuild iteration path)
    listed = a.list({}, 10)
    assert any(r.namespaced_id == f"turn:{turn_id}" for r in listed)
    # every turn record is unique -> not salience-eligible
    assert all(r.recurrence_key is None for r in listed)


def test_real_substrate_only_turn_has_frame_none():
    """A turn with an outcome but NO frame yields a joined with frame=None
    (turn_trace.py:1313-1316). Confirms the real read produces the shape the
    _to_record frame-None guard handles — get() must not crash and ts falls
    back to the outcome's recorded_at."""
    turn_id = "substrate-only-0002"
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id=turn_id,
        recorded_at="2026-06-03T10:00:00+00:00",
        kind="thumbs_down",
    ))
    a = TurnTraceAdapter()
    g = a.get(f"turn:{turn_id}")
    assert g is not None
    assert g.namespaced_id == f"turn:{turn_id}"
    assert "effort_tier" not in g.metadata  # no frame -> no frame-only fields
    assert "backend" not in g.metadata
    assert isinstance(g.ts, float)  # fell back to the outcome's recorded_at
