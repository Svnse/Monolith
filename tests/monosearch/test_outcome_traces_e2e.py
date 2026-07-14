"""End-to-end regression lock for "monosearch premise -> 0".

Proves the outcome_traces ledger is reachable through the real service+router+
registry path: a partial-keyword search scoped to ratings surfaces the
premise_unchecked rating, and rated failures flow into the salience ledger
(recurring -> monopulse hotspots).
"""
import pytest

from core import turn_trace as tt
from core.monosearch import bootstrap as ms_bootstrap, registry, service
from core.monosearch.adapters.outcome_traces import OutcomeTraceAdapter


@pytest.fixture(autouse=True)
def _stub_monothink(monkeypatch):
    monkeypatch.setattr(
        "core.monothink.maybe_evolve_after_rating", lambda *a, **k: None, raising=False
    )


def _seed_two_ratings() -> None:
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="t_bad", recorded_at="2026-06-05T00:00:00+00:00", kind="rating",
        rating_value=48,
        reason="Reasoning-failure(s) flagged — [premise_unchecked] a premise was used "
               "without being compared against the evidence present in the turn.",
        metadata={"failure_tags": ["premise_unchecked"]},
    ))
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="t_ok", recorded_at="2026-06-05T01:00:00+00:00", kind="rating",
        rating_value=88, reason="", metadata={},
    ))


def test_service_search_premise_finds_rating_scoped_to_ratings():
    # Full boot wiring + scoped search (router fans only to outcome_traces).
    registry.clear()
    ms_bootstrap.init_monosearch()
    _seed_two_ratings()
    recs = service.search("premise", {"source": "ratings"}, limit=10)
    assert recs, "regression: 'monosearch premise' still returns 0 from the ratings ledger"
    assert all(r.source == "outcome_traces" for r in recs)
    assert any(r.recurrence_key == "premise_unchecked" for r in recs)


def test_recurring_surfaces_rated_failure_tag():
    # Rated failures must reach the salience ledger so recurring (and monopulse)
    # can see them. Register only the outcome adapter to keep the rebuild hermetic.
    registry.clear()
    registry.register(OutcomeTraceAdapter())
    _seed_two_ratings()
    keys = {row["recurrence_key"] for row in service.recurring(limit=50)}
    assert "premise_unchecked" in keys
