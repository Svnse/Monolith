"""OutcomeTraceAdapter — the ratings-ledger source for monosearch.

Regression home for "monosearch premise -> 0": premise_unchecked lives in
outcome_traces, which no adapter read. These tests prove the adapter maps rows
correctly and that a partial-keyword search surfaces the tagged rating.
"""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from core import turn_trace as tt
from core.monosearch.adapters.outcome_traces import OutcomeTraceAdapter
from core.monosearch.record import EvidenceTier, Provenance


@pytest.fixture(autouse=True)
def _stub_monothink(monkeypatch):
    """record_outcome fires the monothink evolution hook on a tagged rating;
    stub it so these adapter tests stay hermetic (no DeepSeek call)."""
    monkeypatch.setattr(
        "core.monothink.maybe_evolve_after_rating", lambda *a, **k: None, raising=False
    )


def _row(**kw):
    base = dict(
        id=7, turn_id="t1", recorded_at="2026-06-05T16:00:00+00:00",
        kind="rating", rating_value=48,
        reason="Reasoning-failure(s) flagged — [premise_unchecked] a premise was used "
               "without being compared against the evidence present in the turn.",
        metadata={"failure_tags": ["premise_unchecked"], "surface_note": "conceded"},
    )
    base.update(kw)
    return tt.OutcomeReadRow(**base)


def test_to_record_shape():
    a = OutcomeTraceAdapter()
    r = a._to_record(_row())
    assert r.namespaced_id == "outcome:7"
    assert r.source == "outcome_traces"
    assert r.provenance is Provenance.USER          # external evaluation, not self-detection
    assert r.evidence_tier == EvidenceTier.DERIVED  # a graded judgment, under LITERAL faults
    assert r.recurrence_key == "premise_unchecked"
    assert "premise_unchecked" in r.text            # tag visible to the model
    assert r.metadata["failure_tags"] == ["premise_unchecked"]
    assert r.metadata["rating_value"] == 48
    assert r.ts == datetime(2026, 6, 5, 16, 0, tzinfo=timezone.utc).timestamp()


def test_recurrence_key_is_primary_failure_tag():
    a = OutcomeTraceAdapter()
    row = _row(metadata={"failure_tags": ["premise_unchecked", "missing_branch_pressure"]})
    assert a._recurrence_key(row) == "premise_unchecked"


def test_untagged_rating_is_not_salience_eligible():
    # A clean/positive rating (no failure_tag) is not a recurring failure -> key None.
    a = OutcomeTraceAdapter()
    r = a._to_record(_row(rating_value=88, reason="", metadata={}))
    assert r.recurrence_key is None


def test_search_finds_failure_tag_by_partial_keyword():
    # THE BUG: searching the partial word 'premise' surfaces the premise_unchecked
    # rating from the outcome_traces ledger (and not a clean rating).
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
    a = OutcomeTraceAdapter()
    recs = a.search("premise", {}, 10)
    assert [r.metadata["turn_id"] for r in recs] == ["t_bad"]
    assert recs[0].recurrence_key == "premise_unchecked"


def test_get_parses_namespaced_id():
    a = OutcomeTraceAdapter()
    with patch.object(tt, "read_outcome", return_value=_row()) as m:
        r = a.get("outcome:7")
    m.assert_called_once_with(7)
    assert r.namespaced_id == "outcome:7"
    assert a.get("fault:7") is None  # wrong namespace -> declined
