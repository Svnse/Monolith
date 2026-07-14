"""Tests for the continuity (pins) adapter — spec §5/§5.1.

Real end-to-end where it counts: continuity's store is a single JSON file, so we
isolate it exactly like tests/test_continuity.py:12 (monkeypatch _STORE_PATH at a
tmp file) and seed REAL pins through continuity.pin()/retire(). The conftest
autouse fixtures isolate turn_trace + salience but NOT continuity, so we add our
own per-test redirect here (conftest is off-limits).
"""
from __future__ import annotations

import pytest

from core import continuity
from core.monosearch.adapters.continuity import ContinuityAdapter
from core.monosearch.record import EvidenceTier, Provenance


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    """Redirect the continuity store to a temp file for each test."""
    store_path = tmp_path / "continuity.json"
    monkeypatch.setattr(continuity, "_STORE_PATH", store_path)
    yield store_path


def test_list_maps_active_pins_to_records(tmp_store):
    p = continuity.pin("keep the user honest", category="lesson")
    a = ContinuityAdapter()
    recs = a.list({}, 50)
    assert len(recs) == 1
    r = recs[0]
    assert r.namespaced_id == f"continuity:{p['id']}"
    assert r.source == "continuity"
    assert r.provenance is Provenance.SELF
    assert r.evidence_tier is EvidenceTier.DERIVED
    assert r.text == "keep the user honest"
    assert r.metadata["active"] is True
    assert r.metadata["category"] == "lesson"
    assert r.ts is not None and r.ts > 0  # parsed from ISO created_at


def test_list_includes_retired_pins_flagged(tmp_store):
    active = continuity.pin("anchor that stays", category="anchor")
    doomed = continuity.pin("lesson to retire", category="lesson")
    continuity.retire(doomed["id"], "stale")

    a = ContinuityAdapter()
    recs = a.list({}, 50)
    by_id = {r.namespaced_id: r for r in recs}

    assert f"continuity:{active['id']}" in by_id
    assert f"continuity:{doomed['id']}" in by_id
    assert by_id[f"continuity:{active['id']}"].metadata["active"] is True
    retired_rec = by_id[f"continuity:{doomed['id']}"]
    assert retired_rec.metadata["active"] is False
    # retired pins carry retired_at — ts must come from it and be set.
    assert retired_rec.ts is not None and retired_rec.ts > 0


def test_get_finds_active_and_retired(tmp_store):
    keeper = continuity.pin("still active", category="pending")
    gone = continuity.pin("now retired", category="lesson")
    continuity.retire(gone["id"], "resolved")

    a = ContinuityAdapter()
    r_active = a.get(f"continuity:{keeper['id']}")
    r_retired = a.get(f"continuity:{gone['id']}")
    assert r_active is not None and r_active.text == "still active"
    assert r_retired is not None and r_retired.text == "now retired"
    assert r_retired.metadata["active"] is False


def test_get_returns_none_for_unknown_or_foreign_id(tmp_store):
    continuity.pin("only pin")
    a = ContinuityAdapter()
    assert a.get("continuity:9999") is None
    assert a.get("fault:1") is None
    assert a.get("continuity:notanint") is None


def test_search_filters_by_query(tmp_store):
    continuity.pin("the model must read its own grades", category="lesson")
    continuity.pin("anchor about the user", category="anchor")
    a = ContinuityAdapter()
    hits = a.search("grades", {}, 50)
    assert len(hits) == 1
    assert "grades" in hits[0].text
    # empty query returns everything
    assert len(a.search("", {}, 50)) == 2


def test_recurrence_key_is_none_for_all_pins(tmp_store):
    """A pin is a UNIQUE self-curated commitment, not a recurring event, so
    recurrence_key is None for every pin (active or retired) — continuity is a
    lookup/search source, never salience-eligible. Hashing the text would yield
    all-count-1 keys and only pollute the `recurring` selector."""
    text = "I keep forgetting to verify before claiming done"
    first = continuity.pin(text, category="lesson")
    continuity.retire(first["id"], "stale")
    second = continuity.pin(text, category="lesson")  # identical re-pin

    a = ContinuityAdapter()
    recs = {r.namespaced_id: r for r in a.list({}, 50)}
    assert recs[f"continuity:{first['id']}"].recurrence_key is None
    assert recs[f"continuity:{second['id']}"].recurrence_key is None
    # distinct text — still None
    other = continuity.pin("a different lesson entirely", category="lesson")
    rec_other = a.get(f"continuity:{other['id']}")
    assert rec_other.recurrence_key is None


def test_list_retrieves_all_retired_not_just_default_five(tmp_store):
    """read() defaults retired_limit=5; the adapter must pass retired_limit=0 so
    a faithful map exposes every retired pin (store-capped at 16)."""
    ids = []
    for i in range(8):
        p = continuity.pin(f"lesson {i}", category="lesson")
        continuity.retire(p["id"], "stale")
        ids.append(p["id"])
    a = ContinuityAdapter()
    recs = a.list({}, 50)
    retired_ids = {r.namespaced_id for r in recs if r.metadata["active"] is False}
    # all 8 retired pins surface, not just the last 5
    for pid in ids:
        assert f"continuity:{pid}" in retired_ids
