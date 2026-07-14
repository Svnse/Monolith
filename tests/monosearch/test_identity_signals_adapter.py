"""Tests for the identity_signals adapter (spec §5.2 — the `pulling` source).

Surfaces the CURRENT curiosity/emergence signals from the identity_milestones
ledger as Records — one per top-pull / top-candidate, with the per-item CANONICAL
CLAIM as the text (NOT the aggregate count message, which would just be a nudge).

The ledger getters read the module global STORE_PATH at call time, so monkeypatch
+ a real set_latest_*_signal write is the real read path (load_ledger -> JSON).
The conftest autouse only isolates turn_trace + salience, NOT identity_milestones,
so each test points STORE_PATH at tmp_path itself.
"""
from __future__ import annotations

import pytest

from core import identity_milestones as _m
from core.monosearch.adapters.identity_signals import IdentitySignalAdapter
from core.monosearch.record import EvidenceTier, Provenance


@pytest.fixture(autouse=True)
def tmp_ledger(monkeypatch, tmp_path):
    """Isolate the identity_milestones ledger per test (conftest does not)."""
    monkeypatch.setattr(_m, "STORE_PATH", tmp_path / "identity_milestones.json")
    yield


_CURIOSITY_SIGNAL = {
    "detected_at": "2026-06-02T18:00:00+00:00",
    "pull_count": 2,
    "top": [
        {"id": 42, "canonical": "I am drawn to non-performance",
         "pull_strength": 0.31, "confidentity": 0.44, "stability": 0.3,
         "provenance": "self"},
        {"id": 17, "canonical": "I keep returning to the read-side blindness",
         "pull_strength": 0.12, "confidentity": 0.4, "stability": 0.7,
         "provenance": "self"},
    ],
    "message": "2 curiosity pull(s) — fresh, identity-aligned claims.",
}

_EMERGENCE_SIGNAL = {
    "detected_at": "2026-06-02T19:00:00+00:00",
    "new_acu_count": 9,
    "candidate_count": 1,
    "threshold": 0.2,
    "top": [
        {"id": 88, "canonical": "I treat the LLM as the worst component",
         "confidentity": 0.41, "provenance": "self", "reinforcement": 3},
    ],
    "message": "1 self-derived claim(s) at or above 0.2 confidentity.",
}


def test_search_surfaces_canonical_claim_not_the_nudge_message():
    _m.set_latest_curiosity_signal(_CURIOSITY_SIGNAL)
    a = IdentitySignalAdapter()
    recs = a.search("", {}, 10)
    # one record per top-pull
    cur = [r for r in recs if r.namespaced_id.startswith("curiosity:")]
    assert len(cur) == 2
    top = cur[0]
    assert top.namespaced_id == "curiosity:42"
    assert top.source == "identity_signals"
    # The TEXT is the per-pull canonical claim — the real signal CONTENT, not the
    # "2 curiosity pull(s)" count-summary nudge.
    assert top.text == "I am drawn to non-performance"
    assert "curiosity pull(s)" not in top.text
    assert top.provenance is Provenance.SELF
    assert top.evidence_tier == EvidenceTier.DERIVED
    assert top.recurrence_key is None  # current-state signal, not recurrence
    assert top.ts == 1780423200.0  # 2026-06-02T18:00:00Z -> epoch
    # aggregate context preserved in metadata, nothing lost
    assert top.metadata["pull_count"] == 2
    assert top.metadata["signal_message"] == _CURIOSITY_SIGNAL["message"]
    assert top.metadata["signal_kind"] == "curiosity"


def test_search_preserves_top_ordering():
    _m.set_latest_curiosity_signal(_CURIOSITY_SIGNAL)
    a = IdentitySignalAdapter()
    cur = [r for r in a.search("", {}, 10) if r.namespaced_id.startswith("curiosity:")]
    assert [r.namespaced_id for r in cur] == ["curiosity:42", "curiosity:17"]


def test_search_includes_emergence_candidates():
    _m.set_latest_curiosity_signal(_CURIOSITY_SIGNAL)
    _m.set_latest_emergence_signal(_EMERGENCE_SIGNAL)
    a = IdentitySignalAdapter()
    recs = a.search("", {}, 10)
    em = [r for r in recs if r.namespaced_id.startswith("emergence:")]
    assert len(em) == 1
    assert em[0].namespaced_id == "emergence:88"
    assert em[0].text == "I treat the LLM as the worst component"
    assert em[0].metadata["signal_kind"] == "emergence"
    assert em[0].ts == 1780426800.0  # 2026-06-02T19:00:00Z -> epoch


def test_search_keyword_filters_on_text():
    _m.set_latest_curiosity_signal(_CURIOSITY_SIGNAL)
    a = IdentitySignalAdapter()
    recs = a.search("non-performance", {}, 10)
    assert len(recs) == 1
    assert recs[0].namespaced_id == "curiosity:42"


def test_get_roundtrips_by_acu_id():
    _m.set_latest_curiosity_signal(_CURIOSITY_SIGNAL)
    a = IdentitySignalAdapter()
    r = a.get("curiosity:17")
    assert r is not None
    assert r.text == "I keep returning to the read-side blindness"
    # wrong namespace / unknown id -> None
    assert a.get("fault:17") is None
    assert a.get("curiosity:9999") is None


def test_list_is_the_rebuild_iteration_path():
    _m.set_latest_curiosity_signal(_CURIOSITY_SIGNAL)
    _m.set_latest_emergence_signal(_EMERGENCE_SIGNAL)
    a = IdentitySignalAdapter()
    recs = a.list({}, 100)
    # all current signals surfaced; none salience-eligible (current-state, not recurrence)
    assert len(recs) == 3
    assert all(r.recurrence_key is None for r in recs)


def test_empty_when_no_signal():
    a = IdentitySignalAdapter()
    assert a.search("", {}, 10) == []
    assert a.list({}, 10) == []
    assert a.get("curiosity:42") is None


def test_recurrence_key_is_always_none():
    a = IdentitySignalAdapter()
    pull = {"id": 1, "canonical": "x"}
    assert a._recurrence_key(pull) is None


def test_provenance_maps_real_item_provenance():
    # The item's OWN provenance is mapped self/user/world -> SELF/USER/WORLD,
    # not hardcoded SELF. Absent / unknown defaults to SELF.
    a = IdentitySignalAdapter()
    assert a._provenance({"provenance": "self"}) is Provenance.SELF
    assert a._provenance({"provenance": "user"}) is Provenance.USER
    assert a._provenance({"provenance": "world"}) is Provenance.WORLD
    assert a._provenance({"provenance": "WORLD"}) is Provenance.WORLD  # case-insensitive
    assert a._provenance({}) is Provenance.SELF  # absent -> SELF
    assert a._provenance({"provenance": "bogus"}) is Provenance.SELF  # unknown -> SELF


def test_search_record_carries_mapped_provenance():
    # End-to-end through a real signal write: a user-provenance pull surfaces as
    # Provenance.USER on the Record, not SELF.
    signal = {
        "detected_at": "2026-06-02T18:00:00+00:00",
        "pull_count": 1,
        "top": [
            {"id": 7, "canonical": "E asked me to stop performing",
             "pull_strength": 0.3, "confidentity": 0.4, "stability": 0.2,
             "provenance": "user"},
        ],
        "message": "1 curiosity pull.",
    }
    _m.set_latest_curiosity_signal(signal)
    a = IdentitySignalAdapter()
    rec = a.get("curiosity:7")
    assert rec is not None
    assert rec.provenance is Provenance.USER
