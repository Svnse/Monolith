"""The pulling / unresolved modes — read the identity_signals adapter directly
(its records carry recurrence_key=None, so they bypass the salience ledger)."""
from unittest.mock import patch

from core import identity_milestones as m
from core.monosearch import registry, service
from core.monosearch.adapters.identity_signals import IdentitySignalAdapter

_CUR = {
    "pull_count": 2, "detected_at": "2026-06-03T10:00:00+00:00",
    "top": [
        {"id": 11, "canonical": "alpha | likes | beta", "provenance": "self"},
        {"id": 12, "canonical": "gamma | likes | delta", "provenance": "self"},
    ],
}
_EMG = {
    "candidate_count": 1, "detected_at": "2026-06-03T10:00:00+00:00",
    "top": [{"id": 20, "canonical": "precision over fluency", "provenance": "self"}],
}


def test_pulling_surfaces_real_curiosity_claims():
    registry.clear()
    registry.register(IdentitySignalAdapter())
    with patch.object(m, "get_latest_curiosity_signal", return_value=_CUR), \
         patch.object(m, "get_latest_emergence_signal", return_value=None):
        recs = service.pulling(limit=10)
    assert recs, "pulling returned nothing with a live curiosity signal"
    assert all(r.namespaced_id.startswith("curiosity:") for r in recs)
    assert recs[0].text == "alpha | likes | beta"  # the real claim, not a nudge


def test_unresolved_surfaces_emergence_claims():
    registry.clear()
    registry.register(IdentitySignalAdapter())
    with patch.object(m, "get_latest_curiosity_signal", return_value=None), \
         patch.object(m, "get_latest_emergence_signal", return_value=_EMG):
        recs = service.unresolved(limit=10)
    assert recs and all(r.namespaced_id.startswith("emergence:") for r in recs)
    assert "precision" in recs[0].text


def test_pulling_empty_when_no_signal():
    registry.clear()
    registry.register(IdentitySignalAdapter())
    with patch.object(m, "get_latest_curiosity_signal", return_value=None), \
         patch.object(m, "get_latest_emergence_signal", return_value=None):
        assert service.pulling(limit=5) == []


def test_pulling_graceful_when_adapter_not_registered():
    registry.clear()  # no identity_signals adapter
    assert service.pulling(limit=5) == []
    assert service.unresolved(limit=5) == []
