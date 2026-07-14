"""Salience ledger store. Isolation (own temp DB + connection close) is provided
by the autouse `isolated_salience_db` fixture in conftest.py."""
from core.monosearch import salience
from core.monosearch.record import Provenance


def test_record_observation_counts_and_tracks_first_last():
    salience.record_observation("k1", Provenance.SELF, "fault_traces", ts=100.0)
    salience.record_observation("k1", Provenance.SELF, "fault_traces", ts=300.0)
    row = salience.get_row("k1", "fault_traces")
    assert row["count"] == 2
    assert row["first_seen"] == 100.0
    assert row["last_seen"] == 300.0


def test_distinct_keys_are_separate_rows():
    salience.record_observation("a", Provenance.SELF, "fault_traces", ts=1.0)
    salience.record_observation("b", Provenance.SELF, "fault_traces", ts=1.0)
    assert salience.get_row("a", "fault_traces")["count"] == 1
    assert salience.get_row("b", "fault_traces")["count"] == 1


def test_none_recurrence_key_is_ignored():
    salience.record_observation(None, Provenance.SELF, "canonical_log", ts=1.0)
    assert salience.get_row("anything", "canonical_log") is None
