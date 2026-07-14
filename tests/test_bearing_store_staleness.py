"""Tests for the pending_staleness persistence API in addons/system/bearing/store.py.

Mirrors the pending_rejection trio. Must coexist with (never clobber) rejection
state and the bearing snapshot.
"""
from __future__ import annotations

import pytest

from addons.system.bearing import schema as bs
from addons.system.bearing import store


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    yield tmp_path


def test_pending_staleness_defaults_to_none(tmp_store) -> None:
    assert store.get_pending_staleness() is None


def test_set_and_get_pending_staleness_round_trip(tmp_store) -> None:
    store.set_pending_staleness({"signal_id": "channel:user", "streak": 2})
    assert store.get_pending_staleness() == {"signal_id": "channel:user", "streak": 2}


def test_clear_pending_staleness(tmp_store) -> None:
    store.set_pending_staleness({"signal_id": "channel:user", "streak": 2})
    store.clear_pending_staleness()
    assert store.get_pending_staleness() is None


def test_pending_staleness_coexists_with_pending_rejection(tmp_store) -> None:
    store.set_pending_rejection(["D1"], turn_id="t1", ts="2026-06-06T00:00:00+00:00")
    store.set_pending_staleness({"signal_id": "channel:user", "streak": 1})
    rej = store.get_pending_rejection()
    assert rej is not None and rej["failed_rules"] == ["D1"]
    assert store.get_pending_staleness() == {"signal_id": "channel:user", "streak": 1}


def test_clear_pending_staleness_leaves_rejection_intact(tmp_store) -> None:
    store.set_pending_rejection(["D1"], turn_id="t1", ts="2026-06-06T00:00:00+00:00")
    store.set_pending_staleness({"signal_id": "channel:user", "streak": 1})
    store.clear_pending_staleness()
    assert store.get_pending_staleness() is None
    assert store.get_pending_rejection() is not None


def test_set_bearing_preserves_pending_staleness(tmp_store) -> None:
    store.set_pending_staleness({"signal_id": "channel:user", "streak": 1})
    store.set_bearing(bs.Bearing(active_goal="g"))
    assert store.get_pending_staleness() == {"signal_id": "channel:user", "streak": 1}
