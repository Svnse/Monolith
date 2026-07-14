from __future__ import annotations

import pytest

from addons.system.bearing import schema as bs
from addons.system.bearing import store


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    """Redirect the bearing store to a temp file per test."""
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    yield tmp_path


# ── empty / missing-file behavior ────────────────────────────────────


def test_get_bearing_returns_empty_when_file_missing(tmp_store) -> None:
    b = store.get_bearing()
    assert b == bs.Bearing()
    assert b.is_empty()


def test_get_pending_rejection_returns_none_when_file_missing(tmp_store) -> None:
    assert store.get_pending_rejection() is None


def test_get_rejection_streak_zero_when_file_missing(tmp_store) -> None:
    assert store.get_rejection_streak() == 0


# ── set / get roundtrip ──────────────────────────────────────────────


def test_set_then_get_bearing_roundtrips(tmp_store) -> None:
    b = bs.Bearing(active_goal="x", current_frame="y", updated_at_turn="t1")
    store.set_bearing(b)
    assert store.get_bearing() == b


def test_set_pending_rejection_roundtrip(tmp_store) -> None:
    store.set_pending_rejection(["D1", "D3"], turn_id="t1", ts="2026-05-20T00:00:00+00:00")
    pr = store.get_pending_rejection()
    assert pr is not None
    assert pr["failed_rules"] == ["D1", "D3"]
    assert pr["turn_id"] == "t1"


def test_clear_pending_rejection(tmp_store) -> None:
    store.set_pending_rejection(["D1"], turn_id="t1", ts="2026-05-20T00:00:00+00:00")
    store.clear_pending_rejection()
    assert store.get_pending_rejection() is None


# ── rejection streak ─────────────────────────────────────────────────


def test_increment_rejection_streak_returns_new_value(tmp_store) -> None:
    assert store.increment_rejection_streak() == 1
    assert store.increment_rejection_streak() == 2
    assert store.increment_rejection_streak() == 3
    assert store.get_rejection_streak() == 3


def test_reset_rejection_streak_zeros_value(tmp_store) -> None:
    store.increment_rejection_streak()
    store.increment_rejection_streak()
    store.reset_rejection_streak()
    assert store.get_rejection_streak() == 0


# ── clear_bearing preserves rejection state ──────────────────────────


def test_clear_bearing_resets_bearing_only(tmp_store) -> None:
    store.set_bearing(bs.Bearing(active_goal="x"))
    store.set_pending_rejection(["D1"], turn_id="t1", ts="ts1")
    store.increment_rejection_streak()
    store.clear_bearing()
    assert store.get_bearing().is_empty()
    # pending_rejection and streak are not touched
    assert store.get_pending_rejection() is not None
    assert store.get_rejection_streak() == 1


# ── cross-session persistence (the key contract vs WORKING_MEMORY) ────


def test_bearing_survives_simulated_session_cycle(tmp_store) -> None:
    """Bearing must NOT clear on session boundary — distinguishes it from
    WORKING_MEMORY, which clears in continuity_interceptor first-turn."""
    b = bs.Bearing(active_goal="long-running goal", current_frame="frame")
    store.set_bearing(b)

    # Simulate a session-boundary cycle by re-importing or re-loading via
    # the public API (no explicit clear should happen).
    same = store.get_bearing()
    assert same.active_goal == "long-running goal"
    assert same.current_frame == "frame"


# ── atomic write ─────────────────────────────────────────────────────


def test_set_bearing_writes_via_tempfile_then_rename(tmp_store) -> None:
    """Verify the temp file is removed after rename (atomic-write contract)."""
    store.set_bearing(bs.Bearing(active_goal="x"))
    tmp_file = (tmp_store / "bearing.json").with_name("bearing.json.tmp")
    assert not tmp_file.exists()
    assert (tmp_store / "bearing.json").exists()


# ── malformed file recovery ──────────────────────────────────────────


def test_get_bearing_recovers_from_malformed_json(tmp_store) -> None:
    (tmp_store / "bearing.json").write_text("{not valid json", encoding="utf-8")
    assert store.get_bearing() == bs.Bearing()


def test_get_bearing_recovers_from_non_dict_root(tmp_store) -> None:
    (tmp_store / "bearing.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert store.get_bearing() == bs.Bearing()
