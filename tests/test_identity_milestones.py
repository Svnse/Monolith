from __future__ import annotations

import pytest

from core import identity_milestones as m


@pytest.fixture
def tmp_ledger(monkeypatch, tmp_path):
    p = tmp_path / "identity_milestones.json"
    monkeypatch.setattr(m, "STORE_PATH", p)
    yield p


def test_defaults_when_empty(tmp_ledger) -> None:
    assert m.get_watermark() == 0
    assert m.get_milestone() == 0
    assert m.get_latest_emergence_signal() is None
    assert m.get_origin0_hash() == ""


def test_watermark_roundtrip(tmp_ledger) -> None:
    m.set_watermark(42)
    assert m.get_watermark() == 42


def test_emergence_signal_roundtrip(tmp_ledger) -> None:
    sig = {"detected_at": "2026-06-02T00:00:00+00:00", "new_acu_count": 5, "message": "x"}
    m.set_latest_emergence_signal(sig)
    got = m.get_latest_emergence_signal()
    assert got is not None and got["new_acu_count"] == 5
    m.set_latest_emergence_signal(None)
    assert m.get_latest_emergence_signal() is None


def test_origin0_hash_roundtrip(tmp_ledger) -> None:
    m.set_origin0_hash("abc123")
    assert m.get_origin0_hash() == "abc123"


def test_bump_milestone(tmp_ledger) -> None:
    assert m.get_milestone() == 0
    assert m.bump_milestone() == 1
    assert m.get_milestone() == 1


def test_curiosity_signal_roundtrip(tmp_ledger) -> None:
    assert m.get_latest_curiosity_signal() is None
    m.set_latest_curiosity_signal({"pull_count": 3, "message": "x"})
    got = m.get_latest_curiosity_signal()
    assert got is not None and got["pull_count"] == 3


def test_curiosity_surfaced_seen_set_counts(tmp_ledger) -> None:
    assert m.get_curiosity_surfaced() == {}
    m.bump_curiosity_surfaced(["a", "b"])
    m.bump_curiosity_surfaced(["a"])
    s = m.get_curiosity_surfaced()
    assert s["a"] == 2
    assert s["b"] == 1
