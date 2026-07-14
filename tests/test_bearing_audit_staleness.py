"""The staleness audit kinds must be accepted by audit.append (it silently drops
any kind not in VALID_KINDS). V1 adds nudged + cleared only; escalated is
deferred with the fault path (V2)."""
from __future__ import annotations

import pytest

from addons.system.bearing import audit


@pytest.fixture
def tmp_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    yield tmp_path


def test_staleness_nudged_kind_is_accepted(tmp_audit) -> None:
    audit.append("staleness_nudged", turn_id="t1", signal_id="channel:user", streak=2)
    rows = audit.read_recent()
    assert any(r.get("kind") == "staleness_nudged" for r in rows)


def test_staleness_cleared_kind_is_accepted(tmp_audit) -> None:
    audit.append("staleness_cleared", turn_id="t1", signal_id="channel:user")
    rows = audit.read_recent()
    assert any(r.get("kind") == "staleness_cleared" for r in rows)


def test_unknown_kind_still_dropped(tmp_audit) -> None:
    # guard: the VALID_KINDS gate still rejects junk — we opened it for two kinds, not wide
    audit.append("totally_made_up_kind", turn_id="t1")
    rows = audit.read_recent()
    assert not any(r.get("kind") == "totally_made_up_kind" for r in rows)
