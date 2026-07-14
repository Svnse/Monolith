from __future__ import annotations

import pytest

from addons.system.bearing import audit


@pytest.fixture
def tmp_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    yield tmp_path


# ── valid kinds ──────────────────────────────────────────────────────


def test_valid_kinds_set() -> None:
    assert {
        "applied", "rejected", "grounding_failed", "escalated", "cleared",
        "staleness_nudged", "staleness_cleared",
    } == set(audit.VALID_KINDS)


# ── append + read ────────────────────────────────────────────────────


def test_append_single_row(tmp_audit) -> None:
    audit.append("applied", turn_id="t1", slots_changed=["current_frame"])
    rows = audit.read_recent()
    assert len(rows) == 1
    assert rows[0]["kind"] == "applied"
    assert rows[0]["turn_id"] == "t1"
    assert rows[0]["slots_changed"] == ["current_frame"]
    assert "ts" in rows[0]


def test_append_preserves_order(tmp_audit) -> None:
    audit.append("rejected", turn_id="t1", failed_rules=["D1"])
    audit.append("rejected", turn_id="t2", failed_rules=["D3"])
    audit.append("applied", turn_id="t3", slots_changed=["next_move"])
    rows = audit.read_recent(limit=10)
    kinds = [r["kind"] for r in rows]
    assert kinds == ["rejected", "rejected", "applied"]


def test_read_recent_with_limit(tmp_audit) -> None:
    for i in range(5):
        audit.append("applied", turn_id=f"t{i}")
    rows = audit.read_recent(limit=2)
    assert len(rows) == 2
    turn_ids = [r["turn_id"] for r in rows]
    assert turn_ids == ["t3", "t4"]


# ── invalid kind is silently dropped ─────────────────────────────────


def test_append_with_invalid_kind_drops(tmp_audit) -> None:
    audit.append("not-a-kind", turn_id="t1")
    assert audit.read_recent() == []


# ── read_recent on empty / missing file ──────────────────────────────


def test_read_recent_missing_file(tmp_audit) -> None:
    assert audit.read_recent() == []


def test_read_recent_skips_malformed_lines(tmp_audit) -> None:
    path = tmp_audit / "bearing.audit.jsonl"
    path.write_text(
        '{"kind": "applied", "turn_id": "t1"}\n'
        'this is not json\n'
        '{"kind": "rejected", "turn_id": "t2"}\n',
        encoding="utf-8",
    )
    rows = audit.read_recent()
    assert len(rows) == 2
    assert rows[0]["turn_id"] == "t1"
    assert rows[1]["turn_id"] == "t2"


# ── failure containment ─────────────────────────────────────────────


def test_append_does_not_raise_on_unwritable_path(monkeypatch, tmp_path) -> None:
    # Point to a path whose parent is a file (invalid mkdir target).
    bogus_parent = tmp_path / "wedge"
    bogus_parent.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(audit, "_AUDIT_PATH", bogus_parent / "child" / "audit.jsonl")
    # Must not raise.
    audit.append("applied", turn_id="t1")
