"""SP3 of the monothink training loop — the reversible event ledger backend.

git-like any-version revert (not just last-apply) + a ledger-list API the omnibar
undo addon renders. Append-only: a revert is itself journaled, never rewriting
history. The scaffold full snapshot stored on every applied entry makes restore to
any point deterministic.

Spec: docs/superpowers/specs/2026-06-03-monothink-training-loop-sp1-rating-contract-design.md
"""
from __future__ import annotations

import pytest

import core.monothink as mt


@pytest.fixture
def paths(tmp_path, monkeypatch):
    j = tmp_path / "monothink.journal.jsonl"
    s = tmp_path / "monothink.md"
    monkeypatch.setattr(mt, "_JOURNAL_PATH", j)
    monkeypatch.setattr(mt, "_SCAFFOLD_PATH", s)
    return j, s


def _applied(turn_id, scaffold_full, parent, tag, *, diff=10, rating=50):
    return {
        "ts": "2026-06-03T00:00:00Z",
        "turn_id": turn_id,
        "applied": True,
        "applied_scaffold_full": scaffold_full,
        "parent_scaffold_version": parent,
        "primary_failure_tag": tag,
        "adjudicated_tag": tag,
        "rating_value": rating,
        "old_chars": 0,
        "new_chars": len(scaffold_full),
        "diff_chars": diff,
        "reject_reason": None,
    }


def test_revert_to_version_restores_that_snapshot(paths):
    mt._append_journal(_applied("v1", "SCAFFOLD V1", None, "restatement_unpruned"))
    mt._append_journal(_applied("v2", "SCAFFOLD V2", "v1", "missing_branch_pressure"))
    mt._write_scaffold("SCAFFOLD V2")  # live scaffold is v2

    entry = mt.revert_to_version("v1")
    assert entry is not None
    assert mt.read_scaffold() == "SCAFFOLD V1"
    assert entry["kind"] == "rollback"
    assert entry["reverted_to"] == "v1"


def test_revert_is_append_only_and_shows_as_current(paths):
    mt._append_journal(_applied("v1", "AAA", None, "restatement_unpruned"))
    mt._append_journal(_applied("v2", "BBBB", "v1", "missing_branch_pressure"))
    mt._write_scaffold("BBBB")
    before = len(mt.read_journal(limit=100))

    mt.revert_to_version("v1")
    after = mt.read_journal(limit=100)
    assert len(after) == before + 1  # revert appended, nothing rewritten
    ledger = mt.list_ledger()
    assert ledger[0]["kind"] == "rollback"
    assert ledger[0]["is_current"] is True   # live scaffold == v1 snapshot
    # the original v2 row is still there, no longer current
    v2 = next(r for r in ledger if r["turn_id"] == "v2")
    assert v2["is_current"] is False


def test_revert_to_unknown_turn_returns_none(paths):
    mt._append_journal(_applied("v1", "X", None, "restatement_unpruned"))
    assert mt.revert_to_version("does_not_exist") is None


def test_revert_to_rejected_entry_returns_none(paths):
    # a rejected (applied=False) entry has no usable snapshot to restore
    mt._append_journal({
        "ts": "2026-06-03T00:00:00Z", "turn_id": "rej", "applied": False,
        "reject_reason": "scope_mismatch:proposed=line,actual=section",
        "primary_failure_tag": "premise_unchecked",
    })
    assert mt.revert_to_version("rej") is None


def test_list_ledger_newest_first_with_display_fields(paths):
    mt._append_journal(_applied("v1", "AAA", None, "restatement_unpruned", diff=3))
    mt._append_journal(_applied("v2", "BBBB", "v1", "missing_branch_pressure", diff=4, rating=70))
    mt._write_scaffold("BBBB")

    ledger = mt.list_ledger()
    assert [r["turn_id"] for r in ledger] == ["v2", "v1"]  # newest first
    top = ledger[0]
    assert top["tag"] == "missing_branch_pressure"
    assert top["rating_value"] == 70
    assert top["diff_chars"] == 4
    assert top["is_current"] is True
    assert top["revertable"] is True
    assert ledger[1]["is_current"] is False


def test_list_ledger_marks_rejected_not_revertable(paths):
    mt._append_journal(_applied("v1", "AAA", None, "restatement_unpruned"))
    mt._append_journal({
        "ts": "2026-06-03T00:00:00Z", "turn_id": "rej", "applied": False,
        "reject_reason": "diff_cap_exceeded:400>300",
        "primary_failure_tag": "premise_unchecked", "diff_chars": 400,
    })
    mt._write_scaffold("AAA")
    ledger = mt.list_ledger()
    rej = next(r for r in ledger if r["turn_id"] == "rej")
    assert rej["revertable"] is False
    assert rej["applied"] is False


def test_list_ledger_skips_async_reservations(paths):
    """scheduled_async rows are transient reservation placeholders — the omnibar
    ledger must not show them as events."""
    mt._append_journal(_applied("v1", "AAA", None, "restatement_unpruned"))
    mt._append_journal({
        "ts": "2026-06-03T00:00:00Z", "turn_id": "resv", "applied": False,
        "reject_reason": "scheduled_async", "primary_failure_tag": "premise_unchecked",
    })
    mt._write_scaffold("AAA")
    ledger = mt.list_ledger()
    assert all(r["turn_id"] != "resv" for r in ledger)
    assert any(r["turn_id"] == "v1" for r in ledger)


def test_list_ledger_empty_when_no_journal(paths):
    assert mt.list_ledger() == []


def test_ledger_addon_is_registered_in_omnibar():
    """The Monothink Ledger addon is wired into the builtin registry so it shows
    in the omnibar. Verified headlessly — the factory is lazy, so no Qt widget is
    constructed here."""
    from ui.addons.builtin import build_builtin_registry
    registry = build_builtin_registry()
    specs = {s.id: s for s in registry.all()}
    assert "monothink_ledger" in specs
    spec = specs["monothink_ledger"]
    assert spec.kind == "module"
    assert spec.title == "MONOTHINK LEDGER"
    assert callable(spec.factory)
