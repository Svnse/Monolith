"""Tests for core.fault_response — emit/read API and run_all_detectors."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

import core.turn_trace as tt
from core.fault_response import (
    KNOWN_KINDS,
    FaultRecord,
    emit_fault,
    read_by_kind,
    read_recent,
    run_all_detectors,
)


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Give each test its own SQLite database."""
    db = tmp_path / "test_fault_response.sqlite3"
    tt.set_db_path(db)
    os.environ["MONOLITH_TURN_TRACE_V1"] = "1"
    yield db
    tt.set_db_path(None)


# ── KNOWN_KINDS ─────────────────────────────────────────────────────────────


def test_known_kinds_contains_core_four_and_bearing_two():
    # The original four kernel fault kinds.
    assert "markdown_corruption" in KNOWN_KINDS
    assert "tool_no_fire" in KNOWN_KINDS
    assert "think_leak" in KNOWN_KINDS
    assert "regen_mismatch" in KNOWN_KINDS
    # The two Bearing addon fault kinds (added per Bearing V0 plan §7).
    # Bearing escalates via emit_fault using these kinds — they MUST be in
    # KNOWN_KINDS or emit_fault rejects them.
    assert "bearing_structural_unrecoverable" in KNOWN_KINDS
    assert "bearing_grounding_failed" in KNOWN_KINDS
    # MonoExplore expedition-runner escalation kind.
    assert "expedition_halted" in KNOWN_KINDS
    # Grounded-verdict V1: fabricated-cite (the laundering fault) — emit_fault
    # rejects it unless it's a KNOWN_KIND, which would silently dark the feed.
    assert "fabricated_cite" in KNOWN_KINDS
    # Self-maintenance runner escalation kind.
    assert "self_maint_halted" in KNOWN_KINDS
    assert len(KNOWN_KINDS) == 9


# ── emit_fault ──────────────────────────────────────────────────────────────


def test_emit_fault_returns_positive_id():
    row_id = emit_fault(
        turn_id="turn-001",
        fault_kind="markdown_corruption",
        detector_name="detect_markdown_corruption",
        evidence="fence_imbalance(count=3)",
    )
    assert row_id > 0


def test_emit_fault_unknown_kind_returns_minus_one():
    row_id = emit_fault(
        turn_id="turn-001",
        fault_kind="not_a_real_kind",
        detector_name="test",
        evidence="x",
    )
    assert row_id == -1


def test_emit_fault_stores_evidence_in_payload():
    emit_fault(
        turn_id="turn-002",
        fault_kind="tool_no_fire",
        detector_name="detect_tool_no_fire",
        evidence="I will check",
        metadata={"intent_span": [0, 12]},
    )
    records = read_by_kind("tool_no_fire", limit=5)
    assert len(records) >= 1
    r = records[0]
    assert r.evidence == "I will check"
    assert r.metadata.get("intent_span") == [0, 12]


def test_emit_fault_multiple_kinds_all_readable():
    for kind in KNOWN_KINDS:
        emit_fault(
            turn_id="turn-multi",
            fault_kind=kind,
            detector_name=f"detect_{kind}",
            evidence=f"evidence-{kind}",
        )
    all_records = read_recent(limit=10)
    kinds_found = {r.fault_kind for r in all_records}
    assert kinds_found == KNOWN_KINDS


# ── read_recent ─────────────────────────────────────────────────────────────


def test_read_recent_empty_store():
    records = read_recent(limit=10)
    assert records == []


def test_read_recent_returns_newest_first():
    emit_fault("turn-a", "markdown_corruption", "d", "ev-a")
    emit_fault("turn-b", "think_leak", "d", "ev-b")
    records = read_recent(limit=10)
    assert len(records) >= 2
    # Newest first: turn-b inserted last
    assert records[0].turn_id == "turn-b"
    assert records[1].turn_id == "turn-a"


def test_read_recent_respects_limit():
    for i in range(10):
        emit_fault(f"turn-{i}", "regen_mismatch", "d", "ev")
    records = read_recent(limit=3)
    assert len(records) == 3


# ── read_by_kind ────────────────────────────────────────────────────────────


def test_read_by_kind_filters_correctly():
    emit_fault("turn-k1", "markdown_corruption", "d", "ev")
    emit_fault("turn-k2", "tool_no_fire", "d", "ev")
    emit_fault("turn-k3", "markdown_corruption", "d", "ev")

    md_records = read_by_kind("markdown_corruption", limit=10)
    kinds = {r.fault_kind for r in md_records}
    assert kinds == {"markdown_corruption"}
    assert len(md_records) == 2


def test_read_by_kind_empty_for_missing_kind():
    emit_fault("turn-x", "think_leak", "d", "ev")
    records = read_by_kind("regen_mismatch", limit=10)
    assert records == []


# ── run_all_detectors ────────────────────────────────────────────────────────


def test_run_all_detectors_clean_response_returns_empty():
    text = "Here is a simple answer with no problems."
    records = run_all_detectors(text, turn_id="turn-clean", context={})
    assert records == []


def test_run_all_detectors_all_four_faults_detected():
    """Synthetic response with all four fault conditions embedded."""
    # markdown_corruption: odd fence count
    # tool_no_fire: intent phrase without <tool_call>
    # think_leak: unbalanced think tags (open > close)
    # regen_mismatch: tool_result_5 with only 0 tool messages in frame
    text = (
        "```python\ncode here\n"           # unclosed fence (1 triple-backtick)
        "I'll check the result.\n"          # tool intent, no <tool_call>
        "<think>some reasoning\n"           # open without close
        "See tool_result_5 for details.\n"  # regen ref out of range
    )
    context = {"frame_traces": []}  # no tool results in frame
    records = run_all_detectors(text, turn_id="turn-all", context=context)
    kinds_found = {r.fault_kind for r in records}
    assert "markdown_corruption" in kinds_found
    assert "tool_no_fire" in kinds_found
    assert "think_leak" in kinds_found
    assert "regen_mismatch" in kinds_found
    assert len(records) == 4


def test_run_all_detectors_returns_fault_records_not_yet_persisted():
    """Records returned have id=-1 (not yet emitted)."""
    text = "```unclosed fence"
    records = run_all_detectors(text, turn_id="turn-x", context={})
    assert len(records) >= 1
    for r in records:
        assert isinstance(r, FaultRecord)
        assert r.id == -1


def test_run_all_detectors_emit_then_read_roundtrip():
    """Caller emits the returned records; they appear in read_recent."""
    text = "```unclosed"
    records = run_all_detectors(text, turn_id="turn-rt", context={})
    for r in records:
        emit_fault(r.turn_id, r.fault_kind, r.detector_name, r.evidence, r.metadata)
    stored = read_recent(limit=10)
    kinds = {r.fault_kind for r in stored}
    assert "markdown_corruption" in kinds
