"""Verify Bearing escalation respects the territorial constraint:

  - Uses the existing FaultDetectedEvent kind (emitted via emit_fault).
  - Does NOT introduce a new PipelineEvent subclass.
  - Does NOT introduce a new SQLite table.
"""
from __future__ import annotations

from pathlib import Path


def test_no_BearingTransitionEvent_in_turn_pipeline_events() -> None:
    from core import turn_pipeline_events as tpe
    forbidden = "BearingTransitionEvent"
    assert not hasattr(tpe, forbidden), (
        f"Plan §2 forbids adding {forbidden} to core/turn_pipeline_events.py"
    )


def test_no_bearing_table_in_turn_trace_schema() -> None:
    """The Bearing plan forbids new SQLite tables. Scan turn_trace.py source
    for any CREATE TABLE statement naming 'bearing'."""
    src = Path("core/turn_trace.py").read_text(encoding="utf-8")
    src_lower = src.lower()
    assert "create table bearing" not in src_lower, (
        "Plan §2 forbids any 'bearing' SQLite table in turn_trace.sqlite3"
    )
    assert "create table if not exists bearing" not in src_lower


def test_bearing_fault_kinds_registered_in_known_kinds() -> None:
    """For emit_fault to accept Bearing's fault kinds, they must be in
    KNOWN_KINDS — verified mechanically."""
    from core.fault_response import KNOWN_KINDS
    assert "bearing_structural_unrecoverable" in KNOWN_KINDS
    assert "bearing_grounding_failed" in KNOWN_KINDS


def test_no_bearing_sqlite_path_created_outside_turn_trace() -> None:
    """The Bearing addon owns bearing.json + bearing.audit.jsonl only. No
    new SQLite file. Verify by scanning addon source for sqlite3 imports."""
    bearing_dir = Path("addons/system/bearing")
    for py_file in bearing_dir.rglob("*.py"):
        src = py_file.read_text(encoding="utf-8")
        assert "import sqlite3" not in src, (
            f"{py_file} imports sqlite3 — Plan §2 forbids new persistence outside bearing.json/jsonl"
        )
        assert "from sqlite3" not in src
