"""Offscreen render test for the MonoSearch companion panel.

This test lives at the TOP of the tests tree, so the autouse isolation
fixtures in tests/monosearch/conftest.py do NOT run here — the turn_trace +
salience isolation is set up manually below.

The panel wraps every service call in try/except, so "nothing raised" is
trivially satisfied and ">0 rows" is satisfied even by a "(no results)"
placeholder or an "[error: ...]" sentinel. To prove the panel actually
rendered REAL data we additionally assert the first row is neither an error
sentinel nor the empty placeholder, and contains the seeded fault kind.
"""
from __future__ import annotations

import os

# Must be set before any PySide6/Qt import so Qt picks the headless platform.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

import core.turn_trace as turn_trace
from core import fault_response as fr
from core.monosearch import registry, salience
from core.monosearch.adapters.faults import FaultAdapter
from ui.addons.monosearch import MonoSearchPanel


def _qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


def test_monosearch_panel_renders_failing_and_search(tmp_path):
    app = _qapp()

    # ── isolation: turn_trace (where faults live) + salience ledger ──────
    turn_trace.set_db_path(tmp_path / "test_panel_turn_trace.sqlite3")
    os.environ["MONOLITH_TURN_TRACE_V1"] = "1"
    salience.set_db_path(tmp_path / "test_panel_salience.sqlite3")
    salience.ensure_schema()

    # Only the fault adapter is registered for this test, so the panel reads
    # an empty-but-isolated registry plus our seeded faults.
    registry.clear()
    registry.register(FaultAdapter())

    # Seed >=2 faults of the SAME kind so kind-level recurrence aggregates
    # them into one salience row with count > 1 (the "what I keep failing"
    # signal). Different evidence each time mirrors production reality.
    fault_kind = "tool_no_fire"
    for i in range(3):
        rid = fr.emit_fault(
            turn_id=f"t{i}",
            fault_kind=fault_kind,
            detector_name="detect_tool_no_fire",
            evidence=f"stated intent {i}, no tool_call emitted",
        )
        assert rid > 0, "fault seeding failed — turn_trace isolation not wired"

    panel = None
    try:
        panel = MonoSearchPanel()
        panel.show()
        app.processEvents()

        # ── Failing mode (default on open, re-run explicitly) ───────────
        panel._run_mode("Failing")
        app.processEvents()
        results = panel._results
        assert results.count() > 0, "Failing rendered zero rows"
        first = results.item(0).text()
        # False-pass guard: a real data row, not an error sentinel or the
        # empty placeholder, and it must carry the seeded fault kind.
        assert not first.startswith("[error"), f"Failing surfaced an error row: {first}"
        assert first != "(no results)", "Failing rendered the empty placeholder"
        assert fault_kind in first, f"Failing row lacks the seeded kind: {first}"
        # Salience-dict shape: 'x3' (three aggregated faults) + the source tag.
        assert "×3" in first, f"recurrence count not aggregated to 3: {first}"
        assert "[fault_traces]" in first, f"source tag missing: {first}"

        # ── Search mode — query the fault kind, scope to faults ─────────
        panel._on_mode_clicked("Search")
        app.processEvents()
        assert panel._search_row.isVisible(), "Search mode did not reveal the query row"
        panel._query_edit.setText(fault_kind)
        panel._source_combo.setCurrentText("faults")
        panel._run_search()
        app.processEvents()
        s_results = panel._results
        assert s_results.count() > 0, "Search rendered zero rows"
        s_first = s_results.item(0).text()
        assert not s_first.startswith("[error"), f"Search surfaced an error row: {s_first}"
        assert s_first != "(no results)", "Search rendered the empty placeholder"
        assert s_first.startswith("fault:"), f"Search row is not a fault Record: {s_first}"
        assert fault_kind in s_first, f"Search row lacks the queried kind: {s_first}"

        # ── sources count reflects the live registry (1 adapter here) ───
        assert "1 sources" in panel._sources_label.text()
    finally:
        if panel is not None:
            panel.close()
            panel.deleteLater()
            app.processEvents()
        # Teardown: close the salience connection so Windows can delete the
        # temp dir, and clear the turn_trace path so the rest of the suite is
        # not left pointing at a deleted temp DB.
        salience.close()
        turn_trace.set_db_path(None)
