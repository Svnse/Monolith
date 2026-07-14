from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QFrame, QLabel, QListWidget, QListWidgetItem,
                               QScrollArea, QVBoxLayout, QWidget)

import core.style as _s
import core.turn_trace as _tt
from core.run_model import live_runs
from ui.components.run_view import RunView


class WorkshopPane(QWidget):
    """Companion RUN BROWSER (Workshop Pane v2): recent Monoline runs (live + historical), each
    opening the unified RunView. A live run binds the in-memory RunModel (updates in real time);
    a past run rehydrates from turn_trace. Poll-primary, gated on visibility, churn-guarded.
    Human-only read surface -- nothing here feeds prompt assembly (non-performative)."""

    _DOT = {"running": "● ", "done": "✓ ", "error": "✕ "}

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        # Status-only header (run count) — identity lives in the pane header
        # (UI_CONTRACT §2: status is content, the panel name is not).
        self._header = QLabel("no runs")
        self._header.setStyleSheet(f"color:{_s.FG_DIM}; font-size:11px;")
        root.addWidget(self._header)

        self._list = QListWidget()
        self._list.setProperty("panelInset", True)      # shared inset style (theme QSS)
        self._list.setMaximumHeight(140)
        self._list.currentRowChanged.connect(self._on_row_changed)
        root.addWidget(self._list)

        # the selected run's RunView, scrollable (the pane height is fixed; a long run scrolls).
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._view = RunView()
        self._scroll.setWidget(self._view)
        root.addWidget(self._scroll, 1)

        self._run_ids: list[str] = []        # parallel to the list rows
        self._rendered_sig: tuple = ()       # churn guard: (run_id, status) per row
        self._selected_id: str | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self.refresh)

    # -- lifecycle (poll only while visible) --
    def showEvent(self, event) -> None:  # noqa: N802
        self._timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    # -- introspection --
    def run_count(self) -> int:
        return len(self._run_ids)

    def run_ids(self) -> list[str]:
        return list(self._run_ids)

    def current_run_view(self) -> RunView:
        return self._view

    # -- live + historical merge --
    def _gather_runs(self) -> list[tuple[str, str, str | None]]:
        """(run_id, display_name, status) -- live runs (from the shared registry) first, then
        historical (turn_trace), deduped by run_id. status is None for purely-historical rows."""
        out: list[tuple[str, str, str | None]] = []
        seen: set[str] = set()
        for m in live_runs.list_runs():
            if m.run_id in seen:
                continue
            seen.add(m.run_id)
            out.append((m.run_id, m.name or m.flow_id or m.run_id, m.status))
        try:
            for s in _tt.list_recent_runs(20):
                if s.run_id in seen:
                    continue
                seen.add(s.run_id)
                out.append((s.run_id, s.name or s.flow_id or s.run_id, None))
        except Exception:
            pass
        return out

    def refresh(self) -> None:
        runs = self._gather_runs()
        sig = tuple((rid, status) for rid, _name, status in runs)
        if sig == self._rendered_sig:
            return  # churn guard: nothing changed (id set + per-run status both stable)
        prev_selected = self._selected_id
        self._list.blockSignals(True)
        self._list.clear()
        self._run_ids = []
        newest_live: str | None = None
        for rid, name, status in runs:
            QListWidgetItem(self._DOT.get(status or "", "   ") + str(name), self._list)
            self._run_ids.append(rid)
            if status == "running" and newest_live is None:
                newest_live = rid
        self._list.blockSignals(False)
        self._rendered_sig = sig
        self._header.setText(
            f"{len(self._run_ids)} run(s)" if self._run_ids else "no runs")
        # selection: keep the prior pick if still present; else auto-show a running run; else newest.
        target = (prev_selected if prev_selected in self._run_ids
                  else (newest_live or (self._run_ids[0] if self._run_ids else None)))
        if target is not None:
            self.select_run(target)

    def select_run(self, run_id: str) -> None:
        if run_id not in self._run_ids:
            return
        self._selected_id = run_id
        idx = self._run_ids.index(run_id)
        if self._list.currentRow() != idx:
            self._list.blockSignals(True)
            self._list.setCurrentRow(idx)
            self._list.blockSignals(False)
        self._show_run(run_id)

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._run_ids):
            self._selected_id = self._run_ids[row]
            self._show_run(self._run_ids[row])

    def _show_run(self, run_id: str) -> None:
        model = live_runs.get(run_id)   # prefer the live model (instant + real-time updates)
        if model is None:
            try:
                model = _tt.rehydrate_run(run_id)   # past run -> rebuild from turn_trace
            except Exception:
                model = None
        if model is not None:
            self._view.bind(model)
