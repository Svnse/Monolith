"""The agent strip — the little live UI under the chat that shows sub-agents.

Fed by the active-agents spine (PageChat.sig_agents_changed). Each chip shows
live status (running… → done) and, on click, zooms into that agent's own trace
(its child_turn_id) — the "zoom into what each agent does" ask (E, 2026-06-18).

Pure-display: the source of truth is the PageChat workers; this only renders the
records it's handed and emits a zoom request.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton


class AgentsStrip(QFrame):
    sig_zoom_agent = Signal(str, str)  # (child_turn_id, frame)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AgentsStrip")
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(10, 3, 10, 3)
        self._row.setSpacing(6)
        self._chips: list[QPushButton] = []
        self.setVisible(False)

    def update_agents(self, records) -> None:
        """Rebuild the chips from the live records. Hidden when none."""
        for chip in self._chips:
            chip.setParent(None)
            chip.deleteLater()
        self._chips = []
        while self._row.count():
            self._row.takeAt(0)

        records = list(records or [])
        if not records:
            self.setVisible(False)
            return
        # running first, then done — most-relevant left
        records.sort(key=lambda r: 0 if getattr(r, "is_running", False) else 1)
        for rec in records:
            self._chips.append(self._make_chip(rec))
            self._row.addWidget(self._chips[-1])
        self._row.addStretch(1)
        self.setVisible(True)

    def _make_chip(self, rec) -> QPushButton:
        running = bool(getattr(rec, "is_running", False))
        label = str(getattr(rec, "frame", "") or "").strip() or f"L{getattr(rec, 'level', 2)}"
        state = "running…" if running else "done"
        cid = str(getattr(rec, "child_turn_id", "") or "")
        chip = QPushButton(f"● {label} · {state}")
        chip.setCursor(Qt.PointingHandCursor if cid else Qt.ArrowCursor)
        chip.setFlat(True)
        chip.setEnabled(bool(cid))  # zoomable once the trace id is known (done)
        chip.setToolTip("Click to zoom into this agent's trace" if cid
                        else "Agent is still running…")
        accent = "#4a9eff" if running else "#7a7f87"
        chip.setStyleSheet(
            "QPushButton { border: 1px solid %s; border-radius: 9px; padding: 2px 9px; "
            "text-align: left; color: %s; } QPushButton:hover { border-color: #4a9eff; }"
            % (accent, accent)
        )
        chip.clicked.connect(lambda _=False, c=cid, f=label: self.sig_zoom_agent.emit(c, f))
        return chip
