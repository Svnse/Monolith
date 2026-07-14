from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

import core.style as _s
from ui.components.atoms import MonoButton


class AuditLogPanel(QWidget):
    _OUTCOME_COLOR = {
        "auto_approved": _s.FG_OK,
        "approved": _s.ACCENT_PRIMARY,
        "rejected": _s.FG_ERROR,
        "blocked": _s.FG_WARN,
    }

    def __init__(self, world_state, parent=None):
        super().__init__(parent)
        self._world_state = world_state

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Controls row only — identity lives in the pane header (UI_CONTRACT §2).
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addStretch()

        self._clear = MonoButton("CLEAR")
        self._clear.clicked.connect(self.clear)
        header.addWidget(self._clear)
        root.addLayout(header)

        self.audit_list = QListWidget()
        self.audit_list.setObjectName("audit_list")
        self.audit_list.setProperty("panelInset", True)  # shared inset style (theme QSS)
        self.audit_list.setStyleSheet("font-size: 9px; font-family: Consolas;")
        root.addWidget(self.audit_list, 1)

    def bind_controller(self, controller) -> None:
        return

    def refresh(self) -> None:
        self.audit_list.clear()
        if self._world_state is None:
            return
        for entry in reversed(self._world_state.get_action_log()):
            ts = str(entry.get("ts", ""))[:19].replace("T", " ")
            outcome = str(entry.get("outcome", "?"))
            cmd = str(entry.get("command") or entry.get("type", "?"))
            engine = str(entry.get("engine") or "—")
            source = str(entry.get("source", ""))
            text = f"{ts}  {outcome:<14}  {cmd:<18}  eng:{engine}  src:{source}"
            item = QListWidgetItem(text)
            item.setForeground(QColor(self._OUTCOME_COLOR.get(outcome, _s.FG_DIM)))
            self.audit_list.addItem(item)

    def clear(self) -> None:
        if self._world_state is not None:
            self._world_state.clear_action_log()
        self.audit_list.clear()
