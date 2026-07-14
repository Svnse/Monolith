from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

import core.style as _s


class GenerationTracePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._controller = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # No "LOG" label — identity lives in the pane header (UI_CONTRACT §2).
        self.trace = QTextEdit()
        self.trace.setReadOnly(True)
        self.trace.setMinimumHeight(80)
        # Shared inset style (theme QSS), NOT the chat-side #trace_log rule —
        # an ID selector would out-rank [panelInset] and repaint the dark box.
        self.trace.setProperty("panelInset", True)
        self.trace.setStyleSheet("font-family: Consolas; font-size: 10px;")
        root.addWidget(self.trace, 1)

        self._update_label = QLabel("")
        self._update_label.setObjectName("lbl_config_update")
        self._update_label.hide()
        root.addWidget(self._update_label)

        self._fade = QTimer(self)
        self._fade.setSingleShot(True)
        self._fade.timeout.connect(self._update_label.hide)

    def bind_controller(self, controller) -> None:
        self._controller = controller

    def append_html(self, msg: str, tag: str = "INFO", error: bool = False) -> None:
        arrow_color = _s.FG_ERROR if error else _s.ACCENT_PRIMARY
        tag_color = _s.FG_ERROR if error else _s.FG_PLACEHOLDER
        self.trace.append(
            f"<table width='100%' cellpadding='0' cellspacing='0'><tr>"
            f"<td><span style='color:{arrow_color}'>→</span> {msg}</td>"
            f"<td align='right' style='color:{tag_color}; white-space:nowrap'>[{tag}]</td>"
            f"</tr></table>"
        )

    def append_plain(self, msg: str) -> None:
        self.trace.append(f"<span style='color:{_s.FG_PLACEHOLDER}'>{msg}</span>")

    def clear(self) -> None:
        self.trace.clear()
        self._update_label.hide()

    def show_config_saved(self, stamp: str) -> None:
        self._update_label.setText(f"USER (UPDATED): {stamp}")
        self._update_label.show()
        self._fade.start(2500)

    def refresh(self) -> None:
        return
