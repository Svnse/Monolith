from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

import core.style as _s


class ActionReviewPanel(QWidget):
    sig_approved = Signal(dict)
    sig_rejected = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_action: dict | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        panel = QFrame()
        panel.setObjectName("action_review_panel")
        panel.setStyleSheet(
            f"""
            QFrame#action_review_panel {{
                background: {_s.BG_SURFACE_1};
                border: none;
                border-radius: 8px;
            }}
            """
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        title = QLabel("ACTION PROPOSED")
        title.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 10px; font-family: Consolas; font-weight: bold;"
        )
        header.addWidget(title)
        header.addStretch()

        self._close = QPushButton("CLOSE")
        self._close.setCursor(Qt.PointingHandCursor)
        self._close.setFlat(True)
        self._close.setStyleSheet(
            f"QPushButton {{ color: {_s.FG_DIM}; border: none; background: transparent; font-size: 9px; }}"
            f"QPushButton:hover {{ color: {_s.FG_TEXT}; }}"
        )
        self._close.clicked.connect(lambda: self.sig_rejected.emit(self._current_action))
        header.addWidget(self._close)
        layout.addLayout(header)

        self._lbl_type = QLabel("No pending action.")
        self._lbl_type.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 10px; font-family: Consolas;"
        )
        layout.addWidget(self._lbl_type)

        self._lbl_engine = QLabel("")
        self._lbl_engine.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas;"
        )
        layout.addWidget(self._lbl_engine)

        self._lbl_payload = QLabel("")
        self._lbl_payload.setWordWrap(True)
        self._lbl_payload.setStyleSheet(
            f"color: {_s.FG_SECONDARY}; font-size: 9px; font-family: Consolas;"
        )
        layout.addWidget(self._lbl_payload)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 4, 0, 0)
        buttons.setSpacing(8)
        buttons.addStretch()

        self._reject = QPushButton("REJECT")
        self._reject.setCursor(Qt.PointingHandCursor)
        self._reject.setFixedHeight(24)
        self._reject.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                color: {_s.FG_ERROR};
                border: 1px solid {_s.FG_ERROR};
                border-radius: 6px;
                padding: 0 10px;
                font-size: 9px;
                font-family: Consolas;
            }}
            QPushButton:hover {{
                background: {_s.FG_ERROR}22;
            }}
            """
        )
        self._reject.clicked.connect(lambda: self.sig_rejected.emit(self._current_action))
        buttons.addWidget(self._reject)

        self._approve = QPushButton("APPROVE")
        self._approve.setCursor(Qt.PointingHandCursor)
        self._approve.setFixedHeight(24)
        self._approve.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                color: {_s.ACCENT_PRIMARY};
                border: 1px solid {_s.ACCENT_PRIMARY};
                border-radius: 6px;
                padding: 0 10px;
                font-size: 9px;
                font-family: Consolas;
            }}
            QPushButton:hover {{
                background: {_s.ACCENT_PRIMARY}22;
            }}
            """
        )
        self._approve.clicked.connect(self._emit_approved)
        buttons.addWidget(self._approve)
        layout.addLayout(buttons)

        root.addWidget(panel)
        self.hide()

    def bind_controller(self, controller) -> None:
        return

    def _emit_approved(self) -> None:
        if isinstance(self._current_action, dict):
            self.sig_approved.emit(self._current_action)

    def show_action(self, action: dict) -> None:
        self._current_action = dict(action or {})
        kind = self._current_action.get("type", "—")
        cmd = self._current_action.get("command", "")
        engine = self._current_action.get("engine", "—")
        payload = self._current_action.get("payload") or {}
        payload_str = json.dumps(payload, ensure_ascii=False)
        if len(payload_str) > 180:
            payload_str = payload_str[:177] + "..."
        self._lbl_type.setText(f"type: {kind}    command: {cmd or '—'}")
        self._lbl_engine.setText(f"engine: {engine}")
        self._lbl_payload.setText(f"payload: {payload_str}")
        self.show()

    def clear(self) -> None:
        self._current_action = None
        self._lbl_type.setText("No pending action.")
        self._lbl_engine.clear()
        self._lbl_payload.clear()
        self.hide()

    def refresh(self) -> None:
        return
