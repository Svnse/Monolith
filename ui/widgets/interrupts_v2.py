from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QHBoxLayout, QVBoxLayout

import core.style as _s


class _InterruptCardBase(QFrame):
    sig_decision = Signal(dict)

    def __init__(self, title: str, request_id: str, parent=None):
        super().__init__(parent)
        self.request_id = str(request_id or "")
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"""
            QFrame {{
                background: {_s.BG_INPUT};
                border: 1px solid {_s.BORDER_LIGHT};
                border-radius: 4px;
            }}
            QLabel {{ color: {_s.FG_TEXT}; }}
            QPushButton {{
                background: {_s.BG_BUTTON};
                border: 1px solid {_s.BORDER_LIGHT};
                color: {_s.FG_TEXT};
                padding: 4px 10px;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: {_s.BG_BUTTON_HOVER}; color: {_s.ACCENT_PRIMARY}; }}
            """
        )

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setSpacing(8)

        heading = QLabel(title)
        heading.setStyleSheet(f"color: {_s.ACCENT_PRIMARY}; font-weight: bold;")
        self._layout.addWidget(heading)

    def _emit(self, payload: dict):
        self.sig_decision.emit(payload)


class CapabilityInterruptCard(_InterruptCardBase):
    def __init__(self, request_id: str, request: dict, parent=None):
        super().__init__("Capability Request", request_id=request_id, parent=parent)
        request = request if isinstance(request, dict) else {}

        details = QLabel(
            f"Scope: {request.get('scope', 'read')}\n"
            f"Path: {request.get('path_pattern', '**')}\n"
            f"Reason: {request.get('reason', '')}"
        )
        details.setWordWrap(True)
        details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._layout.addWidget(details)

        row = QHBoxLayout()
        deny_btn = QPushButton("Deny")
        approve_btn = QPushButton("Approve")
        deny_btn.clicked.connect(lambda: self._emit({"request_id": self.request_id, "approved": False}))
        approve_btn.clicked.connect(lambda: self._emit({"request_id": self.request_id, "approved": True}))
        row.addWidget(deny_btn)
        row.addWidget(approve_btn)
        self._layout.addLayout(row)


class ClarificationInterruptCard(_InterruptCardBase):
    def __init__(self, request_id: str, question: str, options: list[str] | None = None, parent=None):
        super().__init__("Clarification Request", request_id=request_id, parent=parent)
        prompt = QLabel(question or "The agent asked for clarification.")
        prompt.setWordWrap(True)
        prompt.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._layout.addWidget(prompt)

        choices = [str(option) for option in (options or []) if str(option).strip()] or ["Yes", "No"]
        row = QHBoxLayout()
        for choice in choices:
            btn = QPushButton(choice)
            btn.clicked.connect(lambda _checked=False, selected=choice: self._emit({"request_id": self.request_id, "choice": selected}))
            row.addWidget(btn)
        self._layout.addLayout(row)


class DestructiveInterruptCard(_InterruptCardBase):
    def __init__(self, request_id: str, warning: str, parent=None):
        super().__init__("Destructive Confirmation", request_id=request_id, parent=parent)
        alert = QLabel("⚠ This action is destructive.")
        alert.setStyleSheet(f"color: {_s.ACCENT_DANGER}; font-weight: bold;")
        self._layout.addWidget(alert)

        details = QLabel(warning or "This action may overwrite files. Proceed?")
        details.setWordWrap(True)
        details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._layout.addWidget(details)

        row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        proceed_btn = QPushButton("Proceed")
        cancel_btn.clicked.connect(lambda: self._emit({"request_id": self.request_id, "approved": False}))
        proceed_btn.clicked.connect(lambda: self._emit({"request_id": self.request_id, "approved": True}))
        row.addWidget(cancel_btn)
        row.addWidget(proceed_btn)
        self._layout.addLayout(row)
