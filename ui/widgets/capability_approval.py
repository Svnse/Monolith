from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFormLayout, QLabel, QPushButton, QVBoxLayout

import core.style as _s


class CapabilityApprovalDialog(QDialog):
    def __init__(self, request_id: str, request: dict, parent=None):
        super().__init__(parent)
        self._request_id = request_id
        self._request = request if isinstance(request, dict) else {}
        self._approved = False

        self.setWindowTitle("Capability Request")
        self.setModal(True)
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumWidth(460)
        self.setStyleSheet(
            f"""
            QDialog {{ background: {_s.BG_MAIN}; color: {_s.FG_TEXT}; }}
            QLabel {{ color: {_s.FG_TEXT}; }}
            QPushButton {{
                background: {_s.BG_BUTTON}; border: 1px solid {_s.BORDER_LIGHT};
                color: {_s.FG_TEXT}; padding: 8px 14px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {_s.BG_BUTTON_HOVER}; color: {_s.ACCENT_PRIMARY}; }}
            """
        )

        layout = QVBoxLayout(self)
        intro = QLabel("The agent is requesting additional capabilities.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        lbl_scope = QLabel(str(self._request.get("scope", "read")))
        lbl_scope.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl_pattern = QLabel(str(self._request.get("path_pattern", "**")))
        lbl_pattern.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl_reason = QLabel(str(self._request.get("reason", "")))
        lbl_reason.setWordWrap(True)
        lbl_reason.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Scope:", lbl_scope)
        form.addRow("Path Pattern:", lbl_pattern)
        form.addRow("Reason:", lbl_reason)
        layout.addLayout(form)

        button_row = QVBoxLayout()
        btn_yes = QPushButton("Yes")
        btn_no = QPushButton("No")
        btn_yes.clicked.connect(self._approve)
        btn_no.clicked.connect(self._deny)
        button_row.addWidget(btn_yes)
        button_row.addWidget(btn_no)
        layout.addLayout(button_row)

    @property
    def decision_payload(self) -> dict:
        return {
            "approved": self._approved,
            "request_id": self._request_id,
        }

    def _approve(self):
        self._approved = True
        self.accept()

    def _deny(self):
        self._approved = False
        self.reject()
