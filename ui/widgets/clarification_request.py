from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

import core.style as _s


class ClarificationRequestDialog(QDialog):
    def __init__(self, question: str, options: list[str] | None = None, parent=None):
        super().__init__(parent)
        self._choice: str | None = None
        self._options = [str(opt) for opt in (options or []) if str(opt).strip()]

        self.setWindowTitle("Clarification Needed")
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
                text-align: left;
            }}
            QPushButton:hover {{ background: {_s.BG_BUTTON_HOVER}; color: {_s.ACCENT_PRIMARY}; }}
            """
        )

        layout = QVBoxLayout(self)
        prompt = QLabel(question or "The agent requested clarification.")
        prompt.setWordWrap(True)
        prompt.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(prompt)

        choices = self._options if self._options else ["Yes", "No"]
        for choice in choices:
            btn = QPushButton(choice)
            btn.clicked.connect(lambda _checked=False, selected=choice: self._select(selected))
            layout.addWidget(btn)

    @property
    def choice(self) -> str | None:
        return self._choice

    def _select(self, choice: str):
        self._choice = choice
        self.accept()
