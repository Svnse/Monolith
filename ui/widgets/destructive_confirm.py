from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QHBoxLayout, QVBoxLayout

import core.style as _s


class DestructiveConfirmationDialog(QDialog):
    def __init__(self, warning: str, parent=None):
        super().__init__(parent)
        self._approved = False

        self.setWindowTitle("Destructive Confirmation")
        self.setModal(True)
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumWidth(500)
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
        title = QLabel("⚠ This action is destructive.")
        title.setStyleSheet(f"color: {_s.ACCENT_DANGER}; font-weight: bold;")
        layout.addWidget(title)

        detail = QLabel(warning or "This operation may overwrite or delete files. Proceed?")
        detail.setWordWrap(True)
        detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(detail)

        row = QHBoxLayout()
        btn_proceed = QPushButton("Proceed")
        btn_cancel = QPushButton("Cancel")
        btn_proceed.clicked.connect(self._proceed)
        btn_cancel.clicked.connect(self._cancel)
        row.addWidget(btn_cancel)
        row.addWidget(btn_proceed)
        layout.addLayout(row)

    @property
    def approved(self) -> bool:
        return self._approved

    def _proceed(self):
        self._approved = True
        self.accept()

    def _cancel(self):
        self._approved = False
        self.reject()
