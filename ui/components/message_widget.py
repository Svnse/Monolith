from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.style import ACCENT_GOLD, BG_INPUT, FG_DIM, FG_TEXT


class MessageWidget(QWidget):
    sig_delete = Signal(int)
    sig_edit = Signal(int)
    sig_regen = Signal(int)

    def __init__(self, index: int, role: str, text: str, timestamp: str):
        super().__init__()
        self._index = index
        self._role = role
        self._content = text or ""

        self.setAttribute(Qt.WA_Hover, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(8)

        self.lbl_role = QLabel((role or "").upper())
        self.lbl_role.setStyleSheet(
            f"color: {ACCENT_GOLD if role == 'assistant' else FG_TEXT}; font-size: 10px; font-weight: bold;"
        )
        head.addWidget(self.lbl_role)

        self.lbl_time = QLabel(timestamp or "")
        self.lbl_time.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")
        head.addWidget(self.lbl_time)
        head.addStretch()

        self.actions = QWidget()
        actions_layout = QHBoxLayout(self.actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)

        self.btn_edit = QPushButton("Edit")
        self.btn_edit.clicked.connect(lambda: self.sig_edit.emit(self._index))
        self.btn_edit.setVisible(role == "user")

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(lambda: self.sig_delete.emit(self._index))

        self.btn_regen = QPushButton("Regen")
        self.btn_regen.clicked.connect(lambda: self.sig_regen.emit(self._index))
        self.btn_regen.setVisible(role == "assistant")

        btn_style = (
            f"QPushButton {{"
            f"background: {BG_INPUT}; color: {FG_DIM}; border: 1px solid #333;"
            f"padding: 2px 6px; font-size: 9px; border-radius: 2px;"
            f"}}"
            f"QPushButton:hover {{ color: {FG_TEXT}; border: 1px solid {ACCENT_GOLD}; }}"
        )
        for btn in (self.btn_edit, self.btn_delete, self.btn_regen):
            btn.setStyleSheet(btn_style)
            actions_layout.addWidget(btn)

        self.actions.setVisible(False)
        head.addWidget(self.actions)

        root.addLayout(head)

        self.lbl_content = QLabel()
        self.lbl_content.setTextFormat(Qt.PlainText)
        self.lbl_content.setWordWrap(True)
        self.lbl_content.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")
        self.lbl_content.setText(self._content)
        root.addWidget(self.lbl_content)

        self.setStyleSheet(f"MessageWidget {{ border: 1px solid #222; background: {BG_INPUT}; }}")

    def enterEvent(self, event):
        self.actions.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.actions.setVisible(False)
        super().leaveEvent(event)

    def append_token(self, token: str):
        if not token:
            return
        self._content += token
        self.lbl_content.setText(self._content)

    def finalize(self):
        self.lbl_content.setText(self._content)

    def set_index(self, idx: int):
        self._index = idx
