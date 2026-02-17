from PySide6.QtCore import Qt, QSize, Signal, QTimer
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget, QTextEdit,
)
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QTextOption

import core.style as _s


class _AutoTextView(QTextEdit):
    """Read-only text display that auto-sizes to its content height."""
    heightChanged = Signal(int)

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.setCursor(Qt.IBeamCursor)
        self.document().setDocumentMargin(0)
        self.document().contentsChanged.connect(self._schedule_update)
        self.setStyleSheet("""
            QTextEdit {
                background: transparent; border: none; padding: 0px; margin: 0px;
                selection-background-color: rgba(255, 255, 255, 0.15);
                selection-color: inherit;
            }
        """)
        font = QFont("Consolas, Segoe UI Emoji, Noto Color Emoji, Apple Color Emoji", 12)
        self.setFont(font)
        self._last_h = 18
        self.setFixedHeight(18)
        self._pending = False

    def _schedule_update(self):
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._do_update_height)

    def _do_update_height(self):
        self._pending = False
        self._update_height()

    def _update_height(self):
        vw = self.viewport().width()
        if vw < 10:
            vw = self.width() - 4
        doc = self.document()
        doc.setTextWidth(max(vw, 20))
        h = int(doc.size().height()) + 6
        h = max(h, 18)
        if h != self._last_h:
            self._last_h = h
            self.setFixedHeight(h)
            self.heightChanged.emit(h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_height()

    def ideal_height(self, available_width: int) -> int:
        doc = self.document().clone()
        doc.setDocumentMargin(0)
        doc.setTextWidth(max(available_width, 20))
        h = int(doc.size().height()) + 6
        return max(h, 18)


class _IconAction(QPushButton):
    """Tiny icon-only action button for message hover bar."""

    def __init__(self, icon_char: str, tooltip: str):
        super().__init__(icon_char)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(24, 24)
        self.setProperty("class", "msg_icon_action")


class MessageWidget(QFrame):
    sig_delete = Signal(int)
    sig_edit = Signal(int)
    sig_regen = Signal(int)
    sig_height_changed = Signal()  # emitted when internal height changes

    _HEADER_H = 16
    _MARGINS = (8, 4, 8, 4)  # left, top, right, bottom
    _SPACING = 1

    def __init__(self, index: int, role: str, text: str, timestamp: str):
        super().__init__()
        self._index = index
        self._role = role
        self._content = text or ""

        self.setAttribute(Qt.WA_Hover, True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("class", "MessageWidget")
        self.setProperty("role", role)
        self.setStyleSheet(f"border-left: 2px solid {_s.BORDER_SUBTLE};")

        is_assistant = role == "assistant"
        is_system = role == "system"

        root = QVBoxLayout(self)
        root.setContentsMargins(*self._MARGINS)
        root.setSpacing(self._SPACING)

        head = QHBoxLayout()
        head.setSpacing(6)

        self.lbl_role = QLabel((role or "").upper())
        self.lbl_role.setObjectName("msg_role")
        self.lbl_role.setProperty("role", role)
        self.lbl_role.setFixedHeight(self._HEADER_H)
        head.addWidget(self.lbl_role)

        pretty_ts = (timestamp or "")
        if "T" in pretty_ts and len(pretty_ts) >= 16:
            pretty_ts = pretty_ts[11:16]
        self.lbl_time = QLabel(pretty_ts)
        self.lbl_time.setObjectName("msg_time")
        self.lbl_time.setFixedHeight(self._HEADER_H)
        head.addWidget(self.lbl_time)
        head.addStretch()

        self.actions = QWidget()
        self.actions.setObjectName("msg_actions")
        actions_layout = QHBoxLayout(self.actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)

        if not is_system:
            if role == "user":
                self.btn_edit = _IconAction("\u270e", "Edit")
                self.btn_edit.clicked.connect(lambda: self.sig_edit.emit(self._index))
                actions_layout.addWidget(self.btn_edit)

            if is_assistant:
                self.btn_regen = _IconAction("\u27f2", "Regenerate")
                self.btn_regen.clicked.connect(lambda: self.sig_regen.emit(self._index))
                actions_layout.addWidget(self.btn_regen)

            self.btn_delete = _IconAction("\u2715", "Delete")
            self.btn_delete.clicked.connect(lambda: self.sig_delete.emit(self._index))
            actions_layout.addWidget(self.btn_delete)

        self.actions.setVisible(False)
        head.addWidget(self.actions)
        root.addLayout(head)

        self.text_view = _AutoTextView()
        self.text_view.setObjectName("msg_content")
        self.text_view.setProperty("role", role)
        self.text_view.setPlainText(self._content)
        self.text_view.heightChanged.connect(self._on_text_height_changed)
        root.addWidget(self.text_view)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def _on_text_height_changed(self, text_h):
        m = self._MARGINS
        total = m[1] + self._HEADER_H + self._SPACING + text_h + m[3]
        self.setFixedHeight(max(total, 28))
        self.sig_height_changed.emit()

    def sizeHint(self):
        w = self.width() if self.width() > 50 else 600
        m = self._MARGINS
        content_w = w - m[0] - m[2] - 2  # 2px for border-left
        text_h = self.text_view.ideal_height(content_w)
        total = m[1] + self._HEADER_H + self._SPACING + text_h + m[3]
        return QSize(w, max(total, 28))

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
        cursor = self.text_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)

    def finalize(self):
        self.text_view.setPlainText(self._content)

    def set_index(self, idx: int):
        self._index = idx
