from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class _IconAction(QPushButton):
    """Tiny icon-only action button for message hover bar."""

    def __init__(self, icon_char: str, tooltip: str):
        super().__init__(icon_char)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(22, 22)
        self.setProperty("class", "msg_icon_action")


class MessageWidget(QFrame):
    sig_delete = Signal(int)
    sig_edit = Signal(int)
    sig_regen = Signal(int)

    def __init__(self, index: int, role: str, text: str, timestamp: str):
        super().__init__()
        self._index = index
        self._role = role
        self._content = text or ""

        self.setAttribute(Qt.WA_Hover, True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("class", "MessageWidget")
        self.setProperty("role", role)

        is_assistant = role == "assistant"
        is_system = role == "system"

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 3, 10, 3 if not is_assistant else 8)
        root.setSpacing(4)

        head = QHBoxLayout()
        head.setSpacing(6)

        self.lbl_role = QLabel((role or "").upper())
        self.lbl_role.setObjectName("msg_role")
        self.lbl_role.setProperty("role", role)
        head.addWidget(self.lbl_role)

        pretty_ts = (timestamp or "")
        if "T" in pretty_ts and len(pretty_ts) >= 16:
            pretty_ts = pretty_ts[11:16]
        self.lbl_time = QLabel(pretty_ts)
        self.lbl_time.setObjectName("msg_time")
        head.addWidget(self.lbl_time)
        head.addStretch()

        self.actions = QWidget()
        self.actions.setObjectName("msg_actions")
        actions_layout = QHBoxLayout(self.actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(2)

        if not is_system:
            if role == "user":
                self.btn_edit = _IconAction("✎", "Edit")
                self.btn_edit.clicked.connect(lambda: self.sig_edit.emit(self._index))
                actions_layout.addWidget(self.btn_edit)

            if is_assistant:
                self.btn_regen = _IconAction("⟲", "Regenerate")
                self.btn_regen.clicked.connect(lambda: self.sig_regen.emit(self._index))
                actions_layout.addWidget(self.btn_regen)

            self.btn_delete = _IconAction("✕", "Delete")
            self.btn_delete.clicked.connect(lambda: self.sig_delete.emit(self._index))
            actions_layout.addWidget(self.btn_delete)

        self.actions.setVisible(False)
        head.addWidget(self.actions)
        root.addLayout(head)

        self.lbl_content = QLabel()
        self.lbl_content.setObjectName("msg_content")
        self.lbl_content.setProperty("role", role)
        self.lbl_content.setTextFormat(Qt.PlainText)
        self.lbl_content.setWordWrap(True)
        self.lbl_content.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.lbl_content.setCursor(Qt.IBeamCursor)
        self.lbl_content.setText(self._content)
        root.addWidget(self.lbl_content)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def sizeHint(self):
        w = self.width() if self.width() > 50 else 600
        margins = self.layout().contentsMargins()
        content_w = w - margins.left() - margins.right() - 2
        content_h = self.lbl_content.heightForWidth(max(content_w, 60))
        if content_h <= 0:
            content_h = self.lbl_content.sizeHint().height()
        header_h = 20
        spacing = self.layout().spacing()
        total = margins.top() + header_h + spacing + content_h + margins.bottom()
        return QSize(w, max(total, 30))

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
