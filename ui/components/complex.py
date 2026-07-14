import re

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FlameLabel(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self._text = text
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(50)
        self.font_obj = QFont("Segoe UI", 14, QFont.Bold)
        self.setFixedHeight(30)
        self.setFixedWidth(120)

    def _animate(self):
        self.phase -= 0.08
        if self.phase < -1.0:
            self.phase = 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        h = self.height()
        grad = QLinearGradient(0, h + (h * self.phase), 0, -h + (h * self.phase))
        grad.setSpread(QLinearGradient.RepeatSpread)

        import core.style as _s

        grad.setColorAt(0.0, QColor(_s.GRADIENT_COLOR))
        grad.setColorAt(0.4, QColor(_s.GRADIENT_COLOR))
        grad.setColorAt(0.5, QColor("white"))
        grad.setColorAt(0.6, QColor(_s.GRADIENT_COLOR))
        grad.setColorAt(1.0, QColor(_s.FG_DIM))

        path = QPainterPath()
        fm = QFontMetrics(self.font_obj)
        text_w = fm.horizontalAdvance(self._text)
        text_h = fm.ascent()
        x = (self.width() - text_w) / 2
        y = (self.height() + text_h) / 2 - fm.descent()
        path.addText(x, y, self.font_obj, self._text)

        painter.setBrush(grad)
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)


class ModeSelector(QWidget):
    modeChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)

        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 5, 20, 5)

        self.btn_op = self._make_box("OPERATOR", True)
        self.btn_ov = self._make_box("MONITOR", False)

        layout.addStretch()
        layout.addWidget(self.btn_op)
        layout.addWidget(self.btn_ov)
        layout.addStretch()

    def _make_box(self, title, active):
        btn = QPushButton(title)
        btn.setFixedSize(120, 35)
        btn.setCheckable(True)
        btn.setChecked(active)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setObjectName("mode_btn_active" if active else "mode_btn_inactive")
        btn.clicked.connect(lambda: self._select(title))
        return btn

    def _select(self, mode):
        is_op = mode == "OPERATOR"
        self.btn_op.setChecked(is_op)
        self.btn_ov.setChecked(not is_op)

        self.btn_op.setObjectName("mode_btn_active" if is_op else "mode_btn_inactive")
        self.btn_ov.setObjectName("mode_btn_active" if not is_op else "mode_btn_inactive")
        self.btn_op.style().unpolish(self.btn_op)
        self.btn_op.style().polish(self.btn_op)
        self.btn_ov.style().unpolish(self.btn_ov)
        self.btn_ov.style().polish(self.btn_ov)

        self.modeChanged.emit(mode)


class GradientLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(2)
        self.offset = 0.0
        self._glow_active = True
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._step)
        self.timer.start(33)

    def set_glow_active(self, active: bool) -> None:
        active = bool(active)
        if self._glow_active == active:
            return
        self._glow_active = active
        if active:
            self.timer.start(33)
        else:
            self.timer.stop()
        self.update()

    def _step(self):
        self.offset = (self.offset + 0.015) % 1.0
        self.repaint()

    def paintEvent(self, event):
        import core.style as _s

        painter = QPainter(self)
        if not self._glow_active:
            painter.fillRect(self.rect(), QColor(_s.BORDER_SUBTLE))
            return

        grad = QLinearGradient(0, 0, self.width(), 0)
        c_gold = QColor(_s.GRADIENT_COLOR)
        c_dark = QColor(_s.BG_SIDEBAR)
        grad.setSpread(QLinearGradient.RepeatSpread)
        w = self.width()
        start_x = -self.offset * w
        grad.setStart(start_x, 0)
        grad.setFinalStop(start_x + w, 0)
        grad.setColorAt(0.0, c_dark)
        grad.setColorAt(0.5, c_gold)
        grad.setColorAt(1.0, c_dark)
        painter.fillRect(self.rect(), grad)


class TagLineEdit(QLineEdit):
    backspaceOnEmpty = Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace and not self.text():
            self.backspaceOnEmpty.emit()
            return
        super().keyPressEvent(event)


class BehaviorTagInput(QFrame):
    tagsChanged = Signal(list)

    def __init__(self, known_tags=None, parent=None):
        super().__init__(parent)
        self._known_tags = {tag.lower() for tag in (known_tags or [])}
        self._tags = []

        self.setObjectName("tag_input_frame")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._chip_layout = QHBoxLayout()
        self._chip_layout.setContentsMargins(0, 0, 0, 0)
        self._chip_layout.setSpacing(6)
        layout.addLayout(self._chip_layout)

        self._input = TagLineEdit()
        self._input.setPlaceholderText("Type tags...")
        self._input.textEdited.connect(self._on_text_edited)
        self._input.returnPressed.connect(self._commit_current_text)
        self._input.backspaceOnEmpty.connect(self._remove_last_tag)
        layout.addWidget(self._input, stretch=1)

    def set_tags(self, tags):
        self._clear_tags()
        for tag in tags:
            self._add_tag(tag, emit_signal=False)
        self.tagsChanged.emit(self._tags.copy())

    def tags(self):
        return self._tags.copy()

    def _clear_tags(self):
        while self._chip_layout.count():
            item = self._chip_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._tags = []

    def _normalize_tag(self, tag):
        return tag.strip().lower()

    def _add_tag(self, tag, emit_signal=True):
        normalized = self._normalize_tag(tag)
        if not normalized or normalized in self._tags:
            return
        chip = QPushButton(normalized)
        chip.setCursor(Qt.PointingHandCursor)
        chip.setProperty("class", "tag_chip")
        chip.clicked.connect(lambda _, t=normalized: self._remove_tag(t))
        self._chip_layout.addWidget(chip)
        self._tags.append(normalized)
        if emit_signal:
            self.tagsChanged.emit(self._tags.copy())

    def _remove_tag(self, tag):
        if tag not in self._tags:
            return
        self._tags = [t for t in self._tags if t != tag]
        for index in range(self._chip_layout.count() - 1, -1, -1):
            widget = self._chip_layout.itemAt(index).widget()
            if widget and widget.text() == tag:
                self._chip_layout.takeAt(index)
                widget.deleteLater()
                break
        self.tagsChanged.emit(self._tags.copy())

    def _remove_last_tag(self):
        if not self._tags:
            return
        self._remove_tag(self._tags[-1])

    def _commit_current_text(self):
        text = self._input.text()
        if text:
            self._add_tag(text)
        self._input.clear()

    def _on_text_edited(self, text):
        if not text:
            return
        if "," not in text and " " not in text:
            return
        parts = [part for part in re.split(r"[,\s]+", text) if part]
        trailing = ""
        if text and text[-1] not in {",", " "}:
            trailing = parts.pop() if parts else text
        for part in parts:
            self._add_tag(part)
        self._input.blockSignals(True)
        self._input.setText(trailing)
        self._input.blockSignals(False)
