from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QSlider, QHBoxLayout, QVBoxLayout,
    QPushButton, QScrollArea, QSizePolicy, QTextEdit
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QDragEnterEvent, QPainter, QPen, QColor, QFont


def import_vbox(widget, l=15, t=25, r=15, b=15):
    from PySide6.QtWidgets import QVBoxLayout
    v = QVBoxLayout(widget)
    v.setContentsMargins(l, t, r, b)
    v.setSpacing(10)
    return v


class MonoGroupBox(QFrame):
    """Group box with a bordered rectangle and title embedded in the top border."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._title = title
        self._header_buttons: list[QPushButton] = []
        self.setObjectName("mono_group_box")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout_main = import_vbox(self)
        self.layout_main.setContentsMargins(10, 22, 10, 8)

    def add_widget(self, widget):
        self.layout_main.addWidget(widget)

    def add_layout(self, layout):
        self.layout_main.addLayout(layout)

    def add_header_action(self, text, callback):
        """Add a small text button in the top-right of the group box title bar."""
        import core.style as s
        btn = QPushButton(text, self)
        btn.setFixedHeight(16)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"color: {s.FG_DIM}; font-family: Consolas; font-size: 8px; "
            f"font-weight: bold; letter-spacing: 1px; padding: 0 6px; }}"
            f"QPushButton:hover {{ color: {s.ACCENT_PRIMARY}; }}"
        )
        btn.clicked.connect(callback)
        self._header_buttons.append(btn)
        self._position_header_buttons()
        return btn

    def _position_header_buttons(self):
        """Position header action buttons at the top-right of the frame."""
        x_offset = self.width() - 10
        for btn in reversed(self._header_buttons):
            btn.adjustSize()
            bw = max(btn.sizeHint().width(), 40)
            btn.setFixedWidth(bw)
            x_offset -= bw + 2
            btn.move(x_offset, 2)
            btn.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._header_buttons:
            self._position_header_buttons()

    def _resolve_parent_bg(self):
        """Walk up the parent chain to find the first opaque background color."""
        import core.style as s
        widget = self.parentWidget()
        while widget is not None:
            role = widget.backgroundRole()
            palette_color = widget.palette().color(role)
            if palette_color.alpha() == 255 and palette_color != QColor(0, 0, 0, 255):
                return palette_color
            widget = widget.parentWidget()
        return QColor(s.BG_MAIN)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        import core.style as s
        from PySide6.QtCore import QRectF

        border_color = QColor(s.BORDER_LIGHT)
        bg_color = QColor(s.BG_MAIN)
        radius = 4
        y_top = 10  # the top border line sits at y=10

        # Draw the full rounded rectangle border (all 4 sides)
        pen = QPen(border_color)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        box_rect = QRectF(0.5, y_top, self.width() - 1, self.height() - y_top - 0.5)
        painter.drawRoundedRect(box_rect, radius, radius)

        if not self._title:
            painter.end()
            return

        # Title font
        font = QFont("Consolas", 8)
        font.setBold(True)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        painter.setFont(font)
        fm = painter.fontMetrics()
        title_text = self._title
        text_width = fm.horizontalAdvance(title_text)
        text_height = fm.height()

        # Position title on the top border
        x_start = 12
        pad_h = 6  # horizontal padding around title text

        # Fill a background rect behind the title to "cut" the border
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg_color)
        painter.drawRect(QRectF(x_start, y_top - text_height / 2,
                                text_width + pad_h * 2, text_height))

        # Draw the title text
        text_color = QColor(s.FG_DIM)
        painter.setPen(text_color)
        text_y = y_top + text_height // 2 - fm.descent()
        painter.drawText(int(x_start + pad_h), int(text_y), title_text)

        painter.end()


class MonoButton(QPushButton):
    def __init__(self, text, accent=False):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("accent", "true" if accent else "false")


class MonoTriangleButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(18, 18)
        self.setFocusPolicy(Qt.NoFocus)


class MonoSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label, min_v, max_v, init_v, is_int=False):
        super().__init__()
        self.is_int = is_int
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.lbl = QLabel(label)
        self.lbl.setObjectName("slider_label")
        val_str = str(int(init_v) if is_int else f"{init_v:.2f}")
        self.val_lbl = QLabel(val_str)
        self.val_lbl.setObjectName("slider_value")
        self.val_lbl.setFixedWidth(40)
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = QSlider(Qt.Horizontal)
        if is_int:
            self.slider.setRange(int(min_v), int(max_v))
            self.slider.setValue(int(init_v))
        else:
            self.slider.setRange(int(min_v * 100), int(max_v * 100))
            self.slider.setValue(int(init_v * 100))
        self.slider.valueChanged.connect(self._on_change)
        layout.addWidget(self.lbl)
        layout.addWidget(self.slider)
        layout.addWidget(self.val_lbl)

    def _on_change(self, val):
        real_val = val if self.is_int else val / 100.0
        val_str = str(int(real_val) if self.is_int else f"{real_val:.2f}")
        self.val_lbl.setText(val_str)
        self.valueChanged.emit(float(real_val))


class SidebarButton(QPushButton):
    def __init__(self, icon_char, text, checkable=True):
        super().__init__()
        self.setCheckable(checkable)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(30)
        self.setMinimumWidth(60)
        if checkable:
            self.setAutoExclusive(False)
        self.setAcceptDrops(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_text = QLabel(text)
        self.lbl_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_text)
        self.update_style(False)

    def nextCheckState(self):
        pass

    def setChecked(self, checked):
        super().setChecked(checked)
        self.update_style(checked)

    def update_style(self, checked):
        import core.style as s
        self.setProperty("checked", checked)
        self.style().unpolish(self)
        self.style().polish(self)
        color = s.ACCENT_PRIMARY if checked else s.FG_DIM
        self.lbl_text.setStyleSheet(
            f"color: {color}; font-size: 9px; font-weight: bold;"
            f" letter-spacing: 1px; background: transparent;"
        )

    def enterEvent(self, event):
        if not self.isChecked():
            import core.style as s
            self.lbl_text.setStyleSheet(
                f"color: {s.FG_TEXT}; font-size: 9px; font-weight: bold;"
                f" letter-spacing: 1px; background: transparent;"
            )
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update_style(self.isChecked())
        super().leaveEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()


class CollapsibleSection(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.layout_main = import_vbox(self, 0, 0, 0, 0)
        self.layout_main.setSpacing(0)
        self.btn_toggle = QPushButton(title)
        self.btn_toggle.setObjectName("collapsible_toggle")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.clicked.connect(self.toggle_animation)
        self.layout_main.addWidget(self.btn_toggle)
        self.content_area = QScrollArea()
        self.content_area.setObjectName("collapsible_content")
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setFrameShape(QFrame.NoFrame)
        self.content_area.setWidgetResizable(True)
        self.layout_main.addWidget(self.content_area)
        self.anim = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.anim.setDuration(300)
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)

    def set_content_layout(self, layout):
        w = QWidget()
        w.setLayout(layout)
        self.content_area.setWidget(w)

    def toggle_animation(self):
        checked = self.btn_toggle.isChecked()
        content_height = self.content_area.widget().layout().sizeHint().height() if self.content_area.widget() else 100
        self.anim.setStartValue(0 if checked else content_height)
        self.anim.setEndValue(content_height if checked else 0)
        self.anim.start()


class CollapsibleStepWidget(QFrame):
    """Unified collapsible agent step block with expand/collapse arrow."""

    sig_height_changed = Signal()

    def __init__(self, step_number: int, label: str, status_icon: str = "…", parent=None):
        super().__init__(parent)
        import core.style as s
        self._expanded = False
        self._step_number = step_number
        self._label = label
        self._status_icon = status_icon
        self._detail_text = ""

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"CollapsibleStepWidget {{ background: {s.BG_INPUT}; "
            f"border: 1px solid {s.BORDER_SUBTLE}; border-radius: 2px; }}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header row (always visible)
        self._header = QFrame()
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setStyleSheet(f"border: none; background: transparent;")
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(6)

        self._arrow = QLabel("▸")
        self._arrow.setFixedWidth(12)
        self._arrow.setStyleSheet(
            f"color: {s.FG_DIM}; font-size: 10px; border: none; background: transparent;"
        )

        self._lbl_number = QLabel(f"[{step_number}]")
        self._lbl_number.setFixedWidth(28)
        self._lbl_number.setStyleSheet(
            f"color: {s.FG_DIM}; font-family: Consolas; font-size: 9px; "
            f"border: none; background: transparent;"
        )

        self._lbl_icon = QLabel(status_icon)
        self._lbl_icon.setFixedWidth(14)
        self._lbl_icon.setStyleSheet(
            f"color: {s.ACCENT_PRIMARY}; font-size: 10px; "
            f"border: none; background: transparent;"
        )

        self._lbl_label = QLabel(label)
        self._lbl_label.setStyleSheet(
            f"color: {s.FG_TEXT}; font-family: Consolas; font-size: 10px; "
            f"border: none; background: transparent;"
        )

        header_layout.addWidget(self._arrow)
        header_layout.addWidget(self._lbl_number)
        header_layout.addWidget(self._lbl_icon)
        header_layout.addWidget(self._lbl_label)
        header_layout.addStretch()
        root.addWidget(self._header)

        # Detail area (hidden by default)
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setVisible(False)
        self._detail.setMaximumHeight(200)
        self._detail.setStyleSheet(
            f"QTextEdit {{ background: {s.BG_MAIN}; color: {s.FG_TEXT}; "
            f"border: none; border-top: 1px solid {s.BORDER_SUBTLE}; "
            f"font-family: Consolas; font-size: 9px; padding: 6px; }}"
        )
        root.addWidget(self._detail)

        self._header.mousePressEvent = self._on_header_click

    def _on_header_click(self, event):
        self._expanded = not self._expanded
        self._detail.setVisible(self._expanded)
        self._arrow.setText("▾" if self._expanded else "▸")
        self.sig_height_changed.emit()

    def update_status(self, status_icon: str):
        import core.style as s
        self._status_icon = status_icon
        self._lbl_icon.setText(status_icon)
        color_map = {"✓": s.ACCENT_PRIMARY, "✗": s.FG_ERROR, "▶": s.FG_WARN, "…": s.FG_DIM}
        self._lbl_icon.setStyleSheet(
            f"color: {color_map.get(status_icon, s.FG_DIM)}; font-size: 10px; "
            f"border: none; background: transparent;"
        )

    def update_label(self, label: str):
        self._label = label
        self._lbl_label.setText(label)

    def set_detail(self, text: str):
        self._detail_text = text
        self._detail.setPlainText(text)

    def append_detail(self, text: str):
        self._detail_text += text
        self._detail.setPlainText(self._detail_text)
