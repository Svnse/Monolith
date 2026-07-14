from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QSlider, QHBoxLayout,
    QPushButton, QScrollArea, QSizePolicy, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QDragEnterEvent, QPainter, QPen, QColor, QFont


def import_vbox(widget, l=15, t=25, r=15, b=15):
    from PySide6.QtWidgets import QVBoxLayout
    v = QVBoxLayout(widget)
    v.setContentsMargins(l, t, r, b)
    v.setSpacing(10)
    return v


def apply_elevation(widget: QWidget, level: int = 1) -> None:
    import core.style as s

    colors = [s.BG_MAIN, getattr(s, "BG_SURFACE_1", s.BG_PANEL), getattr(s, "BG_SURFACE_2", s.BG_GROUP)]
    bg = colors[min(max(level, 0), len(colors) - 1)]
    widget.setStyleSheet(
        f"""
        background: {bg};
        border: none;
        border-radius: 8px;
        """
    )


class MonoGroupBox(QFrame):
    """Section, not a box (UI_CONTRACT §2): a dim label + hairline divider.

    Historically this painted a bordered rectangle with the title cut into the
    top border — a literal panel drawn inside whatever pane hosted it, the
    core of the "panel within a panel" effect. Same API (add_widget /
    add_layout / title), flat rendering: content sits directly on the host
    surface, separated only by the section header."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._title = title
        self.setObjectName("mono_group_box")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout_main = import_vbox(self)
        self.layout_main.setContentsMargins(0, 22 if title else 8, 0, 8)

    def add_widget(self, widget):
        self.layout_main.addWidget(widget)

    def add_layout(self, layout):
        self.layout_main.addLayout(layout)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        import core.style as s

        y_top = 10  # header baseline band (kept from the old geometry)
        divider_start = 0

        if self._title:
            font = QFont("Consolas", 8)
            font.setBold(True)
            font.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
            painter.setFont(font)
            fm = painter.fontMetrics()
            painter.setPen(QColor(s.FG_DIM))
            text_y = y_top + fm.height() // 2 - fm.descent()
            painter.drawText(0, int(text_y), self._title)
            divider_start = fm.horizontalAdvance(self._title) + 8

        pen = QPen(QColor(s.BORDER_SUBTLE))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.drawLine(divider_start, y_top, self.width(), y_top)

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


class MonoDragSpin(QFrame):
    """Drag-to-change value control.

    Click and drag up to increase the value, down to decrease it.
    Double-click to type a value directly.
    A small \u2195 handle on the right indicates the drag axis.
    """

    valueChanged = Signal(float)
    _SENS = 4  # pixels of drag per step

    def __init__(self, *, minimum: float = 0, maximum: float = 100,
                 step: float = 1, decimals: int = 0, parent=None):
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._decimals = decimals
        self._value = float(minimum)
        self._dragging = False
        self._drag_y = 0
        self._drag_accum = 0.0
        self._editing = False

        self.setObjectName("drag_spin")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(26)
        self.setCursor(Qt.SizeVerCursor)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 4, 0)
        lay.setSpacing(0)

        self._lbl = QLabel(self._fmt(self._value))
        self._lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._edit = QLineEdit()
        self._edit.setFrame(False)
        self._edit.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._edit.hide()
        self._edit.editingFinished.connect(self._commit_edit)
        self._edit.installEventFilter(self)

        self._handle = QLabel("\u2195")
        self._handle.setFixedWidth(14)
        self._handle.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lay.addWidget(self._lbl)
        lay.addWidget(self._edit)
        lay.addWidget(self._handle)

        self.refresh_style()

    # ---- public API ----

    def value(self) -> float:
        return self._value

    def setValue(self, v: float) -> None:
        v = max(self._min, min(self._max, float(v)))
        if v != self._value:
            self._value = v
            if not self._editing:
                self._lbl.setText(self._fmt(v))
            self.valueChanged.emit(v)

    def setRange(self, minimum: float, maximum: float) -> None:
        self._min = float(minimum)
        self._max = float(maximum)
        self.setValue(self._value)

    def setSingleStep(self, step: float) -> None:
        self._step = float(step)

    def setDecimals(self, n: int) -> None:
        self._decimals = n
        self._lbl.setText(self._fmt(self._value))

    def refresh_style(self) -> None:
        import core.style as s
        self.setStyleSheet(
            f"QFrame#drag_spin {{ background: {s.BG_INPUT}; border: 1px solid {s.BORDER_DARK}; "
            f"border-radius: 3px; }}"
        )
        self._lbl.setStyleSheet(
            f"color: {s.FG_TEXT}; font-size: 12px; background: transparent; border: none;"
        )
        self._edit.setStyleSheet(
            f"background: transparent; border: none; color: {s.FG_TEXT}; font-size: 12px;"
        )
        self._handle.setStyleSheet(
            f"color: {s.FG_DIM}; font-size: 10px; background: transparent; border: none;"
        )

    # ---- internal ----

    def _fmt(self, v: float) -> str:
        if self._decimals == 0:
            return str(int(round(v)))
        return f"{v:.{self._decimals}f}"

    # ---- mouse events ----

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._editing:
            self._dragging = True
            self._drag_y = event.globalPosition().toPoint().y()
            self._drag_accum = 0.0
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging and not self._editing:
            dy = self._drag_y - event.globalPosition().toPoint().y()
            self._drag_y = event.globalPosition().toPoint().y()
            self._drag_accum += dy
            while abs(self._drag_accum) >= self._SENS:
                step = self._step if self._drag_accum > 0 else -self._step
                self._drag_accum -= self._SENS if self._drag_accum > 0 else -self._SENS
                self.setValue(self._value + step)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start_edit()
            event.accept()

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self._edit and event.type() == QEvent.FocusOut:
            self._commit_edit()
        return super().eventFilter(obj, event)

    def _start_edit(self):
        if self._editing:
            return
        self._editing = True
        self._edit.setText(self._fmt(self._value))
        self._lbl.hide()
        self._edit.show()
        self._edit.setFocus()
        self._edit.selectAll()

    def _commit_edit(self):
        if not self._editing:
            return
        txt = self._edit.text().strip()
        try:
            val = float(txt)
        except Exception:
            val = self._value
        self._editing = False
        self._edit.hide()
        self._lbl.show()
        self.setValue(val)
