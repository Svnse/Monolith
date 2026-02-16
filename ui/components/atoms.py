from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QSlider, QHBoxLayout,
    QPushButton, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QDragEnterEvent


def import_vbox(widget, l=15, t=25, r=15, b=15):
    from PySide6.QtWidgets import QVBoxLayout
    v = QVBoxLayout(widget)
    v.setContentsMargins(l, t, r, b)
    v.setSpacing(10)
    return v


class MonoGroupBox(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("mono_group_box")
        self.layout_main = import_vbox(self)
        self.lbl_title = QLabel(title, self)
        self.lbl_title.setObjectName("group_title")
        self.lbl_title.move(10, -3)

    def add_widget(self, widget):
        self.layout_main.addWidget(widget)

    def add_layout(self, layout):
        self.layout_main.addLayout(layout)


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
        self.setFixedSize(60, 45)
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
        self.setProperty("checked", checked)
        self.style().unpolish(self)
        self.style().polish(self)
        self.lbl_text.setProperty("active", "true" if checked else "false")
        self.lbl_text.style().unpolish(self.lbl_text)
        self.lbl_text.style().polish(self.lbl_text)

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
