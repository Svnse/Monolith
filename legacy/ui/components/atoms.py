from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QSlider, QHBoxLayout, 
    QPushButton, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QDragEnterEvent

from core.style import BG_GROUP, BORDER_DARK, FG_TEXT, FG_DIM, FG_ACCENT, ACCENT_GOLD

# ======================
# HELPER
# ======================
def import_vbox(widget, l=15, t=25, r=15, b=15):
    from PySide6.QtWidgets import QVBoxLayout
    v = QVBoxLayout(widget)
    v.setContentsMargins(l, t, r, b)
    v.setSpacing(10)
    return v

# ======================
# BASIC UI PRIMITIVES
# ======================

class SkeetGroupBox(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            SkeetGroupBox {{
                background: {BG_GROUP};
                border: 1px solid {BORDER_DARK};
                margin-top: 10px; 
            }}
        """)
        self.layout_main = import_vbox(self)
        self.lbl_title = QLabel(title, self)
        self.lbl_title.setStyleSheet(f"""
            color: {FG_TEXT}; 
            font-weight: bold; font-size: 11px;
            background: {BG_GROUP}; padding: 0 4px;
        """)
        self.lbl_title.adjustSize()
        self.lbl_title.move(10, -3)

    def add_widget(self, widget):
        self.layout_main.addWidget(widget)

    def add_layout(self, layout):
        self.layout_main.addLayout(layout)

class SkeetButton(QPushButton):
    def __init__(self, text, accent=False):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        col = FG_ACCENT if accent else FG_TEXT
        self.setStyleSheet(f"""
            QPushButton {{ background: #181818; border: 1px solid #333; color: {FG_DIM}; padding: 6px 12px; font-size: 11px; font-weight: bold; border-radius: 2px; }}
            QPushButton:hover {{ background: #222; color: {col}; border: 1px solid {col}; }}
            QPushButton:disabled {{ background: #111; color: #333; border: 1px solid #222; }}
        """)

class SkeetSlider(QWidget):
    valueChanged = Signal(float)
    def __init__(self, label, min_v, max_v, init_v, is_int=False):
        super().__init__()
        self.is_int = is_int
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.lbl = QLabel(label)
        self.lbl.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")
        val_str = str(int(init_v) if is_int else f"{init_v:.2f}")
        self.val_lbl = QLabel(val_str)
        self.val_lbl.setStyleSheet(f"color: {FG_TEXT}; font-size: 11px; font-weight: bold;")
        self.val_lbl.setFixedWidth(40)
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = QSlider(Qt.Horizontal)
        if is_int:
            self.slider.setRange(int(min_v), int(max_v))
            self.slider.setValue(int(init_v))
        else:
            self.slider.setRange(int(min_v*100), int(max_v*100))
            self.slider.setValue(int(init_v*100))
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height: 4px; background: #222; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {FG_DIM}; width: 8px; margin: -2px 0; border-radius: 4px; }}
            QSlider::handle:horizontal:hover {{ background: {FG_ACCENT}; }}
            QSlider::sub-page:horizontal {{ background: {FG_ACCENT}; border-radius: 2px; }}
        """)
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
        self.setFixedSize(60, 45) # Shorter height
        if checkable: self.setAutoExclusive(False)
        self.setAcceptDrops(True) 
        
        # Text Only Layout (Skeet Style)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        self.lbl_text = QLabel(text)
        self.lbl_text.setAlignment(Qt.AlignCenter)
        self.lbl_text.setStyleSheet(f"color: {FG_DIM}; font-size: 9px; font-weight: bold;")
        
        layout.addWidget(self.lbl_text)
        self.update_style(False)

    def nextCheckState(self): pass 

    def setChecked(self, checked):
        super().setChecked(checked)
        self.update_style(checked)

    def update_style(self, checked):
        color = ACCENT_GOLD if checked else FG_DIM
        bg = "#1a1a1a" if checked else "transparent"
        # Pure text style, no icon char
        self.lbl_text.setStyleSheet(f"color: {color}; background: transparent; font-size: 10px; font-weight: bold;")
        # Add a left border indicator for active state
        border = f"border-left: 2px solid {ACCENT_GOLD};" if checked else "border: none;"
        self.setStyleSheet(f"background: {bg}; {border}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()

class CollapsibleSection(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.layout_main = import_vbox(self, 0, 0, 0, 0)
        self.layout_main.setSpacing(0)
        self.btn_toggle = QPushButton(title)
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setStyleSheet(f"""
            QPushButton {{
                color: {FG_DIM}; background: transparent; 
                border: none; text-align: left; font-weight: bold; font-size: 10px;
            }}
            QPushButton:checked {{ color: {ACCENT_GOLD}; }}
            QPushButton:hover {{ color: {FG_TEXT}; }}
        """)
        self.btn_toggle.clicked.connect(self.toggle_animation)
        self.layout_main.addWidget(self.btn_toggle)
        self.content_area = QScrollArea()
        self.content_area.setMaximumHeight(0) 
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setFrameShape(QFrame.NoFrame)
        self.content_area.setWidgetResizable(True)
        self.content_area.setStyleSheet("background: transparent;")
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
