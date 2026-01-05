import math
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF

from core.style import ACCENT_GOLD, FG_DIM, FG_TEXT
from .atoms import SidebarButton

class OverflowArrow(QWidget):
    clicked = Signal()
    def __init__(self):
        super().__init__()
        self.setFixedHeight(15)
        self.setCursor(Qt.PointingHandCursor)
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._anim)
        self.timer.start(50)
        
    def _anim(self):
        self.phase += 0.2
        self.update()
        
    def mousePressEvent(self, e): self.clicked.emit()
        
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        y_off = math.sin(self.phase) * 2
        c = QColor(ACCENT_GOLD)
        c.setAlpha(150)
        p.setPen(QPen(c, 1.5))
        p.setBrush(Qt.NoBrush)
        cx = self.width() / 2
        cy = self.height() / 2 + y_off
        path = QPolygonF([QPoint(cx - 4, cy - 2), QPoint(cx, cy + 3), QPoint(cx + 4, cy - 2)])
        p.drawPolyline(path)

class ModuleIcon(SidebarButton):
    sig_close = Signal(str)
    sig_select = Signal(str)

    def __init__(self, mod_id, icon_char, label_text):
        super().__init__(icon_char, label_text, checkable=True)
        self.mod_id = mod_id
        self.code = icon_char
        self.is_pulsing = False
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda: self.sig_close.emit(self.mod_id))
        
        self.pulse_phase = 0.0
        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self._step_pulse)
        self.pulse_timer.start(50)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton: self.sig_close.emit(self.mod_id)
        elif e.button() == Qt.LeftButton: self.sig_select.emit(self.mod_id)

    def set_active(self, val):
        self.setChecked(val)
        if val: self.is_pulsing = False

    def flash(self):
        if not self.isChecked():
            self.is_pulsing = True
            QTimer.singleShot(2000, lambda: self.set_pulsing(False))

    def set_pulsing(self, val):
        self.is_pulsing = val
        if val: self.pulse_phase = 0.0
        self.update()

    def _step_pulse(self):
        if self.is_pulsing:
            self.pulse_phase += 0.15
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.is_pulsing and not self.isChecked():
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            glow = QColor(ACCENT_GOLD)
            alpha = int(((math.sin(self.pulse_phase) + 1) / 2) * 50)
            glow.setAlpha(alpha)
            p.fillRect(self.rect(), glow)

class ModuleGroup(QWidget):
    sig_item_select = Signal(str)
    sig_item_close = Signal(str)

    def __init__(self, code):
        super().__init__()
        self.code = code
        self.expanded = False
        self.items = {} 
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(2)
        
        self.header = QPushButton(f"{code} ▸")
        self.header.setFixedSize(50, 40)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setStyleSheet(f"""
            QPushButton {{
                color: {FG_DIM}; background: transparent; 
                border: 1px solid #222; font-weight: bold; font-size: 10px;
            }}
            QPushButton:hover {{ color: {FG_TEXT}; border-color: #444; }}
        """)
        self.header.clicked.connect(self.toggle)
        self.layout.addWidget(self.header)
        
        self.container = QWidget()
        self.cont_layout = QVBoxLayout(self.container)
        self.cont_layout.setContentsMargins(10, 0, 0, 5)
        self.cont_layout.setSpacing(2)
        self.container.setVisible(False)
        self.layout.addWidget(self.container)

    def add_item(self, mod_id, icon_char, label_text):
        icon = ModuleIcon(mod_id, icon_char, label_text)
        icon.sig_select.connect(self.sig_item_select)
        icon.sig_close.connect(self.sig_item_close)
        icon.setFixedSize(40, 40)
        icon.lbl_icon.setStyleSheet("font-size: 14px;")
        icon.lbl_text.setVisible(False)
        self.items[mod_id] = icon
        self.cont_layout.addWidget(icon)
        self.update_recent(mod_id)

    def remove_item(self, mod_id):
        if mod_id in self.items:
            icon = self.items.pop(mod_id)
            icon.deleteLater()

    def update_recent(self, mod_id):
        for mid, icon in self.items.items():
            icon.set_pulsing(mid == mod_id)

    def toggle(self):
        self.expanded = not self.expanded
        self.container.setVisible(self.expanded)
        self.header.setText(f"{self.code} {'▾' if self.expanded else '▸'}")

class ModuleStrip(QWidget):
    sig_module_selected = Signal(str)
    sig_module_closed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        self.content = QWidget()
        self.vbox = QVBoxLayout(self.content)
        self.vbox.setContentsMargins(5, 5, 5, 5)
        self.vbox.setSpacing(5)
        self.vbox.addStretch() 
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll)
        self.arrow = OverflowArrow()
        self.arrow.clicked.connect(self.scroll_down)
        self.arrow.setVisible(False)
        layout.addWidget(self.arrow)
        self.groups = {}
        self.singles = {}
        self.registry = {}

    def add_module(self, mod_id, icon_char, label_text):
        self.registry[mod_id] = (icon_char, label_text)
        existing = [mid for mid, (c, l) in self.registry.items() if c == icon_char and mid != mod_id]
        if not existing:
            icon = ModuleIcon(mod_id, icon_char, label_text)
            icon.sig_select.connect(self.sig_module_selected)
            icon.sig_close.connect(self.sig_module_closed)
            self.vbox.insertWidget(self.vbox.count()-1, icon)
            self.singles[mod_id] = icon
        else:
            if icon_char not in self.groups:
                prev_id = existing[0]
                prev_data = self.registry[prev_id]
                prev_icon = self.singles.pop(prev_id)
                self.vbox.removeWidget(prev_icon)
                prev_icon.deleteLater()
                group = ModuleGroup(icon_char)
                group.sig_item_select.connect(self.sig_module_selected)
                group.sig_item_close.connect(self.sig_module_closed)
                self.groups[icon_char] = group
                self.vbox.insertWidget(self.vbox.count()-1, group)
                group.add_item(prev_id, prev_data[0], prev_data[1])
                group.add_item(mod_id, icon_char, label_text)
                if not group.expanded: group.toggle()
            else:
                group = self.groups[icon_char]
                group.add_item(mod_id, icon_char, label_text)
                if not group.expanded: group.toggle()
                group.update_recent(mod_id)
        self._check_overflow()

    def remove_module(self, mod_id):
        if mod_id not in self.registry: return
        icon_char, _ = self.registry.pop(mod_id)
        if mod_id in self.singles:
            icon = self.singles.pop(mod_id)
            icon.deleteLater()
        elif icon_char in self.groups:
            group = self.groups[icon_char]
            group.remove_item(mod_id)
            remaining = [mid for mid, (c, l) in self.registry.items() if c == icon_char]
            if len(remaining) == 1:
                last_id = remaining[0]
                last_data = self.registry[last_id]
                self.vbox.removeWidget(group)
                del self.groups[icon_char]
                group.deleteLater()
                icon = ModuleIcon(last_id, last_data[0], last_data[1])
                icon.sig_select.connect(self.sig_module_selected)
                icon.sig_close.connect(self.sig_module_closed)
                self.vbox.insertWidget(self.vbox.count()-1, icon)
                self.singles[last_id] = icon
        self._check_overflow()

    def select_module(self, mod_id):
        self.deselect_all()
        if mod_id in self.singles:
            self.singles[mod_id].set_active(True)
        else:
            data = self.registry.get(mod_id)
            if data and data[0] in self.groups:
                grp = self.groups[data[0]]
                if mod_id in grp.items:
                    grp.items[mod_id].set_active(True)
    
    def flash_module(self, mod_id):
        if mod_id in self.singles:
            self.singles[mod_id].flash()
        else:
            data = self.registry.get(mod_id)
            if data and data[0] in self.groups:
                grp = self.groups[data[0]]
                if mod_id in grp.items:
                    grp.items[mod_id].flash()

    def deselect_all(self):
        for icon in self.singles.values(): icon.set_active(False)
        for grp in self.groups.values():
            for icon in grp.items.values(): icon.set_active(False)

    def _check_overflow(self):
        self.content.adjustSize()
        overflow = self.content.sizeHint().height() > self.scroll.height()
        self.arrow.setVisible(overflow)

    def scroll_down(self):
        vs = self.scroll.verticalScrollBar()
        vs.setValue(vs.value() + 50)
