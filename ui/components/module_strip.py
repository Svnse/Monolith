import math
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QPoint, QEvent, QMimeData
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF, QDrag

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
        import core.style as s
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        y_off = math.sin(self.phase) * 2
        c = QColor(s.ACCENT_PRIMARY)
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
        self._drag_start_pos = None

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton: self.sig_close.emit(self.mod_id)
        elif e.button() == Qt.LeftButton:
            self._drag_start_pos = e.position().toPoint()
            self.sig_select.emit(self.mod_id)

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.LeftButton) or self._drag_start_pos is None:
            return
        if (e.position().toPoint() - self._drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-monolith-module", self.mod_id.encode())
        drag.setMimeData(mime)
        drag.exec(Qt.MoveAction)

    def set_active(self, val):
        self.setChecked(val)
        if val: self.is_pulsing = False

    def flash(self):
        if not self.isChecked():
            self.is_pulsing = True
            self.pulse_phase = 0.0
            self.update()
            QTimer.singleShot(250, lambda: self.set_pulsing(False))

    def set_pulsing(self, val):
        self.is_pulsing = val
        if val: self.pulse_phase = 0.0
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.is_pulsing and not self.isChecked():
            import core.style as s
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            glow = QColor(s.ACCENT_PRIMARY)
            alpha = int(((math.sin(self.pulse_phase) + 1) / 2) * 50)
            glow.setAlpha(alpha)
            p.fillRect(self.rect(), glow)

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
        self.modules = {}
        self.order = []
        self.content.setAcceptDrops(True)
        self.content.installEventFilter(self)

    def add_module(self, mod_id, icon_char, label_text):
        icon = ModuleIcon(mod_id, icon_char, label_text)
        icon.sig_select.connect(self.sig_module_selected)
        icon.sig_close.connect(self.sig_module_closed)
        self.modules[mod_id] = icon
        self.order.append(mod_id)
        self.vbox.insertWidget(self.vbox.count()-1, icon)
        self._check_overflow()

    def remove_module(self, mod_id):
        if mod_id not in self.modules:
            return
        icon = self.modules.pop(mod_id)
        if mod_id in self.order:
            self.order.remove(mod_id)
        self.vbox.removeWidget(icon)
        icon.deleteLater()
        self._check_overflow()

    def select_module(self, mod_id):
        self.deselect_all()
        if mod_id in self.modules:
            self.modules[mod_id].set_active(True)
    
    def flash_module(self, mod_id):
        if mod_id in self.modules:
            self.modules[mod_id].flash()

    def deselect_all(self):
        for icon in self.modules.values(): icon.set_active(False)

    def get_order(self):
        return list(self.order)

    def eventFilter(self, obj, event):
        if obj is self.content:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if event.mimeData().hasFormat("application/x-monolith-module"):
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Drop:
                if event.mimeData().hasFormat("application/x-monolith-module"):
                    mod_id = bytes(event.mimeData().data("application/x-monolith-module")).decode()
                    target = self._module_at_pos(event.position().toPoint())
                    self.reorder_module(mod_id, target)
                    event.acceptProposedAction()
                    return True
        return super().eventFilter(obj, event)

    def reorder_module(self, mod_id, target_id):
        if mod_id not in self.order or mod_id == target_id:
            return
        self.order.remove(mod_id)
        if target_id and target_id in self.order:
            target_index = self.order.index(target_id)
            self.order.insert(target_index, mod_id)
        else:
            self.order.append(mod_id)
        self._rebuild_layout()

    def _module_at_pos(self, pos):
        widget = self.content.childAt(pos)
        while widget and widget is not self.content:
            if isinstance(widget, ModuleIcon):
                return widget.mod_id
            widget = widget.parent()
        return None

    def _rebuild_layout(self):
        while self.vbox.count():
            item = self.vbox.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        for mod_id in self.order:
            self.vbox.addWidget(self.modules[mod_id])
        self.vbox.addStretch()
        self._check_overflow()

    def _check_overflow(self):
        self.content.adjustSize()
        overflow = self.content.sizeHint().height() > self.scroll.height()
        self.arrow.setVisible(overflow)

    def scroll_down(self):
        vs = self.scroll.verticalScrollBar()
        vs.setValue(vs.value() + 50)
