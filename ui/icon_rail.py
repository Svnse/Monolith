from __future__ import annotations

import math
from collections.abc import Callable

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

import core.style as _s


PaintFn = Callable[[QPainter, QRectF, QColor], None]


def _round_pen(color: QColor, width: float = 1.2) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _paint_config(p: QPainter, _rect: QRectF, color: QColor) -> None:
    p.setPen(_round_pen(color))
    p.setBrush(Qt.NoBrush)
    cx = cy = 8.0
    p.drawEllipse(QRectF(cx - 3, cy - 3, 6, 6))
    for index in range(8):
        angle = math.radians(index * 45)
        x1 = cx + 4.8 * math.cos(angle)
        y1 = cy + 4.8 * math.sin(angle)
        x2 = cx + 6.2 * math.cos(angle)
        y2 = cy + 6.2 * math.sin(angle)
        p.drawLine(x1, y1, x2, y2)


def _paint_trace(p: QPainter, _rect: QRectF, color: QColor) -> None:
    p.setPen(_round_pen(color))
    path = QPainterPath()
    path.moveTo(3, 12)
    path.lineTo(6, 6)
    path.lineTo(9, 9)
    path.lineTo(13, 3)
    p.drawPath(path)


def _paint_history(p: QPainter, _rect: QRectF, color: QColor) -> None:
    p.setPen(_round_pen(color))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QRectF(3, 3.5, 10, 10))
    p.drawLine(8, 5.5, 8, 8)
    p.drawLine(8, 8, 10, 9.5)


def _paint_stats(p: QPainter, _rect: QRectF, color: QColor) -> None:
    """Bar-chart silhouette: three vertical bars of increasing height."""
    from PySide6.QtCore import QRect
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    bar_w = 4
    gap = 2
    heights = [8, 12, 16]
    total_w = len(heights) * bar_w + (len(heights) - 1) * gap
    cx, cy = 8, 8
    x = cx - total_w // 2
    for h in heights:
        p.drawRect(QRect(x, cy + 8 - h, bar_w, h))
        x += bar_w + gap


def _paint_audit(p: QPainter, _rect: QRectF, color: QColor) -> None:
    p.setPen(_round_pen(color))
    p.drawLine(4, 4.5, 12, 4.5)
    p.drawLine(4, 8, 9.5, 8)
    p.drawLine(4, 11.5, 11, 11.5)


def _paint_databank(p: QPainter, _rect: QRectF, color: QColor) -> None:
    p.setPen(_round_pen(color))
    p.setBrush(Qt.NoBrush)
    for x, y in ((3, 3), (8.8, 3), (3, 8.8), (8.8, 8.8)):
        p.drawRoundedRect(QRectF(x, y, 4.2, 4.2), 1, 1)


def _paint_overseer(p: QPainter, _rect: QRectF, color: QColor) -> None:
    p.setPen(_round_pen(color))
    p.setBrush(Qt.NoBrush)
    path = QPainterPath()
    path.moveTo(2, 8)
    path.cubicTo(2, 5, 5.5, 2.5, 8, 2.5)
    path.cubicTo(10.5, 2.5, 14, 5, 14, 8)
    path.cubicTo(14, 11, 10.5, 13.5, 8, 13.5)
    path.cubicTo(5.5, 13.5, 2, 11, 2, 8)
    p.drawPath(path)
    p.drawEllipse(QRectF(5.5, 5.5, 5, 5))


def _paint_generic(p: QPainter, _rect: QRectF, color: QColor) -> None:
    p.setPen(_round_pen(color))
    p.setBrush(Qt.NoBrush)
    path = QPainterPath()
    path.moveTo(8, 2.5)
    path.lineTo(13.5, 8)
    path.lineTo(8, 13.5)
    path.lineTo(2.5, 8)
    path.closeSubpath()
    p.drawPath(path)


class _RailIcon(QWidget):
    clicked = Signal()

    def __init__(self, paint_fn: PaintFn, tooltip: str, state_name: str, parent=None):
        super().__init__(parent)
        self._paint_fn = paint_fn
        self._state_name = state_name
        self._active = False
        self._badge = False
        self.setFixedSize(34, 34)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)

    def state_name(self) -> str:
        return self._state_name

    def set_active(self, active: bool) -> None:
        if self._active == bool(active):
            return
        self._active = bool(active)
        self.update()

    def set_badge(self, visible: bool) -> None:
        if self._badge == bool(visible):
            return
        self._badge = bool(visible)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._active:
            painter.fillRect(self.rect(), QColor(_s.BG_PANEL))
        elif self.underMouse():
            painter.fillRect(self.rect(), QColor(_s.BG_BUTTON_HOVER))

        if self._active:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(_s.ACCENT_PRIMARY))
            painter.drawRoundedRect(QRectF(0, 7, 2.5, self.height() - 14), 1, 1)

        icon_color = QColor(_s.ACCENT_PRIMARY if self._active else _s.FG_DIM)
        if self.underMouse() and not self._active:
            icon_color = QColor(_s.FG_SECONDARY)

        painter.translate(9, 9)
        self._paint_fn(painter, QRectF(0, 0, 16, 16), icon_color)
        painter.resetTransform()

        if self._badge:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(_s.FG_WARN))
            painter.drawEllipse(self.width() - 10, 3, 6, 6)

        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)


class IconRail(QWidget):
    sig_panel_requested = Signal(str)

    # Expedition / Self-Maintenance / Workshop / Reasoning Tree are NOT pinned:
    # they open from the omni bar (search or > command) — E's 2026-07-08 declutter.
    TOP_ICONS = [
        (_paint_config, "Config", "CONFIG"),
        (_paint_trace, "Generating", "GENERATING"),
        (_paint_history, "History", "ARCHIVE"),
        (_paint_stats, "Stats", "STATS"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("icon_rail")
        self.setFixedWidth(44)
        self._buttons: dict[str, _RailIcon] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(6)
        self._layout = layout

        for paint_fn, tooltip, state_name in self.TOP_ICONS:
            self._add_button(paint_fn, tooltip, state_name)

        divider = QFrame()
        divider.setFixedSize(18, 1)
        divider.setStyleSheet(f"background: {_s.BORDER_SUBTLE}; border: none;")
        layout.addWidget(divider, alignment=Qt.AlignHCenter)

        self._addon_insert_index = layout.count()
        layout.addStretch()
        self._add_button(_paint_overseer, "Overseer", "OVERSEER")

    def _add_button(self, paint_fn: PaintFn, tooltip: str, state_name: str, *, index: int | None = None) -> None:
        btn = _RailIcon(paint_fn, tooltip, state_name)
        btn.clicked.connect(lambda name=state_name: self.sig_panel_requested.emit(name))
        self._buttons[state_name] = btn
        if index is None:
            self._layout.addWidget(btn, alignment=Qt.AlignHCenter)
        else:
            self._layout.insertWidget(index, btn, alignment=Qt.AlignHCenter)

    def add_addon_icon(self, mod_id: str, _icon_char: str, label: str) -> None:
        key = f"addon:{mod_id}"
        if key in self._buttons:
            return
        btn = _RailIcon(_paint_generic, label, key)
        btn.clicked.connect(lambda name=key: self.sig_panel_requested.emit(name))
        self._buttons[key] = btn
        self._layout.insertWidget(self._addon_insert_index, btn, alignment=Qt.AlignHCenter)
        self._addon_insert_index += 1

    def remove_addon_icon(self, mod_id: str) -> None:
        key = f"addon:{mod_id}"
        btn = self._buttons.pop(key, None)
        if btn is None:
            return
        index = self._layout.indexOf(btn)
        if 0 <= index < self._addon_insert_index:
            self._addon_insert_index -= 1
        self._layout.removeWidget(btn)
        btn.deleteLater()

    def set_active(self, name: str | None) -> None:
        for key, btn in self._buttons.items():
            if key == "OVERSEER":
                btn.set_active(False)
                continue
            btn.set_active(key == name)
