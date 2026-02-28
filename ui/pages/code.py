from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    Signal, Qt, QEvent, QTimer, QVariantAnimation, QEasingCurve,
)
from PySide6.QtGui import (
    QColor, QCursor, QPainter, QPen, QTextCursor,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QComboBox,
    QSplitter,
    QStackedWidget,
    QPushButton,
    QButtonGroup,
    QFrame,
    QApplication,
)
from shiboken6 import isValid

import core.style as _s
from core.config import DEFAULT_WORKSPACE_ROOT
from core.paths import CONFIG_DIR
from core.state import SystemStatus
from engine.loop.events import (
    EVENT_CONTROL,
    EVENT_LOOP,
    RUN_STATE_COMPLETED,
    RUN_STATE_FAILED,
    RUN_STATE_REDIRECTED,
    RUN_STATE_RUNNING,
    RUN_STATE_STOPPED,
    RUN_STATE_STOPPING,
    RUN_STATE_WAITING_APPROVAL,
    REASON_APPROVAL_DENIED,
    REASON_APPROVAL_GRANTED,
    REASON_APPROVAL_PROMPT,
    REASON_COMPLETED,
    REASON_CYCLE_START,
    REASON_ERROR,
    REASON_STOP_REQUESTED,
    REASON_USER_REDIRECT,
    REASON_WALL_HIT,
)
from engine.loop.commands import CMD_APPROVAL_RESPONSE, CMD_SET_WORKSPACE_ROOT
from engine.loop.lifecycle import apply_lifecycle_transition
from ui.components.atoms import MonoButton, MonoGroupBox, MonoSlider


_CONFIG_PATH = CONFIG_DIR / "code_loop_config.json"

_TOOL_PRESETS: dict[str, list[str]] = {
    "read_only": [
        "read_file", "list_dir", "grep_search", "glob_files", "git_status", "git_diff",
    ],
    "code_write": [
        "read_file", "list_dir", "grep_search", "glob_files", "git_status", "git_diff",
        "write_file", "apply_patch", "mkdir", "copy_path", "move_path",
    ],
    "full_exec": [
        "read_file", "list_dir", "grep_search", "glob_files", "git_status", "git_diff",
        "write_file", "apply_patch", "mkdir", "copy_path", "move_path",
        "run_cmd", "run_tests",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────

def _short_path_text(path: str, *, max_len: int = 72) -> str:
    text = str(path or "").strip()
    if len(text) <= max_len:
        return text
    keep = max(12, max_len - 3)
    return "..." + text[-keep:]


def _tab_style() -> str:
    return f"""
        QPushButton {{
            background: {_s.BG_BUTTON}; border: 1px solid {_s.BORDER_LIGHT}; color: {_s.FG_DIM};
            padding: 6px 12px; font-size: 10px; font-weight: bold; border-radius: 2px;
        }}
        QPushButton:checked {{
            background: {_s.BG_BUTTON_HOVER}; color: {_s.ACCENT_PRIMARY};
            border: 1px solid {_s.ACCENT_PRIMARY};
        }}
        QPushButton:hover {{ color: {_s.FG_TEXT}; border: 1px solid {_s.FG_TEXT}; }}
    """


def _mini_tab_style() -> str:
    return f"""
        QPushButton {{
            background: {_s.BG_BUTTON}; border: 1px solid {_s.BORDER_LIGHT}; color: {_s.FG_DIM};
            padding: 4px 10px; font-size: 9px; font-weight: bold; border-radius: 2px;
        }}
        QPushButton:checked {{
            background: {_s.BG_BUTTON_HOVER}; color: {_s.ACCENT_PRIMARY};
            border: 1px solid {_s.ACCENT_PRIMARY};
        }}
        QPushButton:hover {{ color: {_s.FG_TEXT}; border: 1px solid {_s.FG_TEXT}; }}
    """


def _output_style() -> str:
    return f"""
        QTextEdit {{
            background: {_s.BG_INPUT}; color: {_s.FG_TEXT};
            border: 1px solid {_s.BORDER_SUBTLE};
            font-family: 'Consolas', monospace; font-size: 10px;
            padding: 4px;
        }}
        {_s.SCROLLBAR_STYLE}
    """


def _input_style() -> str:
    return f"""
        QTextEdit {{
            background: {_s.BG_INPUT}; color: white;
            border: 1px solid {_s.BORDER_LIGHT};
            padding: 8px; font-family: 'Verdana'; font-size: 11px;
        }}
        QTextEdit:focus {{ border: 1px solid {_s.ACCENT_PRIMARY}; }}
        {_s.SCROLLBAR_STYLE}
    """


def _path_style() -> str:
    return (
        f"background: {_s.BG_INPUT}; color: {_s.FG_PLACEHOLDER};"
        f" border: 1px solid {_s.BORDER_LIGHT}; padding: 5px;"
    )


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {_s.FG_DIM}; font-size: 8px; font-weight: bold; "
        f"letter-spacing: 2px; padding-top: 6px;"
    )
    return lbl


def _combo_style() -> str:
    return (
        f"QComboBox {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT}; "
        f"border: 1px solid {_s.BORDER_LIGHT}; padding: 3px 6px; font-size: 10px; }}"
        f"QComboBox::drop-down {{ border: none; }}"
        f"QComboBox QAbstractItemView {{ background: {_s.BG_BUTTON}; "
        f"color: {_s.FG_TEXT}; border: 1px solid {_s.BORDER_LIGHT}; }}"
    )


def _timeline_list_style() -> str:
    return f"""
        QListWidget {{
            background: {_s.BG_INPUT};
            border: 1px solid {_s.BORDER_SUBTLE};
            padding: 4px;
        }}
        QListWidget::item {{
            border: none;
            margin: 3px 0px;
        }}
        {_s.SCROLLBAR_STYLE}
    """


# ─────────────────────────────────────────────────────────────────────────────
# Curtain + Detachable Panel System
# ─────────────────────────────────────────────────────────────────────────────

class _TimelineCard(QFrame):
    def __init__(
        self,
        kind: str,
        title: str,
        body: str = "",
        details: str = "",
        actions: list[tuple[str, Any]] | None = None,
        on_click: Any | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._timeline_item = None
        self._timeline_list = None
        self._on_click = on_click if callable(on_click) else None
        self.setAttribute(Qt.WA_StyledBackground, True)
        colors = {
            "user": (_s.ACCENT_PRIMARY, _s.FG_TEXT),
            "agent": (_s.FG_ACCENT, _s.FG_TEXT),
            "thinking": (_s.FG_DIM, _s.FG_TEXT),
            "tool": (_s.FG_TEXT, _s.FG_TEXT),
            "approval": (_s.FG_ERROR, _s.FG_TEXT),
            "system": (_s.FG_DIM, _s.FG_DIM),
        }
        accent, body_fg = colors.get(kind, (_s.FG_DIM, _s.FG_TEXT))
        self.setStyleSheet(
            f"QFrame {{ background: {_s.BG_BUTTON}; border: 1px solid {_s.BORDER_LIGHT}; border-left: 3px solid {accent}; border-radius: 3px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8 if kind == "approval" else 6, 8, 8 if kind == "approval" else 6)
        lay.setSpacing(4)
        self._lbl_title = QLabel(title)
        self._lbl_title.setWordWrap(True)
        self._lbl_title.setStyleSheet(
            f"color: {accent}; font-size: 9px; font-weight: bold; letter-spacing: 1px; border: none; background: transparent;"
        )
        lay.addWidget(self._lbl_title)
        if body:
            lbl_body = QLabel(body)
            lbl_body.setWordWrap(True)
            lbl_body.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl_body.setStyleSheet(
                f"color: {body_fg}; font-size: 10px; border: none; background: transparent;"
            )
            lay.addWidget(lbl_body)
        if details:
            btn_details = QPushButton("DETAILS")
            btn_details.setCheckable(True)
            btn_details.setCursor(Qt.PointingHandCursor)
            btn_details.setFixedHeight(22)
            btn_details.setStyleSheet(
                f"QPushButton {{ background: {_s.BG_INPUT}; color: {_s.FG_DIM}; "
                f"border: 1px solid {_s.BORDER_LIGHT}; padding: 1px 8px; font-size: 8px; font-weight: bold; }}"
                f"QPushButton:hover {{ color: {_s.ACCENT_PRIMARY}; border: 1px solid {_s.ACCENT_PRIMARY}; }}"
                f"QPushButton:checked {{ color: {_s.FG_TEXT}; border: 1px solid {_s.FG_TEXT}; }}"
            )
            txt_details = QTextEdit()
            txt_details.setReadOnly(True)
            txt_details.setMaximumHeight(90)
            txt_details.setPlainText(details)
            txt_details.setStyleSheet(_output_style())
            txt_details.setVisible(False)

            def _toggle_details(checked: bool, txt=txt_details, btn=btn_details):
                txt.setVisible(bool(checked))
                btn.setText("HIDE DETAILS" if checked else "DETAILS")
                self.adjustSize()
                if self._timeline_item is not None:
                    self._timeline_item.setSizeHint(self.sizeHint())
                if self._timeline_list is not None:
                    self._timeline_list.updateGeometries()

            btn_details.toggled.connect(_toggle_details)
            lay.addWidget(btn_details)
            lay.addWidget(txt_details)
        if actions:
            row = QHBoxLayout()
            row.setContentsMargins(0, 2, 0, 0)
            row.setSpacing(6)
            row.addStretch()
            for label, fn in actions:
                btn = QPushButton(str(label))
                btn.setCursor(Qt.PointingHandCursor)
                btn.setFixedHeight(24)
                btn.setStyleSheet(
                    f"QPushButton {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT}; "
                    f"border: 1px solid {_s.BORDER_LIGHT}; padding: 2px 8px; font-size: 9px; }}"
                    f"QPushButton:hover {{ border: 1px solid {_s.ACCENT_PRIMARY}; color: {_s.ACCENT_PRIMARY}; }}"
                )
                if callable(fn):
                    btn.clicked.connect(fn)
                row.addWidget(btn)
            lay.addLayout(row)
        is_approval_prompt = kind == "approval" and str(title or "").upper().startswith("APPROVAL NEEDED")
        if is_approval_prompt:
            self.setMinimumHeight(88)
        if self._on_click is not None:
            self.setCursor(Qt.PointingHandCursor)

    def set_title_text(self, title: str) -> None:
        self._lbl_title.setText(str(title or ""))
        self.adjustSize()
        if self._timeline_item is not None:
            self._timeline_item.setSizeHint(self.sizeHint())
        if self._timeline_list is not None:
            self._timeline_list.updateGeometries()

    def mousePressEvent(self, event):
        if self._on_click is not None and event.button() == Qt.LeftButton:
            try:
                self._on_click()
            except Exception:
                pass
        super().mousePressEvent(event)


class _LauncherButton(QFrame):
    """Box button shown inside the curtain drawer."""

    sig_clicked = Signal(str)

    def __init__(self, key: str, label: str, parent=None):
        super().__init__(parent)
        self._key = key
        self._active = False
        self.setFixedSize(110, 44)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        self._refresh_style(hover=False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(2)

        self._label = QLabel(label)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-weight: bold; "
            f"letter-spacing: 1.5px; border: none; background: transparent;"
        )
        lay.addWidget(self._label)

        self._indicator = QLabel("●")
        self._indicator.setAlignment(Qt.AlignCenter)
        self._indicator.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 6px; border: none; background: transparent;"
        )
        self._indicator.setVisible(False)
        lay.addWidget(self._indicator)

    def _refresh_style(self, hover: bool):
        left = (
            f"border-left: 3px solid {_s.ACCENT_PRIMARY};"
            if hover
            else f"border-left: 1px solid {_s.BORDER_LIGHT};"
        )
        self.setStyleSheet(
            f"QFrame {{ background: {_s.BG_BUTTON}; border: 1px solid {_s.BORDER_LIGHT}; "
            f"{left} border-radius: 3px; }}"
        )
        if hasattr(self, "_label"):
            color = _s.ACCENT_PRIMARY if hover else _s.FG_DIM
            self._label.setStyleSheet(
                f"color: {color}; font-size: 9px; font-weight: bold; "
                f"letter-spacing: 1.5px; border: none; background: transparent;"
            )

    def set_active(self, active: bool):
        self._active = active
        if hasattr(self, "_indicator"):
            self._indicator.setVisible(active)

    def enterEvent(self, event):
        if self.isEnabled():
            self._refresh_style(hover=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._refresh_style(hover=False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.sig_clicked.emit(self._key)
        super().mousePressEvent(event)


class _CurtainHandle(QWidget):
    """Always-visible 18px strip at the right edge that triggers the curtain."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drawer: "_CurtainDrawer | None" = None
        self._open_timer = QTimer(self)
        self._open_timer.setSingleShot(True)
        self._open_timer.setInterval(350)
        self._open_timer.timeout.connect(self._trigger_open)
        self._hovered = False
        self.setFixedWidth(18)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"background: {_s.BG_BUTTON}; border-left: 1px solid {_s.BORDER_SUBTLE};"
        )

    def set_drawer(self, drawer: "_CurtainDrawer"):
        self._drawer = drawer

    def cancel_close(self):
        self._open_timer.stop()

    def _trigger_open(self):
        if self._drawer:
            self._drawer.open()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        accent_col = QColor(_s.ACCENT_PRIMARY if self._hovered else _s.BORDER_SUBTLE)
        painter.setPen(QPen(accent_col, 1))
        painter.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        arrow_col = QColor(_s.ACCENT_PRIMARY if self._hovered else _s.FG_DIM)
        painter.setPen(QPen(arrow_col, 1))
        cx = self.width() // 2
        cy = self.height() // 2
        painter.drawText(cx - 5, cy + 5, "◀")
        painter.end()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        if self._drawer:
            self._drawer.cancel_close()
        self._open_timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        self._open_timer.stop()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._drawer:
            self._open_timer.stop()
            if self._drawer._is_open:
                self._drawer.close_drawer()
            else:
                self._drawer.open()
        super().mousePressEvent(event)


class _CurtainDrawer(QWidget):
    """Sliding panel that reveals OUTPUT / CONTROLS / LLM CALLS / PAD launcher buttons."""

    sig_launch = Signal(str)

    def __init__(self, handle: _CurtainHandle, parent=None):
        super().__init__(parent)
        self._handle = handle
        self._is_open = False

        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.setInterval(300)
        self._close_timer.timeout.connect(self.close_drawer)

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_value)
        self._anim.finished.connect(self._on_anim_finished)

        self.setFixedWidth(0)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"background: {_s.BG_BUTTON}; border-left: 1px solid {_s.BORDER_LIGHT};"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 20, 10, 20)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        self._btn_output   = _LauncherButton("output",   "OUTPUT",   self)
        self._btn_controls = _LauncherButton("controls", "CONTROLS", self)
        self._btn_pad = _LauncherButton("pad", "AGENT DEBUG", self)
        self._btn_output.setEnabled(False)
        self._btn_controls.setEnabled(False)
        self._btn_pad.setEnabled(False)
        self._btn_output.sig_clicked.connect(self.sig_launch)
        self._btn_controls.sig_clicked.connect(self.sig_launch)
        self._btn_pad.sig_clicked.connect(self.sig_launch)

        lay.addWidget(self._btn_output)
        lay.addWidget(self._btn_controls)
        lay.addWidget(self._btn_pad)
        lay.addStretch()

        self._buttons = [self._btn_output, self._btn_controls, self._btn_pad]

    def open(self):
        if self._is_open:
            return
        self._close_timer.stop()
        self._is_open = True
        for b in self._buttons:
            b.setEnabled(False)
        self._anim.stop()
        self._anim.setStartValue(float(self.width()))
        self._anim.setEndValue(130.0)
        self._anim.start()

    def close_drawer(self):
        if not self._is_open:
            return
        self._is_open = False
        for b in self._buttons:
            b.setEnabled(False)
        self._anim.stop()
        self._anim.setStartValue(float(self.width()))
        self._anim.setEndValue(0.0)
        self._anim.start()

    def cancel_close(self):
        self._close_timer.stop()

    def schedule_close(self):
        self._close_timer.start()

    def set_active(self, key: str, active: bool):
        if key == "output":
            btn = self._btn_output
        elif key == "controls":
            btn = self._btn_controls
        else:
            btn = self._btn_pad
        btn.set_active(active)

    def _on_anim_value(self, value):
        self.setFixedWidth(int(value))

    def _on_anim_finished(self):
        if self._is_open:
            for b in self._buttons:
                b.setEnabled(True)

    def enterEvent(self, event):
        self._close_timer.stop()
        if self._handle:
            self._handle.cancel_close()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._handle and self._handle.underMouse():
            pass
        else:
            self.schedule_close()
        super().leaveEvent(event)


class _SpawnedPanel(QWidget):
    """A panel that can be docked in the splitter or floated as a top-level window."""

    sig_close  = Signal()
    sig_detach = Signal()
    sig_dock   = Signal()

    def __init__(self, key: str, title: str, content: QWidget, parent=None):
        super().__init__(parent)
        self._key = key
        self._title = title
        self._content = content
        self._dragging = False
        self._drag_start_pos = None
        self._drag_bar_offset = None
        self._is_floating = False
        self._main_win = None

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#spawned_panel {{ background: {_s.BG_MAIN}; border: 1px solid {_s.BORDER_LIGHT}; }}"
        )
        self.setObjectName("spawned_panel")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Drag / title bar
        self._drag_bar = QWidget()
        self._drag_bar.setFixedHeight(30)
        self._drag_bar.setAttribute(Qt.WA_StyledBackground, True)
        self._drag_bar.setStyleSheet(
            f"background: {_s.BG_BUTTON}; border-bottom: 1px solid {_s.BORDER_LIGHT};"
        )
        self._drag_bar.setCursor(Qt.SizeAllCursor)

        drag_bar_lay = QHBoxLayout(self._drag_bar)
        drag_bar_lay.setContentsMargins(10, 0, 8, 0)
        drag_bar_lay.setSpacing(6)

        grip = QLabel("≡")
        grip.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 13px; border: none; background: transparent;"
        )
        self._name_lbl = QLabel(title)
        self._name_lbl.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 9px; font-weight: bold; "
            f"letter-spacing: 2px; border: none; background: transparent;"
        )
        drag_bar_lay.addWidget(grip)
        drag_bar_lay.addWidget(self._name_lbl)
        drag_bar_lay.addStretch()

        self._btn_float_toggle = QPushButton("⊞")
        self._btn_float_toggle.setFixedSize(22, 22)
        self._btn_float_toggle.setCursor(Qt.PointingHandCursor)
        self._btn_float_toggle.setToolTip("Float panel")
        self._btn_float_toggle.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"color: {_s.FG_DIM}; font-size: 12px; }}"
            f"QPushButton:hover {{ color: {_s.ACCENT_PRIMARY}; }}"
        )
        self._btn_float_toggle.clicked.connect(self._on_float_toggle_clicked)

        # Close button — clearly visible
        self._btn_close = QPushButton("✕")
        self._btn_close.setFixedSize(24, 24)
        self._btn_close.setCursor(Qt.PointingHandCursor)
        self._btn_close.setToolTip("Close panel")
        self._btn_close.setStyleSheet(
            f"QPushButton {{ background: transparent; border: 1px solid transparent; "
            f"color: {_s.FG_TEXT}; font-size: 13px; font-weight: bold; border-radius: 3px; }}"
            f"QPushButton:hover {{ background: {_s.FG_ERROR}; color: white; "
            f"border: 1px solid {_s.FG_ERROR}; }}"
        )
        self._btn_close.clicked.connect(self._on_close)

        drag_bar_lay.addWidget(self._btn_float_toggle)
        drag_bar_lay.addWidget(self._btn_close)

        root.addWidget(self._drag_bar)
        root.addWidget(content, 1)

        self._drag_bar.installEventFilter(self)
        self.setMinimumWidth(200)

    def _set_floating_style(self, floating: bool):
        if floating:
            self._drag_bar.setStyleSheet(
                f"background: {_s.BG_BUTTON}; border-bottom: 2px solid {_s.ACCENT_PRIMARY};"
            )
            self._btn_float_toggle.setText("⊟")
            self._btn_float_toggle.setToolTip("Dock back")
        else:
            self._drag_bar.setStyleSheet(
                f"background: {_s.BG_BUTTON}; border-bottom: 1px solid {_s.BORDER_LIGHT};"
            )
            self._btn_float_toggle.setText("⊞")
            self._btn_float_toggle.setToolTip("Float panel")

    def _on_float_toggle_clicked(self):
        if self._is_floating:
            self._dock_back()
        else:
            self._detach()

    def eventFilter(self, obj, event):
        if obj is self._drag_bar:
            t = event.type()
            if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._dragging = True
                self._drag_start_pos = event.globalPosition().toPoint()
                self._drag_bar_offset = event.pos()
                return False
            elif t == QEvent.MouseMove and self._dragging:
                global_pos = event.globalPosition().toPoint()
                if not self._is_floating:
                    delta = global_pos - self._drag_start_pos
                    if delta.manhattanLength() > 8:
                        main_win = self._find_main_window()
                        if main_win and not main_win.geometry().contains(global_pos):
                            self._detach()
                            return True
                elif self._is_floating and self._drag_bar_offset is not None:
                    self.move(global_pos - self._drag_bar_offset)
                    if self._main_win and self._main_win.geometry().contains(global_pos):
                        self._dock_back()
                        return True
                return False
            elif t == QEvent.MouseButtonRelease:
                self._dragging = False
                return False
        return super().eventFilter(obj, event)

    def _find_main_window(self):
        w = self.parent()
        while w is not None:
            if w.isWindow():
                return w
            w = w.parent()
        return None

    def _detach(self):
        if self._is_floating:
            return
        self._main_win = self._find_main_window()
        self._is_floating = True
        self.setParent(None)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.move(QCursor.pos())
        self.setMinimumSize(280, 400)
        self.resize(360, 560)
        self._set_floating_style(True)
        self.show()
        self.sig_detach.emit()

    def _dock_back(self):
        self._dragging = False
        self._is_floating = False
        self._set_floating_style(False)
        self.sig_dock.emit()

    def _on_close(self):
        self.sig_close.emit()
        self.hide()
        self.setParent(None)


# ─────────────────────────────────────────────────────────────────────────────
# PageCode
# ─────────────────────────────────────────────────────────────────────────────

class PageCode(QWidget):
    # Public signals — unchanged
    sig_generate        = Signal(dict)
    sig_load            = Signal()
    sig_unload          = Signal()
    sig_stop            = Signal()
    sig_runtime_command = Signal(str, dict)
    sig_set_model_path  = Signal(str)
    sig_set_ctx_limit   = Signal(int)
    sig_debug           = Signal(str)

    def __init__(self, state, ui_bridge):
        super().__init__()
        self.state     = state
        self.ui_bridge = ui_bridge
        self._last_status: SystemStatus   = SystemStatus.READY
        self._is_model_loaded: bool       = False
        self._run_active: bool            = False
        self._current_result: dict | None = None
        self._pending_approval_id: str | None = None
        self._pending_redirect_prompt: str | None = None
        self._pending_redirect_parent_run_id: str | None = None
        self._active_run_id: str | None = None
        self._conversation_turn: int = 0
        self._session_id: str = f"code_{uuid.uuid4().hex[:10]}"
        self._run_lifecycle: dict[str, dict[str, Any]] = {}
        self._effect_journal: dict[str, list[dict[str, Any]]] = {}
        self._run_cycle_headers: dict[str, set[int]] = {}
        self._run_cycle_collapsed: dict[tuple[str, int], bool] = {}
        self._llm_call_seq_by_run: dict[str, int] = {}
        self._approval_prompt_ts: dict[str, datetime] = {}
        self._debug_log_path: Path | None = None
        self._panels: dict[str, _SpawnedPanel] = {}
        self.config = self._load_config()
        self._workspace_root: Path = self._resolve_workspace_root_from_config()
        self._init_debug_log()
        self._build_ui()
        self._apply_config()
        self._wire_signals()
        self._refresh_status_ui()
        if self._debug_log_path is not None:
            self.append_trace(f"[DEBUG] session log: {self._debug_log_path}")
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._cleanup_spawned_panels)

    # ─────────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(0)

        # content_splitter: main view (slot 0) + spawned panels (slots 1-2)
        self.content_splitter = QSplitter(Qt.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(2)

        # Curtain system
        self._curtain_handle = _CurtainHandle(self)
        self._curtain = _CurtainDrawer(self._curtain_handle, self)
        self._curtain_handle.set_drawer(self._curtain)
        self._curtain.sig_launch.connect(self._on_launcher_clicked)

        # Panel contents (built now, shown when spawned)
        self._output_content   = self._build_output_content()
        self._controls_content = self._build_controls_content()
        self._pad_content = self._build_pad_content()

        # Main view always at slot 0
        self.content_splitter.addWidget(self._build_main_view())

        # outer row: [content_splitter | curtain_drawer | curtain_handle]
        outer_row = QHBoxLayout()
        outer_row.setContentsMargins(0, 0, 0, 0)
        outer_row.setSpacing(0)
        outer_row.addWidget(self.content_splitter, 1)
        outer_row.addWidget(self._curtain, 0)
        outer_row.addWidget(self._curtain_handle, 0)
        root.addLayout(outer_row)

    # ── main view: MESSAGE + LOOP EVENTS ─────────────────────────────────────

    def _build_main_view(self) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        # MESSAGE composer group
        grp_task = MonoGroupBox("MESSAGE")
        task_lay = QVBoxLayout()
        task_lay.setSpacing(8)

        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText("Ask Monolith to inspect, fix, refactor, or explain something…  (Enter = Send, Shift+Enter = New line)")
        self.txt_prompt.setMinimumHeight(70)
        self.txt_prompt.setMaximumHeight(120)
        self.txt_prompt.setAcceptRichText(False)
        self.txt_prompt.setStyleSheet(_input_style())
        self.txt_prompt.installEventFilter(self)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_run  = MonoButton("SEND", accent=True)
        self.btn_stop = MonoButton("STOP")
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()

        task_lay.addWidget(self.txt_prompt)
        task_lay.addLayout(btn_row)
        grp_task.add_layout(task_lay)
        lay.addWidget(grp_task)

        # LOOP EVENTS group — always visible, high-signal
        grp_events = MonoGroupBox("LOOP EVENTS")
        ev_lay = QVBoxLayout()
        self.txt_events = QTextEdit()
        self.txt_events.setReadOnly(True)
        self.txt_events.setStyleSheet(_output_style())
        ev_lay.addWidget(self.txt_events)
        grp_events.add_layout(ev_lay)
        lay.addWidget(grp_events, 1)

        return container

    # ── output panel content: AGENT STREAM + TRACE ───────────────────────────

    def _build_output_content(self) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        out_split = QSplitter(Qt.Vertical)
        out_split.setChildrenCollapsible(False)

        grp_stream = MonoGroupBox("TIMELINE")
        stream_lay = QVBoxLayout()
        self.lst_timeline = QListWidget()
        self.lst_timeline.setStyleSheet(_timeline_list_style())
        self.lst_timeline.setSelectionMode(QListWidget.NoSelection)
        self.lst_timeline.setFocusPolicy(Qt.NoFocus)
        self.lst_timeline.viewport().installEventFilter(self)
        stream_lay.addWidget(self.lst_timeline)
        grp_stream.add_layout(stream_lay)
        out_split.addWidget(grp_stream)

        grp_trace = MonoGroupBox("TRACE")
        trace_lay = QVBoxLayout()
        self.txt_trace = QTextEdit()
        self.txt_trace.setReadOnly(True)
        self.txt_trace.setStyleSheet(_output_style())
        trace_lay.addWidget(self.txt_trace)
        grp_trace.add_layout(trace_lay)
        out_split.addWidget(grp_trace)

        # Legacy stream buffer kept for compatibility with any remaining calls
        # that expect a QTextEdit-like field. The timeline is the primary view.
        self.txt_stream = QTextEdit()
        self.txt_stream.setReadOnly(True)

        out_split.setStretchFactor(0, 2)
        out_split.setStretchFactor(1, 1)
        out_split.setSizes([300, 150])
        lay.addWidget(out_split, 1)
        return container

    # ── controls panel content: MODEL/CONFIG tabs + LOOP STATUS/APPROVAL ─────

    def _build_controls_content(self) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        ctrl_split = QSplitter(Qt.Vertical)
        ctrl_split.setChildrenCollapsible(False)
        ctrl_split.addWidget(self._build_code_group())
        ctrl_split.addWidget(self._build_loop_status_group())
        ctrl_split.setStretchFactor(0, 2)
        ctrl_split.setStretchFactor(1, 1)
        ctrl_split.setSizes([340, 200])

        lay.addWidget(ctrl_split, 1)
        return container

    def _build_pad_content(self) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        grp_pad = MonoGroupBox("AGENT DEBUG")
        pad_lay = QVBoxLayout()
        view_row = QHBoxLayout()
        self.btn_pad_snapshot = QPushButton("PAD SNAPSHOT")
        self.btn_pad_snapshot.setCheckable(True)
        self.btn_pad_snapshot.setChecked(True)
        self.btn_pad_snapshot.setStyleSheet(_mini_tab_style())
        self.btn_pad_inject = QPushButton("PAD INJECT")
        self.btn_pad_inject.setCheckable(True)
        self.btn_pad_inject.setStyleSheet(_mini_tab_style())
        self.btn_pad_llm_calls = QPushButton("LLM CALLS")
        self.btn_pad_llm_calls.setCheckable(True)
        self.btn_pad_llm_calls.setStyleSheet(_mini_tab_style())
        self.btn_pad_trace = QPushButton("TRACE")
        self.btn_pad_trace.setCheckable(True)
        self.btn_pad_trace.setStyleSheet(_mini_tab_style())
        pad_view_group = QButtonGroup(self)
        pad_view_group.setExclusive(True)
        pad_view_group.addButton(self.btn_pad_snapshot)
        pad_view_group.addButton(self.btn_pad_inject)
        pad_view_group.addButton(self.btn_pad_llm_calls)
        pad_view_group.addButton(self.btn_pad_trace)
        view_row.addWidget(self.btn_pad_snapshot)
        view_row.addWidget(self.btn_pad_inject)
        view_row.addWidget(self.btn_pad_llm_calls)
        view_row.addWidget(self.btn_pad_trace)
        view_row.addStretch()

        self.pad_stack = QStackedWidget()
        self.txt_pad = QTextEdit()
        self.txt_pad.setReadOnly(True)
        self.txt_pad.setStyleSheet(_output_style())
        self.txt_pad_inject = QTextEdit()
        self.txt_pad_inject.setReadOnly(True)
        self.txt_pad_inject.setStyleSheet(_output_style())
        self.txt_llm_calls = QTextEdit()
        self.txt_llm_calls.setReadOnly(True)
        self.txt_llm_calls.setStyleSheet(_output_style())
        self.txt_debug_trace = QTextEdit()
        self.txt_debug_trace.setReadOnly(True)
        self.txt_debug_trace.setStyleSheet(_output_style())
        self.pad_stack.addWidget(self.txt_pad)
        self.pad_stack.addWidget(self.txt_pad_inject)
        self.pad_stack.addWidget(self.txt_llm_calls)
        self.pad_stack.addWidget(self.txt_debug_trace)
        self.btn_pad_snapshot.toggled.connect(
            lambda checked: self.pad_stack.setCurrentIndex(0) if checked else None
        )
        self.btn_pad_inject.toggled.connect(
            lambda checked: self.pad_stack.setCurrentIndex(1) if checked else None
        )
        self.btn_pad_llm_calls.toggled.connect(
            lambda checked: self.pad_stack.setCurrentIndex(2) if checked else None
        )
        self.btn_pad_trace.toggled.connect(
            lambda checked: self.pad_stack.setCurrentIndex(3) if checked else None
        )
        pad_lay.addLayout(view_row)
        pad_lay.addWidget(self.pad_stack, 1)
        grp_pad.add_layout(pad_lay)
        lay.addWidget(grp_pad, 1)
        return container

    # ── CODE v2 group — MODEL tab (minimal) + CONFIG tab (sliders) ───────────

    def _build_code_group(self) -> QWidget:
        grp_code = MonoGroupBox("CODE v2")
        code_lay = QVBoxLayout()
        code_lay.setSpacing(10)

        # Tab row
        tab_row = QHBoxLayout()
        self.btn_tab_model  = QPushButton("MODEL")
        self.btn_tab_model.setCheckable(True)
        self.btn_tab_model.setChecked(True)
        self.btn_tab_model.setStyleSheet(_tab_style())
        self.btn_tab_config = QPushButton("CONFIG")
        self.btn_tab_config.setCheckable(True)
        self.btn_tab_config.setStyleSheet(_tab_style())
        tab_grp = QButtonGroup(self)
        tab_grp.setExclusive(True)
        tab_grp.addButton(self.btn_tab_model)
        tab_grp.addButton(self.btn_tab_config)
        tab_row.addWidget(self.btn_tab_model)
        tab_row.addWidget(self.btn_tab_config)
        tab_row.addStretch()
        code_lay.addLayout(tab_row)

        self.ops_stack = QStackedWidget()
        code_lay.addWidget(self.ops_stack, 1)

        # ── MODEL tab: path + load/unload (clean, minimal) ───────────────────
        model_tab = QWidget()
        ml = QVBoxLayout(model_tab)
        ml.setSpacing(10)
        ml.setContentsMargins(0, 0, 0, 0)

        path_row = QHBoxLayout()
        self.inp_model = QLineEdit()
        self.inp_model.setReadOnly(True)
        self.inp_model.setPlaceholderText("No GGUF selected")
        self.inp_model.setStyleSheet(_path_style())
        btn_browse = MonoButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self._pick_model)
        path_row.addWidget(self.inp_model)
        path_row.addWidget(btn_browse)

        load_row = QHBoxLayout()
        self.btn_load = MonoButton("LOAD MODEL")
        load_row.addWidget(self.btn_load)
        load_row.addStretch()

        self.btn_load.clicked.connect(self._toggle_load)

        self.btn_workspace = MonoButton("")
        self.btn_workspace.clicked.connect(self._pick_workspace_root)
        self._refresh_workspace_button()

        ml.addLayout(path_row)
        ml.addLayout(load_row)
        ml.addWidget(self.btn_workspace)
        ml.addStretch()

        # ── CONFIG tab: status + all sliders + tool/infer combos ─────────────
        config_tab = QWidget()
        cl = QVBoxLayout(config_tab)
        cl.setSpacing(6)
        cl.setContentsMargins(0, 0, 0, 0)

        # Status row
        status_row = QHBoxLayout()
        lbl_s = QLabel("STATUS")
        lbl_s.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-weight: bold; "
            f"letter-spacing: 1px; min-width: 54px;"
        )
        self.lbl_status = QLabel("READY")
        self.lbl_status.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 10px; font-weight: bold;"
        )
        self.lbl_caps = QLabel("")
        self.lbl_caps.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 9px;")
        status_row.addWidget(lbl_s)
        status_row.addWidget(self.lbl_status)
        status_row.addStretch()
        status_row.addWidget(self.lbl_caps)
        cl.addLayout(status_row)

        # ── Context ──────────────────────────────────────────────────────────
        cl.addWidget(_section_label("CONTEXT"))
        self.s_ctx = MonoSlider(
            "Context Limit", 1024, 32768,
            self.config.get("ctx_limit", 8192), is_int=True
        )
        cl.addWidget(self.s_ctx)

        # ── Loop policy ───────────────────────────────────────────────────────
        cl.addWidget(_section_label("LOOP POLICY"))
        self.s_cycles = MonoSlider(
            "Max Cycles", 1, 200,
            self.config.get("max_cycles", 16), is_int=True
        )
        self.s_tool_calls = MonoSlider(
            "Max Tool Calls", 1, 500,
            self.config.get("max_tool_calls", 40), is_int=True
        )
        cl.addWidget(self.s_cycles)
        cl.addWidget(self.s_tool_calls)

        # ── Inference ─────────────────────────────────────────────────────────
        cl.addWidget(_section_label("INFERENCE"))
        self.s_temp = MonoSlider(
            "Temperature", 0.0, 2.0,
            self.config.get("loop_temperature", 0.2)
        )
        self.s_top_p = MonoSlider(
            "Top-P", 0.0, 1.0,
            self.config.get("loop_top_p", 0.9)
        )
        self.s_max_tokens = MonoSlider(
            "Step Tokens", 32, 16384,
            self.config.get("loop_max_tokens", 1024), is_int=True
        )
        cl.addWidget(self.s_temp)
        cl.addWidget(self.s_top_p)
        cl.addWidget(self.s_max_tokens)

        # ── Tools / Infer backend ─────────────────────────────────────────────
        cl.addWidget(_section_label("TOOLS"))

        tools_row = QHBoxLayout()
        lbl_tools = QLabel("Preset")
        lbl_tools.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 9px;")
        lbl_tools.setFixedWidth(72)
        self.cmb_tools = QComboBox()
        self.cmb_tools.addItem("Read Only",               "read_only")
        self.cmb_tools.addItem("Code Write  (! write)",   "code_write")
        self.cmb_tools.addItem("Full Exec   (! exec)",    "full_exec")
        self.cmb_tools.setStyleSheet(_combo_style())
        tools_row.addWidget(lbl_tools)
        tools_row.addWidget(self.cmb_tools, 1)
        cl.addLayout(tools_row)

        infer_row = QHBoxLayout()
        lbl_infer = QLabel("Infer")
        lbl_infer.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 9px;")
        lbl_infer.setFixedWidth(72)
        self.cmb_infer_backend = QComboBox()
        self.cmb_infer_backend.addItem("JSON Parser", "json")
        self.cmb_infer_backend.addItem("BAML",        "baml")
        self.cmb_infer_backend.setStyleSheet(_combo_style())
        infer_row.addWidget(lbl_infer)
        infer_row.addWidget(self.cmb_infer_backend, 1)
        cl.addLayout(infer_row)

        baml_row = QHBoxLayout()
        lbl_baml = QLabel("BAML Call")
        lbl_baml.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 9px;")
        lbl_baml.setFixedWidth(72)
        self.inp_baml_call = QLineEdit()
        self.inp_baml_call.setPlaceholderText("module.callable (or module:callable)")
        self.inp_baml_call.setStyleSheet(_path_style())
        baml_row.addWidget(lbl_baml)
        baml_row.addWidget(self.inp_baml_call, 1)
        cl.addLayout(baml_row)

        cl.addStretch()

        self.ops_stack.addWidget(model_tab)
        self.ops_stack.addWidget(config_tab)
        self.btn_tab_model.toggled.connect(
            lambda checked: self.ops_stack.setCurrentIndex(0) if checked else None
        )
        self.btn_tab_config.toggled.connect(
            lambda checked: self.ops_stack.setCurrentIndex(1) if checked else None
        )

        grp_code.add_layout(code_lay)
        return grp_code

    # ── LOOP STATUS group ─────────────────────────────────────────────────────

    def _build_loop_status_group(self) -> QWidget:
        grp_loop = MonoGroupBox("LOOP STATUS")
        loop_lay = QVBoxLayout()
        loop_lay.setSpacing(10)

        stat_row = QHBoxLayout()
        stat_row.setSpacing(8)
        self.lbl_run_status = QLabel("—")
        self.lbl_run_cycles = QLabel("cycles: —")
        self.lbl_run_tools  = QLabel("tools: —")
        self.lbl_run_wall   = QLabel("wall: —")
        for lbl in (self.lbl_run_status, self.lbl_run_cycles, self.lbl_run_tools, self.lbl_run_wall):
            lbl.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 9px;")
            stat_row.addWidget(lbl)
        stat_row.addStretch()

        self.txt_result_summary = QTextEdit()
        self.txt_result_summary.setReadOnly(True)
        self.txt_result_summary.setMaximumHeight(80)
        self.txt_result_summary.setPlaceholderText("Run summary / handoff notes will appear here…")
        self.txt_result_summary.setStyleSheet(_output_style())

        lbl_appr = QLabel("APPROVAL (FALLBACK)")
        lbl_appr.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-weight: bold; letter-spacing: 1px;"
        )
        self.lbl_approval = QLabel("No pending approval (use inline timeline buttons when available)")
        self.lbl_approval.setWordWrap(True)
        self.lbl_approval.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 9px;")

        appr_row = QHBoxLayout()
        appr_row.setSpacing(6)
        self.btn_approve = MonoButton("APPROVE")
        self.btn_deny    = MonoButton("DENY")
        self.btn_approve.setEnabled(False)
        self.btn_deny.setEnabled(False)
        self.btn_approve.clicked.connect(lambda: self._send_approval(True))
        self.btn_deny.clicked.connect(lambda: self._send_approval(False))
        appr_row.addWidget(self.btn_approve)
        appr_row.addWidget(self.btn_deny)
        appr_row.addStretch()

        loop_lay.addLayout(stat_row)
        loop_lay.addWidget(self.txt_result_summary)
        loop_lay.addWidget(lbl_appr)
        loop_lay.addWidget(self.lbl_approval)
        loop_lay.addLayout(appr_row)
        loop_lay.addStretch()
        grp_loop.add_layout(loop_lay)
        return grp_loop

    # ─────────────────────────────────────────────────────────────────────────
    # Panel lifecycle — curtain drawer + detachable panels
    # ─────────────────────────────────────────────────────────────────────────

    def _on_launcher_clicked(self, key: str):
        panel = self._panels.get(key)
        if panel is not None and not isValid(panel):
            self._panels.pop(key, None)
            panel = None
        if panel is not None:
            if panel.isVisible():
                panel.raise_()
            else:
                self._dock_panel(key)
            return
        docked = sum(
            1 for p in self._panels.values()
            if not p._is_floating and p.parent() is not None
        )
        if docked >= 2:
            return
        self._spawn_panel(key)

    def _spawn_panel(self, key: str):
        if key == "output":
            content = self._output_content
            title = "OUTPUT"
        elif key == "controls":
            content = self._controls_content
            title = "CONTROLS"
        else:
            content = self._pad_content
            title = "AGENT DEBUG"
        panel = _SpawnedPanel(key, title, content, self)
        panel.sig_close.connect(lambda: self._on_panel_closed(key))
        panel.sig_detach.connect(lambda: self._on_panel_detached(key))
        panel.sig_dock.connect(lambda: self._on_panel_dock_requested(key))
        self._panels[key] = panel
        self._dock_panel(key)

    def _dock_panel(self, key: str):
        panel = self._panels.get(key)
        if panel is None or not isValid(panel):
            self._panels.pop(key, None)
            return
        panel._is_floating = False
        panel.setWindowFlags(Qt.Widget)
        panel.setParent(self)
        self.content_splitter.addWidget(panel)
        n = self.content_splitter.count()
        if n >= 2:
            total = self.content_splitter.width() or 900
            main_w = int(total * 0.50)
            rest = total - main_w
            if n == 2:
                self.content_splitter.setSizes([main_w, rest])
            else:
                panel_w = rest // (n - 1)
                self.content_splitter.setSizes([main_w] + [panel_w] * (n - 1))
        panel.show()
        self._curtain.set_active(key, True)

    def _on_panel_detached(self, key: str):
        panel = self._panels.get(key)
        if panel is None or not isValid(panel):
            self._panels.pop(key, None)
            self._curtain.set_active(key, False)
            return
        self._curtain.set_active(key, True)

    def _on_panel_dock_requested(self, key: str):
        panel = self._panels.get(key)
        if panel is None:
            return
        docked = sum(
            1 for k, p in self._panels.items()
            if k != key and not p._is_floating and p.parent() is not None
        )
        if docked >= 2:
            return
        self._dock_panel(key)

    def _on_panel_closed(self, key: str):
        panel = self._panels.pop(key, None)
        if panel is not None:
            self._preserve_panel_content(panel)
            panel.setParent(None)
            panel.hide()
            panel.deleteLater()
        self._curtain.set_active(key, False)

    def _preserve_panel_content(self, panel: _SpawnedPanel) -> None:
        if panel is None:
            return
        content = getattr(panel, "_content", None)
        if content is None or not isValid(content):
            return
        try:
            content.hide()
        except Exception:
            pass
        try:
            content.setParent(self)
        except Exception:
            pass

    def _cleanup_spawned_panels(self) -> None:
        for key, panel in list(self._panels.items()):
            try:
                self._curtain.set_active(key, False)
            except Exception:
                pass
            try:
                self._preserve_panel_content(panel)
            except Exception:
                pass
            try:
                panel.hide()
            except Exception:
                pass
            try:
                panel.setParent(None)
            except Exception:
                pass
            try:
                panel.close()
            except Exception:
                pass
            try:
                panel.deleteLater()
            except Exception:
                pass
        self._panels.clear()

    def closeEvent(self, event) -> None:
        self._cleanup_spawned_panels()
        super().closeEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_signals(self) -> None:
        self.btn_run.clicked.connect(self._emit_generate)
        self.btn_stop.clicked.connect(self.sig_stop.emit)
        # Sliders → save config + emit side-effects
        self.s_ctx.valueChanged.connect(self._on_ctx_changed)
        self.s_cycles.valueChanged.connect(self._save_config)
        self.s_tool_calls.valueChanged.connect(self._save_config)
        self.s_temp.valueChanged.connect(self._save_config)
        self.s_top_p.valueChanged.connect(self._save_config)
        self.s_max_tokens.valueChanged.connect(self._save_config)
        # Combos + text
        self.inp_model.textChanged.connect(self._save_config)
        self.inp_baml_call.textChanged.connect(self._save_config)
        self.txt_prompt.textChanged.connect(self._save_config)
        self.cmb_tools.currentIndexChanged.connect(self._save_config)
        self.cmb_infer_backend.currentIndexChanged.connect(self._save_config)

    def eventFilter(self, obj: object, event: object) -> bool:
        try:
            timeline = getattr(self, "lst_timeline", None)
            if timeline is not None and isValid(timeline):
                viewport = timeline.viewport()
                if viewport is not None and isValid(viewport) and obj is viewport:
                    if hasattr(event, "type") and event.type() == QEvent.Resize:
                        self._refresh_timeline_layout()
                        return False

            prompt = getattr(self, "txt_prompt", None)
            if prompt is not None and isValid(prompt) and obj is prompt and hasattr(event, "type"):
                if event.type() == QEvent.KeyPress:
                    if event.key() == Qt.Key_Return and not (event.modifiers() & Qt.ShiftModifier):
                        self._emit_generate()
                        return True
                    if event.key() == Qt.Key_Return and (event.modifiers() & Qt.ShiftModifier):
                        return False
            return super().eventFilter(obj, event)
        except RuntimeError:
            # Widget tear-down race: ignore late events from deleted C++ objects.
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Config persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _default_config(self) -> dict[str, Any]:
        return {
            "gguf_path":        "",
            "ctx_limit":        8192,
            "workspace_root":   str(DEFAULT_WORKSPACE_ROOT),
            "tool_preset":      "read_only",
            "infer_backend":    "json",
            "baml_call":       "",
            "max_cycles":       16,
            "max_tool_calls":   40,
            "loop_temperature": 0.2,
            "loop_top_p":       0.9,
            "loop_max_tokens":  1024,
            "prompt":           "",
        }

    def _init_debug_log(self) -> None:
        try:
            debug_dir = CONFIG_DIR / "debug_logs"
            debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._debug_log_path = debug_dir / f"code_debug_{stamp}_{self._session_id}.jsonl"
            self._write_debug_entry(
                "session_start",
                {
                    "session_id": self._session_id,
                    "workspace_root": str(self._workspace_root),
                    "cwd": str(Path(os.getcwd()).resolve()),
                    "config_path": str(_CONFIG_PATH),
                },
            )
        except Exception:
            self._debug_log_path = None

    def _write_debug_entry(self, kind: str, data: dict[str, Any] | None = None) -> None:
        path = self._debug_log_path
        if path is None:
            return
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": str(kind or "unknown"),
            "session_id": self._session_id,
            "data": dict(data or {}),
        }
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _load_config(self) -> dict[str, Any]:
        cfg = self._default_config()
        try:
            if _CONFIG_PATH.exists():
                data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    cfg.update(data)
        except Exception:
            pass
        return cfg

    def _slider_val(self, s: MonoSlider) -> float:
        """Read the true value from a MonoSlider (respects is_int scaling)."""
        if s.is_int:
            return float(s.slider.value())
        return s.slider.value() / 100.0

    def _set_slider_val(self, s: MonoSlider, val: float) -> None:
        """Set a MonoSlider value without triggering spurious saves."""
        s.slider.blockSignals(True)
        if s.is_int:
            s.slider.setValue(int(val))
            s.val_lbl.setText(str(int(val)))
        else:
            s.slider.setValue(int(val * 100))
            s.val_lbl.setText(f"{val:.2f}")
        s.slider.blockSignals(False)

    def _save_config(self, *_args) -> None:
        self.config.update({
            "gguf_path":        self.inp_model.text().strip(),
            "ctx_limit":        int(self._slider_val(self.s_ctx)),
            "workspace_root":   str(self._workspace_root),
            "tool_preset":      str(self.cmb_tools.currentData() or "read_only"),
            "infer_backend":    str(self.cmb_infer_backend.currentData() or "json"),
            "baml_call":       self.inp_baml_call.text().strip(),
            "max_cycles":       int(self._slider_val(self.s_cycles)),
            "max_tool_calls":   int(self._slider_val(self.s_tool_calls)),
            "loop_temperature": float(self._slider_val(self.s_temp)),
            "loop_top_p":       float(self._slider_val(self.s_top_p)),
            "loop_max_tokens":  int(self._slider_val(self.s_max_tokens)),
            "prompt":           self.txt_prompt.toPlainText(),
        })
        try:
            _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CONFIG_PATH.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        except Exception as exc:
            self.append_trace(f"config save failed: {exc}")

    def _apply_config(self) -> None:
        self._workspace_root = self._resolve_workspace_root_from_config()
        self._refresh_workspace_button()
        self.inp_model.setText(str(self.config.get("gguf_path") or ""))

        self._set_slider_val(self.s_ctx,        float(self.config.get("ctx_limit", 8192)))
        self._set_slider_val(self.s_cycles,     float(self.config.get("max_cycles", 16)))
        self._set_slider_val(self.s_tool_calls, float(self.config.get("max_tool_calls", 40)))
        self._set_slider_val(self.s_temp,       float(self.config.get("loop_temperature", 0.2)))
        self._set_slider_val(self.s_top_p,      float(self.config.get("loop_top_p", 0.9)))
        self._set_slider_val(self.s_max_tokens, float(self.config.get("loop_max_tokens", 1024)))

        preset = str(self.config.get("tool_preset", "read_only"))
        idx = self.cmb_tools.findData(preset)
        self.cmb_tools.setCurrentIndex(idx if idx >= 0 else 0)

        backend = str(self.config.get("infer_backend", "json"))
        idx = self.cmb_infer_backend.findData(backend)
        self.cmb_infer_backend.setCurrentIndex(idx if idx >= 0 else 0)
        self.inp_baml_call.setText(str(self.config.get("baml_call") or ""))

        self.txt_prompt.setPlainText(str(self.config.get("prompt") or ""))

    # ─────────────────────────────────────────────────────────────────────────
    # Outgoing
    # ─────────────────────────────────────────────────────────────────────────

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", "", "GGUF Files (*.gguf);;All Files (*)"
        )
        if not path:
            return
        self.inp_model.setText(path)
        self.sig_set_model_path.emit(path)
        self._save_config()

    def _on_ctx_changed(self, value: float) -> None:
        self.sig_set_ctx_limit.emit(int(value))
        self._save_config()

    def _resolve_workspace_root_from_config(self) -> Path:
        raw = str(self.config.get("workspace_root") or "").strip()
        if not raw:
            return DEFAULT_WORKSPACE_ROOT
        try:
            path = Path(raw).expanduser().resolve()
        except Exception:
            return DEFAULT_WORKSPACE_ROOT
        return path if path.exists() else DEFAULT_WORKSPACE_ROOT

    def _refresh_workspace_button(self) -> None:
        workspace_path = str(self._workspace_root)
        self.btn_workspace.setText(f"WORKSPACE: {_short_path_text(workspace_path)}")
        self.btn_workspace.setToolTip(workspace_path)

    def _pick_workspace_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Workspace Root",
            str(self._workspace_root),
        )
        if not selected:
            return
        try:
            resolved = Path(selected).expanduser().resolve()
        except Exception:
            self.append_trace(f"[LOOP] invalid workspace path: {selected}")
            return
        if not resolved.exists() or not resolved.is_dir():
            self.append_trace(f"[LOOP] workspace must be an existing directory: {resolved}")
            return
        self._workspace_root = resolved
        self._refresh_workspace_button()
        self._save_config()
        self.sig_runtime_command.emit(
            CMD_SET_WORKSPACE_ROOT,
            {"path": str(self._workspace_root)},
        )
        self.append_trace(f"[LOOP] workspace root updated: {self._workspace_root}")

    def _apply_workspace_root_runtime(self) -> None:
        self.sig_runtime_command.emit(
            CMD_SET_WORKSPACE_ROOT,
            {"path": str(self._workspace_root)},
        )

    def _append_stream_line(self, line: str = "") -> None:
        text = str(line or "")
        if not text:
            self._timeline_add_card("system", " ", "")
            return
        if text.startswith("USER: "):
            self._timeline_add_card("user", "USER", text[6:].strip())
            return
        if text.startswith("AGENT: "):
            self._timeline_add_card("agent", "AGENT", text[7:].strip())
            return
        if text.startswith("SYSTEM: "):
            self._timeline_add_card("system", "SYSTEM", text[8:].strip())
            return
        if text.startswith("APPROVAL NEEDED: "):
            self._timeline_add_card("approval", "APPROVAL", text[len("APPROVAL NEEDED: "):].strip())
            return
        self._timeline_add_card("system", "SYSTEM", text)

    def _timeline_has_items(self) -> bool:
        return bool(getattr(self, "lst_timeline", None)) and self.lst_timeline.count() > 0

    def _timeline_clear(self) -> None:
        if getattr(self, "lst_timeline", None) is not None:
            self.lst_timeline.clear()
        if getattr(self, "txt_stream", None) is not None:
            self.txt_stream.clear()

    def _timeline_add_card(
        self,
        kind: str,
        title: str,
        body: str = "",
        details: str = "",
        actions: list[tuple[str, Any]] | None = None,
        on_click: Any | None = None,
        meta: dict[str, Any] | None = None,
    ) -> QListWidgetItem | None:
        if getattr(self, "lst_timeline", None) is None:
            # Fallback to legacy text stream if timeline is unavailable.
            line = f"{title}: {body}".strip()
            self.append_token(line + ("\n" if line else ""))
            return None
        item = QListWidgetItem(self.lst_timeline)
        card = _TimelineCard(kind, title, body, details, actions, on_click=on_click, parent=self.lst_timeline)
        card._timeline_item = item
        card._timeline_list = self.lst_timeline
        item_meta = dict(meta) if isinstance(meta, dict) else {}
        item_meta.setdefault("kind", str(kind or ""))
        item.setData(Qt.UserRole, item_meta)
        item.setSizeHint(card.sizeHint())
        self.lst_timeline.addItem(item)
        self.lst_timeline.setItemWidget(item, card)
        self._refresh_timeline_layout()
        self.lst_timeline.scrollToBottom()
        return item

    def _refresh_timeline_layout(self) -> None:
        if getattr(self, "lst_timeline", None) is None:
            return
        for i in range(self.lst_timeline.count()):
            item = self.lst_timeline.item(i)
            if item is None:
                continue
            widget = self.lst_timeline.itemWidget(item)
            if widget is None:
                continue
            try:
                widget.adjustSize()
                item.setSizeHint(widget.sizeHint())
            except Exception:
                continue
        self.lst_timeline.updateGeometries()

    def _set_cycle_items_hidden(self, run_id: str, cycle: int, hidden: bool) -> None:
        rid = str(run_id or "").strip()
        if not rid or cycle <= 0 or getattr(self, "lst_timeline", None) is None:
            return
        for i in range(self.lst_timeline.count()):
            item = self.lst_timeline.item(i)
            if item is None:
                continue
            meta = item.data(Qt.UserRole)
            if not isinstance(meta, dict):
                continue
            if str(meta.get("run_id") or "").strip() != rid:
                continue
            try:
                cnum = int(meta.get("cycle") or 0)
            except Exception:
                cnum = 0
            if cnum != int(cycle):
                continue
            if str(meta.get("card_type") or "") == "cycle_header":
                continue
            if bool(meta.get("pin_visible", False)):
                # Keep approval prompts visible even when the cycle collapses.
                continue
            item.setHidden(bool(hidden))
        self._run_cycle_collapsed[(rid, int(cycle))] = bool(hidden)
        self._set_cycle_header_state(rid, int(cycle), bool(hidden))
        self._refresh_timeline_layout()

    def _toggle_cycle_group(self, run_id: str, cycle: int) -> None:
        rid = str(run_id or "").strip()
        if not rid or cycle <= 0:
            return
        hidden = not bool(self._run_cycle_collapsed.get((rid, int(cycle)), False))
        self._set_cycle_items_hidden(rid, int(cycle), hidden)

    def _cycle_header_title(self, cycle: int, collapsed: bool) -> str:
        arrow = "▶" if collapsed else "▼"
        return f"{arrow} CYCLE {int(cycle)}"

    def _set_cycle_header_state(self, run_id: str, cycle: int, collapsed: bool) -> None:
        rid = str(run_id or "").strip()
        if not rid or cycle <= 0 or getattr(self, "lst_timeline", None) is None:
            return
        title = self._cycle_header_title(cycle, collapsed)
        for i in range(self.lst_timeline.count()):
            item = self.lst_timeline.item(i)
            if item is None:
                continue
            meta = item.data(Qt.UserRole)
            if not isinstance(meta, dict):
                continue
            if str(meta.get("run_id") or "").strip() != rid:
                continue
            try:
                cnum = int(meta.get("cycle") or 0)
            except Exception:
                cnum = 0
            if cnum != int(cycle):
                continue
            if str(meta.get("card_type") or "") != "cycle_header":
                continue
            widget = self.lst_timeline.itemWidget(item)
            if isinstance(widget, _TimelineCard):
                widget.set_title_text(title)

    def _transition_run_state(
        self,
        run_id: str,
        next_state: str,
        reason: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            return
        result = apply_lifecycle_transition(
            current=self._run_lifecycle.get(rid),
            next_state=next_state,
            reason=reason,
            extra=extra,
        )
        for warn in result.warnings:
            self.append_trace(f"[LIFECYCLE] {rid}: {warn}")
        self._run_lifecycle[rid] = result.record

    def _effect_add(self, run_id: str, entry: dict[str, Any]) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            return
        bucket = self._effect_journal.setdefault(rid, [])
        bucket.append(dict(entry or {}))
        if len(bucket) > 200:
            self._effect_journal[rid] = bucket[-200:]

    def _mark_cycle_seen(self, run_id: str, cycle: int) -> bool:
        rid = str(run_id or "").strip()
        if not rid:
            return False
        try:
            cnum = int(cycle)
        except Exception:
            return False
        if cnum <= 0:
            return False
        seen = self._run_cycle_headers.setdefault(rid, set())
        if cnum in seen:
            return False
        seen.add(cnum)
        if len(seen) > 500:
            # Defensive bound; runs should be much smaller than this.
            self._run_cycle_headers[rid] = set(sorted(seen)[-500:])
        return True

    @staticmethod
    def _extract_touched_files_from_entry(entry: dict[str, Any]) -> list[str]:
        if not isinstance(entry, dict):
            return []
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        paths: list[str] = []
        for key in ("path", "src", "dst", "from", "to", "target", "file"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())
        return paths

    def _carry_forward_block(self, *, include_unsuccessful: bool = False) -> str:
        if not isinstance(self._current_result, dict):
            return ""
        success = bool(self._current_result.get("success", False))
        if not success and not include_unsuccessful:
            return ""
        if self._conversation_turn <= 0:
            return ""

        run_id = str(self._current_result.get("run_id") or "").strip()
        summary = str(self._current_result.get("summary") or "").strip()
        if not summary:
            return ""
        wall_hit = str(self._current_result.get("wall_hit") or "").strip()

        entries = list(self._effect_journal.get(run_id, [])) if run_id else []
        tool_results = [e for e in entries if str(e.get("kind") or "") == "tool_result"]
        self_checks = [e for e in entries if str(e.get("kind") or "") == "step_self_check"]

        evidence_lines: list[str] = []
        for e in tool_results[-5:]:
            tool = str(e.get("tool") or "tool")
            status = str(e.get("status") or "?").upper()
            preview = str(e.get("output_preview") or "").strip().replace("\n", " ")
            if len(preview) > 140:
                preview = preview[:140] + "..."
            evidence_lines.append(f"- {tool} [{status}]: {preview}" if preview else f"- {tool} [{status}]")

        touched: list[str] = []
        seen_paths: set[str] = set()
        for e in tool_results[-12:]:
            for p in self._extract_touched_files_from_entry(e):
                if p not in seen_paths:
                    seen_paths.add(p)
                    touched.append(p)
        touched = touched[:8]

        check_lines: list[str] = []
        for e in self_checks[-4:]:
            ok = e.get("step_ok")
            status = "UNKNOWN"
            if ok is True:
                status = "PASS"
            elif ok is False:
                status = "FAIL"
            msg = str(e.get("self_check") or "").strip()
            if msg:
                check_lines.append(f"- {status}: {msg}")

        parts: list[str] = ["PREVIOUS RUN CONTEXT:"]
        parts.append(f"STATUS: {'success' if success else 'stopped'}")
        if wall_hit:
            parts.append(f"WALL/STOP REASON: {wall_hit}")
        parts.append("SUMMARY:")
        parts.append(summary)
        if touched:
            parts.append("")
            parts.append("TOUCHED FILES:")
            parts.extend(f"- {p}" for p in touched)
        if evidence_lines:
            parts.append("")
            parts.append("RECENT TOOL EVIDENCE:")
            parts.extend(evidence_lines)
        if check_lines:
            parts.append("")
            parts.append("RECENT SELF-CHECKS:")
            parts.extend(check_lines)
        return "\n".join(parts).strip()

    def _workspace_tree_snapshot(self, root: Path, *, max_depth: int = 2, max_entries: int = 80) -> list[str]:
        lines: list[str] = []
        emitted = 0

        def walk(path: Path, depth: int, prefix: str) -> None:
            nonlocal emitted
            if emitted >= max_entries or depth > max_depth:
                return
            try:
                entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except Exception:
                lines.append(f"{prefix}<unreadable>")
                emitted += 1
                return
            for entry in entries:
                if emitted >= max_entries:
                    break
                name = entry.name + ("/" if entry.is_dir() else "")
                lines.append(f"{prefix}{name}")
                emitted += 1
                if entry.is_dir() and depth < max_depth:
                    walk(entry, depth + 1, prefix + "  ")

        walk(root, 1, "")
        return lines

    def _build_environment_snapshot_block(self, preset: str) -> str:
        workspace = self._workspace_root if self._workspace_root.exists() else DEFAULT_WORKSPACE_ROOT
        cwd = Path(os.getcwd()).resolve()
        tree_lines = self._workspace_tree_snapshot(workspace, max_depth=2, max_entries=80)
        model_path = str(self.inp_model.text().strip() or "")

        lines: list[str] = [
            "ENVIRONMENT SNAPSHOT (DETERMINISTIC)",
            f"workspace_root: {workspace}",
            f"working_directory: {workspace}",
            "",
            "SCOPED FILE TREE (workspace_root, depth<=2):",
        ]
        if tree_lines:
            lines.extend(f"- {line}" for line in tree_lines)
        else:
            lines.append("- <empty>")

        lines.extend([
            "",
            "RELEVANT CONFIG:",
            f"- tool_preset: {preset}",
            f"- infer_backend: {str(self.cmb_infer_backend.currentData() or 'json')}",
            f"- baml_call: {self.inp_baml_call.text().strip() or '<env:MONOLITH_LOOP_BAML_CALL>'}",
            f"- model_path: {model_path or '<not selected>'}",
            f"- ctx_limit: {int(self._slider_val(self.s_ctx))}",
            f"- max_cycles: {int(self._slider_val(self.s_cycles))}",
            f"- max_tool_calls: {int(self._slider_val(self.s_tool_calls))}",
            f"- temperature: {float(self._slider_val(self.s_temp))}",
            f"- top_p: {float(self._slider_val(self.s_top_p))}",
            f"- max_tokens: {int(self._slider_val(self.s_max_tokens))}",
        ])
        return "\n".join(lines).strip()

    def _format_event_line(self, event_type: str, kind: str, data: dict[str, Any], meta: dict[str, Any]) -> str:
        et = str(event_type or "").strip()
        kd = str(kind or "unknown").strip()
        rid = str((data or {}).get("run_id") or "").strip()
        seq = meta.get("seq")
        seq_s = f"#{seq}" if isinstance(seq, int) else "#?"
        try:
            cycle = int((data or {}).get("cycle") or 0)
        except Exception:
            cycle = 0
        indent = "  " * max(0, min(cycle - 1, 10))
        run_s = f" run={rid}" if rid else ""
        cyc_s = f" c={cycle}" if cycle > 0 else ""

        if et == EVENT_CONTROL:
            state = str((data or {}).get("state") or "").strip()
            reason = str((data or {}).get("reason_code") or "").strip()
            extra: list[str] = []
            if state:
                extra.append(f"state={state}")
            if reason:
                extra.append(f"reason={reason}")
            if "request_id" in (data or {}):
                extra.append(f"request={str((data or {}).get('request_id') or '')}")
            if "allow" in (data or {}):
                extra.append("allow=yes" if bool((data or {}).get("allow")) else "allow=no")
            extras = (" " + " ".join(extra)) if extra else ""
            return f"{seq_s} => control.{kd}{run_s}{cyc_s}{extras}"

        if et == EVENT_LOOP:
            if kd == "step_parsed":
                step = (data or {}).get("step") if isinstance((data or {}).get("step"), dict) else {}
                intent = str(step.get("intent") or "").strip()
                actions = step.get("actions_count")
                task_finished = step.get("task_finished")
                if not isinstance(task_finished, bool):
                    task_finished = bool(step.get("finish", False))
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} intent={intent or '?'} actions={actions if actions is not None else 0} done={'yes' if task_finished else 'no'}"
            if kd == "llm_call":
                call_index = int((data or {}).get("call_index") or 0)
                ok = bool((data or {}).get("ok", False))
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} n={call_index} ok={'yes' if ok else 'no'}"
            if kd == "llm_input":
                call_index = int((data or {}).get("call_index") or 0)
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} n={call_index}"
            if kd == "action_result":
                tool = str((data or {}).get("tool") or "tool")
                status = str((data or {}).get("status") or ("ok" if bool((data or {}).get("ok")) else "failed"))
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} tool={tool} status={status}"
            if kd == "policy_decision":
                tool = str((data or {}).get("tool") or "tool")
                action = str((data or {}).get("action") or "")
                reason = str((data or {}).get("reason_code") or (data or {}).get("reason") or "")
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} tool={tool} action={action or '?'} reason={reason or '?'}"
            if kd in {"approval_prompt", "approval_required"}:
                tool = str((data or {}).get("tool") or "tool")
                scope = str((data or {}).get("scope") or "")
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} tool={tool} scope={scope or '?'}"
            if kd == "tool_started":
                tool = str((data or {}).get("tool") or "tool")
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} tool={tool}"
            if kd == "tool_failure_guidance":
                tool = str((data or {}).get("tool") or "tool")
                fclass = str((data or {}).get("failure_class") or "")
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} tool={tool} class={fclass or '?'}"
            if kd == "circuit_breaker":
                tool = str((data or {}).get("tool") or "tool")
                fclass = str((data or {}).get("failure_class") or "")
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} tool={tool} class={fclass or '?'}"
            if kd == "routing_blocked":
                tool = str((data or {}).get("tool") or "tool")
                dep = str((data or {}).get("dependency") or "")
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} tool={tool} dep={dep or '?'}"
            if kd == "noop_blocked":
                tool = str((data or {}).get("tool") or "tool")
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} tool={tool}"
            if kd == "mission_check":
                ok = bool((data or {}).get("ok", False))
                score = (data or {}).get("score")
                return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s} ok={'yes' if ok else 'no'} score={score}"
            if kd == "finish":
                result = (data or {}).get("result") if isinstance((data or {}).get("result"), dict) else {}
                success = bool(result.get("success", False))
                wall = str(result.get("wall_hit") or "").strip()
                tail = f" wall={wall}" if wall else ""
                return f"{seq_s} -> loop.{kd}{run_s}{cyc_s} success={'yes' if success else 'no'}{tail}"
            return f"{indent}{seq_s} -> loop.{kd}{run_s}{cyc_s}"

        return f"{seq_s} .. {et}.{kd}{run_s}{cyc_s}"

    def _build_generate_payload(self, prompt: str) -> dict[str, Any]:
        preset = str(self.cmb_tools.currentData() or "read_only")
        goal = str(prompt or "").strip()
        parent_run_id = str(self._pending_redirect_parent_run_id or "")
        carry_block = self._carry_forward_block(include_unsuccessful=bool(parent_run_id))
        if carry_block:
            goal = carry_block + "\n\n" + goal
        turn_id = f"turn_{uuid.uuid4().hex[:10]}"
        redirected = bool(parent_run_id)
        return {
            "goal": goal,
            "tool_names": list(_TOOL_PRESETS.get(preset, _TOOL_PRESETS["read_only"])),
            "infer_backend": str(self.cmb_infer_backend.currentData() or "json"),
            "baml_call": self.inp_baml_call.text().strip(),
            "policy": {
                "max_cycles": int(self._slider_val(self.s_cycles)),
                "max_tool_calls": int(self._slider_val(self.s_tool_calls)),
            },
            "infer": {
                "temperature": float(self._slider_val(self.s_temp)),
                "top_p": float(self._slider_val(self.s_top_p)),
                "max_tokens": int(self._slider_val(self.s_max_tokens)),
            },
            "_turn_meta": {
                "session_id": self._session_id,
                "turn_id": turn_id,
                "parent_run_id": parent_run_id,
                "redirected": redirected,
                "user_prompt": str(prompt or ""),
            },
        }

    def _start_prompt_run(self, prompt: str, *, redirected: bool = False, echo_user: bool = True) -> None:
        payload = self._build_generate_payload(prompt)
        self._write_debug_entry(
            "prompt_run_start",
            {
                "prompt": str(prompt or ""),
                "redirected": bool(redirected),
                "echo_user": bool(echo_user),
                "payload": payload,
            },
        )
        self._apply_workspace_root_runtime()
        self._pending_redirect_parent_run_id = None
        preset = str(self.cmb_tools.currentData() or "read_only")
        self._conversation_turn += 1
        if self._timeline_has_items():
            self._append_stream_line("-" * 40)
        if echo_user:
            self._append_stream_line(f"USER: {prompt}")
        if redirected:
            self._append_stream_line("SYSTEM: Redirecting with your updated instruction.")
        self._append_stream_line("AGENT: Working on it...")
        if preset != "read_only":
            self.append_trace("INFO: non-read-only preset - tool approvals will prompt before execution")
        self.txt_result_summary.clear()
        self._current_result = None
        self._set_run_summary(None)
        self._clear_approval()
        self.sig_generate.emit(payload)

    def _emit_generate(self) -> None:
        prompt = self.txt_prompt.toPlainText().strip()
        if not prompt:
            self.append_trace("ERROR: message is empty")
            return
        self.txt_prompt.clear()
        if self._run_active:
            self._pending_redirect_prompt = prompt
            self._pending_redirect_parent_run_id = str(self._active_run_id or "")
            if self._pending_approval_id:
                # Redirect invalidates pending approval; fail closed and continue
                # with explicit restart semantics on the next READY.
                self._send_approval(False)
            self._append_stream_line(f"USER: {prompt}")
            self._append_stream_line(
                "SYSTEM: Interrupt requested. Stopping current run, then continuing with your correction..."
            )
            self.append_trace("[LOOP] redirect requested by user; stopping active run")
            self.sig_stop.emit()
            return
        self._start_prompt_run(prompt)

    # Incoming — public interface (guard / main window)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_model_capabilities(self, payload: dict) -> None:
        try:
            ctx_len = int(payload.get("model_ctx_length") or 0)
            actual  = int(payload.get("actual_ctx") or 0)
            self.lbl_caps.setText(f"ctx={ctx_len} actual={actual}")
        except Exception:
            self.lbl_caps.setText("")

    def update_status(self, engine_key: str, status: SystemStatus) -> None:
        if not isValid(self):
            return
        if engine_key != getattr(self, "_engine_key", ""):
            return
        self._last_status = status
        if status == SystemStatus.READY:
            if self._run_active:
                self._run_active = False
            self.lbl_status.setText("READY")
        elif status == SystemStatus.LOADING:
            self.lbl_status.setText("LOADING MODEL")
        elif status == SystemStatus.UNLOADING:
            self.lbl_status.setText("UNLOADING")
        elif status == SystemStatus.RUNNING:
            self._run_active = True
            self.lbl_status.setText("WORKING")
        elif status == SystemStatus.ERROR:
            self.lbl_status.setText("ERROR")
        self._refresh_status_ui()
        if status == SystemStatus.READY and self._pending_redirect_prompt:
            queued = self._pending_redirect_prompt
            self._pending_redirect_prompt = None
            self._start_prompt_run(queued, redirected=True, echo_user=False)

    def append_token(self, text: str) -> None:
        if not isValid(self):
            return
        if not text:
            return
        # Keep a hidden/raw buffer for compatibility and debugging.
        stream = getattr(self, "txt_stream", None)
        if stream is None or not isValid(stream):
            return
        cursor = stream.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(str(text))
        stream.setTextCursor(cursor)
        stream.ensureCursorVisible()

    def append_trace(self, message: str) -> None:
        if not isValid(self):
            return
        trace = getattr(self, "txt_trace", None)
        if trace is None or not isValid(trace):
            return
        msg = str(message)
        trace.append(msg)
        debug_trace = getattr(self, "txt_debug_trace", None)
        if debug_trace is not None and isValid(debug_trace):
            debug_trace.append(msg)
        self._write_debug_entry("trace", {"message": msg})
        low = msg.lower()
        if "system online" in low:
            self._is_model_loaded = True
            self._update_load_button_text()
        elif "model unloaded" in low:
            self._is_model_loaded = False
            self._update_load_button_text()
        self._refresh_status_ui()

    def on_guard_finished(self, engine_key: str, _task_id: str) -> None:
        if not isValid(self):
            return
        if engine_key != getattr(self, "_engine_key", ""):
            return
        self._run_active = False
        self._refresh_status_ui()

    def on_agent_event(self, engine_key: str, event: dict) -> None:
        if engine_key != getattr(self, "_engine_key", ""):
            return
        if not isValid(self):
            return
        self._append_loop_event(event)

    def apply_operator(self, _operator_data: dict) -> None:
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _append_loop_event(self, event: dict) -> None:
        try:
            if not isinstance(event, dict):
                return
            self._write_debug_entry("agent_event", {"event": event})
            event_type = str(event.get("event") or "")
            meta = event.get("_meta") if isinstance(event.get("_meta"), dict) else {}
            if event_type == EVENT_CONTROL:
                kind = str(event.get("kind") or "unknown")
                data = event.get("data") if isinstance(event.get("data"), dict) else {}
                data_state = str(data.get("state") or "").strip()
                data_reason = str(data.get("reason_code") or "").strip()
                control_run_id = str(data.get("run_id") or self._active_run_id or self._pending_redirect_parent_run_id or "").strip()
                if kind == "turn_started":
                    parent = str(meta.get("parent_run_id") or data.get("parent_run_id") or "").strip()
                    if parent:
                        self._transition_run_state(parent, RUN_STATE_REDIRECTED, data_reason or REASON_USER_REDIRECT)
                        self._timeline_add_card("system", "RUN STATE", f"{parent}: {RUN_STATE_REDIRECTED} ({data_reason or REASON_USER_REDIRECT})")
                    started_body_parts: list[str] = []
                    infer_backend = str(data.get("infer_backend") or "").strip()
                    if infer_backend:
                        started_body_parts.append(f"infer={infer_backend}")
                    tool_names = data.get("tool_names") if isinstance(data.get("tool_names"), list) else []
                    if tool_names:
                        started_body_parts.append(f"tools={len(tool_names)}")
                    policy_data = data.get("policy") if isinstance(data.get("policy"), dict) else {}
                    if policy_data:
                        mc = policy_data.get("max_cycles")
                        mt = policy_data.get("max_tool_calls")
                        if mc is not None and mt is not None:
                            started_body_parts.append(f"walls: cycles={mc}, tools={mt}")
                    details_parts: list[str] = []
                    if tool_names:
                        details_parts.append("[tools]\n" + ", ".join(str(t) for t in tool_names))
                    if policy_data:
                        details_parts.append("[policy]\n" + json.dumps(policy_data, ensure_ascii=False, indent=2))
                    self._timeline_add_card(
                        "system",
                        "RUN STARTED",
                        " | ".join(started_body_parts),
                        "\n\n".join(details_parts),
                    )
                    if control_run_id:
                        self._llm_call_seq_by_run[control_run_id] = 0
                elif kind == "redirect_started":
                    parent = str(data.get("parent_run_id") or "")
                    body = f"Continuing after redirect from {parent}" if parent else "Continuing after redirect"
                    self._timeline_add_card("system", "REDIRECT", body)
                elif kind == "turn_stop_requested":
                    if control_run_id:
                        self._transition_run_state(control_run_id, data_state or RUN_STATE_STOPPING, data_reason or REASON_STOP_REQUESTED)
                    self._timeline_add_card("system", "STOP", "Stop requested")
                elif kind == "turn_completed":
                    c_run = str(data.get("run_id") or self._active_run_id or "").strip()
                    if c_run:
                        success = bool(data.get("success", False))
                        reason = data_reason or str(data.get("wall_hit") or (REASON_COMPLETED if success else "unknown"))
                        state = data_state or (RUN_STATE_COMPLETED if success else RUN_STATE_STOPPED)
                        self._transition_run_state(
                            c_run,
                            state,
                            reason,
                            data,
                        )
                        self._timeline_add_card(
                            "system",
                            "RUN STATE",
                            f"{c_run}: {state}" + (f" ({reason})" if reason else ""),
                        )
                elif kind == "turn_failed":
                    if control_run_id:
                        self._transition_run_state(
                            control_run_id,
                            data_state or RUN_STATE_FAILED,
                            data_reason or REASON_ERROR,
                            {"error": str(data.get("error") or "unknown error")},
                        )
                        self._run_cycle_headers.pop(control_run_id, None)
                        for k in [k for k in list(self._run_cycle_collapsed.keys()) if k[0] == control_run_id]:
                            self._run_cycle_collapsed.pop(k, None)
                    self._timeline_add_card("system", "RUN FAILED", str(data.get("error") or "unknown error"))
                elif kind == "approval_response":
                    approved = bool(data.get("allow", False))
                    req_id = str(data.get("request_id") or "").strip()
                    latency_ms = None
                    if req_id:
                        started = self._approval_prompt_ts.pop(req_id, None)
                        if isinstance(started, datetime):
                            latency_ms = int(max(0.0, (datetime.now(timezone.utc) - started).total_seconds() * 1000.0))
                    if control_run_id:
                        self._transition_run_state(
                            control_run_id,
                            data_state or (RUN_STATE_RUNNING if approved else RUN_STATE_STOPPING),
                            data_reason or (REASON_APPROVAL_GRANTED if approved else REASON_APPROVAL_DENIED),
                        )
                        self._effect_add(control_run_id, {
                            "kind": "approval_decision",
                            "request_id": req_id,
                            "allow": approved,
                            "latency_ms": latency_ms,
                            "ts": str(meta.get("ts") or ""),
                            "seq": int(meta.get("seq") or 0) if str(meta.get("seq") or "").isdigit() else meta.get("seq"),
                        })
                    self._write_debug_entry("approval_latency", {
                        "run_id": control_run_id,
                        "request_id": req_id,
                        "allow": approved,
                        "latency_ms": latency_ms,
                    })
                    self._timeline_add_card("system", "APPROVAL", ("Granted" if approved else "Denied") + (f" ({req_id})" if req_id else ""))
                self.txt_events.append(self._format_event_line(event_type, kind, data, meta))
                return
            if event_type != EVENT_LOOP:
                self.txt_events.append(self._format_event_line(event_type, str(event.get("kind") or "unknown"), {}, meta))
                return
            kind = str(event.get("kind") or "unknown")
            data = event.get("data") or {}
            run_id = str((data or {}).get("run_id") or "").strip()
            if kind == "cycle_start" and run_id:
                self._active_run_id = run_id
                cycle_num = int((data or {}).get("cycle") or 0)
                self._transition_run_state(run_id, RUN_STATE_RUNNING, REASON_CYCLE_START, {"last_cycle": cycle_num})
                self._append_pad_snapshot(data if isinstance(data, dict) else {})
                if self._mark_cycle_seen(run_id, cycle_num):
                    for prev_cycle in sorted(self._run_cycle_headers.get(run_id, set())):
                        if prev_cycle != cycle_num:
                            self._set_cycle_items_hidden(run_id, prev_cycle, True)
                    self._timeline_add_card(
                        "system",
                        self._cycle_header_title(cycle_num, False),
                        f"Run {run_id}",
                        on_click=(lambda self=self, rid=run_id, c=cycle_num: self._toggle_cycle_group(rid, c)),
                        meta={"run_id": run_id, "cycle": cycle_num, "card_type": "cycle_header"},
                    )
            if kind == "step_parsed":
                cycle_num = int((data or {}).get("cycle") or 0)
                response = str((data or {}).get("response") or "").strip()
                reasoning = str((data or {}).get("reasoning") or "").strip()
                self_check = str((data or {}).get("self_check") or "").strip()
                step_ok_raw = (data or {}).get("step_ok")
                step = (data or {}).get("step") if isinstance((data or {}).get("step"), dict) else {}
                if run_id and (self_check or isinstance(step_ok_raw, bool)):
                    self._effect_add(run_id, {
                        "kind": "step_self_check",
                        "step_ok": step_ok_raw if isinstance(step_ok_raw, bool) else None,
                        "self_check": self_check,
                        "cycle": int((data or {}).get("cycle") or 0),
                        "ts": str(meta.get("ts") or ""),
                        "seq": int(meta.get("seq") or 0) if str(meta.get("seq") or "").isdigit() else meta.get("seq"),
                    })
                if response:
                    self._timeline_add_card("agent", "AGENT", response, meta={"run_id": run_id, "cycle": cycle_num, "card_type": "cycle_item"})
                elif reasoning:
                    self._timeline_add_card("thinking", "THINKING", reasoning, meta={"run_id": run_id, "cycle": cycle_num, "card_type": "cycle_item"})
                else:
                    intent = str(step.get("intent") or "").strip()
                    if intent:
                        self._timeline_add_card("agent", "AGENT", intent, meta={"run_id": run_id, "cycle": cycle_num, "card_type": "cycle_item"})
                if self_check or isinstance(step_ok_raw, bool):
                    status = "UNKNOWN"
                    kind_name = "thinking"
                    if step_ok_raw is True:
                        status = "PASS"
                        kind_name = "system"
                    elif step_ok_raw is False:
                        status = "FAIL"
                        kind_name = "approval" if self_check else "system"
                    body = self_check or "No self-check note provided."
                    self._timeline_add_card(
                        kind_name,
                        f"SELF-CHECK [{status}]",
                        body,
                        meta={"run_id": run_id, "cycle": cycle_num, "card_type": "cycle_item"},
                    )
            elif kind == "llm_input":
                self._append_pad_inject_entry(data if isinstance(data, dict) else {})
            elif kind == "llm_call":
                self._append_llm_call_entry(data if isinstance(data, dict) else {})
            elif kind == "policy_decision":
                cycle_num = int((data or {}).get("cycle") or 0)
                tool = str((data or {}).get("tool") or "tool")
                action = str((data or {}).get("action") or "")
                reason = str((data or {}).get("reason_code") or (data or {}).get("reason") or "")
                scope = str((data or {}).get("scope") or "")
                self._effect_add(run_id, {
                    "kind": "policy_decision",
                    "tool": tool,
                    "action": action,
                    "reason": reason,
                    "scope": scope,
                    "cycle": int((data or {}).get("cycle") or 0),
                    "ts": str(meta.get("ts") or ""),
                    "seq": int(meta.get("seq") or 0) if str(meta.get("seq") or "").isdigit() else meta.get("seq"),
                })
                if action == "deny":
                    self._timeline_add_card(
                        "system",
                        "POLICY DENY",
                        f"{tool} [{scope}] - {reason}",
                        meta={"run_id": run_id, "cycle": cycle_num, "card_type": "cycle_item"},
                    )
            elif kind == "tool_started":
                tool = str((data or {}).get("tool") or "tool")
                scope = str((data or {}).get("scope") or "")
                self._effect_add(run_id, {
                    "kind": "tool_started",
                    "tool": tool,
                    "scope": scope,
                    "args": (data or {}).get("args") if isinstance((data or {}).get("args"), dict) else {},
                    "cycle": int((data or {}).get("cycle") or 0),
                    "ts": str(meta.get("ts") or ""),
                    "seq": int(meta.get("seq") or 0) if str(meta.get("seq") or "").isdigit() else meta.get("seq"),
                })
            elif kind == "action_result":
                cycle_num = int((data or {}).get("cycle") or 0)
                tool = str((data or {}).get("tool") or "tool")
                status = str((data or {}).get("status") or ("ok" if bool((data or {}).get("ok")) else "failed"))
                args = (data or {}).get("args") if isinstance((data or {}).get("args"), dict) else {}
                preview = str((data or {}).get("output_preview") or "").strip().replace("\n", " ")
                if len(preview) > 180:
                    preview = preview[:180] + "..."
                line = f"TOOL [{status.upper()}] {tool}"
                details_parts = []
                if args:
                    details_parts.append("[args]\n" + json.dumps(args, ensure_ascii=False, indent=2))
                full_output = str((data or {}).get("output") or "")
                if full_output:
                    details_parts.append("[output]\n" + full_output[:2000])
                self._timeline_add_card(
                    "tool",
                    line,
                    preview,
                    "\n\n".join(details_parts),
                    meta={"run_id": run_id, "cycle": cycle_num, "card_type": "cycle_item"},
                )
                self._effect_add(run_id, {
                    "kind": "tool_result",
                    "tool": tool,
                    "status": status,
                    "args": args,
                    "output_preview": preview,
                    "cycle": int((data or {}).get("cycle") or 0),
                    "ts": str(meta.get("ts") or ""),
                    "seq": int(meta.get("seq") or 0) if str(meta.get("seq") or "").isdigit() else meta.get("seq"),
                })
            if kind == "finish":
                if run_id:
                    self._active_run_id = None
                    self._run_cycle_headers.pop(run_id, None)
                    for k in [k for k in list(self._run_cycle_collapsed.keys()) if k[0] == run_id]:
                        self._run_cycle_collapsed.pop(k, None)
                result = (data or {}).get("result") or {}
                self._current_result = result if isinstance(result, dict) else None
                if run_id and isinstance(self._current_result, dict):
                    success = bool(self._current_result.get("success", False))
                    self._transition_run_state(
                        run_id,
                        RUN_STATE_COMPLETED if success else RUN_STATE_STOPPED,
                        REASON_COMPLETED if success else REASON_WALL_HIT,
                        {
                            "wall_hit": str(self._current_result.get("wall_hit") or ""),
                            "summary": str(self._current_result.get("summary") or ""),
                            "cycles_used": self._current_result.get("cycles_used"),
                            "tool_calls_used": self._current_result.get("tool_calls_used"),
                        },
                    )
                self._set_run_summary(self._current_result)
                if isinstance(self._current_result, dict):
                    summary_text = str(self._current_result.get("summary") or "")
                    if summary_text:
                        self.txt_result_summary.setPlainText(summary_text)
                        status_label = "FINAL" if bool(self._current_result.get("success", False)) else "STOPPED"
                        self._timeline_add_card("agent", status_label, summary_text)
                self._clear_approval()
            elif kind == "approval_prompt":
                self._show_approval_prompt(data if isinstance(data, dict) else {})
                cycle_num = int((data or {}).get("cycle") or 0)
                tool = str((data or {}).get("tool") or "unknown")
                scope = str((data or {}).get("scope") or "")
                args = (data or {}).get("args") if isinstance((data or {}).get("args"), dict) else {}
                details = json.dumps(args, ensure_ascii=False, indent=2) if args else ""
                self._timeline_add_card(
                    "approval",
                    f"APPROVAL NEEDED: {tool} [{scope}]",
                    "",
                    details,
                    actions=[
                        ("Approve", lambda _checked=False, self=self: self._send_approval(True)),
                        ("Deny", lambda _checked=False, self=self: self._send_approval(False)),
                    ],
                    meta={"run_id": run_id, "cycle": cycle_num, "card_type": "cycle_item", "pin_visible": True},
                )
            self.txt_events.append(self._format_event_line(event_type, kind, data, meta))
        except Exception as exc:
            self.txt_events.append(f"[event parse error] {exc}")

    def _append_llm_call_entry(self, data: dict[str, Any]) -> None:
        txt = getattr(self, "txt_llm_calls", None)
        if txt is None or not isValid(txt):
            return
        run_id = str(data.get("run_id") or "").strip()
        try:
            cycle = int(data.get("cycle") or 0)
        except Exception:
            cycle = 0
        try:
            attempt = int(data.get("attempt") or 0)
        except Exception:
            attempt = 0
        backend = str(data.get("backend") or "").strip() or "json"
        ok = bool(data.get("ok", False))
        error = str(data.get("error") or "").strip()
        output = str(data.get("output") or "")
        call_index_raw = data.get("call_index")
        if isinstance(call_index_raw, int) and call_index_raw > 0:
            call_index = call_index_raw
            if run_id:
                self._llm_call_seq_by_run[run_id] = call_index
        else:
            call_index = int(self._llm_call_seq_by_run.get(run_id, 0)) + 1
            if run_id:
                self._llm_call_seq_by_run[run_id] = call_index

        header = f"LLM CALL #{call_index}  run={run_id or '?'}  cycle={cycle}  attempt={attempt}  backend={backend}  ok={'yes' if ok else 'no'}"
        parts = [header]
        if error:
            parts.append(f"error: {error}")
        parts.append("response:")
        parts.append(output if output else "<empty>")
        parts.append("-" * 80)
        txt.append("\n".join(parts))
        txt.moveCursor(QTextCursor.End)
        self._write_debug_entry(
            "llm_call",
            {
                "run_id": run_id,
                "call_index": call_index,
                "cycle": cycle,
                "attempt": attempt,
                "backend": backend,
                "ok": ok,
                "error": error,
                "output": output,
            },
        )

    def _append_pad_snapshot(self, data: dict[str, Any]) -> None:
        txt = getattr(self, "txt_pad", None)
        if txt is None or not isValid(txt):
            return
        run_id = str(data.get("run_id") or "").strip()
        try:
            cycle = int(data.get("cycle") or 0)
        except Exception:
            cycle = 0
        pad = data.get("pad") if isinstance(data.get("pad"), dict) else {}
        todo_state = pad.get("todo_state") if isinstance(pad.get("todo_state"), list) else []
        last_check = pad.get("last_check") if isinstance(pad.get("last_check"), dict) else {}
        block = {
            "run_id": run_id,
            "cycle": cycle,
            "pad": pad,
        }
        progress = pad.get("progress")
        try:
            progress_pct = f"{float(progress) * 100:.0f}%"
        except Exception:
            progress_pct = "?"

        goal_text = str(pad.get("goal") or "").strip()
        goal_preview = goal_text
        if len(goal_preview) > 320:
            goal_preview = goal_preview[:320] + "..."

        lines: list[str] = [
            f"PAD SNAPSHOT  run={run_id or '?'}  cycle={cycle}  progress={progress_pct}",
            "",
            "goal_preview:",
            goal_preview or "<empty>",
            "",
            "plan:",
            str(pad.get("plan") or "<empty>"),
            "",
        ]

        steps = pad.get("steps") if isinstance(pad.get("steps"), list) else []
        lines.append(f"steps ({len(steps)}):")
        if steps:
            for item in steps[-8:]:
                lines.append(f"- {str(item)}")
        else:
            lines.append("- <none>")
        lines.append("")

        questions = pad.get("open_questions") if isinstance(pad.get("open_questions"), list) else []
        lines.append(f"open_questions ({len(questions)}):")
        if questions:
            for q in questions[-8:]:
                lines.append(f"- {str(q)}")
        else:
            lines.append("- <none>")
        lines.append("")

        lines.append(f"todo_state ({len(todo_state)}):")
        if todo_state:
            for idx, item in enumerate(todo_state[:12], start=1):
                done = "x" if bool(item.get("crystallized")) else " "
                directive = str(item.get("directive") or "<todo>")
                tool_hint = str(item.get("tool_hint") or "")
                suffix = f" -> {tool_hint}" if tool_hint else ""
                done_cycle = item.get("cycle_crystallized")
                done_note = f" (cycle {done_cycle})" if done_cycle else ""
                lines.append(f"- [{done}] {idx}. {directive}{suffix}{done_note}")
        else:
            lines.append("<none>")
        lines.append("")

        lines.append("last_check:")
        lines.append(json.dumps(last_check, ensure_ascii=False, indent=2) if last_check else "<none>")
        lines.append("")

        artifacts = pad.get("artifacts") if isinstance(pad.get("artifacts"), dict) else {}
        lines.append(f"artifacts ({len(artifacts)}):")
        if artifacts:
            lines.append(json.dumps(artifacts, ensure_ascii=False, indent=2))
        else:
            lines.append("<none>")
        lines.append("")
        lines.append("-" * 80)
        txt.append("\n".join(lines))
        txt.moveCursor(QTextCursor.End)
        self._write_debug_entry("pad_snapshot", block)

    def _append_pad_inject_entry(self, data: dict[str, Any]) -> None:
        txt = getattr(self, "txt_pad_inject", None)
        if txt is None or not isValid(txt):
            return
        run_id = str(data.get("run_id") or "").strip()
        try:
            cycle = int(data.get("cycle") or 0)
        except Exception:
            cycle = 0
        try:
            attempt = int(data.get("attempt") or 0)
        except Exception:
            attempt = 0
        try:
            call_index = int(data.get("call_index") or 0)
        except Exception:
            call_index = 0
        backend = str(data.get("backend") or "").strip() or "json"
        digest = str(data.get("prompt_digest") or "").strip()
        prompt_chars = int(data.get("prompt_chars") or 0) if str(data.get("prompt_chars") or "").isdigit() else data.get("prompt_chars")
        prompt_messages = int(data.get("prompt_messages") or 0) if str(data.get("prompt_messages") or "").isdigit() else data.get("prompt_messages")
        messages = data.get("messages") if isinstance(data.get("messages"), list) else []
        block = {
            "run_id": run_id,
            "cycle": cycle,
            "attempt": attempt,
            "call_index": call_index,
            "backend": backend,
            "prompt_digest": digest,
            "prompt_chars": prompt_chars,
            "prompt_messages": prompt_messages,
            "messages": messages,
        }
        lines: list[str] = [f"PAD INJECT  run={run_id or '?'}  cycle={cycle}  call={call_index}  attempt={attempt}  backend={backend}"]
        lines.append(f"prompt_digest={digest or '?'}  prompt_messages={prompt_messages}  prompt_chars={prompt_chars}")
        for idx, msg in enumerate(messages, start=1):
            role = str(msg.get("role") or "unknown") if isinstance(msg, dict) else "unknown"
            content = ""
            if isinstance(msg, dict):
                content = str(msg.get("content") or "")
            lines.extend([
                "",
                f"[MESSAGE {idx}] role={role}",
                "-" * 24,
                content if content else "<empty>",
            ])
        lines.extend(["", "-" * 80])
        txt.append("\n".join(lines))
        txt.moveCursor(QTextCursor.End)
        self._write_debug_entry("pad_inject", block)

    def _show_approval_prompt(self, data: dict[str, Any]) -> None:
        request_id = str(data.get("request_id") or "").strip()
        tool  = str(data.get("tool") or "unknown")
        scope = str(data.get("scope") or "")
        args  = data.get("args") if isinstance(data.get("args"), dict) else {}
        run_id = str(data.get("run_id") or self._active_run_id or "").strip()
        self._pending_approval_id = request_id or None
        if request_id:
            self._approval_prompt_ts[request_id] = datetime.now(timezone.utc)
        lbl = getattr(self, "lbl_approval", None)
        btn_approve = getattr(self, "btn_approve", None)
        btn_deny = getattr(self, "btn_deny", None)
        if lbl is not None and isValid(lbl):
            lbl.setText(
                f"Approve {tool} [{scope}]?\n"
                + json.dumps(args, ensure_ascii=False, indent=2)[:400]
            )
        if btn_approve is not None and isValid(btn_approve):
            btn_approve.setEnabled(bool(self._pending_approval_id))
        if btn_deny is not None and isValid(btn_deny):
            btn_deny.setEnabled(bool(self._pending_approval_id))
        if run_id:
            self._transition_run_state(run_id, RUN_STATE_WAITING_APPROVAL, REASON_APPROVAL_PROMPT)
            self._effect_add(run_id, {
                "kind": "approval_prompt",
                "request_id": request_id,
                "tool": tool,
                "scope": scope,
                "args": args,
            })
        self.append_trace(f"[LOOP] approval requested: {tool} [{scope}]")

    def _send_approval(self, allow: bool) -> None:
        if not self._pending_approval_id:
            return
        req_id = self._pending_approval_id
        self.sig_runtime_command.emit(
            CMD_APPROVAL_RESPONSE,
            {"request_id": req_id, "allow": bool(allow)},
        )
        self.append_trace(
            "[LOOP] approval " + ("granted" if allow else "denied") + ": " + req_id
        )
        self._clear_approval()

    def _clear_approval(self) -> None:
        self._pending_approval_id = None
        lbl = getattr(self, "lbl_approval", None)
        btn_approve = getattr(self, "btn_approve", None)
        btn_deny = getattr(self, "btn_deny", None)
        if lbl is not None and isValid(lbl):
            lbl.setText("No pending approval (use inline timeline buttons when available)")
        if btn_approve is not None and isValid(btn_approve):
            btn_approve.setEnabled(False)
        if btn_deny is not None and isValid(btn_deny):
            btn_deny.setEnabled(False)

    def _set_run_summary(self, result: dict[str, Any] | None) -> None:
        dim = f"color: {_s.FG_DIM}; font-size: 9px;"
        if not isinstance(result, dict):
            self.lbl_run_status.setText("—")
            self.lbl_run_cycles.setText("cycles: —")
            self.lbl_run_tools.setText("tools: —")
            self.lbl_run_wall.setText("wall: —")
            for lbl in (self.lbl_run_status, self.lbl_run_cycles,
                        self.lbl_run_tools, self.lbl_run_wall):
                lbl.setStyleSheet(dim)
            return
        success  = bool(result.get("success"))
        wall     = str(result.get("wall_hit") or "—")
        cycles   = result.get("cycles_used")
        tools    = result.get("tool_calls_used")
        ok_color = _s.FG_ACCENT if success else _s.FG_ERROR
        self.lbl_run_status.setText("OK" if success else "STOPPED")
        self.lbl_run_status.setStyleSheet(
            f"color: {ok_color}; font-size: 9px; font-weight: bold;"
        )
        self.lbl_run_cycles.setText("cycles: " + (str(cycles) if cycles is not None else "—"))
        self.lbl_run_tools.setText("tools: " + (str(tools) if tools is not None else "—"))
        self.lbl_run_wall.setText("wall: " + wall)

    def _toggle_load(self) -> None:
        if self._is_model_loaded:
            self.sig_unload.emit()
        else:
            self.sig_load.emit()

    def _update_load_button_text(self) -> None:
        self.btn_load.setText("UNLOAD MODEL" if self._is_model_loaded else "LOAD MODEL")

    def _refresh_status_ui(self) -> None:
        if not isValid(self):
            return
        btn_run = getattr(self, "btn_run", None)
        btn_stop = getattr(self, "btn_stop", None)
        btn_load = getattr(self, "btn_load", None)
        if btn_run is None or not isValid(btn_run):
            return
        if btn_stop is None or not isValid(btn_stop):
            return
        if btn_load is None or not isValid(btn_load):
            return
        busy = self._last_status in (
            SystemStatus.LOADING, SystemStatus.RUNNING, SystemStatus.UNLOADING
        )
        btn_run.setEnabled(not busy and self._is_model_loaded)
        btn_stop.setEnabled(bool(self._run_active))
        btn_load.setEnabled(not busy)
        self._update_load_button_text()
