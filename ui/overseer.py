from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent, QFont, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QFrame,
    QLabel,
    QHeaderView,
    QSplitter,
)

from core.overseer_db import OverseerDB
from core.style import (
    ACCENT_GOLD, FG_DIM, FG_TEXT, FG_ERROR, FG_WARN, FG_ACCENT,
    OVERSEER_BG, OVERSEER_FG, OVERSEER_DIM, OVERSEER_BORDER, BG_INPUT,
)
from monokernel.guard import MonoGuard
from ui.bridge import UIBridge

# Severity colors
_SEV_COLORS = {
    "ERROR": FG_ERROR,
    "WARNING": FG_WARN,
    "INFO": OVERSEER_FG,
    "DEBUG": FG_DIM,
    "STATUS": ACCENT_GOLD,
    "FINISHED": FG_ACCENT,
}

_FILTER_STYLE_ON = """
    QPushButton {{
        background: {bg};
        border: 1px solid {color};
        color: {color};
        padding: 4px 8px; font-size: 9px; font-weight: bold;
        border-radius: 2px;
    }}
    QPushButton:hover {{ background: #1a1a1a; }}
"""

_FILTER_STYLE_OFF = f"""
    QPushButton {{
        background: transparent;
        border: 1px solid #222;
        color: #333;
        padding: 4px 8px; font-size: 9px; font-weight: bold;
        border-radius: 2px;
    }}
    QPushButton:hover {{ color: {FG_DIM}; border: 1px solid #333; }}
"""

_RECIPE_PRESETS = {
    "ALL": {"ERROR", "WARNING", "INFO", "DEBUG", "STATUS", "FINISHED"},
    "ERRORS ONLY": {"ERROR"},
    "KERNEL": {"ERROR", "WARNING", "STATUS", "FINISHED"},
    "PERFORMANCE": {"INFO", "FINISHED"},
}


class _SeverityFilter(QPushButton):
    """Toggle button for a log severity level."""

    def __init__(self, label: str, color: str):
        super().__init__(label)
        self._label = label
        self._color = color
        self._active = True
        self.setCheckable(True)
        self.setChecked(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(24)
        self.clicked.connect(self._toggle)
        self._apply_style()

    def _toggle(self):
        self._active = self.isChecked()
        self._apply_style()

    def _apply_style(self):
        if self._active:
            self.setStyleSheet(
                _FILTER_STYLE_ON.format(bg=OVERSEER_BG, color=self._color)
            )
        else:
            self.setStyleSheet(_FILTER_STYLE_OFF)

    def is_active(self) -> bool:
        return self._active

    def set_active(self, val: bool):
        self._active = val
        self.setChecked(val)
        self._apply_style()


class ActiveTasksPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel("ACTIVE TASKS")
        lbl.setStyleSheet(
            f"color: {OVERSEER_DIM}; font-size: 9px; font-weight: bold; "
            f"letter-spacing: 2px; background: transparent;"
        )
        layout.addWidget(lbl)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["TASK", "ENGINE", "STATUS"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: {OVERSEER_BG};
                color: {OVERSEER_FG};
                border: 1px solid {OVERSEER_BORDER};
                gridline-color: {OVERSEER_BORDER};
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }}
            QTableWidget::item {{
                padding: 4px;
                border-bottom: 1px solid {OVERSEER_BORDER};
            }}
            QHeaderView::section {{
                background: {OVERSEER_BG};
                color: {OVERSEER_DIM};
                border: none;
                border-bottom: 1px solid {OVERSEER_BORDER};
                font-size: 9px;
                font-weight: bold;
                padding: 4px;
            }}
        """)
        layout.addWidget(self.table)

    def set_tasks(self, rows: list[tuple[str, str, str]]) -> None:
        self.table.setRowCount(len(rows))
        for idx, (task_id, engine_key, status) in enumerate(rows):
            self.table.setItem(idx, 0, QTableWidgetItem(task_id))
            self.table.setItem(idx, 1, QTableWidgetItem(engine_key))
            item = QTableWidgetItem(status)
            color = _SEV_COLORS.get(status.upper(), OVERSEER_FG)
            item.setForeground(QColor(color))
            self.table.setItem(idx, 2, item)


class OverseerWindow(QMainWindow):
    def __init__(self, guard: MonoGuard, ui_bridge: UIBridge):
        super().__init__()
        self.guard = guard
        self.ui_bridge = ui_bridge
        self.db = OverseerDB()
        self._paused = False
        self._last_task_state: dict[str, tuple[str, str]] = {}
        self._severity_filters: dict[str, _SeverityFilter] = {}

        self.setWindowTitle("OVERSEER")
        self.resize(1000, 560)
        self.setStyleSheet(f"background: {OVERSEER_BG};")

        main = QWidget()
        self.setCentralWidget(main)
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # --- Top bar: title + recipe buttons ---
        top_bar = QHBoxLayout()
        lbl_title = QLabel("⬡ OVERSEER")
        lbl_title.setStyleSheet(
            f"color: {OVERSEER_FG}; font-size: 12px; font-weight: bold; "
            f"letter-spacing: 2px; background: transparent;"
        )
        top_bar.addWidget(lbl_title)
        top_bar.addStretch()

        # Recipe presets
        for recipe_name in _RECIPE_PRESETS:
            btn = QPushButton(recipe_name)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(22)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    border: 1px solid {OVERSEER_BORDER};
                    color: {OVERSEER_DIM};
                    padding: 2px 8px; font-size: 8px; font-weight: bold;
                    border-radius: 2px;
                }}
                QPushButton:hover {{
                    border: 1px solid {OVERSEER_FG};
                    color: {OVERSEER_FG};
                }}
            """)
            btn.clicked.connect(lambda _=False, r=recipe_name: self._apply_recipe(r))
            top_bar.addWidget(btn)

        main_layout.addLayout(top_bar)

        # --- Severity filter row ---
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        for sev, color in _SEV_COLORS.items():
            f = _SeverityFilter(sev, color)
            self._severity_filters[sev] = f
            filter_row.addWidget(f)
        filter_row.addStretch()
        main_layout.addLayout(filter_row)

        # --- Separator ---
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {OVERSEER_BORDER};")
        main_layout.addWidget(sep)

        # --- Content: tasks panel + log display ---
        content_split = QSplitter(Qt.Horizontal)
        content_split.setStyleSheet(f"""
            QSplitter::handle {{ background: {OVERSEER_BORDER}; width: 1px; }}
        """)
        content_split.setChildrenCollapsible(False)

        self.panel = ActiveTasksPanel()
        content_split.addWidget(self.panel)

        # Log display — command prompt style
        log_wrap = QWidget()
        log_layout = QVBoxLayout(log_wrap)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(4)

        lbl_log = QLabel("EVENT LOG")
        lbl_log.setStyleSheet(
            f"color: {OVERSEER_DIM}; font-size: 9px; font-weight: bold; "
            f"letter-spacing: 2px; background: transparent;"
        )
        log_layout.addWidget(lbl_log)

        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 10))
        self.log_display.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {OVERSEER_BG};
                color: {OVERSEER_FG};
                border: 1px solid {OVERSEER_BORDER};
                selection-background-color: #1a3a1a;
            }}
            QPlainTextEdit::viewport {{
                background: {OVERSEER_BG};
            }}
        """)
        log_layout.addWidget(self.log_display)
        content_split.addWidget(log_wrap)

        content_split.setStretchFactor(0, 1)
        content_split.setStretchFactor(1, 3)
        content_split.setSizes([250, 700])
        main_layout.addWidget(content_split, 1)

        # --- Bottom controls ---
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        ctrl_style = f"""
            QCheckBox {{
                color: {OVERSEER_DIM}; font-size: 9px; font-weight: bold;
                spacing: 4px;
            }}
            QCheckBox::indicator {{
                width: 10px; height: 10px;
                border: 1px solid {OVERSEER_DIM};
                background: {OVERSEER_BG};
                border-radius: 2px;
            }}
            QCheckBox::indicator:checked {{
                background: {OVERSEER_FG};
                border: 1px solid {OVERSEER_FG};
            }}
        """

        self.chk_pause = QCheckBox("PAUSE")
        self.chk_pause.setStyleSheet(ctrl_style)
        self.chk_pause.toggled.connect(self._on_pause_toggled)

        self.btn_clear = QPushButton("CLEAR")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.setFixedHeight(22)
        self.btn_clear.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {OVERSEER_BORDER};
                color: {OVERSEER_DIM};
                padding: 2px 10px; font-size: 9px; font-weight: bold;
                border-radius: 2px;
            }}
            QPushButton:hover {{ border: 1px solid {FG_ERROR}; color: {FG_ERROR}; }}
        """)
        self.btn_clear.clicked.connect(self.log_display.clear)

        self.chk_viz = QCheckBox("VIZTRACER")
        self.chk_viz.setStyleSheet(ctrl_style)
        self.chk_viz.toggled.connect(self.ui_bridge.sig_overseer_viz_toggle.emit)

        controls_layout.addWidget(self.chk_pause)
        controls_layout.addWidget(self.btn_clear)
        controls_layout.addWidget(self.chk_viz)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # --- Signal connections ---
        self.guard.sig_trace.connect(lambda _engine_key, msg: self._on_trace(msg))
        self.guard.sig_status.connect(self._on_status)
        self.guard.sig_finished.connect(self._on_finished)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(300)
        self._poll_timer.timeout.connect(self._refresh_active_tasks)
        self._poll_timer.start()

    # ---- Filtering ----

    def _is_severity_visible(self, severity: str) -> bool:
        f = self._severity_filters.get(severity.upper())
        return f.is_active() if f else True

    def _apply_recipe(self, recipe_name: str):
        active = _RECIPE_PRESETS.get(recipe_name, set())
        for sev, filt in self._severity_filters.items():
            filt.set_active(sev in active)

    # ---- Log helpers ----

    def _append_line(self, severity: str, text: str) -> None:
        if self._paused:
            return
        if not self._is_severity_visible(severity):
            return
        color = _SEV_COLORS.get(severity.upper(), OVERSEER_FG)
        self.log_display.appendHtml(
            f'<span style="color:{OVERSEER_DIM}">[{self._now_label()}]</span> '
            f'<span style="color:{color}">[{severity}]</span> '
            f'<span style="color:{OVERSEER_FG}">{text}</span>'
        )

    def _now_label(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _on_pause_toggled(self, checked: bool) -> None:
        self._paused = checked

    # ---- Signal handlers ----

    def _on_trace(self, msg: str) -> None:
        self.db.log_event("guard", "trace", {"message": msg})
        sev = "INFO"
        lowered = msg.lower()
        if "error" in lowered:
            sev = "ERROR"
        elif "warn" in lowered:
            sev = "WARNING"
        self._append_line(sev, msg)

    def _on_status(self, engine_key: str, status) -> None:
        status_val = status.value if hasattr(status, "value") else str(status)
        self.db.log_event(engine_key, "status", {"status": status_val})
        self._append_line("STATUS", f"{engine_key} → {status_val}")

    def _on_finished(self, engine_key: str, task_id: str) -> None:
        self.db.log_task(str(task_id), engine_key, "DONE")
        self.db.log_event(engine_key, "finished", {"task_id": str(task_id)})
        self._append_line("FINISHED", f"{engine_key} task={task_id}")

    def _refresh_active_tasks(self) -> None:
        rows = []
        for engine_key, task in self.guard.active_tasks.items():
            if task is None:
                if engine_key in self._last_task_state:
                    prev = self._last_task_state.pop(engine_key)
                    self.db.log_task(prev[0], engine_key, "CLEARED")
                continue
            status_val = task.status.value if hasattr(task.status, "value") else str(task.status)
            current = (str(task.id), status_val)
            if self._last_task_state.get(engine_key) != current:
                self._last_task_state[engine_key] = current
                self.db.log_task(str(task.id), engine_key, status_val)
            rows.append((str(task.id), engine_key, status_val))
        self.panel.set_tasks(rows)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._poll_timer.stop()
        if getattr(self.guard, "_viztracer", None) is not None:
            self.guard.enable_viztracer(False)
        self.db.close()
        super().closeEvent(event)
