from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QFont
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
)

from core.overseer_db import OverseerDB
from monokernel.guard import MonoGuard
from ui.bridge import UIBridge


class ActiveTasksPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["task_id", "engine", "status"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def set_tasks(self, rows: list[tuple[str, str, str]]) -> None:
        self.table.setRowCount(len(rows))
        for idx, (task_id, engine_key, status) in enumerate(rows):
            self.table.setItem(idx, 0, QTableWidgetItem(task_id))
            self.table.setItem(idx, 1, QTableWidgetItem(engine_key))
            self.table.setItem(idx, 2, QTableWidgetItem(status))


class OverseerWindow(QMainWindow):
    def __init__(self, guard: MonoGuard, ui_bridge: UIBridge):
        super().__init__()
        self.guard = guard
        self.ui_bridge = ui_bridge
        self.db = OverseerDB()
        self._paused = False
        self._last_task_state: dict[str, tuple[str, str]] = {}

        self.setWindowTitle("OVERSEER")
        self.resize(980, 520)

        main = QWidget()
        self.setCentralWidget(main)
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        content_layout = QHBoxLayout()
        self.panel = ActiveTasksPanel()
        content_layout.addWidget(self.panel, 1)

        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Courier New", 10))
        content_layout.addWidget(self.log_display, 2)
        main_layout.addLayout(content_layout, 1)

        controls_layout = QHBoxLayout()
        self.chk_pause = QCheckBox("Pause")
        self.chk_pause.toggled.connect(self._on_pause_toggled)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.log_display.clear)

        self.chk_viz = QCheckBox("VizTracer")
        self.chk_viz.toggled.connect(self.ui_bridge.sig_overseer_viz_toggle.emit)

        controls_layout.addWidget(self.chk_pause)
        controls_layout.addWidget(self.btn_clear)
        controls_layout.addWidget(self.chk_viz)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        self.guard.sig_trace.connect(self._on_trace)
        self.guard.sig_status.connect(self._on_status)
        self.guard.sig_finished.connect(self._on_finished)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(300)
        self._poll_timer.timeout.connect(self._refresh_active_tasks)
        self._poll_timer.start()

    def _append_line(self, text: str) -> None:
        if not self._paused:
            self.log_display.appendPlainText(text)

    def _now_label(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _on_pause_toggled(self, checked: bool) -> None:
        self._paused = checked

    def _on_trace(self, msg: str) -> None:
        self.db.log_event("guard", "trace", {"message": msg})
        self._append_line(f"[{self._now_label()}] {msg}")

    def _on_status(self, engine_key: str, status) -> None:
        status_val = status.value if hasattr(status, "value") else str(status)
        self.db.log_event(engine_key, "status", {"status": status_val})
        self._append_line(f"[{self._now_label()}] STATUS {engine_key} -> {status_val}")

    def _on_finished(self, engine_key: str, task_id: str) -> None:
        self.db.log_task(str(task_id), engine_key, "DONE")
        self.db.log_event(engine_key, "finished", {"task_id": str(task_id)})
        self._append_line(f"[{self._now_label()}] FINISHED {engine_key} task={task_id}")

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
