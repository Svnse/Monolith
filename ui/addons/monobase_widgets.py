"""Small UI pieces for the MonoBase companion pane."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

import core.style as _s
from core.acatalepsy import monobase_status as _status_model


__all__ = ("AcatalepsyMapDot", "AcatalepsyMapWidget", "MonoBaseStatusStrip")


@dataclass(frozen=True)
class AcatalepsyMapDot:
    """One clickable point in the Acatalepsy map."""

    kind: str
    item_id: int
    label: str
    tone: str = "idle"
    tooltip: str = ""


class AcatalepsyMapWidget(QFrame):
    """Clickable overview of the MonoBase flow: log -> candidates -> ACUs."""

    dotActivated = Signal(str, int)

    _STAGES = (
        ("log", "LOG"),
        ("candidate", "CANDIDATES"),
        ("run", "RUNS"),
        ("acu", "ACUS"),
    )
    _MAX_DOTS_PER_STAGE = 12

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("acatalepsy_map")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 7, 8, 7)
        root.setSpacing(7)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self._title = QLabel("ACATALEPSY MAP")
        self._title.setObjectName("map_title")
        self._summary = QLabel("idle")
        self._summary.setObjectName("map_summary")
        self._summary.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._title)
        header.addWidget(self._summary, 1)
        root.addLayout(header)

        self._stage_labels: dict[str, QLabel] = {}
        self._dot_grids: dict[str, QGridLayout] = {}

        stage_row = QHBoxLayout()
        stage_row.setContentsMargins(0, 0, 0, 0)
        stage_row.setSpacing(6)
        for key, title in self._STAGES:
            stage = QFrame()
            stage.setObjectName("map_stage")
            stage.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            stage_layout = QVBoxLayout(stage)
            stage_layout.setContentsMargins(5, 5, 5, 5)
            stage_layout.setSpacing(4)

            label = QLabel(f"{title} 0")
            label.setObjectName("map_stage_label")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stage_layout.addWidget(label)
            self._stage_labels[key] = label

            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(3)
            grid.setVerticalSpacing(3)
            stage_layout.addLayout(grid)
            self._dot_grids[key] = grid
            stage_row.addWidget(stage, 1)
        root.addLayout(stage_row)
        self._refresh_style()

    def apply_map(
        self,
        groups: dict[str, list[AcatalepsyMapDot]],
        *,
        totals: dict[str, int] | None = None,
        summary: str = "",
    ) -> None:
        totals = totals or {}
        for key, title in self._STAGES:
            dots = list(groups.get(key, ()))[: self._MAX_DOTS_PER_STAGE]
            total = int(totals.get(key, len(groups.get(key, ()))))
            self._stage_labels[key].setText(f"{title} {total}")
            grid = self._dot_grids[key]
            self._clear_grid(grid)

            if not dots:
                empty = QLabel("-")
                empty.setObjectName("map_empty")
                empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(empty, 0, 0, 1, 6)
                continue

            for idx, dot in enumerate(dots):
                btn = QPushButton("")
                btn.setObjectName("map_dot")
                btn.setFixedSize(11, 11)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setToolTip(dot.tooltip or dot.label)
                btn.setProperty("tone", dot.tone)
                btn.setStyleSheet(self._dot_style(dot.tone))
                btn.clicked.connect(
                    lambda _checked=False, kind=dot.kind, item_id=dot.item_id:
                    self.dotActivated.emit(kind, int(item_id))
                )
                grid.addWidget(btn, idx // 6, idx % 6)

        self._summary.setText(summary or "idle")

    def _clear_grid(self, grid: QGridLayout) -> None:
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_style(self) -> None:
        self.setStyleSheet(
            f"""
            QFrame#acatalepsy_map {{
                background: {_s.BG_PANEL};
                border: 1px solid {_s.BORDER_SUBTLE};
                border-radius: 4px;
            }}
            QFrame#map_stage {{
                background: {_s.BG_INPUT};
                border: 1px solid {_s.BORDER_SUBTLE};
                border-radius: 3px;
            }}
            QLabel#map_title {{
                color: {_s.FG_TEXT};
                font-size: 10px;
                font-family: Consolas;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
            QLabel#map_summary {{
                color: {_s.FG_DIM};
                font-size: 9px;
                font-family: Consolas;
                background: transparent;
                border: none;
            }}
            QLabel#map_stage_label {{
                color: {_s.FG_DIM};
                font-size: 8px;
                font-family: Consolas;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
            QLabel#map_empty {{
                color: {_s.BORDER_LIGHT};
                font-size: 9px;
                font-family: Consolas;
                background: transparent;
                border: none;
            }}
            """
        )

    def _dot_style(self, tone: str) -> str:
        color = {
            "active": _s.ACCENT_PRIMARY,
            "warn": _s.FG_WARN,
            "ok": _s.FG_INFO,
            "error": _s.FG_ERROR,
        }.get(tone, _s.FG_DIM)
        return (
            f"QPushButton#map_dot {{ background: {color}; border: none;"
            " border-radius: 5px; padding: 0px; }"
            f"QPushButton#map_dot:hover {{ border: 1px solid {_s.FG_TEXT}; }}"
        )


class MonoBaseStatusStrip(QFrame):
    """Persistent, truthful status surface for MonoBase activity."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("monobase_status_strip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 7, 8, 7)
        root.setSpacing(5)

        self._phase = QLabel("Auditor off")
        self._phase.setObjectName("monobase_phase")
        self._phase.setMinimumHeight(24)
        self._phase.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(self._phase)

        self._detail = QLabel("No worker is registered.")
        self._detail.setWordWrap(True)
        self._detail.setObjectName("monobase_detail")
        root.addWidget(self._detail)

        self._stage_line = QLabel("LOG -> LLM -> CANDIDATES -> ACU")
        self._stage_line.setObjectName("monobase_stage_line")
        root.addWidget(self._stage_line)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        self._run = self._metric("Run clock", "--")
        self._llm = self._metric("LLM request", "--")
        self._candidates = self._metric("Candidates", "pending 0")
        self._worker = self._metric("Worker", "off")
        self._acu = self._metric("Last ACU write", "none")
        for idx, label in enumerate(
            (self._run, self._llm, self._candidates, self._worker, self._acu)
        ):
            grid.addWidget(label, idx // 2, idx % 2)
        root.addLayout(grid)
        self._refresh_shell_style()

    def apply_snapshot(self, snapshot: _status_model.MonobaseSnapshot) -> None:
        self._phase.setText(snapshot.phase_title)
        self._phase.setStyleSheet(self._phase_style(snapshot.phase_tone))
        self._detail.setText(snapshot.phase_detail)
        self._stage_line.setText(self._stage_text(snapshot.phase))
        self._run.setText(self._metric_text(
            "Run clock",
            _status_model.format_elapsed(snapshot.run_elapsed_secs),
        ))
        llm_value = _status_model.format_elapsed(snapshot.llm_elapsed_secs)
        if snapshot.phase == "calling_llm":
            llm_value = f"{llm_value} live"
        self._llm.setText(self._metric_text("LLM request", llm_value))
        counts = snapshot.candidate_counts
        self._candidates.setText(self._metric_text(
            "Candidates",
            (
                f"pending {counts.get('pending', 0)}  "
                f"accepted {counts.get('accepted', 0)}  "
                f"rejected {counts.get('rejected', 0)}"
            ),
        ))
        worker = snapshot.worker
        if worker.thread_alive:
            queue = worker.queue_size if worker.queue_size is not None else "?"
            max_events = worker.max_events_per_run if worker.max_events_per_run else "?"
            self._worker.setText(self._metric_text("Worker", f"on  queue {queue}  max {max_events}"))
        elif worker.registered:
            self._worker.setText(self._metric_text("Worker", "paused"))
        else:
            self._worker.setText(self._metric_text("Worker", "off"))

        if snapshot.recent_acu_writes:
            self._acu.setText(
                self._metric_text(
                    "Last ACU write",
                    _status_model.format_recent_acu_write(
                    snapshot.recent_acu_writes[0],
                    now=snapshot.now,
                    ),
                )
            )
        else:
            self._acu.setText(self._metric_text("Last ACU write", "none yet"))

    def show_error(self, message: str) -> None:
        self._phase.setText("Status unavailable")
        self._phase.setStyleSheet(self._phase_style("warn"))
        self._detail.setText(message)

    def _metric(self, label: str, value: str) -> QLabel:
        widget = QLabel(self._metric_text(label, value))
        widget.setObjectName("monobase_metric")
        widget.setWordWrap(True)
        return widget

    def _metric_text(self, label: str, value: str) -> str:
        return f"{label}\n{value}"

    def _stage_text(self, phase: str) -> str:
        if phase in {"auditing_log", "queued", "watching", "needs_audit"}:
            return "[LOG] -> LLM -> CANDIDATES -> ACU"
        if phase == "calling_llm":
            return "LOG -> [LLM] -> CANDIDATES -> ACU"
        if phase in {"updating_candidates", "candidates_pending"}:
            return "LOG -> LLM -> [CANDIDATES] -> ACU"
        if phase in {"caught_up", "off", "paused"}:
            return "LOG -> LLM -> CANDIDATES -> ACU"
        return "LOG -> LLM -> CANDIDATES -> ACU"

    def _refresh_shell_style(self) -> None:
        self.setStyleSheet(
            f"""
            QFrame#monobase_status_strip {{
                background: {_s.BG_INPUT};
                border: 1px solid {_s.BORDER_SUBTLE};
                border-radius: 4px;
            }}
            QLabel#monobase_detail {{
                color: {_s.FG_DIM};
                font-size: 10px;
                font-family: Consolas;
                background: transparent;
                border: none;
            }}
            QLabel#monobase_metric {{
                color: {_s.FG_TEXT};
                font-size: 9px;
                font-family: Consolas;
                background: transparent;
                border: none;
            }}
            QLabel#monobase_stage_line {{
                color: {_s.FG_DIM};
                font-size: 9px;
                font-family: Consolas;
                background: transparent;
                border: none;
                padding: 2px 0px;
            }}
            """
        )

    def _phase_style(self, tone: str) -> str:
        if tone == "active":
            return (
                f"color: {_s.BG_MAIN}; background: {_s.ACCENT_PRIMARY};"
                " border: none; border-radius: 3px; padding: 3px 6px;"
                " font-size: 11px; font-family: Consolas; font-weight: bold;"
            )
        if tone == "warn":
            return (
                f"color: {_s.FG_WARN}; background: transparent;"
                f" border: 1px solid {_s.FG_WARN}; border-radius: 3px; padding: 3px 6px;"
                " font-size: 11px; font-family: Consolas; font-weight: bold;"
            )
        return (
            f"color: {_s.FG_TEXT}; background: transparent;"
            f" border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 3px; padding: 3px 6px;"
            " font-size: 11px; font-family: Consolas; font-weight: bold;"
        )
