from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEasingCurve, QDateTime, QPropertyAnimation, QTimer, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QProgressBar, QVBoxLayout, QWidget

import core.style as _s


def _fmt_pct(value) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.0f}%"
    except Exception:
        return "--"


def _fmt_mb(value) -> str:
    if value is None:
        return "--"
    try:
        mb = float(value)
    except Exception:
        return "--"
    if mb >= 1024:
        return f"{mb / 1024:.1f}G"
    return f"{mb:.0f}M"


def _clear_layout(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            while child_layout.count():
                child = child_layout.takeAt(0)
                child_widget = child.widget()
                if child_widget is not None:
                    child_widget.deleteLater()


class VitalsFooter(QFrame):
    sig_unload_requested = Signal(str)

    def __init__(self, state, ui_bridge=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.ui_bridge = ui_bridge
        self._pinned = False
        self._engine_status: dict[str, str] = {}
        self._session_title = "Untitled Chat"
        self.setObjectName("vitals_footer")
        self.setMaximumHeight(28)
        self.setMinimumHeight(28)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 6, 12, 6)
        root.setSpacing(6)

        summary_row = QHBoxLayout()
        summary_row.setContentsMargins(0, 0, 0, 0)
        summary_row.setSpacing(10)

        self._summary = QLabel("")
        self._summary.setTextFormat(Qt.RichText)
        self._summary.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 11px; font-family: Consolas; background: transparent;"
        )
        summary_row.addWidget(self._summary, 1)

        self._session_meta = QLabel("")
        self._session_meta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._session_meta.setStyleSheet(
            f"color: {_s.FG_SECONDARY}; font-size: 10px; font-family: Consolas; background: transparent;"
        )
        summary_row.addWidget(self._session_meta, 0, Qt.AlignRight)

        root.addLayout(summary_row)

        self._details_host = QWidget()
        details_root = QVBoxLayout(self._details_host)
        details_root.setContentsMargins(0, 0, 0, 0)
        details_root.setSpacing(6)

        self._resources = QLabel("")
        self._resources.setWordWrap(True)
        self._resources.setStyleSheet(
            f"color: {_s.FG_SECONDARY}; font-size: 10px; font-family: Consolas; background: transparent;"
        )
        details_root.addWidget(self._resources)

        self._ctx_label = QLabel("Context")
        self._ctx_label.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; background: transparent;"
        )
        details_root.addWidget(self._ctx_label)

        self._ctx_bar = QProgressBar()
        self._ctx_bar.setTextVisible(False)
        self._ctx_bar.setRange(0, 100)
        self._ctx_bar.setFixedHeight(8)
        self._ctx_bar.setStyleSheet(
            f"""
            QProgressBar {{
                background: {_s.BG_INPUT};
                border: 1px solid {_s.BORDER_SUBTLE};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {_s.ACCENT_PRIMARY};
                border-radius: 3px;
            }}
            """
        )
        details_root.addWidget(self._ctx_bar)

        self._engine_rows = QWidget()
        self._engine_rows_layout = QVBoxLayout(self._engine_rows)
        self._engine_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._engine_rows_layout.setSpacing(4)
        details_root.addWidget(self._engine_rows)

        self._details_host.setVisible(False)
        root.addWidget(self._details_host)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        # Theme refresh: per-widget stylesheets f-string theme tokens at
        # construction, which would freeze to the construction-time theme.
        # Subscribe to sig_theme_changed (when ui_bridge is wired) so the
        # outer stylesheets re-evaluate. Inner spans in the summary RichText
        # rebuild every refresh() tick, so they self-update on theme change
        # within ≤1s even without this handler — but the outer labels
        # (resources line, ctx label, summary base color) need the explicit
        # re-apply to follow theme changes.
        self._apply_theme_styles()
        if self.ui_bridge is not None and hasattr(self.ui_bridge, "sig_theme_changed"):
            self.ui_bridge.sig_theme_changed.connect(self._on_theme_changed)
        self.refresh()

    def attach_ui_bridge(self, ui_bridge) -> None:
        """Late-bind ui_bridge for theme refresh, for call sites that build
        VitalsFooter before the bridge is available. Safe to call twice;
        idempotent connection."""
        if self.ui_bridge is ui_bridge:
            return
        self.ui_bridge = ui_bridge
        if ui_bridge is not None and hasattr(ui_bridge, "sig_theme_changed"):
            try:
                ui_bridge.sig_theme_changed.connect(self._on_theme_changed)
            except Exception:
                pass
            self._apply_theme_styles()

    def _on_theme_changed(self, *_args) -> None:
        self._apply_theme_styles()
        self.refresh()

    def _apply_theme_styles(self) -> None:
        """Re-apply the theme-tokened stylesheets so the footer follows
        theme changes instead of freezing to construction-time colors."""
        self._summary.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 11px; font-family: Consolas; background: transparent;"
        )
        self._session_meta.setStyleSheet(
            f"color: {_s.FG_SECONDARY}; font-size: 10px; font-family: Consolas; background: transparent;"
        )
        self._resources.setStyleSheet(
            f"color: {_s.FG_SECONDARY}; font-size: 10px; font-family: Consolas; background: transparent;"
        )
        self._ctx_label.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; background: transparent;"
        )
        self._ctx_bar.setStyleSheet(
            f"""
            QProgressBar {{
                background: {_s.BG_INPUT};
                border: 1px solid {_s.BORDER_SUBTLE};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {_s.ACCENT_PRIMARY};
                border-radius: 3px;
            }}
            """
        )

    def set_session_meta(self, title: str | None) -> None:
        clean = str(title or "").strip()
        self._session_title = clean or "Untitled Chat"
        self.refresh()

    def refresh(self) -> None:
        world_state = getattr(self.state, "world_state", None)
        snapshot = world_state.snapshot() if world_state is not None else {}
        resources = snapshot.get("resources", {})
        engines = snapshot.get("engines", {})
        tasks = snapshot.get("tasks", {})

        active_model = Path(self.state.gguf_path or "").name if getattr(self.state, "gguf_path", None) else ""
        active_model = active_model or str(getattr(self.state, "api_model", "") or "").strip() or "no-model"

        active_status = "READY"
        for engine_key in sorted(engines):
            active_status = str(engines[engine_key].get("status") or self._engine_status.get(engine_key) or "READY")
            if engine_key.startswith("llm"):
                break

        vram_used = resources.get("vram_used_mb")
        vram_free = resources.get("vram_free_mb")
        vram_total = None
        if vram_used is not None and vram_free is not None:
            try:
                vram_total = float(vram_used) + float(vram_free)
            except Exception:
                vram_total = None

        status_color = self._status_color(active_status)
        summary_chunks = [
            f"CPU {_fmt_pct(resources.get('cpu_pct'))}",
            f"RAM {_fmt_mb(resources.get('ram_used_mb'))}/{_fmt_mb(resources.get('ram_total_mb'))}",
            f"VRAM {_fmt_mb(vram_used)}/{_fmt_mb(vram_total if vram_total is not None else vram_free)}",
            f"CTX {int(getattr(self.state, 'ctx_used', 0))}/{int(getattr(self.state, 'ctx_limit', 0) or 0)}",
            active_model,
        ]
        summary_chunks.append(
            f"<span style=\"color:{status_color}\">{active_status}</span>"
        )
        self._summary.setText(" | ".join(summary_chunks))
        self._session_meta.setText(
            f"{self._session_title} | {QDateTime.currentDateTime().toString('dddd | h:mm AP')}"
        )

        self._resources.setText(
            " | ".join(
                [
                    f"CPU {_fmt_pct(resources.get('cpu_pct'))}",
                    f"RAM {_fmt_mb(resources.get('ram_used_mb'))}/{_fmt_mb(resources.get('ram_total_mb'))}",
                    f"VRAM {_fmt_mb(vram_used)}/{_fmt_mb(vram_total if vram_total is not None else vram_free)} free={_fmt_mb(vram_free)}",
                    f"Model {active_model}",
                ]
            )
        )

        ctx_used = int(getattr(self.state, "ctx_used", 0) or 0)
        ctx_limit = int(getattr(self.state, "ctx_limit", 0) or 0)
        self._ctx_label.setText(f"Context {ctx_used}/{ctx_limit}")
        pct = 0
        if ctx_limit > 0:
            pct = max(0, min(100, int((ctx_used / ctx_limit) * 100)))
        self._ctx_bar.setValue(pct)

        _clear_layout(self._engine_rows_layout)
        if not engines:
            empty = QLabel("No engine activity recorded.")
            empty.setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; background: transparent;"
            )
            self._engine_rows_layout.addWidget(empty)
            return

        for engine_key in sorted(engines):
            status = str(engines[engine_key].get("status") or self._engine_status.get(engine_key, "READY"))
            queue_len = int(tasks.get(engine_key, {}).get("queue_len", 0) or 0)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            lbl_name = QLabel(f"{engine_key:<18}")
            lbl_name.setStyleSheet(
                f"color: {_s.FG_TEXT}; font-size: 10px; font-family: Consolas; background: transparent;"
            )
            row_layout.addWidget(lbl_name)

            lbl_status = QLabel(status)
            lbl_status.setStyleSheet(
                f"color: {self._status_color(status)}; font-size: 10px; font-family: Consolas; background: transparent;"
            )
            row_layout.addWidget(lbl_status)

            lbl_queue = QLabel(f"queue={queue_len}")
            lbl_queue.setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; background: transparent;"
            )
            row_layout.addWidget(lbl_queue)
            row_layout.addStretch()

            if self._show_unload(engine_key, status):
                btn = QPushButton("UNLOAD")
                btn.setCursor(Qt.PointingHandCursor)
                btn.setFixedHeight(22)
                btn.setStyleSheet(
                    f"""
                    QPushButton {{
                        background: transparent;
                        color: {_s.ACCENT_PRIMARY};
                        border: 1px solid {_s.BORDER_SUBTLE};
                        border-radius: 4px;
                        padding: 0 8px;
                        font-size: 9px;
                        font-family: Consolas;
                    }}
                    QPushButton:hover {{
                        border-color: {_s.ACCENT_PRIMARY};
                        background: {_s.BG_BUTTON_HOVER};
                    }}
                    """
                )
                btn.clicked.connect(lambda _checked=False, ek=engine_key: self.sig_unload_requested.emit(ek))
                row_layout.addWidget(btn)

            self._engine_rows_layout.addWidget(row)

    def update_engine(self, engine_key: str, status) -> None:
        text = status.value if hasattr(status, "value") else str(status)
        self._engine_status[engine_key] = text
        self.refresh()

    def toggle_expanded(self) -> None:
        self._pinned = not self._pinned
        self._set_expanded(self._pinned)

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._set_expanded(True)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        if not self._pinned:
            self._set_expanded(False)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.toggle_expanded()
            event.accept()
            return
        super().mousePressEvent(event)

    def _set_expanded(self, expanded: bool) -> None:
        target = 124 if expanded else 28
        if expanded:
            self._details_host.setVisible(True)
        anim = QPropertyAnimation(self, b"maximumHeight", self)
        anim.setDuration(150)
        anim.setStartValue(self.maximumHeight())
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        if not expanded:
            anim.finished.connect(lambda: self._details_host.setVisible(False))
        anim.start()
        self._anim = anim

    def _show_unload(self, engine_key: str, status: str) -> bool:
        text = str(status or "").upper()
        if text in {"UNLOADED", "ERROR"}:
            return False
        return engine_key.startswith("llm") or text in {"READY", "RUNNING", "LOADING", "GENERATING"}

    def _status_color(self, status: str) -> str:
        text = str(status or "").upper()
        if text == "ERROR":
            return _s.FG_ERROR
        if text in {"RUNNING", "LOADING", "GENERATING"}:
            return _s.ACCENT_PRIMARY
        return _s.FG_ACCENT
