from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import core.style as _s


def _section_header(title: str) -> QLabel:
    """Flat section header — small caps label + thin underline. Replaces the
    previous _card() helper that wrapped every section in its own QFrame
    border (the boxes-inside-boxes problem)."""
    lbl = QLabel(title)
    lbl.setStyleSheet(
        f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; "
        f"font-weight: bold; letter-spacing: 1px; "
        f"border-bottom: 1px solid {_s.BORDER_SUBTLE}; padding-bottom: 3px;"
    )
    return lbl


class _MiniWaveform(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._waveform = None
        self.setStyleSheet(
            f"QFrame {{ background: {_s.BG_SURFACE_1}; border: 1px solid {_s.BORDER_SUBTLE}; "
            f"border-radius: 6px; }}"
        )

    def set_waveform(self, waveform) -> None:
        self._waveform = waveform
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._waveform is None or len(self._waveform) == 0:
            painter.setPen(QPen(QColor(_s.FG_DIM)))
            painter.drawText(self.rect(), Qt.AlignCenter, "NO WAVEFORM")
            return
        data = self._waveform
        width = self.width()
        height = self.height()
        mid_y = height / 2.0
        painter.setPen(QPen(QColor(_s.FG_ACCENT), 1))
        count = len(data)
        for i in range(count - 1):
            x1 = int((i / count) * width)
            x2 = int(((i + 1) / count) * width)
            y1 = int(mid_y - (float(data[i]) * mid_y * 0.9))
            y2 = int(mid_y - (float(data[i + 1]) * mid_y * 0.9))
            painter.drawLine(x1, y1, x2, y2)


class AudioPanel(QWidget):
    """Compact AUDIO summary shown in the companion pane.

    Flat layout — no boxes-inside-boxes. Section headers are small-caps
    labels with a thin underline (matches Monolith's tabular feel). The
    waveform widget IS a frame because it needs its own bg; everything
    else flows directly inside the scroll body.
    """

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self._module = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(scroll, 1)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(2, 2, 2, 2)
        body_layout.setSpacing(8)
        scroll.setWidget(body)

        # ── STATUS ────────────────────────────────────────────────────────
        body_layout.addWidget(_section_header("STATUS"))
        self.lbl_status = QLabel("IDLE")
        self.lbl_status.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 14px; font-weight: bold; padding-top: 2px;"
        )
        body_layout.addWidget(self.lbl_status)
        self.lbl_detail = QLabel("Open the Audio addon to generate audio.")
        self.lbl_detail.setWordWrap(True)
        self.lbl_detail.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px;")
        body_layout.addWidget(self.lbl_detail)

        # ── WAVEFORM ──────────────────────────────────────────────────────
        body_layout.addWidget(_section_header("WAVEFORM"))
        self.waveform = _MiniWaveform()
        body_layout.addWidget(self.waveform)
        self.lbl_meta = QLabel("Duration -- | Sample rate --")
        self.lbl_meta.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas;")
        body_layout.addWidget(self.lbl_meta)

        # ── PROMPT ────────────────────────────────────────────────────────
        body_layout.addWidget(_section_header("PROMPT"))
        self.lbl_prompt = QLabel("--")
        self.lbl_prompt.setWordWrap(True)
        self.lbl_prompt.setStyleSheet(f"color: {_s.FG_TEXT}; font-size: 11px;")
        body_layout.addWidget(self.lbl_prompt)

        # ── ACTIONS ───────────────────────────────────────────────────────
        body_layout.addWidget(_section_header("ACTIONS"))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_generate = QPushButton("GENERATE")
        self.btn_play = QPushButton("PLAY")
        self.btn_save = QPushButton("SAVE")
        for btn in (self.btn_generate, self.btn_play, self.btn_save):
            btn.setCursor(Qt.PointingHandCursor)
            row.addWidget(btn)
        body_layout.addLayout(row)

        body_layout.addStretch()

        self.btn_generate.clicked.connect(self._generate)
        self.btn_play.clicked.connect(self._play)
        self.btn_save.clicked.connect(self._save)

        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    def bind_module(self, module) -> None:
        self._module = module
        self.refresh()

    def refresh(self) -> None:
        module = self._module
        world_state = getattr(self._state, "world_state", None)
        snapshot = world_state.snapshot() if world_state is not None else {}
        engine_state = (snapshot.get("engines") or {}).get("audio", {})
        task_state = (snapshot.get("tasks") or {}).get("audio", {})
        active = task_state.get("active") if isinstance(task_state, dict) else None
        active_cmd = str(active.get("command", "")).upper() if isinstance(active, dict) else ""

        status = str(engine_state.get("status") or "IDLE").upper()
        detail = active_cmd or "Open the Audio addon to generate audio."
        prompt = "--"
        meta = "Duration -- | Sample rate --"
        waveform = None
        has_audio = False
        has_module = module is not None
        can_play = False
        can_save = False

        if module is not None:
            status = module.lbl_status.text().strip() or status
            detail = module.current_filepath.name if getattr(module, "current_filepath", None) else detail
            prompt = module.inp_prompt.text().strip() or "--"
            cfg = dict(getattr(module, "config", {}) or {})
            meta = (
                f"Duration {cfg.get('duration', '--')}s | "
                f"Sample rate {cfg.get('sample_rate', '--')} Hz"
            )
            waveform = getattr(module.waveform_widget, "waveform_data", None)
            has_audio = getattr(module, "current_audio", None) is not None
            can_play = bool(has_audio and getattr(module, "current_filepath", None))
            can_save = bool(has_audio and getattr(module, "current_filepath", None))

        self.lbl_status.setText(status)
        self.lbl_detail.setText(detail[:220])
        self.lbl_prompt.setText(prompt[:240] if prompt else "--")
        self.lbl_meta.setText(meta)
        self.waveform.set_waveform(waveform)
        self.btn_generate.setEnabled(has_module)
        self.btn_play.setEnabled(has_module and can_play)
        self.btn_save.setEnabled(has_module and can_save)

    def _generate(self) -> None:
        if self._module is not None and hasattr(self._module, "_start_generate"):
            self._module._start_generate()

    def _play(self) -> None:
        if self._module is not None and hasattr(self._module, "_play_audio"):
            self._module._play_audio()

    def _save(self) -> None:
        if self._module is not None and hasattr(self._module, "_save_audio"):
            self._module._save_audio()
