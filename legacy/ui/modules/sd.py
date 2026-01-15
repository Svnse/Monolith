from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QFrame,
)
from PySide6.QtCore import Qt, QTimer

from core.style import BG_INPUT, BORDER_DARK, FG_DIM, FG_TEXT, FG_ACCENT
from ui.components.atoms import SkeetGroupBox, SkeetButton


class SDModule(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        grp = SkeetGroupBox("STABLE DIFFUSION")
        inner = QVBoxLayout()
        inner.setSpacing(12)

        lbl_prompt = QLabel("Prompt")
        lbl_prompt.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")

        self.inp_prompt = QLineEdit()
        self.inp_prompt.setPlaceholderText("Describe an image to generate...")
        self.inp_prompt.setStyleSheet(f"""
            QLineEdit {{
                background: {BG_INPUT};
                color: {FG_TEXT};
                border: 1px solid {BORDER_DARK};
                padding: 6px;
            }}
        """)

        self.btn_generate = SkeetButton("GENERATE", accent=True)
        self.btn_generate.clicked.connect(self._start_generate)

        preview_frame = QFrame()
        preview_frame.setFixedHeight(220)
        preview_frame.setStyleSheet(f"background: {BG_INPUT}; border: 1px solid {BORDER_DARK};")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_preview = QLabel("PREVIEW")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet(f"color: {FG_DIM}; font-size: 12px;")
        preview_layout.addWidget(self.lbl_preview)

        status_row = QHBoxLayout()
        lbl_status_title = QLabel("Status")
        lbl_status_title.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")
        self.lbl_status = QLabel("IDLE")
        self.lbl_status.setStyleSheet(f"color: {FG_TEXT}; font-size: 10px; font-weight: bold;")
        status_row.addWidget(lbl_status_title)
        status_row.addStretch()
        status_row.addWidget(self.lbl_status)

        inner.addWidget(lbl_prompt)
        inner.addWidget(self.inp_prompt)
        inner.addWidget(self.btn_generate)
        inner.addWidget(preview_frame)
        inner.addLayout(status_row)
        inner.addStretch()

        grp.add_layout(inner)
        layout.addWidget(grp)

        self._done_timer = QTimer(self)
        self._done_timer.setSingleShot(True)
        self._done_timer.timeout.connect(self._finish_generate)

    def _set_status(self, status, color):
        self.lbl_status.setText(status)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")

    def _start_generate(self):
        self.btn_generate.setEnabled(False)
        self._set_status("LOADING", FG_ACCENT)
        self.lbl_preview.setText("RENDERING...")
        self._done_timer.start(900)

    def _finish_generate(self):
        self._set_status("DONE", FG_TEXT)
        self.lbl_preview.setText("IMAGE READY")
        self.btn_generate.setEnabled(True)
