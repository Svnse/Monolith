import os
import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QFileDialog, QPlainTextEdit
)
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Signal, QObject

from core.style import BG_INPUT, FG_DIM
from ui.components.atoms import SkeetGroupBox, SkeetButton, SkeetSlider
from ui.components.complex import ModeSelector

# HELPER: Output Redirector
class EmittingStream(QObject):
    textWritten = Signal(str)
    def write(self, text):
        self.textWritten.emit(str(text))
    def flush(self):
        pass

class PageSettings(QWidget):
    sig_load = Signal()
    sig_unload = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(20)

        # 1. Mode Selector
        self.selector = ModeSelector()
        self.selector.modeChanged.connect(self.set_mode)
        self.layout.addWidget(self.selector)

        # 2. Content Stack
        self.content_area = QWidget()
        self.content_layout = QHBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0,0,0,0)
        
        # --- OPERATOR VIEW ---
        self.view_op = QWidget()
        op_layout = QHBoxLayout(self.view_op)
        op_layout.setContentsMargins(0,0,0,0)
        
        col1 = QVBoxLayout()
        grp_load = SkeetGroupBox("MODEL LOADER")
        self.path_display = QLineEdit()
        self.path_display.setReadOnly(True)
        self.path_display.setPlaceholderText("No GGUF Selected")
        self.path_display.setStyleSheet(f"background: {BG_INPUT}; color: #555; border: 1px solid #333; padding: 5px;")
        btn_browse = SkeetButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self.pick_file)
        row_file = QHBoxLayout()
        row_file.addWidget(self.path_display)
        row_file.addWidget(btn_browse)
        self.btn_load = SkeetButton("LOAD MODEL")
        self.btn_load.clicked.connect(self.toggle_load)
        grp_load.add_layout(row_file)
        grp_load.add_widget(self.btn_load)
        col1.addWidget(grp_load)
        col1.addStretch()
        
        col2 = QVBoxLayout()
        grp_ai = SkeetGroupBox("AI CONFIGURATION")
        self.s_temp = SkeetSlider("Temperature", 0.1, 2.0, state.temp)
        self.s_temp.valueChanged.connect(lambda v: setattr(state, 'temp', v))
        self.s_top = SkeetSlider("Top-P", 0.1, 1.0, state.top_p)
        self.s_top.valueChanged.connect(lambda v: setattr(state, 'top_p', v))
        self.s_tok = SkeetSlider("Max Tokens", 512, 8192, state.max_tokens, is_int=True)
        self.s_tok.valueChanged.connect(lambda v: setattr(state, 'max_tokens', int(v)))
        self.s_ctx = SkeetSlider("Context Limit", 1024, 16384, state.ctx_limit, is_int=True)
        self.s_ctx.valueChanged.connect(lambda v: setattr(state, "ctx_limit", int(v)))
        lbl_sys = QLabel("System Prompt")
        lbl_sys.setStyleSheet(f"color: {FG_DIM}; font-size: 11px; margin-top: 5px;")
        self.inp_sys = QLineEdit(state.system_prompt)
        self.inp_sys.setStyleSheet(f"background: {BG_INPUT}; color: #aaa; border: 1px solid #333; padding: 5px;")
        self.inp_sys.textChanged.connect(lambda: setattr(state, 'system_prompt', self.inp_sys.text()))
        grp_ai.add_widget(self.s_temp)
        grp_ai.add_widget(self.s_top)
        grp_ai.add_widget(self.s_tok)
        grp_ai.add_widget(self.s_ctx)
        grp_ai.add_widget(lbl_sys)
        grp_ai.add_widget(self.inp_sys)
        col2.addWidget(grp_ai)
        col2.addStretch()
        
        op_layout.addLayout(col1)
        op_layout.addLayout(col2)

        # --- OVERSEER VIEW ---
        self.view_ov = QWidget()
        ov_layout = QVBoxLayout(self.view_ov)
        ov_layout.setContentsMargins(0,0,0,0)
        
        grp_sys = SkeetGroupBox("SYSTEM CONSOLE IO")
        self.console_out = QPlainTextEdit()
        self.console_out.setReadOnly(True)
        self.console_out.setStyleSheet(f"""
            background: #080808; color: #555; 
            border: 1px solid #333; font-family: 'Consolas'; font-size: 10px;
        """)
        grp_sys.add_widget(self.console_out)
        ov_layout.addWidget(grp_sys)
        self.view_ov.setVisible(False)

        self.content_layout.addWidget(self.view_op)
        self.content_layout.addWidget(self.view_ov)
        self.layout.addWidget(self.content_area)
        
        # Init Stream Redirect
        self.sys_out = EmittingStream()
        self.sys_out.textWritten.connect(self.append_sys_log)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._capture_enabled = False
        if os.environ.get("MONOLITH_CAPTURE_STDIO") == "1":
            sys.stdout = self.sys_out
            sys.stderr = self.sys_out
            self._capture_enabled = True

    def set_mode(self, mode):
        is_op = (mode == "OPERATOR")
        self.view_op.setVisible(is_op)
        self.view_ov.setVisible(not is_op)

    def append_sys_log(self, text):
        self.console_out.moveCursor(QTextCursor.End)
        self.console_out.insertPlainText(text)

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF", "", "GGUF (*.gguf)")
        if path:
            self.state.gguf_path = path
            self.path_display.setText(os.path.basename(path))

    def toggle_load(self):
        if self.state.model_loaded: self.sig_unload.emit()
        else: self.sig_load.emit()

    def set_loading_state(self, is_loading):
        self.btn_load.setEnabled(not is_loading)
        if is_loading: self.btn_load.setText("PROCESSING...")

        else: self.btn_load.setText("UNLOAD MODEL" if self.state.model_loaded else "LOAD MODEL")

    def closeEvent(self, event):
        if self._capture_enabled:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        super().closeEvent(event)
