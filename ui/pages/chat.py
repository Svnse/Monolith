from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel, QFileDialog
)
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Signal, Qt, QTimer

from core.state import SystemStatus
from core.style import BG_INPUT, FG_DIM, FG_ACCENT, FG_TEXT, ACCENT_GOLD
from ui.components.atoms import SkeetGroupBox, SkeetButton, CollapsibleSection, SkeetSlider
from ui.modules.llm_config import load_config, save_config

class PageChat(QWidget):
    sig_generate = Signal(str)
    sig_load = Signal()
    sig_unload = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.config = load_config()
        self.state.gguf_path = self.config.get("gguf_path")
        self.state.ctx_limit = int(self.config.get("ctx_limit", self.state.ctx_limit))
        self._token_buf: list[str] = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(25)
        self._flush_timer.timeout.connect(self._flush_tokens)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # LEFT COLUMN (Chat)
        left_col = QVBoxLayout()
        chat_group = SkeetGroupBox("TERMINAL")
        chat_layout = QVBoxLayout()
        chat_layout.setSpacing(10)

        # Chat Controls
        controls = QHBoxLayout()
        btn_new = SkeetButton("NEW SESSION")
        btn_new.clicked.connect(self.clear_chat)
        btn_clear = SkeetButton("CLEAR LOG")
        btn_clear.clicked.connect(lambda: self.chat.clear())
        
        controls.addStretch()
        controls.addWidget(btn_new)
        controls.addWidget(btn_clear)
        chat_layout.addLayout(controls)

        config_section = CollapsibleSection("⚙ CONFIGURATION")
        config_section.btn_toggle.setStyleSheet(f"""
            QPushButton {{
                color: {FG_TEXT}; background: transparent;
                border: none; text-align: left; font-weight: bold; font-size: 10px;
            }}
            QPushButton:checked {{ color: {FG_TEXT}; }}
            QPushButton:hover {{ color: {FG_TEXT}; }}
        """)
        config_layout = QVBoxLayout()
        config_layout.setSpacing(12)

        config_row = QHBoxLayout()
        config_row.setSpacing(12)

        loader_col = QVBoxLayout()
        grp_load = SkeetGroupBox("MODEL LOADER")
        self.path_display = QLineEdit()
        self.path_display.setReadOnly(True)
        self.path_display.setPlaceholderText("No GGUF Selected")
        self.path_display.setStyleSheet(
            f"background: {BG_INPUT}; color: #555; border: 1px solid #333; padding: 5px;"
        )
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
        loader_col.addWidget(grp_load)
        loader_col.addStretch()

        ai_col = QVBoxLayout()
        grp_ai = SkeetGroupBox("AI CONFIGURATION")
        self.s_temp = SkeetSlider("Temperature", 0.1, 2.0, self.config.get("temp", 0.7))
        self.s_temp.valueChanged.connect(lambda v: self._update_config_value("temp", v))
        self.s_top = SkeetSlider("Top-P", 0.1, 1.0, self.config.get("top_p", 0.9))
        self.s_top.valueChanged.connect(lambda v: self._update_config_value("top_p", v))
        self.s_tok = SkeetSlider(
            "Max Tokens", 512, 8192, self.config.get("max_tokens", 2048), is_int=True
        )
        self.s_tok.valueChanged.connect(
            lambda v: self._update_config_value("max_tokens", int(v))
        )
        self.s_ctx = SkeetSlider(
            "Context Limit", 1024, 16384, self.config.get("ctx_limit", 8192), is_int=True
        )
        self.s_ctx.valueChanged.connect(self._on_ctx_limit_changed)
        lbl_sys = QLabel("System Prompt")
        lbl_sys.setStyleSheet(f"color: {FG_DIM}; font-size: 11px; margin-top: 5px;")
        self.inp_sys = QLineEdit(self.config.get("system_prompt", "You are Monolith. Be precise."))
        self.inp_sys.setStyleSheet(
            f"background: {BG_INPUT}; color: #aaa; border: 1px solid #333; padding: 5px;"
        )
        self.inp_sys.textChanged.connect(self._on_system_prompt_changed)
        grp_ai.add_widget(self.s_temp)
        grp_ai.add_widget(self.s_top)
        grp_ai.add_widget(self.s_tok)
        grp_ai.add_widget(self.s_ctx)
        grp_ai.add_widget(lbl_sys)
        grp_ai.add_widget(self.inp_sys)
        ai_col.addWidget(grp_ai)
        ai_col.addStretch()

        config_row.addLayout(loader_col)
        config_row.addLayout(ai_col)
        config_layout.addLayout(config_row)
        config_section.set_content_layout(config_layout)
        chat_layout.addWidget(config_section)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setStyleSheet(f"""
            background: {BG_INPUT}; color: #ccc; border: 1px solid #222; 
            font-family: 'Consolas', monospace; font-size: 12px;
        """)
        chat_layout.addWidget(self.chat)
        
        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Enter command...")
        self.input.returnPressed.connect(self.send)
        self.input.setStyleSheet(f"""
            QLineEdit {{
                background: {BG_INPUT}; color: white; border: 1px solid #333;
                padding: 8px; font-family: 'Verdana'; font-size: 11px;
            }}
            QLineEdit:focus {{ border: 1px solid {ACCENT_GOLD}; }}
        """)
        
        self.btn_send = QPushButton("SEND")
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.setFixedWidth(80)
        self.btn_send.setStyleSheet(f"""
            QPushButton {{
                background: {BG_INPUT}; 
                border: 1px solid {ACCENT_GOLD}; 
                color: {ACCENT_GOLD}; 
                padding: 8px; font-size: 11px; font-weight: bold; border-radius: 2px;
            }}
            QPushButton:hover {{ background: {ACCENT_GOLD}; color: black; }}
            QPushButton:pressed {{ background: #b08d2b; }}
        """)
        self.btn_send.clicked.connect(self.send)
        
        input_row.addWidget(self.input)
        input_row.addWidget(self.btn_send)
        chat_layout.addLayout(input_row)
        
        chat_group.add_layout(chat_layout)
        left_col.addWidget(chat_group)
        left_col.addStretch()
        layout.addLayout(left_col, 3)

        right_col = QVBoxLayout()
        trace_group = SkeetGroupBox("REASONING TRACE")
        self.trace = QTextEdit()
        self.trace.setReadOnly(True)
        self.trace.setStyleSheet(f"""
            background: {BG_INPUT}; color: {FG_TEXT}; border: 1px solid #222; 
            font-family: 'Consolas', monospace; font-size: 10px;
        """)
        trace_group.add_widget(self.trace)
        right_col.addWidget(trace_group)
        layout.addLayout(right_col, 2)

        self._sync_path_display()
        self._update_load_button_text()

    def send(self):
        txt = self.input.text().strip()
        if not txt: return
        self.input.clear()
        self.chat.append(f"<span style='color:{ACCENT_GOLD}'><b>USER:</b></span> {txt}")
        self.chat.append(f"<span style='color:{FG_TEXT}'><b>MONOLITH:</b></span>")
        self.chat.moveCursor(QTextCursor.End)
        self.sig_generate.emit(txt)

    def _flush_tokens(self):
        if not self._token_buf:
            self._flush_timer.stop()
            return
        chunk = "".join(self._token_buf)
        self._token_buf.clear()
        self.chat.moveCursor(QTextCursor.End)
        self.chat.insertPlainText(chunk)
        self.chat.moveCursor(QTextCursor.End)

    def append_token(self, t):
        self._token_buf.append(t)
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def append_trace(self, html): self.trace.append(html)
    def clear_chat(self):
        self.chat.clear()
        self.trace.clear()
        self.chat.append(f"<span style='color:{FG_DIM}'>--- SESSION RESET ---</span>")

    def _sync_path_display(self):
        if self.state.gguf_path:
            self.path_display.setText(self.state.gguf_path)
            self.path_display.setToolTip(self.state.gguf_path)
        else:
            self.path_display.clear()
            self.path_display.setToolTip("")

    def _save_config(self):
        save_config(self.config)

    def _update_config_value(self, key, value):
        self.config[key] = value
        self._save_config()

    def _on_ctx_limit_changed(self, value):
        self.state.ctx_limit = int(value)
        self._update_config_value("ctx_limit", int(value))

    def _on_system_prompt_changed(self, text):
        self._update_config_value("system_prompt", text)

    def _on_context_injection_changed(self, text):
        self._update_config_value("context_injection", text)

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF", "", "GGUF (*.gguf)")
        if path:
            self.state.gguf_path = path
            self.config["gguf_path"] = path
            self._sync_path_display()
            self._save_config()

    def toggle_load(self):
        if self.state.model_loaded:
            self.sig_unload.emit()
        else:
            self.sig_load.emit()

    def _update_load_button_text(self):
        self.btn_load.setText("UNLOAD MODEL" if self.state.model_loaded else "LOAD MODEL")

    def update_status(self, status):
        is_loading = status in (SystemStatus.LOADING, SystemStatus.RUNNING)
        self.btn_load.setEnabled(not is_loading)
        if is_loading:
            self.btn_load.setText("PROCESSING...")
        else:
            self._update_load_button_text()
