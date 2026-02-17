from datetime import datetime, timezone

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.state import SystemStatus
import core.style as _s
from ui.components.atoms import MonoButton, MonoGroupBox, MonoSlider
from ui.components.message_widget import MessageWidget
from core.llm_config import DEFAULT_CONFIG, load_config, save_config


class PageCode(QWidget):
    sig_generate = Signal(str)
    sig_load = Signal()
    sig_unload = Signal()
    sig_stop = Signal()
    sig_sync_history = Signal(list)
    sig_set_model_path = Signal(str)
    sig_set_ctx_limit = Signal(int)
    sig_debug = Signal(str)
    sig_finished = Signal()

    def __init__(self, state, ui_bridge):
        super().__init__()
        self.state = state
        self.ui_bridge = ui_bridge
        self.config = load_config()
        self.config.pop("agent_mode", None)
        self._workspace_root = self.config.get("workspace_root")
        self._token_buf: list[str] = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(25)
        self._flush_timer.timeout.connect(self._flush_tokens)
        self._current_session = self._create_session()
        self._active_widget: MessageWidget | None = None
        self._active_assistant_index = None
        self._is_running = False
        self._is_model_loaded = False
        self._last_status = None
        self._agent_step = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        main_split = QSplitter(Qt.Horizontal)
        main_split.setChildrenCollapsible(False)
        layout.addWidget(main_split)

        chat_group = MonoGroupBox("CODE")
        chat_layout = QVBoxLayout()
        chat_layout.setSpacing(10)

        workspace_row = QHBoxLayout()
        self.workspace_input = QLineEdit()
        self.workspace_input.setReadOnly(True)
        self.workspace_input.setPlaceholderText("Select a workspace folder to begin")
        self.workspace_input.setStyleSheet(
            f"background: {_s.BG_INPUT}; color: {_s.FG_PLACEHOLDER}; border: 1px solid {_s.BORDER_LIGHT}; padding: 5px;"
        )
        btn_workspace = MonoButton("...")
        btn_workspace.setFixedWidth(30)
        btn_workspace.clicked.connect(self.pick_workspace)
        workspace_row.addWidget(self.workspace_input)
        workspace_row.addWidget(btn_workspace)
        chat_layout.addLayout(workspace_row)

        self.message_list = QListWidget()
        self.message_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.message_list.setStyleSheet(f"""
            QListWidget {{
                background: transparent; color: {_s.FG_TEXT}; border: 1px solid {_s.BORDER_SUBTLE};
                font-family: 'Consolas', monospace; font-size: 12px;
            }}
            QListWidget::item {{ border: none; background: transparent; padding: 0px; }}
            {_s.SCROLLBAR_STYLE}
        """)
        chat_layout.addWidget(self.message_list)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Select a workspace folder to begin")
        self.input.returnPressed.connect(self.handle_send_click)
        self.input.textChanged.connect(self._on_input_changed)
        self.input.setStyleSheet(f"""
            QLineEdit {{
                background: {_s.BG_INPUT}; color: white; border: 1px solid {_s.BORDER_LIGHT};
                padding: 8px; font-family: 'Verdana'; font-size: 11px;
            }}
            QLineEdit:focus {{ border: 1px solid {_s.ACCENT_PRIMARY}; }}
        """)

        self.btn_send = QPushButton("SEND")
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.setFixedWidth(80)
        self._btn_style_template = f"""
            QPushButton {{{{
                background: {{bg}};
                border: 1px solid {{color}};
                color: {{color}};
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
                border-radius: 2px;
            }}}}
            QPushButton:hover {{{{ background: {{color}}; color: black; }}}}
            QPushButton:pressed {{{{ background: {_s.ACCENT_PRIMARY_DARK}; }}}}
        """
        self.btn_send.clicked.connect(self.handle_send_click)
        input_row.addWidget(self.input)
        input_row.addWidget(self.btn_send)
        chat_layout.addLayout(input_row)

        status_row = QHBoxLayout()
        self.lbl_workspace_status = QLabel("workspace: (none)")
        self.lbl_workspace_status.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px;")
        self.lbl_step_status = QLabel("step: 0/25")
        self.lbl_step_status.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px;")
        status_row.addWidget(self.lbl_workspace_status)
        status_row.addStretch()
        status_row.addWidget(self.lbl_step_status)
        chat_layout.addLayout(status_row)

        session_row = QHBoxLayout()
        btn_new_session = MonoButton("NEW SESSION")
        btn_new_session.clicked.connect(self._new_session)
        session_row.addStretch()
        session_row.addWidget(btn_new_session)
        chat_layout.addLayout(session_row)

        chat_group.add_layout(chat_layout)

        right_stack = QSplitter(Qt.Vertical)
        right_stack.setChildrenCollapsible(False)

        trace_group = MonoGroupBox("REASONING TRACE")
        self.trace = QTextEdit()
        self.trace.setReadOnly(True)
        self.trace.setStyleSheet(f"""
            QTextEdit {{
                background-color: {_s.BG_INPUT};
                color: {_s.FG_TEXT};
                border: 1px solid {_s.BORDER_SUBTLE};
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }}
            QTextEdit::viewport {{ background-color: {_s.BG_INPUT}; }}
            {_s.SCROLLBAR_STYLE}
        """)
        trace_group.add_widget(self.trace)

        controls_group = MonoGroupBox("SETTINGS")
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)

        self.path_display = QLineEdit()
        self.path_display.setReadOnly(True)
        self.path_display.setPlaceholderText("No GGUF Selected")
        self.path_display.setStyleSheet(
            f"background: {_s.BG_INPUT}; color: {_s.FG_PLACEHOLDER}; border: 1px solid {_s.BORDER_LIGHT}; padding: 5px;"
        )
        btn_browse = MonoButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self.pick_file)
        model_row = QHBoxLayout()
        model_row.addWidget(self.path_display)
        model_row.addWidget(btn_browse)
        self.btn_load = MonoButton("LOAD MODEL")
        self.btn_load.clicked.connect(self.toggle_load)
        controls_layout.addLayout(model_row)
        controls_layout.addWidget(self.btn_load)

        self.s_temp = MonoSlider("Temperature", 0.1, 2.0, self.config.get("temp", 0.7))
        self.s_temp.valueChanged.connect(lambda v: self._update_config_value("temp", v))
        self.s_top = MonoSlider("Top-P", 0.1, 1.0, self.config.get("top_p", 0.9))
        self.s_top.valueChanged.connect(lambda v: self._update_config_value("top_p", v))
        self.s_tok = MonoSlider("Max Tokens", 512, 8192, self.config.get("max_tokens", 2048), is_int=True)
        self.s_tok.valueChanged.connect(lambda v: self._update_config_value("max_tokens", int(v)))
        self.s_ctx = MonoSlider("Context Limit", 1024, 16384, self.config.get("ctx_limit", 8192), is_int=True)
        self.s_ctx.valueChanged.connect(self._on_ctx_limit_changed)
        controls_layout.addWidget(self.s_temp)
        controls_layout.addWidget(self.s_top)
        controls_layout.addWidget(self.s_tok)
        controls_layout.addWidget(self.s_ctx)

        save_row = QHBoxLayout()
        self.btn_save_config = MonoButton("SAVE SETTINGS")
        self.btn_save_config.clicked.connect(self._save_config)
        btn_reset = MonoButton("RESET")
        btn_reset.clicked.connect(self._reset_config)
        save_row.addWidget(btn_reset)
        save_row.addStretch()
        save_row.addWidget(self.btn_save_config)
        controls_layout.addLayout(save_row)
        controls_layout.addStretch()
        controls_group.add_layout(controls_layout)

        right_stack.addWidget(trace_group)
        right_stack.addWidget(controls_group)
        right_stack.setStretchFactor(0, 1)
        right_stack.setStretchFactor(1, 1)

        main_split.addWidget(chat_group)
        main_split.addWidget(right_stack)
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 2)

        self._sync_path_display()
        self._sync_workspace_display()
        self._update_load_button_text()
        self._set_send_button_state(False)
        self._sync_send_availability()

    def _create_session(self):
        now = datetime.now(timezone.utc).isoformat()
        return {"id": now, "messages": [], "created_at": now}

    def _add_message(self, role, text):
        self._current_session["messages"].append(
            {"role": role, "text": text, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        return len(self._current_session["messages"]) - 1

    def _append_message_widget(self, idx):
        msg = self._current_session["messages"][idx]
        widget = MessageWidget(idx, msg["role"], msg["text"], msg["timestamp"])
        item = QListWidgetItem(self.message_list)
        item.setSizeHint(widget.sizeHint())
        self.message_list.setItemWidget(item, widget)
        return widget

    def send(self):
        txt = self.input.text().strip()
        if not txt:
            return
        if not self._workspace_root:
            return
        self._agent_step = 0
        self.lbl_step_status.setText("step: 0/25")
        self._set_send_button_state(True)
        self.input.clear()
        user_idx = self._add_message("user", txt)
        self._append_message_widget(user_idx)
        self._active_assistant_index = self._add_message("assistant", "")
        self._active_widget = self._append_message_widget(self._active_assistant_index)
        self.message_list.scrollToBottom()
        self.sig_generate.emit(txt)

    def handle_send_click(self):
        if self._is_running:
            self.sig_stop.emit()
            return
        self.send()

    def _set_send_button_state(self, is_running):
        self._is_running = is_running
        if is_running:
            self.btn_send.setText("■")
            color = _s.FG_ERROR
        else:
            self.btn_send.setText("SEND")
            color = _s.ACCENT_PRIMARY
        self.btn_send.setStyleSheet(self._btn_style_template.format(bg=_s.BG_INPUT, color=color))
        self._sync_send_availability()

    def _sync_send_availability(self):
        ready = bool(self._workspace_root)
        self.btn_send.setEnabled(ready)
        self.input.setEnabled(ready)
        if ready:
            self.input.setPlaceholderText("Enter coding request...")
        else:
            self.input.setPlaceholderText("Select a workspace folder to begin")

    def _on_input_changed(self, _text):
        if self._is_running:
            self.btn_send.setText("■")

    def _flush_tokens(self):
        if not self._token_buf:
            self._flush_timer.stop()
            return
        chunk = "".join(self._token_buf)
        self._token_buf.clear()
        if self._active_widget is None or self._active_assistant_index is None:
            return
        self._current_session["messages"][self._active_assistant_index]["text"] += chunk
        self._active_widget.append_token(chunk)
        self.message_list.scrollToBottom()

    def append_token(self, token):
        if not token:
            return
        self._token_buf.append(token)
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def append_trace(self, msg):
        self.trace.append(msg)
        if msg.startswith("[AGENT ") and "]" in msg:
            try:
                bracket = msg.split("]", 1)
                inner = bracket[0].replace("[AGENT ", "").strip()
                parts = inner.split("|")
                step_part = parts[0].strip()
                payload = bracket[1].strip()
                tool_name = payload.split("→", 1)[0].strip()
                self.lbl_step_status.setText(f"[step {step_part}] {tool_name}")
            except Exception:
                pass

    def update_status(self, status):
        is_processing = status in (SystemStatus.LOADING, SystemStatus.RUNNING, SystemStatus.UNLOADING)
        if status == SystemStatus.RUNNING:
            self._set_send_button_state(True)
        elif status == SystemStatus.READY:
            if self._last_status == SystemStatus.LOADING:
                self._is_model_loaded = True
            elif self._last_status == SystemStatus.UNLOADING:
                self._is_model_loaded = False
            self._set_send_button_state(False)
            if self._active_widget is not None:
                self._active_widget.finalize()
            self._active_widget = None
            self._active_assistant_index = None
        elif status == SystemStatus.LOADING:
            self.btn_send.setEnabled(False)
        elif status in (SystemStatus.UNLOADING, SystemStatus.ERROR):
            self._is_model_loaded = False
            if not is_processing:
                self._update_load_button_text()

        self._update_load_button_text()
        self._last_status = status

    def on_guard_finished(self):
        self._set_send_button_state(False)
        self.sig_finished.emit()

    def _new_session(self):
        self.message_list.clear()
        self._current_session = self._create_session()
        self._active_widget = None
        self._active_assistant_index = None
        self._agent_step = 0
        self.lbl_step_status.setText("step: 0/25")
        self.sig_sync_history.emit([])

    def apply_operator(self, operator_data: dict):
        if not isinstance(operator_data, dict):
            return
        config = operator_data.get("config")
        if not isinstance(config, dict):
            return
        config.pop("agent_mode", None)
        self.config.update(config)
        self.s_temp.slider.setValue(int(float(self.config.get("temp", 0.7)) * 100))
        self.s_top.slider.setValue(int(float(self.config.get("top_p", 0.9)) * 100))
        self.s_tok.slider.setValue(int(self.config.get("max_tokens", 2048)))
        self.s_ctx.slider.setValue(int(self.config.get("ctx_limit", 8192)))
        self.sig_set_ctx_limit.emit(int(self.config.get("ctx_limit", 8192)))

        gguf_path = self.config.get("gguf_path")
        if gguf_path:
            self.sig_set_model_path.emit(str(gguf_path))
        self._workspace_root = self.config.get("workspace_root")
        self._sync_path_display()
        self._sync_workspace_display()
        self._sync_send_availability()

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF", "", "GGUF (*.gguf)")
        if path:
            self.config["gguf_path"] = path
            self.sig_set_model_path.emit(path)
            self._sync_path_display()

    def pick_workspace(self):
        path = QFileDialog.getExistingDirectory(self, "Select workspace", "")
        if path:
            self._workspace_root = path
            self.config["workspace_root"] = path
            self._sync_workspace_display()
            self._sync_send_availability()

    def toggle_load(self):
        if self._is_model_loaded:
            self.sig_unload.emit()
        else:
            self.sig_load.emit()

    def _update_load_button_text(self):
        self.btn_load.setText("UNLOAD MODEL" if self._is_model_loaded else "LOAD MODEL")

    def _sync_path_display(self):
        model_path = self.config.get("gguf_path")
        self.path_display.setText(str(model_path) if model_path else "")

    def _sync_workspace_display(self):
        text = str(self._workspace_root) if self._workspace_root else ""
        self.workspace_input.setText(text)
        self.lbl_workspace_status.setText(f"workspace: {text or '(none)'}")

    def _on_ctx_limit_changed(self, value):
        self._update_config_value("ctx_limit", int(value))
        self.sig_set_ctx_limit.emit(int(value))

    def _update_config_value(self, key, value):
        self.config[key] = value

    def _save_config(self):
        persisted = dict(self.config)
        persisted.pop("agent_mode", None)
        save_config(persisted)

    def _reset_config(self):
        for key, val in DEFAULT_CONFIG.items():
            if key == "agent_mode":
                continue
            self.config[key] = val
        self.s_temp.slider.setValue(int(DEFAULT_CONFIG["temp"] * 100))
        self.s_top.slider.setValue(int(DEFAULT_CONFIG["top_p"] * 100))
        self.s_tok.slider.setValue(int(DEFAULT_CONFIG["max_tokens"]))
        self.s_ctx.slider.setValue(int(DEFAULT_CONFIG["ctx_limit"]))
        self.sig_set_ctx_limit.emit(int(DEFAULT_CONFIG["ctx_limit"]))
