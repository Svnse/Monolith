import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel, QFileDialog,
    QSplitter, QListWidget, QListWidgetItem, QStackedWidget,
    QMessageBox, QButtonGroup
)
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Signal, Qt, QTimer, QDateTime

from core.state import SystemStatus
from core.style import BG_INPUT, FG_DIM, FG_TEXT, ACCENT_GOLD, FG_ERROR, SCROLLBAR_STYLE
from ui.components.atoms import SkeetGroupBox, SkeetButton, CollapsibleSection, SkeetSlider
from ui.components.complex import BehaviorTagInput
from core.llm_config import DEFAULT_CONFIG, MASTER_PROMPT, load_config, save_config
from core.paths import ARCHIVE_DIR

class PageChat(QWidget):
    sig_generate = Signal(str, bool)
    sig_load = Signal()
    sig_unload = Signal()
    sig_stop = Signal()
    sig_sync_history = Signal(list)

    def __init__(self, state, ui_bridge):
        super().__init__()
        self.state = state
        self.ui_bridge = ui_bridge
        self.config = load_config()
        self.state.gguf_path = self.config.get("gguf_path")
        self.state.ctx_limit = int(self.config.get("ctx_limit", self.state.ctx_limit))
        self._token_buf: list[str] = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(25)
        self._flush_timer.timeout.connect(self._flush_tokens)
        self._archive_dir = self._get_archive_dir()
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._session_counter = 0
        self._current_session = self._create_session()
        self._title_generated = False
        self._active_assistant_index = None
        self._rewrite_assistant_index = None
        self._assistant_block_map = {}
        self._last_status = None
        self._is_running = False
        self._pending_update_text = None
        self._awaiting_update_restart = False
        self._update_trace_state = None
        self._update_token_count = 0
        self._update_progress_index = 0
        self._config_dirty = False
        self._thinking_mode = bool(self.config.get("thinking_mode", False))

        capabilities_signal = getattr(self.state, "sig_model_capabilities", None)
        if capabilities_signal is not None:
            capabilities_signal.connect(self._on_model_capabilities)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        main_split = QSplitter(Qt.Horizontal)
        main_split.setChildrenCollapsible(False)
        layout.addWidget(main_split)

        config_section = CollapsibleSection("⚙ CONFIGURATION")
        config_section.btn_toggle.setStyleSheet(f"""
            QPushButton {{
                color: {FG_TEXT}; background: transparent;
                border: none; text-align: left; font-weight: bold; font-size: 10px;
            }}
            QPushButton:checked {{ color: {FG_TEXT}; }}
            QPushButton:hover {{ color: {FG_TEXT}; }}
            QPushButton::indicator {{
                width: 6px; height: 6px; border-radius: 3px;
                border: 1px solid {FG_DIM}; margin-right: 6px;
            }}
            QPushButton::indicator:checked {{
                background: {ACCENT_GOLD}; border: 1px solid {ACCENT_GOLD};
            }}
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
        lbl_sys = QLabel("Behavior Tags")
        lbl_sys.setStyleSheet(f"color: {FG_DIM}; font-size: 11px; margin-top: 5px;")
        self.behavior_tags = BehaviorTagInput([])
        self.behavior_tags.tagsChanged.connect(self._on_behavior_tags_changed)
        grp_ai.add_widget(self.s_temp)
        grp_ai.add_widget(self.s_top)
        grp_ai.add_widget(self.s_tok)
        grp_ai.add_widget(self.s_ctx)
        grp_ai.add_widget(lbl_sys)
        grp_ai.add_widget(self.behavior_tags)
        save_row = QHBoxLayout()
        self.lbl_config_state = QLabel("SAVED")
        self.lbl_config_state.setStyleSheet(f"color: {FG_DIM}; font-size: 10px; font-weight: bold;")
        self.btn_save_config = SkeetButton("SAVE SETTINGS")
        self.btn_save_config.clicked.connect(self._save_config)
        save_row.addWidget(self.lbl_config_state)
        save_row.addStretch()
        save_row.addWidget(self.btn_save_config)
        grp_ai.add_layout(save_row)
        ai_col.addWidget(grp_ai)
        ai_col.addStretch()

        config_row.addLayout(loader_col)
        config_row.addLayout(ai_col)
        config_layout.addLayout(config_row)
        config_section.set_content_layout(config_layout)

        operations_group = SkeetGroupBox("OPERATIONS")
        operations_layout = QVBoxLayout()
        operations_layout.setSpacing(10)

        tab_row = QHBoxLayout()
        tab_style = f"""
            QPushButton {{
                background: #181818; border: 1px solid #333; color: {FG_DIM};
                padding: 6px 12px; font-size: 10px; font-weight: bold; border-radius: 2px;
            }}
            QPushButton:checked {{
                background: #222; color: {ACCENT_GOLD}; border: 1px solid {ACCENT_GOLD};
            }}
            QPushButton:hover {{ color: {FG_TEXT}; border: 1px solid {FG_TEXT}; }}
        """
        self.btn_tab_control = SkeetButton("CONTROL")
        self.btn_tab_control.setCheckable(True)
        self.btn_tab_control.setChecked(True)
        self.btn_tab_control.setStyleSheet(tab_style)
        self.btn_tab_archive = SkeetButton("ARCHIVE")
        self.btn_tab_archive.setCheckable(True)
        self.btn_tab_archive.setStyleSheet(tab_style)
        tab_group = QButtonGroup(self)
        tab_group.setExclusive(True)
        tab_group.addButton(self.btn_tab_control)
        tab_group.addButton(self.btn_tab_archive)
        tab_row.addWidget(self.btn_tab_control)
        tab_row.addWidget(self.btn_tab_archive)
        tab_row.addStretch()
        operations_layout.addLayout(tab_row)

        self.ops_stack = QStackedWidget()
        operations_layout.addWidget(self.ops_stack)

        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)
        control_layout.setSpacing(12)

        control_layout.addWidget(config_section)
        control_layout.addStretch()

        archive_tab = QWidget()
        archive_layout = QVBoxLayout(archive_tab)
        archive_layout.setSpacing(10)

        archive_controls = QHBoxLayout()
        self.btn_save_chat = SkeetButton("SAVE")
        self.btn_save_chat.clicked.connect(self._save_chat_archive)
        self.btn_load_chat = SkeetButton("LOAD")
        self.btn_load_chat.clicked.connect(self._load_chat_archive)
        self.btn_delete_chat = SkeetButton("DELETE")
        self.btn_delete_chat.clicked.connect(self._delete_selected_archive)
        self.btn_clear_chat = SkeetButton("CLEAR")
        self.btn_clear_chat.clicked.connect(lambda: self._clear_current_session(delete_archive=False))
        archive_controls.addWidget(self.btn_save_chat)
        archive_controls.addWidget(self.btn_load_chat)
        archive_controls.addWidget(self.btn_delete_chat)
        archive_controls.addWidget(self.btn_clear_chat)
        archive_controls.addStretch()
        archive_layout.addLayout(archive_controls)

        self.archive_list = QListWidget()
        self.archive_list.setStyleSheet(f"""
            QListWidget {{
                background: {BG_INPUT}; color: {FG_TEXT}; border: 1px solid #222;
                font-family: 'Consolas', monospace; font-size: 10px;
            }}
            QListWidget::item {{ padding: 6px; }}
            QListWidget::item:selected {{ background: #222; color: {ACCENT_GOLD}; }}
            {SCROLLBAR_STYLE}
        """)
        archive_layout.addWidget(self.archive_list)

        self.ops_stack.addWidget(control_tab)
        self.ops_stack.addWidget(archive_tab)
        self.btn_tab_control.toggled.connect(lambda checked: self._switch_ops_tab(0, checked))
        self.btn_tab_archive.toggled.connect(lambda checked: self._switch_ops_tab(1, checked))

        operations_group.add_layout(operations_layout)

        chat_group = SkeetGroupBox("TERMINAL")
        chat_layout = QVBoxLayout()
        chat_layout.setSpacing(10)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setStyleSheet(f"""
            background: {BG_INPUT}; color: #ccc; border: 1px solid #222; 
            font-family: 'Consolas', monospace; font-size: 12px;
            {SCROLLBAR_STYLE}
        """)
        chat_layout.addWidget(self.chat)
        
        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Enter command...")
        self.input.returnPressed.connect(self.handle_send_click)
        self.input.textChanged.connect(self._on_input_changed)
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
        self._btn_style_template = """
            QPushButton {{
                background: {bg};
                border: 1px solid {color};
                color: {color};
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
                border-radius: 2px;
            }}
            QPushButton:hover {{ background: {color}; color: black; }}
            QPushButton:pressed {{ background: #b08d2b; }}
        """
        self._set_send_button_state(is_running=False)
        self.btn_send.clicked.connect(self.handle_send_click)

        self.chk_thinking_mode = QPushButton("THINK")
        self.chk_thinking_mode.setCheckable(True)
        self.chk_thinking_mode.setChecked(self._thinking_mode)
        self.chk_thinking_mode.setCursor(Qt.PointingHandCursor)
        self.chk_thinking_mode.setStyleSheet(f"""
            QPushButton {{
                background: {BG_INPUT};
                border: 1px solid #333;
                color: {FG_DIM};
                padding: 8px;
                font-size: 10px;
                font-weight: bold;
                border-radius: 2px;
            }}
            QPushButton:checked {{
                border: 1px solid {ACCENT_GOLD};
                color: {ACCENT_GOLD};
            }}
            QPushButton:hover {{
                border: 1px solid {FG_TEXT};
                color: {FG_TEXT};
            }}
        """)
        self.chk_thinking_mode.toggled.connect(self._on_thinking_mode_toggled)
        
        input_row.addWidget(self.input)
        input_row.addWidget(self.chk_thinking_mode)
        input_row.addWidget(self.btn_send)
        chat_layout.addLayout(input_row)
        
        chat_group.add_layout(chat_layout)

        right_stack = QSplitter(Qt.Vertical)
        right_stack.setChildrenCollapsible(False)

        trace_group = SkeetGroupBox("REASONING TRACE")
        self.trace = QTextEdit()
        self.trace.setReadOnly(True)
        self.trace.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_INPUT};
                color: {FG_TEXT};
                border: 1px solid #222;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }}
            QTextEdit::viewport {{
                background-color: {BG_INPUT};
            }}
            {SCROLLBAR_STYLE}
        """)
        self.lbl_config_update = QLabel("")
        self.lbl_config_update.setStyleSheet(f"color: {ACCENT_GOLD}; font-size: 10px; font-weight: bold;")
        self.lbl_config_update.hide()
        self._config_update_fade = QTimer(self)
        self._config_update_fade.setSingleShot(True)
        self._config_update_fade.timeout.connect(self.lbl_config_update.hide)
        trace_group.add_widget(self.trace)
        trace_group.add_widget(self.lbl_config_update)

        right_stack.addWidget(trace_group)
        right_stack.addWidget(operations_group)
        right_stack.setStretchFactor(0, 1)
        right_stack.setStretchFactor(1, 1)
        right_stack.setSizes([200, 200])

        main_split.addWidget(chat_group)
        main_split.addWidget(right_stack)
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 2)

        self._sync_path_display()
        self._update_load_button_text()
        self._refresh_archive_list()
        self._apply_behavior_prompt(self.config.get("behavior_tags", []))
        self.behavior_tags.set_tags(self.config.get("behavior_tags", []))
        self._set_config_dirty(False)
        if not self.state.model_loaded:
            self._apply_default_limits()

    def send(self):
        txt = self.input.text().strip()
        if not txt:
            return
        self._set_send_button_state(is_running=True)
        self.input.clear()
        safe_txt = html.escape(txt)
        self.chat.append(
            f"<span style='color:white'><b>USER:</b></span> "
            f"<span style='color:{FG_DIM}'>{safe_txt}</span>"
        )
        self.chat.append(f"<span style='color:{ACCENT_GOLD}'><b>ASSISTANT:</b></span>")
        self.chat.moveCursor(QTextCursor.End)
        self._add_message("user", txt)
        self._active_assistant_index = self._add_message("assistant", "")
        block = self.chat.document().lastBlock().blockNumber()
        self._assistant_block_map[self._active_assistant_index] = block
        self.sig_generate.emit(txt, self._thinking_mode)

    def handle_send_click(self):
        txt = self.input.text().strip()

        if not self._is_running:
            self.send()
            return

        if not txt:
            self._set_send_button_state(is_running=True, stopping=True)
            self._maybe_generate_title()
            self.sig_stop.emit()
            return

        self._pending_update_text = txt
        self._awaiting_update_restart = True
        self.btn_send.setEnabled(False)
        self._begin_update_trace(txt)
        self.sig_stop.emit()

    def _set_send_button_state(self, is_running: bool, stopping: bool = False):
        self._is_running = is_running
        if is_running:
            has_input = bool(self.input.text().strip())
            if has_input:
                self.btn_send.setText("UPDATE")
                color = ACCENT_GOLD
            else:
                self.btn_send.setText("■")
                color = FG_ERROR
            self.btn_send.setStyleSheet(
                self._btn_style_template.format(
                    bg=BG_INPUT,
                    color=color,
                )
            )
            self.btn_send.setEnabled(not stopping)
        else:
            self.btn_send.setText("SEND")
            self.btn_send.setStyleSheet(
                self._btn_style_template.format(
                    bg=BG_INPUT,
                    color=ACCENT_GOLD,
                )
            )
            self.btn_send.setEnabled(True)

    def _on_input_changed(self, text):
        if not self._is_running:
            return
        self._set_send_button_state(is_running=True)

    def _submit_update(self, update_text):
        self._set_send_button_state(is_running=True)
        self._rewrite_assistant_index = self._active_assistant_index
        partial = "(no output yet)"
        if self._active_assistant_index is not None:
            txt = self._current_session["messages"][self._active_assistant_index]["text"]
            if txt:
                partial = txt

        original = ""
        for msg in reversed(self._current_session["messages"]):
            if msg["role"] == "user":
                original = msg["text"]
                break

        injected = f"""
You were interrupted mid-generation.

Original user request:
{original}

Partial assistant output so far:
{partial}

User update:
{update_text}

Continue from the interruption point. Do not repeat earlier content.
"""

        self.input.clear()
        safe_update = html.escape(update_text)
        self.chat.append(
            f"<span style='color:white'><b>USER:</b></span> "
            f"<span style='color:{FG_DIM}'>{safe_update}</span>"
        )
        self._add_message("user", update_text)
        self._start_update_streaming()
        self.sig_generate.emit(injected, self._thinking_mode)

    def _flush_tokens(self):
        if not self._token_buf:
            self._flush_timer.stop()
            return
        chunk = "".join(self._token_buf)
        self._token_buf.clear()
        if self._rewrite_assistant_index is not None:
            block_number = self._assistant_block_map.get(self._rewrite_assistant_index)
            if block_number is not None:
                doc = self.chat.document()
                start_block = doc.findBlockByNumber(block_number)
                if start_block.isValid():
                    msg = self._current_session["messages"][self._rewrite_assistant_index]
                    safe = html.escape(msg.get("text", ""))
                    scroll = self.chat.verticalScrollBar()
                    at_bottom = scroll.value() == scroll.maximum()
                    value = scroll.value()
                    current_cursor = self.chat.textCursor()
                    cursor = QTextCursor(start_block)
                    next_block_number = min(
                        (number for number in self._assistant_block_map.values() if number > block_number),
                        default=None,
                    )
                    if next_block_number is not None:
                        end_block = doc.findBlockByNumber(next_block_number)
                        cursor.setPosition(end_block.position(), QTextCursor.KeepAnchor)
                    else:
                        end_cursor = QTextCursor(doc)
                        end_cursor.movePosition(QTextCursor.End)
                        cursor.setPosition(end_cursor.position(), QTextCursor.KeepAnchor)
                    cursor.insertHtml(
                        f"<span style='color:{ACCENT_GOLD}'><b>ASSISTANT:</b></span> "
                        f"<span style='color:white'>{safe}</span>"
                    )
                    self.chat.setTextCursor(current_cursor)
                    if at_bottom:
                        scroll.setValue(scroll.maximum())
                    else:
                        scroll.setValue(value)
                    return
        scroll = self.chat.verticalScrollBar()
        threshold = scroll.pageStep() * 0.15
        at_bottom = scroll.value() >= (scroll.maximum() - threshold)
        value = scroll.value()
        self.chat.moveCursor(QTextCursor.End)
        safe_chunk = html.escape(chunk)
        cursor = self.chat.textCursor()
        cursor.insertHtml(
            f"<span style='white-space: pre-wrap; color:{FG_DIM}'>{safe_chunk}</span>"
        )
        if at_bottom:
            self.chat.moveCursor(QTextCursor.End)
            scroll.setValue(scroll.maximum())
        else:
            scroll.setValue(value)

    def append_token(self, t):
        self._token_buf.append(t)
        self._append_assistant_token(t)
        self._update_progress_markers()
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def on_guard_finished(self, engine_key, task_id):
        if engine_key != "llm":
            return
        if not self._current_session.get("messages"):
            return
        try:
            self._save_chat_archive()
        except Exception:
            pass

    def append_trace(self, trace_msg):
        lowered = trace_msg.lower()
        if "token" in lowered:
            state = "TOKENIZING"
        elif "inference started" in lowered:
            state = "INFERENCING"
        elif "inference" in lowered and ("complete" in lowered or "aborted" in lowered):
            state = "COMPLETE"
        elif "error" in lowered:
            state = "COMPLETE"
        else:
            state = "GENERATING"
        self.trace.append(f"[{state}] {trace_msg}")

    def clear_chat(self):
        self._set_current_session(self._create_session(), show_reset=True, sync_history=True)

    def _sync_path_display(self):
        if self.state.gguf_path:
            self.path_display.setText(self.state.gguf_path)
            self.path_display.setToolTip(self.state.gguf_path)
        else:
            self.path_display.clear()
            self.path_display.setToolTip("")

    def _set_config_dirty(self, dirty=True):
        self._config_dirty = dirty
        self.lbl_config_state.setText("DIRTY" if dirty else "SAVED")
        self.lbl_config_state.setStyleSheet(
            f"color: {ACCENT_GOLD if dirty else FG_DIM}; font-size: 10px; font-weight: bold;"
        )

    def _save_config(self):
        save_config(self.config)
        self._last_config_update = QDateTime.currentDateTime()
        stamp = self._last_config_update.toString("HH:mm:ss")
        self.lbl_config_update.setText(f"USER (UPDATED): {stamp}")
        self.lbl_config_update.show()
        self._config_update_fade.start(2500)
        self._set_config_dirty(False)

    def _update_config_value(self, key, value):
        self.config[key] = value
        self._set_config_dirty(True)

    def _set_slider_limits(self, slider, max_value, value):
        qt_slider = slider.slider
        qt_slider.blockSignals(True)
        if slider.is_int:
            min_value = qt_slider.minimum()
            if max_value < min_value:
                min_value = max_value
            qt_slider.setRange(int(min_value), int(max_value))
            qt_slider.setValue(int(value))
            slider.val_lbl.setText(str(int(value)))
        else:
            min_value = qt_slider.minimum()
            max_scaled = int(max_value * 100)
            if max_scaled < min_value:
                min_value = max_scaled
            qt_slider.setRange(min_value, max_scaled)
            qt_slider.setValue(int(value * 100))
            slider.val_lbl.setText(f"{value:.2f}")
        qt_slider.blockSignals(False)

    def _apply_default_limits(self):
        self._set_slider_limits(
            self.s_ctx,
            DEFAULT_CONFIG["ctx_limit"],
            DEFAULT_CONFIG["ctx_limit"],
        )
        self._set_slider_limits(
            self.s_tok,
            DEFAULT_CONFIG["max_tokens"],
            DEFAULT_CONFIG["max_tokens"],
        )

    def _on_model_capabilities(self, payload):
        model_ctx_length = payload.get("model_ctx_length")
        if model_ctx_length is None:
            self._apply_default_limits()
            return
        self._set_slider_limits(self.s_ctx, model_ctx_length, model_ctx_length)
        self._set_slider_limits(
            self.s_tok,
            model_ctx_length,
            min(8192, model_ctx_length),
        )

    def _on_ctx_limit_changed(self, value):
        self.state.ctx_limit = int(value)
        self._update_config_value("ctx_limit", int(value))

    def _on_behavior_tags_changed(self, tags):
        self._apply_behavior_prompt(tags)

    def _on_thinking_mode_toggled(self, checked):
        self._thinking_mode = bool(checked)
        self.config["thinking_mode"] = self._thinking_mode
        self._set_config_dirty(True)

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF", "", "GGUF (*.gguf)")
        if path:
            self.state.gguf_path = path
            self.config["gguf_path"] = path
            self._sync_path_display()
            self._set_config_dirty(True)

    def toggle_load(self):
        if self.state.model_loaded:
            self.sig_unload.emit()
        else:
            self.sig_load.emit()

    def _update_load_button_text(self):
        self.btn_load.setText("UNLOAD MODEL" if self.state.model_loaded else "LOAD MODEL")

    def update_status(self, engine_key, status=None):
        if status is None:
            status = engine_key
            engine_key = "llm"
        if engine_key != "llm":
            return
        is_loading = status in (SystemStatus.LOADING, SystemStatus.RUNNING)
        self.btn_load.setEnabled(not is_loading)
        if is_loading:
            self.btn_load.setText("PROCESSING...")
        else:
            self._update_load_button_text()
        if status == SystemStatus.READY and self._awaiting_update_restart:
            self._awaiting_update_restart = False
            self.btn_send.setEnabled(True)

            update_text = self._pending_update_text
            self._pending_update_text = None
            self._submit_update(update_text)
            return
        if status == SystemStatus.RUNNING:
            self._set_send_button_state(is_running=True)
        elif status == SystemStatus.READY:
            self._set_send_button_state(is_running=False)
            self._rewrite_assistant_index = None
            if self._update_trace_state == "streaming":
                self._finalize_update_progress()
            self._maybe_generate_title()
        elif status == SystemStatus.LOADING:
            self._set_send_button_state(is_running=False)
            self.btn_send.setEnabled(False)
        if status == SystemStatus.READY and not self.state.model_loaded:
            self._apply_default_limits()
        self._last_status = status

    def _switch_ops_tab(self, index, checked):
        if checked:
            self.ops_stack.setCurrentIndex(index)

    def _start_new_session(self):
        self._set_current_session(self._create_session(), show_reset=True, sync_history=True)
        self.trace.append("--- TRACE RESET ---")

    def _prompt_clear_session(self):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Clear Session")
        dialog.setText("Choose how to clear the current session.")
        dialog.setStyleSheet(f"""
            QMessageBox {{
                background: {BG_INPUT};
                color: {FG_TEXT};
            }}
            QLabel {{
                color: {FG_TEXT};
            }}
            QPushButton {{
                color: {FG_TEXT};
                background: transparent;
                border: 1px solid #333;
                padding: 6px 12px;
                font-size: 10px;
                font-weight: bold;
                border-radius: 2px;
            }}
            QPushButton:hover {{
                border: 1px solid {ACCENT_GOLD};
                color: {ACCENT_GOLD};
            }}
            QPushButton:checked {{
                border: 1px solid {ACCENT_GOLD};
                color: {ACCENT_GOLD};
            }}
        """)
        btn_clear = dialog.addButton("Clear Logs", QMessageBox.AcceptRole)
        btn_delete = dialog.addButton("Delete Chat", QMessageBox.DestructiveRole)
        dialog.addButton("Cancel", QMessageBox.RejectRole)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked == btn_clear:
            self._clear_current_session(delete_archive=False)
        elif clicked == btn_delete:
            self._clear_current_session(delete_archive=True)

    def _clear_current_session(self, delete_archive):
        archive_path = self._current_session.get("archive_path")
        if delete_archive and archive_path:
            try:
                Path(archive_path).unlink()
            except OSError:
                pass
        self._set_current_session(self._create_session(), show_reset=True, sync_history=True)
        self._refresh_archive_list()

    def _delete_selected_archive(self):
        item = self.archive_list.currentItem()
        if not item:
            return
        archive_path = Path(item.data(Qt.UserRole))
        try:
            archive_path.unlink()
        except OSError:
            return
        if self._current_session.get("archive_path") == str(archive_path):
            self._set_current_session(self._create_session(), show_reset=True, sync_history=True)
        self._refresh_archive_list()

    def _save_chat_archive(self):
        session = self._current_session
        messages = session["messages"]
        now = self._now_iso()
        created_at = session.get("created_at") or now
        updated_at = now
        title = self._current_session.get("title") or self._derive_title(messages)
        summary = self._build_summary(messages)
        message_payload = []
        for idx, msg in enumerate(messages, start=1):
            message_payload.append({
                "i": idx,
                "time": msg.get("time") or now,
                "role": msg.get("role", "user"),
                "text": msg.get("text", "")
            })
        meta = {
            "title": title,
            "created_at": created_at,
            "updated_at": updated_at,
            "message_count": len(message_payload),
            "assistant_tokens": int(session.get("assistant_tokens", 0)),
            "summary": summary
        }
        payload = {"meta": meta, "messages": message_payload}
        archive_path = session.get("archive_path")
        if not archive_path:
            slug = self._slugify(title)
            stamp = now.replace(":", "-").replace(".", "-")
            archive_path = self._archive_dir / f"{slug}_{stamp}.json"
        else:
            archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        session["archive_path"] = str(archive_path)
        session["created_at"] = created_at
        session["updated_at"] = updated_at
        session["summary"] = summary
        self._refresh_archive_list()

    def _load_chat_archive(self):
        item = self.archive_list.currentItem()
        if not item:
            return
        archive_path = Path(item.data(Qt.UserRole))
        try:
            with archive_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            QMessageBox.warning(self, "Load Failed", "Could not read archive file.")
            return
        meta = data.get("meta", {})
        messages = []
        for msg in data.get("messages", []):
            role = msg.get("role", "user")
            text = msg.get("text", "")
            time = msg.get("time", meta.get("updated_at", self._now_iso()))
            messages.append({"i": msg.get("i"), "time": time, "role": role, "text": text})
        session = self._create_session(
            messages=messages,
            created_at=meta.get("created_at"),
            updated_at=meta.get("updated_at"),
            archive_path=str(archive_path),
            summary=meta.get("summary", []),
            title=meta.get("title"),
            assistant_tokens=int(meta.get("assistant_tokens", meta.get("token_count", 0)))
        )
        self._set_current_session(session, show_reset=False, sync_history=True)

    def _refresh_archive_list(self):
        self.archive_list.clear()
        items = []
        for path in sorted(self._archive_dir.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                continue
            meta = data.get("meta", {})
            title = meta.get("title", path.stem)
            summary = meta.get("summary", [])
            tooltip = "\n".join(summary) if summary else title
            updated_at = meta.get("updated_at", "")
            message_count = meta.get("message_count", len(data.get("messages", [])))
            assistant_tokens = int(meta.get("assistant_tokens", meta.get("token_count", 0)))
            items.append((updated_at, title, message_count, assistant_tokens, str(path), tooltip))
        items.sort(key=lambda item: item[0], reverse=True)
        for updated_at, title, message_count, assistant_tokens, path, tooltip in items:
            date_label = updated_at.split("T")[0] if updated_at else "Unknown date"
            subtext = f"{date_label} • {message_count} msgs • {assistant_tokens} assistant tokens"
            list_item = QListWidgetItem(f"{title}\n{subtext}")
            list_item.setData(Qt.UserRole, path)
            list_item.setToolTip(tooltip)
            self.archive_list.addItem(list_item)

    def _create_session(self, messages=None, created_at=None, updated_at=None, archive_path=None, summary=None, title=None, assistant_tokens=0):
        self._session_counter += 1
        now = self._now_iso()
        return {
            "id": self._session_counter,
            "created_at": created_at or now,
            "updated_at": updated_at or now,
            "messages": messages or [],
            "archive_path": archive_path,
            "summary": summary or [],
            "title": title,
            "assistant_tokens": int(assistant_tokens),
        }

    def _set_current_session(self, session, show_reset=False, sync_history=False):
        self._current_session = session
        self._active_assistant_index = None
        self._rewrite_assistant_index = None
        self._assistant_block_map = {}
        self._title_generated = bool(session.get("title"))
        self._render_session(session, show_reset=show_reset)
        if sync_history:
            history = self._build_engine_history_from_session()
            self.sig_sync_history.emit(history)
        self._notify_header_update()

    def _build_engine_history_from_session(self):
        """
        Convert _current_session["messages"] into engine-ready history.

        Always include:
          {"role": "system", "content": MASTER_PROMPT}

        Then append each session message as:
          {"role": msg["role"], "content": msg["text"]}

        Behavior tags are NOT included here.
        Engine recompiles system prompt at generation time.
        """
        history = [{"role": "system", "content": MASTER_PROMPT}]
        for msg in self._current_session.get("messages", []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            text = msg.get("text")
            if not isinstance(role, str) or not isinstance(text, str):
                continue
            history.append({"role": role, "content": text})
        return history

    def _render_session(self, session, show_reset=False):
        self.chat.clear()
        self.trace.clear()
        if not session["messages"]:
            if show_reset:
                self.chat.append(f"<span style='color:{FG_DIM}'>--- SESSION RESET ---</span>")
            return
        for idx, msg in enumerate(session["messages"]):
            safe = html.escape(msg.get("text", ""))
            if msg.get("role") == "user":
                self.chat.append(
                    f"<span style='color:white'><b>USER:</b></span> "
                    f"<span style='color:{FG_DIM}'>{safe}</span>"
                )
            else:
                self.chat.append(
                    f"<span style='color:{ACCENT_GOLD}'><b>ASSISTANT:</b></span> "
                    f"<span style='color:{FG_DIM}'>{safe}</span>"
                )
                block = self.chat.document().lastBlock().blockNumber()
                self._assistant_block_map[idx] = block
        self.chat.moveCursor(QTextCursor.End)

    def _apply_behavior_prompt(self, tags):
        cleaned = [tag.strip() for tag in tags if tag.strip()]
        self.config["behavior_tags"] = cleaned
        self._set_config_dirty(True)

    def _begin_update_trace(self, update_text):
        self._update_trace_state = "requested"
        self._update_token_count = 0
        self._update_progress_index = 0
        self.trace.append("⟐ UPDATE REQUESTED")
        self.trace.append(f'⟐ USER PATCH: "{update_text}"')

    def _start_update_streaming(self):
        self._update_trace_state = "streaming"
        self._update_token_count = 0
        self._update_progress_index = 0

    def _update_progress_markers(self):
        if self._update_trace_state != "streaming":
            return
        self._update_token_count += 1
        thresholds = [25, 50, 100]
        pct = min(
            100,
            int((self._update_token_count / self.config["max_tokens"]) * 100),
        )
        while self._update_progress_index < len(thresholds):
            if pct >= thresholds[self._update_progress_index]:
                percent = thresholds[self._update_progress_index]
                self.trace.append(f"⟐ UPDATE PROGRESS {percent}%")
                self._update_progress_index += 1
                continue
            break

    def _finalize_update_progress(self):
        if self._update_trace_state != "streaming":
            return
        thresholds = [25, 50, 100]
        while self._update_progress_index < len(thresholds):
            percent = thresholds[self._update_progress_index]
            self.trace.append(f"⟐ UPDATE PROGRESS {percent}%")
            self._update_progress_index += 1
        self._update_trace_state = None

    def _add_message(self, role, text):
        now = self._now_iso()
        message = {
            "i": len(self._current_session["messages"]) + 1,
            "time": now,
            "role": role,
            "text": text
        }
        self._current_session["messages"].append(message)
        self._current_session["updated_at"] = now
        return len(self._current_session["messages"]) - 1

    def _append_assistant_token(self, token):
        target_index = self._rewrite_assistant_index
        if target_index is None:
            target_index = self._active_assistant_index
        if target_index is None:
            target_index = self._add_message("assistant", "")
            self._active_assistant_index = target_index
        msg = self._current_session["messages"][target_index]
        msg["text"] += token
        msg["time"] = self._now_iso()
        self._current_session["updated_at"] = msg["time"]
        self._current_session["assistant_tokens"] = int(self._current_session.get("assistant_tokens", 0)) + 1


    def _maybe_generate_title(self):
        if self._title_generated:
            return
        assistant_msgs = [m for m in self._current_session["messages"] if m.get("role") == "assistant" and m.get("text", "").strip()]
        if len(assistant_msgs) < 2 and not self._topic_dominant():
            return
        title = self._derive_title(self._current_session["messages"])
        self._current_session["title"] = title
        self._title_generated = True
        self._notify_header_update()

    def _topic_dominant(self):
        user_text = " ".join([m.get("text", "") for m in self._current_session["messages"] if m.get("role") == "user"])
        words = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", user_text)]
        if not words:
            return False
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        return max(counts.values()) >= 3

    def _notify_header_update(self):
        dt = QDateTime.currentDateTime().toString("ddd • HH:mm")
        title = self._current_session.get("title") or self._derive_title(self._current_session.get("messages", []))
        self.ui_bridge.sig_terminal_header.emit(title, dt)

    def _derive_title(self, messages):
        for msg in messages:
            if msg.get("role") == "user":
                text = msg.get("text", "").strip()
                if text:
                    return text[:60]
        return "Untitled Chat"

    def _build_summary(self, messages):
        summary = []
        title = self._current_session.get("title") or self._derive_title(messages)
        summary.append(f"Title: {title}")
        user_msgs = [m["text"] for m in messages if m.get("role") == "user" and m.get("text")]
        assistant_msgs = [m["text"] for m in messages if m.get("role") == "assistant" and m.get("text")]

        def _trim(text, limit=120):
            return text if len(text) <= limit else f"{text[:limit]}…"

        for msg in user_msgs[-3:]:
            summary.append(f"User: {_trim(msg)}")
        for msg in assistant_msgs[-3:]:
            summary.append(f"Assistant: {_trim(msg)}")
        if len(summary) < 3:
            summary.append(f"Messages: {len(messages)}")
        if len(summary) < 3:
            summary.append("Summary: Not enough messages yet.")
        return summary[:6]

    def _slugify(self, text):
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug or "chat"

    def _now_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _get_archive_dir(self):
        return ARCHIVE_DIR
