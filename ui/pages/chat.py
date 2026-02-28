import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, BadZipFile

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel, QFileDialog,
    QSplitter, QListWidget, QListWidgetItem, QStackedWidget,
    QMessageBox, QButtonGroup, QMenu, QFrame, QScrollArea,
)
from PySide6.QtCore import Signal, Qt, QTimer, QDateTime, QEvent, QPoint
from PySide6.QtGui import QActionGroup, QKeyEvent

from core.state import SystemStatus
import core.style as _s  # dynamic theme bridge
from ui.components.atoms import MonoGroupBox, MonoButton, MonoSlider
from ui.components.complex import BehaviorTagInput
from ui.components.message_widget import MessageWidget
from core.llm_config import DEFAULT_CONFIG, build_system_prompt, load_config, save_config
from core.paths import ARCHIVE_DIR

# ---------------------------------------------------------------------------
# Slash-command registry
# ---------------------------------------------------------------------------
_SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/think",  "Toggle thinking mode (off / standard)"),
    ("/attach", "Attach a file to your message"),
    ("/clear",  "Clear the current chat"),
    ("/vision", "Open the Vision (image gen) module"),
    ("/audio",  "Open the Audio module"),
]

_ASSISTANT_CMD_RE = re.compile(
    r"<monolith_cmd>\s*(\{.*?\})\s*</monolith_cmd>",
    re.IGNORECASE | re.DOTALL,
)
_ASSISTANT_ALLOWED_ADDONS = {"sd", "audiogen", "databank", "terminal"}

# File extensions accepted for drag-and-drop / attach
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".html",
    ".htm", ".xml", ".rst", ".log", ".sh", ".bat", ".css", ".c",
    ".cpp", ".h", ".java", ".go", ".rs", ".rb", ".php", ".sql",
    ".env", ".gitignore", ".dockerfile", ".tf",
}


# ---------------------------------------------------------------------------
# _CommandPopup — floats above the text input, lists matching slash-commands
# ---------------------------------------------------------------------------
class _CommandPopup(QFrame):
    command_selected = Signal(str)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("cmd_popup")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self._list = QListWidget()
        self._list.setFocusPolicy(Qt.NoFocus)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        self._commands: list[tuple[str, str]] = []

    def populate(self, cmds: list[tuple[str, str]]) -> None:
        self._commands = cmds
        self._list.clear()
        for cmd, desc in cmds:
            self._list.addItem(f"{cmd}  —  {desc}")
        if self._list.count():
            self._list.setCurrentRow(0)
        rows = min(8, self._list.count())
        self.setFixedHeight(rows * 26 + 10)

    def move_selection(self, delta: int) -> None:
        row = max(0, min(self._list.count() - 1, self._list.currentRow() + delta))
        self._list.setCurrentRow(row)

    def accept_current(self) -> None:
        row = self._list.currentRow()
        if 0 <= row < len(self._commands):
            self.command_selected.emit(self._commands[row][0])
        self.hide()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        row = self._list.row(item)
        if 0 <= row < len(self._commands):
            self.command_selected.emit(self._commands[row][0])
        self.hide()

    def apply_theme(self) -> None:
        self.setStyleSheet(
            f"QFrame#cmd_popup {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_LIGHT}; "
            f"border-radius: 4px; }}"
        )
        self._list.setStyleSheet(
            f"QListWidget {{ background: transparent; border: none; color: {_s.FG_TEXT}; "
            f"font-family: Consolas; font-size: 10px; }}"
            f"QListWidget::item {{ padding: 3px 8px; border-radius: 2px; }}"
            f"QListWidget::item:selected {{ background: {_s.BG_BUTTON_HOVER}; "
            f"color: {_s.ACCENT_PRIMARY}; }}"
        )


# ---------------------------------------------------------------------------
# PageChat
# ---------------------------------------------------------------------------
class PageChat(QWidget):
    sig_generate = Signal(str, bool)
    sig_load = Signal()
    sig_unload = Signal()
    sig_stop = Signal()
    sig_sync_history = Signal(list)
    sig_set_model_path = Signal(str)
    sig_set_ctx_limit = Signal(int)
    sig_operator_loaded = Signal(str)
    sig_debug = Signal(str)
    sig_launch_addon = Signal(str)  # emitted by slash commands e.g. /vision → "sd"

    def __init__(self, state, ui_bridge):
        super().__init__()
        self.state = state
        self.ui_bridge = ui_bridge
        self.config = load_config()
        self._token_buf: deque[str] = deque()
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(8)
        self._flush_timer.timeout.connect(self._flush_tokens)
        self._archive_dir = self._get_archive_dir()
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._session_counter = 0
        self._current_session = self._create_session()
        self._undo_snapshot = None
        self._title_generated = False
        self._suppress_title_regen = False
        self._active_assistant_index = None
        self._rewrite_assistant_index = None
        self._active_widget: MessageWidget | None = None
        self._last_status = None
        self._is_running = False
        self._is_model_loaded = False
        self._pending_update_text = None
        self._awaiting_update_restart = False
        self._update_trace_state = None
        self._update_token_count = 0
        self._update_progress_index = 0
        self._config_dirty = False
        self._thinking_mode = bool(self.config.get("thinking_mode", False))
        self._pending_mutation = None  # type: ignore[assignment]

        # File attachment state
        self._attached_files: list[dict] = []  # each: {name, content, icon}

        # Right-click double-click tracking for archive list
        self._archive_last_right_click: float = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        main_split = QSplitter(Qt.Horizontal)
        main_split.setChildrenCollapsible(False)
        layout.addWidget(main_split)

        # === MODEL LOADER (lives in MODEL tab) ===
        grp_load = MonoGroupBox("MODEL LOADER")
        self.path_display = QLineEdit()
        self.path_display.setReadOnly(True)
        self.path_display.setPlaceholderText("No GGUF Selected")
        self.path_display.setStyleSheet(
            f"background: {_s.BG_INPUT}; color: {_s.FG_PLACEHOLDER}; border: 1px solid {_s.BORDER_LIGHT}; padding: 5px;"
        )
        btn_browse = MonoButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self.pick_file)
        row_file = QHBoxLayout()
        row_file.addWidget(self.path_display)
        row_file.addWidget(btn_browse)
        self.btn_load = MonoButton("LOAD MODEL")
        self.btn_load.clicked.connect(self.toggle_load)
        grp_load.add_layout(row_file)
        grp_load.add_widget(self.btn_load)

        # === AI CONFIGURATION (lives in CONFIG tab) ===
        self.s_temp = MonoSlider("Temperature", 0.1, 2.0, self.config.get("temp", 0.7))
        self.s_temp.valueChanged.connect(lambda v: self._update_config_value("temp", v))
        self.s_top = MonoSlider("Top-P", 0.1, 1.0, self.config.get("top_p", 0.9))
        self.s_top.valueChanged.connect(lambda v: self._update_config_value("top_p", v))
        self.s_tok = MonoSlider(
            "Max Tokens", 512, 8192, self.config.get("max_tokens", 2048), is_int=True
        )
        self.s_tok.valueChanged.connect(
            lambda v: self._update_config_value("max_tokens", int(v))
        )
        self.s_ctx = MonoSlider(
            "Context Limit", 1024, 16384, self.config.get("ctx_limit", 8192), is_int=True
        )
        self.s_ctx.valueChanged.connect(self._on_ctx_limit_changed)

        save_row = QHBoxLayout()
        self.lbl_config_state = QLabel("SAVED")
        self.lbl_config_state.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px; font-weight: bold;")
        self.btn_save_config = MonoButton("SAVE SETTINGS")
        self.btn_save_config.clicked.connect(self._save_config)
        btn_reset_config = MonoButton("RESET")
        btn_reset_config.clicked.connect(self._reset_config)
        save_row.addWidget(self.lbl_config_state)
        save_row.addStretch()
        save_row.addWidget(btn_reset_config)
        save_row.addWidget(self.btn_save_config)

        # === TOOLS GROUP with 2 tabs ===
        operations_group = MonoGroupBox("TOOLS")
        operations_layout = QVBoxLayout()
        operations_layout.setSpacing(10)

        tab_row = QHBoxLayout()
        tab_style = f"""
            QPushButton {{
                background: {_s.BG_BUTTON}; border: 1px solid {_s.BORDER_LIGHT}; color: {_s.FG_DIM};
                padding: 6px 12px; font-size: 10px; font-weight: bold; border-radius: 2px;
            }}
            QPushButton:checked {{
                background: {_s.BG_BUTTON_HOVER}; color: {_s.ACCENT_PRIMARY}; border: 1px solid {_s.ACCENT_PRIMARY};
            }}
            QPushButton:hover {{ color: {_s.FG_TEXT}; border: 1px solid {_s.FG_TEXT}; }}
        """
        self.btn_tab_control = MonoButton("MODEL")
        self.btn_tab_control.setCheckable(True)
        self.btn_tab_control.setChecked(True)
        self.btn_tab_control.setStyleSheet(tab_style)
        self.btn_tab_settings = MonoButton("CONFIG")
        self.btn_tab_settings.setCheckable(True)
        self.btn_tab_settings.setStyleSheet(tab_style)
        tab_group = QButtonGroup(self)
        tab_group.setExclusive(True)
        tab_group.addButton(self.btn_tab_control)
        tab_group.addButton(self.btn_tab_settings)
        tab_row.addWidget(self.btn_tab_control)
        tab_row.addWidget(self.btn_tab_settings)
        tab_row.addStretch()
        operations_layout.addLayout(tab_row)

        self.ops_stack = QStackedWidget()
        operations_layout.addWidget(self.ops_stack)

        # --- MODEL tab ---
        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)
        control_layout.setSpacing(12)
        control_layout.addWidget(grp_load)

        self._options_expanded = False
        self.btn_options_toggle = QPushButton("▸ ADVANCED")
        self.btn_options_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_options_toggle.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                color: {_s.FG_DIM}; font-size: 9px; font-weight: bold;
                letter-spacing: 1px; text-align: left; padding: 4px 0;
            }}
            QPushButton:hover {{ color: {_s.ACCENT_PRIMARY}; }}
        """)
        self.btn_options_toggle.clicked.connect(self._toggle_options_panel)
        control_layout.addWidget(self.btn_options_toggle)

        self.options_panel = QWidget()
        self.options_panel.setVisible(False)
        options_layout = QVBoxLayout(self.options_panel)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(8)
        control_layout.addWidget(self.options_panel)
        control_layout.addStretch()

        # --- HISTORY group ---
        history_group = MonoGroupBox("HISTORY")
        history_layout = QVBoxLayout()
        history_layout.setSpacing(8)

        self.archive_list = QListWidget()
        self.archive_list.setStyleSheet(f"""
            QListWidget {{
                background: {_s.BG_INPUT}; color: {_s.FG_TEXT}; border: 1px solid {_s.BORDER_SUBTLE};
                font-family: 'Consolas', monospace; font-size: 10px;
            }}
            QListWidget::item {{ padding: 6px; }}
            QListWidget::item:selected {{ background: {_s.BG_BUTTON_HOVER}; color: {_s.ACCENT_PRIMARY}; }}
            {_s.SCROLLBAR_STYLE}
        """)
        # Use event filter for distinguishing left vs right double-click
        self.archive_list.viewport().installEventFilter(self)

        history_btn_row = QHBoxLayout()
        history_btn_row.setSpacing(6)
        btn_clear_history = MonoButton("CLEAR CHAT")
        btn_clear_history.setFixedHeight(24)
        btn_clear_history.clicked.connect(lambda: self._clear_current_session(delete_archive=False))
        btn_delete_history = MonoButton("DELETE")
        btn_delete_history.setFixedHeight(24)
        btn_delete_history.clicked.connect(self._delete_selected_archive)
        history_btn_row.addWidget(btn_clear_history)
        history_btn_row.addWidget(btn_delete_history)
        history_btn_row.addStretch()

        history_layout.addWidget(self.archive_list)
        history_layout.addLayout(history_btn_row)

        self.lbl_behavior = QLabel("BEHAVIOR TAGS")
        self.lbl_behavior.setStyleSheet(
            f"color: {_s.FG_INFO}; font-size: 8px; font-weight: bold; letter-spacing: 1px;"
        )
        self.behavior_tags = BehaviorTagInput([])
        self.behavior_tags.tagsChanged.connect(self._on_behavior_tags_changed)
        self.behavior_tags.setStyleSheet(
            f"background: {_s.BG_SIDEBAR}; border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 2px;"
        )
        self.behavior_tags.setMaximumHeight(36)

        # --- CONFIG tab ---
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setSpacing(10)
        settings_layout.addWidget(self.s_temp)
        settings_layout.addWidget(self.s_top)
        settings_layout.addWidget(self.s_tok)
        settings_layout.addWidget(self.s_ctx)
        settings_layout.addWidget(self.lbl_behavior)
        settings_layout.addWidget(self.behavior_tags)
        settings_layout.addLayout(save_row)
        settings_layout.addStretch()

        self.ops_stack.addWidget(control_tab)
        self.ops_stack.addWidget(settings_tab)
        self.btn_tab_control.toggled.connect(lambda checked: self._switch_ops_tab(0, checked))
        self.btn_tab_settings.toggled.connect(lambda checked: self._switch_ops_tab(1, checked))

        operations_group.add_layout(operations_layout)
        history_group.add_layout(history_layout)

        # ----------------------------------------------------------------
        # CHAT panel (left side)
        # ----------------------------------------------------------------
        chat_group = MonoGroupBox("CHAT")
        chat_group.add_header_action("CLEAR", lambda: self._clear_current_session(delete_archive=False))
        chat_layout = QVBoxLayout()
        chat_layout.setSpacing(8)

        self.message_list = QListWidget()
        self.message_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.message_list.setStyleSheet(f"""
            QListWidget {{
                background: transparent; color: {_s.FG_TEXT}; border: 1px solid {_s.BORDER_SUBTLE};
                font-family: 'Consolas', monospace; font-size: 12px;
            }}
            QListWidget::item {{
                border: none;
                background: transparent;
                padding: 0px;
            }}
            QListWidget::item:selected {{ background: transparent; border: none; }}
            QListWidget::item:focus {{ background: transparent; border: none; outline: none; }}
            {_s.SCROLLBAR_STYLE}
        """)
        self.message_list.setSelectionMode(QListWidget.NoSelection)
        self.message_list.setFocusPolicy(Qt.NoFocus)
        chat_layout.addWidget(self.message_list, 1)

        # --- File attachment chips (shown when files are pending) ---
        self._chips_container = QWidget()
        self._chips_container.hide()
        self._chips_layout = QHBoxLayout(self._chips_container)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(6)
        self._chips_layout.addStretch()
        chat_layout.addWidget(self._chips_container)

        # --- Input row ---
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Message… or / for commands")
        self.text_input.setFixedHeight(38)   # starts at ~1 line; grows as you type
        self.text_input.setAcceptRichText(False)
        self.text_input.setAcceptDrops(True)
        self.text_input.textChanged.connect(self._on_text_input_changed)
        self.text_input.installEventFilter(self)
        self.text_input.setStyleSheet(f"""
            QTextEdit {{
                background: {_s.BG_INPUT}; color: white; border: 1px solid {_s.BORDER_LIGHT};
                padding: 8px; font-family: 'Verdana'; font-size: 11px;
            }}
            QTextEdit:focus {{ border: 1px solid {_s.ACCENT_PRIMARY}; }}
            {_s.SCROLLBAR_STYLE}
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
        self._set_send_button_state(is_running=False)
        self.btn_send.clicked.connect(self.handle_send_click)

        input_row.addWidget(self.text_input)
        input_row.addWidget(self.btn_send)
        chat_layout.addLayout(input_row)

        chat_group.add_layout(chat_layout)

        right_stack = QSplitter(Qt.Vertical)
        right_stack.setChildrenCollapsible(False)

        self.trace = QTextEdit()
        self.trace.hide()

        right_stack.addWidget(history_group)
        right_stack.addWidget(operations_group)
        right_stack.setStretchFactor(0, 1)
        right_stack.setStretchFactor(1, 1)
        right_stack.setSizes([200, 200])

        main_split.addWidget(chat_group)
        main_split.addWidget(right_stack)
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 2)
        main_split.widget(1).setMaximumWidth(500)
        main_split.setSizes([900, 200])
        self._active_assistant_started = False
        self._active_assistant_token_count = 0

        # --- Command popup (child of this widget, positioned dynamically) ---
        self._cmd_popup = _CommandPopup(self)
        self._cmd_popup.command_selected.connect(self._handle_command)
        self._cmd_popup.apply_theme()

        self.setAcceptDrops(True)

        self._sync_path_display()
        self._update_load_button_text()
        self._refresh_archive_list()
        self._apply_behavior_prompt(self.config.get("behavior_tags", []))
        self.behavior_tags.set_tags(self.config.get("behavior_tags", []))
        self._set_config_dirty(False)
        if not self._is_model_loaded:
            self._apply_default_limits()

        if self.ui_bridge is not None:
            self.ui_bridge.sig_theme_changed.connect(lambda _: self._refresh_widget_styles())

    # -----------------------------------------------------------------------
    # Theme refresh
    # -----------------------------------------------------------------------
    def _refresh_widget_styles(self) -> None:
        """Re-apply all inline widget styles after a theme change."""
        self.path_display.setStyleSheet(
            f"background: {_s.BG_INPUT}; color: {_s.FG_PLACEHOLDER}; border: 1px solid {_s.BORDER_LIGHT}; padding: 5px;"
        )
        self.lbl_config_state.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px; font-weight: bold;")
        tab_style = f"""
            QPushButton {{
                background: {_s.BG_BUTTON}; border: 1px solid {_s.BORDER_LIGHT}; color: {_s.FG_DIM};
                padding: 6px 12px; font-size: 10px; font-weight: bold; border-radius: 2px;
            }}
            QPushButton:checked {{
                background: {_s.BG_BUTTON_HOVER}; color: {_s.ACCENT_PRIMARY}; border: 1px solid {_s.ACCENT_PRIMARY};
            }}
            QPushButton:hover {{ color: {_s.FG_TEXT}; border: 1px solid {_s.FG_TEXT}; }}
        """
        self.btn_tab_control.setStyleSheet(tab_style)
        self.btn_tab_settings.setStyleSheet(tab_style)
        self.btn_options_toggle.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                color: {_s.FG_DIM}; font-size: 9px; font-weight: bold;
                letter-spacing: 1px; text-align: left; padding: 4px 0;
            }}
            QPushButton:hover {{ color: {_s.ACCENT_PRIMARY}; }}
        """)
        self.archive_list.setStyleSheet(f"""
            QListWidget {{
                background: {_s.BG_INPUT}; color: {_s.FG_TEXT}; border: 1px solid {_s.BORDER_SUBTLE};
                font-family: 'Consolas', monospace; font-size: 10px;
            }}
            QListWidget::item {{ padding: 6px; }}
            QListWidget::item:selected {{ background: {_s.BG_BUTTON_HOVER}; color: {_s.ACCENT_PRIMARY}; }}
            {_s.SCROLLBAR_STYLE}
        """)
        self.lbl_behavior.setStyleSheet(
            f"color: {_s.FG_INFO}; font-size: 8px; font-weight: bold; letter-spacing: 1px;"
        )
        self.behavior_tags.setStyleSheet(
            f"background: {_s.BG_SIDEBAR}; border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 2px;"
        )
        self.message_list.setStyleSheet(f"""
            QListWidget {{
                background: transparent; color: {_s.FG_TEXT}; border: 1px solid {_s.BORDER_SUBTLE};
                font-family: 'Consolas', monospace; font-size: 12px;
            }}
            QListWidget::item {{
                border: none;
                background: transparent;
                padding: 0px;
            }}
            QListWidget::item:selected {{ background: transparent; border: none; }}
            QListWidget::item:focus {{ background: transparent; border: none; outline: none; }}
            {_s.SCROLLBAR_STYLE}
        """)
        self.text_input.setStyleSheet(f"""
            QTextEdit {{
                background: {_s.BG_INPUT}; color: white; border: 1px solid {_s.BORDER_LIGHT};
                padding: 8px; font-family: 'Verdana'; font-size: 11px;
            }}
            QTextEdit:focus {{ border: 1px solid {_s.ACCENT_PRIMARY}; }}
            {_s.SCROLLBAR_STYLE}
        """)
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
        self._set_send_button_state(is_running=self._is_running)
        self._cmd_popup.apply_theme()
        self._update_file_chips()

    # -----------------------------------------------------------------------
    # Command system + input auto-grow
    # -----------------------------------------------------------------------
    _INPUT_MIN_H = 38
    _INPUT_MAX_H = 160

    def _on_text_input_changed(self) -> None:
        self._grow_input()
        self._on_input_changed(self.text_input.toPlainText())

    def _grow_input(self) -> None:
        """Resize text_input to fit its content, clamped to [_INPUT_MIN_H, _INPUT_MAX_H]."""
        doc_h = self.text_input.document().documentLayout().documentSize().height()
        fw = self.text_input.frameWidth()
        # 16 = 8px top + 8px bottom padding from stylesheet
        needed = int(doc_h) + fw * 2 + 16
        new_h = max(self._INPUT_MIN_H, min(self._INPUT_MAX_H, needed))
        if self.text_input.height() != new_h:
            self.text_input.setFixedHeight(new_h)
            if self._cmd_popup.isVisible():
                self._reposition_popup()

    def _on_input_changed(self, text: str) -> None:
        if self._is_running:
            self._set_send_button_state(is_running=True)

        stripped = text.lstrip()
        if stripped.startswith("/"):
            token = stripped.split()[0] if stripped.split() else "/"
            matches = [(cmd, desc) for cmd, desc in _SLASH_COMMANDS if cmd.startswith(token)]
            if matches:
                self._cmd_popup.populate(matches)
                self._reposition_popup()
                self._cmd_popup.show()
                self._cmd_popup.raise_()
                return
        self._cmd_popup.hide()

    def _reposition_popup(self) -> None:
        pos = self.text_input.mapTo(self, QPoint(0, 0))
        w = self.text_input.width()
        h = self._cmd_popup.height()
        self._cmd_popup.setFixedWidth(w)
        self._cmd_popup.move(pos.x(), pos.y() - h - 4)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._cmd_popup.isVisible():
            self._reposition_popup()

    def _handle_command(self, cmd: str) -> None:
        """Execute a slash command selected from the popup or typed directly."""
        self.text_input.clear()
        self._cmd_popup.hide()

        if cmd == "/think":
            self._set_thinking_mode(not self._thinking_mode)
            mode_label = "ON (Standard)" if self._thinking_mode else "OFF"
            self._trace_html(f"Think mode → {mode_label}", "CMD")

        elif cmd == "/clear":
            self._clear_current_session(delete_archive=False)

        elif cmd == "/attach":
            self._open_attach_dialog()

        elif cmd == "/vision":
            self.sig_launch_addon.emit("sd")

        elif cmd == "/audio":
            self.sig_launch_addon.emit("audiogen")

    # -----------------------------------------------------------------------
    # File attachment
    # -----------------------------------------------------------------------
    def _open_attach_dialog(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(_TEXT_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Attach File", "",
            f"Text & data files ({exts});;ZIP archives (*.zip);;All files (*.*)"
        )
        if path:
            self._attach_file(path)

    def _attach_file(self, path: str) -> None:
        p = Path(path)
        ext = p.suffix.lower()

        if ext == ".zip":
            try:
                with ZipFile(p) as zf:
                    names = zf.namelist()
                    readable = [n for n in names if Path(n).suffix.lower() in _TEXT_EXTENSIONS]
                    parts = [f"[ZIP: {p.name}]", f"Contents: {len(names)} files"]
                    for name in readable[:20]:
                        try:
                            content = zf.read(name).decode("utf-8", errors="replace")
                            parts.append(f"\n--- {name} ---\n{content[:4000]}")
                        except Exception:
                            parts.append(f"[binary: {name}]")
                    full = "\n".join(parts)
                    icon = "📦"
            except BadZipFile:
                return
        elif ext in _TEXT_EXTENSIONS:
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return
            full = f"[FILE: {p.name}]\n{content[:8000]}"
            icon = "📄"
        else:
            # Try reading as text anyway
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                full = f"[FILE: {p.name}]\n{content[:8000]}"
                icon = "📄"
            except Exception:
                return

        self._attached_files.append({"name": p.name, "content": full, "icon": icon})
        self._update_file_chips()

    def _remove_attached_file(self, index: int) -> None:
        if 0 <= index < len(self._attached_files):
            self._attached_files.pop(index)
        self._update_file_chips()

    def _update_file_chips(self) -> None:
        # Remove all widgets except the stretch
        while self._chips_layout.count() > 1:
            item = self._chips_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        for i, f in enumerate(self._attached_files):
            chip = QFrame()
            chip.setStyleSheet(
                f"QFrame {{ background: {_s.BG_BUTTON}; border: 1px solid {_s.BORDER_LIGHT}; "
                f"border-radius: 3px; padding: 2px 4px; }}"
            )
            row = QHBoxLayout(chip)
            row.setContentsMargins(4, 2, 4, 2)
            row.setSpacing(4)
            icon_lbl = QLabel(f["icon"])
            icon_lbl.setStyleSheet(f"border: none; background: transparent; font-size: 11px; color: {_s.FG_TEXT};")
            name_lbl = QLabel(f["name"])
            name_lbl.setStyleSheet(f"border: none; background: transparent; font-size: 10px; color: {_s.FG_TEXT};")
            btn_x = QPushButton("×")
            btn_x.setFixedSize(16, 16)
            btn_x.setStyleSheet(
                f"QPushButton {{ background: transparent; border: none; color: {_s.FG_DIM}; "
                f"font-size: 12px; font-weight: bold; padding: 0; }}"
                f"QPushButton:hover {{ color: {_s.FG_ERROR}; }}"
            )
            btn_x.clicked.connect(lambda _, idx=i: self._remove_attached_file(idx))
            row.addWidget(icon_lbl)
            row.addWidget(name_lbl)
            row.addWidget(btn_x)
            self._chips_layout.insertWidget(self._chips_layout.count() - 1, chip)

        self._chips_container.setVisible(bool(self._attached_files))

    def _build_send_text(self) -> str:
        """Assemble final text from attached files + typed text."""
        typed = self.text_input.toPlainText().strip()
        if not self._attached_files:
            return typed
        parts = []
        for f in self._attached_files:
            parts.append(f["icon"] + " " + f["content"])
        if typed:
            parts.append(typed)
        return "\n\n".join(parts)

    # -----------------------------------------------------------------------
    # Drag and drop
    # -----------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path:
                    self._attach_file(path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    # -----------------------------------------------------------------------
    # Send flow
    # -----------------------------------------------------------------------
    def send(self):
        txt = self._build_send_text()
        if not txt:
            return
        self.sig_debug.emit(f"[CHAT] send: text={repr(txt[:60])}, msgs={len(self._current_session['messages'])}")
        self._set_send_button_state(is_running=True)
        self.text_input.clear()
        # Clear attachments
        self._attached_files.clear()
        self._update_file_chips()

        user_idx = self._add_message("user", txt)
        self._append_message_widget(user_idx)
        # Auto-save with the new user message
        try:
            self._save_chat_archive()
        except Exception:
            pass
        self._start_assistant_stream()
        self.message_list.scrollToBottom()
        self.sig_debug.emit(f"[CHAT] about to emit sig_generate: txt={repr(txt[:60])}")
        self.sig_generate.emit(txt, self._thinking_mode)
        self.sig_debug.emit(f"[CHAT] sig_generate emitted")

    def handle_send_click(self):
        txt = self.text_input.toPlainText().strip()

        if not self._is_running:
            self.send()
            return

        if not txt and not self._attached_files:
            self._set_send_button_state(is_running=True, stopping=True)
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
            has_input = bool(self.text_input.toPlainText().strip())
            if has_input:
                self.btn_send.setText("UPDATE")
                color = _s.ACCENT_PRIMARY
            else:
                self.btn_send.setText("■")
                color = _s.FG_ERROR
            self.btn_send.setStyleSheet(
                self._btn_style_template.format(bg=_s.BG_INPUT, color=color)
            )
            self.btn_send.setEnabled(not stopping)
        else:
            self.btn_send.setText("SEND")
            self.btn_send.setStyleSheet(
                self._btn_style_template.format(bg=_s.BG_INPUT, color=_s.ACCENT_PRIMARY)
            )
            self.btn_send.setEnabled(True)

    # -----------------------------------------------------------------------
    # Event filter — keyboard nav in popup, drag-drop on text_input
    # -----------------------------------------------------------------------
    def eventFilter(self, source, event):
        # Archive list: left double-click = open, right double-click = delete
        if source is self.archive_list.viewport():
            if event.type() == QEvent.MouseButtonDblClick:
                if event.button() == Qt.LeftButton:
                    self._load_chat_archive()
                    return True
                if event.button() == Qt.RightButton:
                    self._delete_selected_archive()
                    return True

        if hasattr(self, "text_input") and source is self.text_input:
            etype = event.type()

            # Keyboard events — command popup navigation
            if etype == QEvent.KeyPress:
                if self._cmd_popup.isVisible():
                    if event.key() == Qt.Key_Up:
                        self._cmd_popup.move_selection(-1)
                        return True
                    if event.key() == Qt.Key_Down:
                        self._cmd_popup.move_selection(1)
                        return True
                    if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab):
                        self._cmd_popup.accept_current()
                        return True
                    if event.key() == Qt.Key_Escape:
                        self._cmd_popup.hide()
                        return True

                if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not event.modifiers() & Qt.ShiftModifier:
                    self.handle_send_click()
                    return True

            # Drag-and-drop on text_input — intercept before QTextEdit pastes as text
            if etype == QEvent.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True

            if etype == QEvent.Drop:
                if event.mimeData().hasUrls():
                    for url in event.mimeData().urls():
                        path = url.toLocalFile()
                        if path:
                            self._attach_file(path)
                    event.acceptProposedAction()
                    return True

        return super().eventFilter(source, event)

    def _send_message(self, text):
        self.text_input.setPlainText(text)
        self.send()

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

Continue from the interruption point. Do not repeat earlier content. Prioritize the user update.
"""
        self.text_input.clear()
        self._start_update_streaming()
        self.sig_generate.emit(injected, self._thinking_mode)

    def _start_assistant_stream(self):
        self.sig_debug.emit(f"[CHAT] _start_assistant_stream: msgs_before={len(self._current_session['messages'])}")
        self._active_assistant_started = True
        self._active_assistant_token_count = 0
        self._active_assistant_index = self._add_message("assistant", "")
        self._active_widget = self._append_message_widget(self._active_assistant_index)

    def _flush_tokens(self, force_all: bool = False):
        if not self._token_buf:
            self._flush_timer.stop()
            return
        chunk = "".join(self._token_buf) if force_all else self._token_buf.popleft()
        if force_all:
            self._token_buf.clear()
        if self._active_widget is None:
            target_index = self._rewrite_assistant_index
            if target_index is None:
                target_index = self._active_assistant_index
            if target_index is not None:
                self._active_widget = self._widget_for_index(target_index)
        if self._active_widget is None:
            return
        sb = self.message_list.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 40

        self._active_widget.append_token(chunk)
        vw = self.message_list.viewport().width()
        for row in range(self.message_list.count()):
            item = self.message_list.item(row)
            widget = self.message_list.itemWidget(item)
            if widget is self._active_widget:
                if vw > 50:
                    widget.setFixedWidth(vw)
                item.setSizeHint(widget.sizeHint())
                break
        if at_bottom:
            self.message_list.scrollToBottom()
        if not self._token_buf:
            self._flush_timer.stop()

    def append_token(self, t):
        self._token_buf.extend(str(t))
        self._append_assistant_token(t)
        self._update_progress_markers()
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def on_guard_finished(self, engine_key, task_id):
        if engine_key != getattr(self, "_engine_key", "llm"):
            return
        if not self._current_session.get("messages"):
            return
        try:
            self._flush_tokens(force_all=True)
        except Exception:
            pass
        try:
            self._parse_assistant_commands()
        except Exception as exc:
            self._trace_html(f"assistant command parse error: {exc}", "CMD", error=True)
        try:
            self._save_chat_archive()
        except Exception:
            pass

    def append_trace(self, trace_msg):
        lowered = trace_msg.lower()
        skip_patterns = [
            "guard", "dispatch", "route", "bridge", "dock",
            "addon", "registry", "host", "mount",
        ]
        for pat in skip_patterns:
            if pat in lowered and "error" not in lowered:
                return

        if "system online" in lowered:
            self._is_model_loaded = True
        elif "model unloaded" in lowered:
            self._is_model_loaded = False

        if "error" in lowered:
            state = "ERROR"
        elif "token" in lowered:
            state = "TOKENIZING"
        elif "inference started" in lowered:
            state = "INFERENCE"
        elif "inference" in lowered and ("complete" in lowered or "aborted" in lowered):
            state = "COMPLETE"
        elif "init backend" in lowered or "system online" in lowered:
            state = "MODEL"
        elif "unload" in lowered or "cancel" in lowered:
            state = "MODEL"
        elif "ctx" in lowered or "context" in lowered:
            state = "CTX"
        else:
            state = "INFO"

        display_msg = trace_msg
        if "→" in display_msg:
            display_msg = display_msg[display_msg.index("→") + 1:].strip()
        elif display_msg.startswith("ERROR"):
            display_msg = display_msg.replace("ERROR:", "").strip()

        self._trace_html(display_msg, state, error=(state == "ERROR"))

    def clear_chat(self):
        self._set_current_session(self._create_session(), show_reset=True, sync_history=True)

    def _sync_path_display(self):
        gguf_path = self.config.get("gguf_path")
        if gguf_path:
            self.path_display.setText(gguf_path)
            self.path_display.setToolTip(str(gguf_path))
        else:
            self.path_display.clear()
            self.path_display.setToolTip("")

    def _set_config_dirty(self, dirty=True):
        self._config_dirty = dirty
        self.lbl_config_state.setText("UNSAVED" if dirty else "SAVED")
        self.lbl_config_state.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY if dirty else _s.FG_DIM}; font-size: 10px; font-weight: bold;"
        )
        if dirty:
            self.btn_save_config.setStyleSheet(f"""
                QPushButton {{ background: {_s.BG_BUTTON}; border: 1px solid {_s.ACCENT_PRIMARY}; color: {_s.ACCENT_PRIMARY}; padding: 6px 12px; font-size: 11px; font-weight: bold; border-radius: 2px; }}
                QPushButton:hover {{ background: {_s.ACCENT_PRIMARY}; color: black; }}
                QPushButton:pressed {{ background: {_s.ACCENT_PRIMARY_DARK}; color: black; }}
            """)
        else:
            self.btn_save_config.setStyleSheet(f"""
                QPushButton {{ background: {_s.BG_BUTTON}; border: 1px solid {_s.BORDER_LIGHT}; color: {_s.FG_DIM}; padding: 6px 12px; font-size: 11px; font-weight: bold; border-radius: 2px; }}
                QPushButton:hover {{ background: {_s.BG_BUTTON_HOVER}; color: {_s.FG_DIM}; }}
            """)

    def _save_config(self):
        save_config(self.config)
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
        self._set_slider_limits(self.s_ctx, DEFAULT_CONFIG["ctx_limit"], DEFAULT_CONFIG["ctx_limit"])
        self._set_slider_limits(self.s_tok, DEFAULT_CONFIG["max_tokens"], DEFAULT_CONFIG["max_tokens"])

    def _trace_html(self, msg, tag="INFO", error=False):
        arrow_color = _s.FG_ERROR if error else _s.ACCENT_PRIMARY
        tag_color = _s.FG_ERROR if error else _s.FG_PLACEHOLDER
        self.trace.append(
            f"<table width='100%' cellpadding='0' cellspacing='0'><tr>"
            f"<td><span style='color:{arrow_color}'>→</span> {msg}</td>"
            f"<td align='right' style='color:{tag_color}; white-space:nowrap'>[{tag}]</td>"
            f"</tr></table>"
        )

    def _trace_plain(self, msg):
        self.trace.append(f"<span style='color:{_s.FG_PLACEHOLDER}'>{msg}</span>")

    def _on_model_capabilities(self, payload):
        model_ctx_length = payload.get("model_ctx_length")
        if model_ctx_length is None:
            self._apply_default_limits()
            return
        configured_ctx = int(self.config.get("ctx_limit", 8192))
        self._set_slider_limits(self.s_ctx, model_ctx_length, model_ctx_length)
        self._set_slider_limits(self.s_tok, model_ctx_length, min(8192, model_ctx_length))
        if configured_ctx < model_ctx_length:
            pct = int((configured_ctx / model_ctx_length) * 100)
            self._trace_html(
                f"Context: {configured_ctx:,} / {model_ctx_length:,} tokens ({pct}% of model capacity)", "CTX"
            )
            self._trace_html(
                f"Increase context limit in CONFIG to use full {model_ctx_length:,} capacity", "CTX"
            )
        else:
            self._trace_html(f"Context: {model_ctx_length:,} tokens (full capacity)", "CTX")

    def _on_ctx_limit_changed(self, value):
        self._update_config_value("ctx_limit", int(value))
        self.sig_set_ctx_limit.emit(int(value))

    def _on_behavior_tags_changed(self, tags):
        self._apply_behavior_prompt(tags)

    def _on_thinking_mode_toggled(self, checked):
        self._thinking_mode = bool(checked)
        self.config["thinking_mode"] = self._thinking_mode
        self._set_config_dirty(True)

    def _set_thinking_mode(self, enabled, label=""):
        self._thinking_mode = enabled
        self.config["thinking_mode"] = enabled
        self._set_config_dirty(True)

    def _toggle_options_panel(self):
        self._options_expanded = not self._options_expanded
        self.options_panel.setVisible(self._options_expanded)
        self.btn_options_toggle.setText("▾ ADVANCED" if self._options_expanded else "▸ ADVANCED")

    def _reset_config(self):
        for key, val in DEFAULT_CONFIG.items():
            self.config[key] = val
        self.s_temp.slider.blockSignals(True)
        self.s_top.slider.blockSignals(True)
        self.s_tok.slider.blockSignals(True)
        self.s_ctx.slider.blockSignals(True)
        self.s_temp.slider.setValue(int(DEFAULT_CONFIG["temp"] * 100))
        self.s_temp.val_lbl.setText(f"{DEFAULT_CONFIG['temp']:.2f}")
        self.s_top.slider.setValue(int(DEFAULT_CONFIG["top_p"] * 100))
        self.s_top.val_lbl.setText(f"{DEFAULT_CONFIG['top_p']:.2f}")
        self.s_tok.slider.setValue(int(DEFAULT_CONFIG["max_tokens"]))
        self.s_tok.val_lbl.setText(str(int(DEFAULT_CONFIG["max_tokens"])))
        self.s_ctx.slider.setValue(int(DEFAULT_CONFIG["ctx_limit"]))
        self.s_ctx.val_lbl.setText(str(int(DEFAULT_CONFIG["ctx_limit"])))
        self.s_temp.slider.blockSignals(False)
        self.s_top.slider.blockSignals(False)
        self.s_tok.slider.blockSignals(False)
        self.s_ctx.slider.blockSignals(False)
        self.sig_set_ctx_limit.emit(int(DEFAULT_CONFIG["ctx_limit"]))
        self.behavior_tags.set_tags(DEFAULT_CONFIG.get("behavior_tags", []))
        self._set_thinking_mode(False)
        self._set_config_dirty(True)

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF", "", "GGUF (*.gguf)")
        if path:
            self.config["gguf_path"] = path
            self.sig_set_model_path.emit(path)
            self._sync_path_display()
            self._set_config_dirty(True)

    def toggle_load(self):
        if self._is_model_loaded:
            self.sig_unload.emit()
        else:
            self.sig_load.emit()

    def _update_load_button_text(self):
        self.btn_load.setText("UNLOAD MODEL" if self._is_model_loaded else "LOAD MODEL")

    def _request_mutation(self, fn):
        if self._is_running:
            self._awaiting_update_restart = False
            self._pending_update_text = None
            self._pending_mutation = fn
            self._set_send_button_state(is_running=True, stopping=True)
            self.sig_stop.emit()
            return
        fn()

    def update_status(self, engine_key: str, status: SystemStatus):
        ek_short = engine_key[-8:] if engine_key else "?"
        prev = getattr(self, "_last_status", None)
        transition = f"{prev.name if prev else '?'}→{status.name}" if hasattr(status, "name") else str(status)
        self.sig_debug.emit(f"[CHAT:{ek_short}] status {transition}, running={self._is_running}, mutation={'yes' if self._pending_mutation else 'no'}")
        if engine_key != getattr(self, "_engine_key", "llm"):
            return
        is_processing = status in (SystemStatus.LOADING, SystemStatus.RUNNING, SystemStatus.UNLOADING)
        self.btn_load.setEnabled(not is_processing)
        if is_processing:
            self.btn_load.setText("PROCESSING...")
        else:
            self._update_load_button_text()
        if status == SystemStatus.READY and self._pending_mutation is not None:
            self._set_send_button_state(is_running=False)
            self._rewrite_assistant_index = None
            if self._token_buf:
                self._flush_tokens(force_all=True)
            if self._active_widget is not None:
                self._active_widget.finalize()
            self._active_widget = None
            if self._update_trace_state == "streaming":
                self._finalize_update_progress()
            self._cleanup_empty_assistant_if_needed()
            self._active_assistant_started = False
            self._active_assistant_token_count = 0
            self._suppress_title_regen = False
            pending = self._pending_mutation
            self._pending_mutation = None
            try:
                pending()
            finally:
                self._last_status = status
            return

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
            if self._last_status == SystemStatus.LOADING:
                self._is_model_loaded = True
                self._update_load_button_text()
            elif self._last_status == SystemStatus.UNLOADING:
                self._is_model_loaded = False
                self._update_load_button_text()
            self._set_send_button_state(is_running=False)
            self._rewrite_assistant_index = None
            if self._token_buf:
                self._flush_tokens(force_all=True)
            if self._active_widget is not None:
                self._active_widget.finalize()
            self._active_widget = None
            if self._update_trace_state == "streaming":
                self._finalize_update_progress()
            self._cleanup_empty_assistant_if_needed()
            self._active_assistant_started = False
            self._active_assistant_token_count = 0
            if self._pending_mutation is None:
                self._maybe_generate_title()
            self._suppress_title_regen = False
        elif status == SystemStatus.LOADING:
            self._set_send_button_state(is_running=False)
            self.btn_send.setEnabled(False)
        elif status in (SystemStatus.UNLOADING, SystemStatus.ERROR):
            self._is_model_loaded = False
            if not is_processing:
                self._update_load_button_text()

        if status == SystemStatus.READY and not self._is_model_loaded:
            self._apply_default_limits()
        self._last_status = status

    def _switch_ops_tab(self, index, checked):
        if checked:
            self.ops_stack.setCurrentIndex(index)

    def apply_operator(self, operator_data: dict):
        if not isinstance(operator_data, dict):
            return
        config = operator_data.get("config")
        if not isinstance(config, dict):
            return

        slider_values = {
            "temp": float(config.get("temp", self.config.get("temp", 0.7))),
            "top_p": float(config.get("top_p", self.config.get("top_p", 0.9))),
            "max_tokens": int(config.get("max_tokens", self.config.get("max_tokens", 2048))),
            "ctx_limit": int(config.get("ctx_limit", self.config.get("ctx_limit", 8192))),
        }
        self.config.update(config)

        for slider, key in (
            (self.s_temp, "temp"),
            (self.s_top, "top_p"),
            (self.s_tok, "max_tokens"),
            (self.s_ctx, "ctx_limit"),
        ):
            slider.slider.blockSignals(True)
        self.s_temp.slider.setValue(int(slider_values["temp"] * 100))
        self.s_temp.val_lbl.setText(f"{slider_values['temp']:.2f}")
        self.s_top.slider.setValue(int(slider_values["top_p"] * 100))
        self.s_top.val_lbl.setText(f"{slider_values['top_p']:.2f}")
        self.s_tok.slider.setValue(int(slider_values["max_tokens"]))
        self.s_tok.val_lbl.setText(str(int(slider_values["max_tokens"])))
        self.s_ctx.slider.setValue(int(slider_values["ctx_limit"]))
        self.s_ctx.val_lbl.setText(str(int(slider_values["ctx_limit"])))
        for slider, key in (
            (self.s_temp, "temp"),
            (self.s_top, "top_p"),
            (self.s_tok, "max_tokens"),
            (self.s_ctx, "ctx_limit"),
        ):
            slider.slider.blockSignals(False)

        self.sig_set_ctx_limit.emit(int(slider_values["ctx_limit"]))
        tags = config.get("behavior_tags", [])
        self.behavior_tags.set_tags(tags if isinstance(tags, list) else [])
        thinking_mode = bool(config.get("thinking_mode", False))
        self._set_thinking_mode(thinking_mode, "Standard" if thinking_mode else "Off")
        gguf_path = config.get("gguf_path")
        if gguf_path:
            self.config["gguf_path"] = gguf_path
            self.sig_set_model_path.emit(str(gguf_path))
        self._sync_path_display()

        messages = operator_data.get("messages")
        if isinstance(messages, list) and messages:
            session = self._create_session(
                messages=messages,
                title=operator_data.get("session_title"),
                assistant_tokens=operator_data.get("assistant_tokens", 0),
            )
            self._set_current_session(session, show_reset=True, sync_history=True)
        else:
            self._start_new_session()

        self._set_config_dirty(True)
        self.sig_operator_loaded.emit(str(operator_data.get("name", "")))

    def _start_new_session(self):
        self._title_generated = False
        self._suppress_title_regen = False
        self.trace.clear()
        self._set_current_session(self._create_session(), show_reset=True, sync_history=True)
        self._trace_plain("--- TRACE RESET ---")

    def _clear_current_session(self, delete_archive):
        archive_path = self._current_session.get("archive_path")
        if delete_archive and archive_path:
            try:
                Path(archive_path).unlink()
            except OSError:
                pass
        self.trace.clear()
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
            t = msg.get("time", meta.get("updated_at", self._now_iso()))
            messages.append({"i": msg.get("i"), "time": t, "role": role, "text": text})
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
        self._notify_header_update()

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

    def _create_session(self, messages=None, created_at=None, updated_at=None,
                        archive_path=None, summary=None, title=None, assistant_tokens=0):
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
        self._undo_snapshot = None
        self._active_assistant_index = None
        self._rewrite_assistant_index = None
        self._active_widget = None
        self._title_generated = bool(session.get("title"))
        self._suppress_title_regen = False
        self._render_session(session, show_reset=show_reset)
        if sync_history:
            history = self._build_engine_history_from_session()
            self.sig_sync_history.emit(history)
        self._notify_header_update()

    def _build_engine_history_from_session(self):
        history = [{"role": "system", "content": build_system_prompt(self.config)}]
        for msg in self._current_session.get("messages", []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            text = msg.get("text")
            if not isinstance(role, str) or not isinstance(text, str):
                continue
            history.append({"role": role, "content": text})
        return history

    def _parse_assistant_commands(self) -> None:
        if not bool(self.config.get("assistant_commands_enabled", False)):
            return
        msgs = self._current_session.get("messages", [])
        if not msgs:
            return
        msg = msgs[-1]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            return
        if msg.get("_assistant_cmds_parsed"):
            return

        text = str(msg.get("text") or "")
        matches = _ASSISTANT_CMD_RE.findall(text)
        if not matches:
            msg["_assistant_cmds_parsed"] = True
            return

        if len(matches) > 1:
            self._trace_html("assistant command rejected: multiple envelopes", "CMD", error=True)
            msg["_assistant_cmds_parsed"] = True
            return

        raw = matches[0].strip()
        try:
            payload = json.loads(raw)
        except Exception:
            self._trace_html("assistant command rejected: invalid JSON", "CMD", error=True)
            msg["_assistant_cmds_parsed"] = True
            return

        op = str(payload.get("op") or "").strip()
        if op != "open_addon":
            self._trace_html(f"assistant command rejected: unsupported op '{op or 'unknown'}'", "CMD", error=True)
            msg["_assistant_cmds_parsed"] = True
            return

        addon_id = str(payload.get("addon") or "").strip()
        if addon_id not in _ASSISTANT_ALLOWED_ADDONS:
            self._trace_html(f"assistant command rejected: addon '{addon_id or 'unknown'}'", "CMD", error=True)
            msg["_assistant_cmds_parsed"] = True
            return

        self.sig_launch_addon.emit(addon_id)
        self._trace_html(f"open_addon({addon_id})", "CMD")
        msg["_assistant_cmds_parsed"] = True

    def _snapshot_session(self):
        self._undo_snapshot = [dict(m) for m in self._current_session["messages"]]

    def _undo_last_mutation(self):
        if not self._undo_snapshot:
            return
        self._current_session["messages"] = self._undo_snapshot
        self._undo_snapshot = None
        self._render_session()
        self.sig_sync_history.emit(self._build_engine_history_from_session())

    def _delete_from_index(self, idx: int):
        self.sig_debug.emit(f"[CHAT] _delete_from_index: idx={idx}, msgs={len(self._current_session['messages'])}, is_running={self._is_running}")
        def _do_delete():
            self._snapshot_session()
            msgs = self._current_session["messages"]
            if idx < 0 or idx >= len(msgs):
                return
            del msgs[idx:]
            self._active_assistant_index = None
            self._rewrite_assistant_index = None
            self._active_widget = None
            self._token_buf.clear()
            if self._flush_timer.isActive():
                self._flush_timer.stop()
            self._render_session()
            self.sig_sync_history.emit(self._build_engine_history_from_session())
        self._request_mutation(_do_delete)

    def _edit_from_index(self, idx: int):
        self.sig_debug.emit(f"[CHAT] _edit_from_index: idx={idx}, msgs={len(self._current_session['messages'])}, is_running={self._is_running}")
        def _do_edit():
            msgs = self._current_session["messages"]
            if idx < 0 or idx >= len(msgs):
                return
            text = msgs[idx].get("text", "")
            self._suppress_title_regen = True
            self._snapshot_session()
            del msgs[idx:]
            self._active_assistant_index = None
            self._rewrite_assistant_index = None
            self._active_widget = None
            self._token_buf.clear()
            if self._flush_timer.isActive():
                self._flush_timer.stop()
            self._render_session()
            self.sig_sync_history.emit(self._build_engine_history_from_session())
            self.text_input.setPlainText(text)
        self._request_mutation(_do_edit)

    def _regen_last_assistant(self):
        self.sig_debug.emit(f"[CHAT] _regen_last_assistant: msgs={len(self._current_session['messages'])}, is_running={self._is_running}")
        def _do_regen():
            msgs = self._current_session["messages"]
            if not msgs or msgs[-1].get("role") != "assistant":
                return
            self._snapshot_session()
            self._suppress_title_regen = True
            del msgs[-1]
            self._active_assistant_index = None
            self._rewrite_assistant_index = None
            self._active_widget = None
            self._token_buf.clear()
            if self._flush_timer.isActive():
                self._flush_timer.stop()
            self._render_session()
            self.sig_sync_history.emit(self._build_engine_history_from_session())
            for m in reversed(msgs):
                if m.get("role") == "user":
                    self._set_send_button_state(is_running=True)
                    self._start_assistant_stream()
                    self.message_list.scrollToBottom()
                    self.sig_generate.emit(m.get("text", ""), self._thinking_mode)
                    break
        self._request_mutation(_do_regen)

    def _render_session(self, session=None, show_reset=False):
        if session is None:
            session = self._current_session
        self.sig_debug.emit(f"[CHAT] _render_session: msgs={len(session['messages'])}, show_reset={show_reset}")
        self.message_list.clear()
        self._active_widget = None
        if not session["messages"]:
            return
        for idx, _msg in enumerate(session["messages"]):
            self._append_message_widget(idx)
        self.message_list.scrollToBottom()

    def _append_message_widget(self, idx: int, role=None, text=None, timestamp=None):
        if idx >= 0:
            msg = self._current_session["messages"][idx]
            role = msg.get("role", "")
            text = msg.get("text", "")
            timestamp = msg.get("time", "")
        item = QListWidgetItem()
        widget = MessageWidget(idx, role or "", text or "", timestamp or "")
        widget.sig_delete.connect(self._delete_from_index)
        widget.sig_edit.connect(self._edit_from_index)
        widget.sig_regen.connect(lambda _idx: self._regen_last_assistant())
        vw = self.message_list.viewport().width()
        if vw > 50:
            widget.setFixedWidth(vw)
        item.setSizeHint(widget.sizeHint())
        self.message_list.addItem(item)
        self.message_list.setItemWidget(item, widget)
        return widget

    def _widget_for_index(self, idx: int):
        for row in range(self.message_list.count()):
            item = self.message_list.item(row)
            widget = self.message_list.itemWidget(item)
            if isinstance(widget, MessageWidget) and getattr(widget, "_index", None) == idx:
                return widget
        return None

    def _apply_behavior_prompt(self, tags):
        cleaned = [tag.strip() for tag in tags if tag.strip()]
        self.config["behavior_tags"] = cleaned
        self._set_config_dirty(True)

    def _begin_update_trace(self, update_text):
        self._update_trace_state = "requested"
        self._update_token_count = 0
        self._update_progress_index = 0
        self._trace_html("UPDATE REQUESTED", "UPDATE")
        self._trace_html(f'USER PATCH: "{update_text}"', "UPDATE")

    def _start_update_streaming(self):
        self._update_trace_state = "streaming"
        self._update_token_count = 0
        self._update_progress_index = 0

    def _update_progress_markers(self):
        if self._update_trace_state != "streaming":
            return
        self._update_token_count += 1
        thresholds = [25, 50, 100]
        pct = min(100, int((self._update_token_count / self.config["max_tokens"]) * 100))
        while self._update_progress_index < len(thresholds):
            if pct >= thresholds[self._update_progress_index]:
                percent = thresholds[self._update_progress_index]
                self._trace_html(f"UPDATE PROGRESS {percent}%", "UPDATE")
                self._update_progress_index += 1
                continue
            break

    def _finalize_update_progress(self):
        if self._update_trace_state != "streaming":
            return
        thresholds = [25, 50, 100]
        while self._update_progress_index < len(thresholds):
            percent = thresholds[self._update_progress_index]
            self._trace_html(f"UPDATE PROGRESS {percent}%", "UPDATE")
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
            return
        msg = self._current_session["messages"][target_index]
        msg["text"] += token
        self._active_assistant_token_count += 1
        msg["time"] = self._now_iso()
        self._current_session["updated_at"] = msg["time"]
        self._current_session["assistant_tokens"] = int(self._current_session.get("assistant_tokens", 0)) + 1

    def _cleanup_empty_assistant_if_needed(self):
        idx = self._active_assistant_index
        if idx is None:
            return
        if not getattr(self, "_active_assistant_started", False):
            return
        if int(getattr(self, "_active_assistant_token_count", 0)) > 0:
            return
        msgs = self._current_session.get("messages", [])
        if 0 <= idx < len(msgs):
            msg = msgs[idx]
            if msg.get("role") == "assistant" and (msg.get("text") or "") == "":
                del msgs[idx]
                self._active_assistant_index = None
                self._rewrite_assistant_index = None
                self._active_widget = None
                self._token_buf.clear()
                if self._flush_timer.isActive():
                    self._flush_timer.stop()
                self._render_session()
                self.sig_sync_history.emit(self._build_engine_history_from_session())

    def _maybe_generate_title(self):
        if self._suppress_title_regen:
            return
        if self._title_generated:
            return
        if self._current_session.get("title"):
            self._title_generated = True
            return
        if not any(m.get("role") == "user" and m.get("text", "").strip() for m in self._current_session["messages"]):
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
        self.ui_bridge.sig_terminal_header.emit(getattr(self, "_mod_id", ""), title, dt)

    def _derive_title(self, messages):
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
            "how", "i", "if", "in", "into", "is", "it", "me", "my", "of", "on", "or",
            "our", "please", "so", "that", "the", "their", "them", "then", "there", "these",
            "this", "to", "us", "we", "what", "when", "where", "which", "who", "why", "with",
            "you", "your",
        }
        user_texts = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            text = " ".join((msg.get("text") or "").lower().split())
            if text:
                user_texts.append(text)
            if len(user_texts) == 3:
                break
        if not user_texts:
            return "chat"
        candidates = []
        counts = {}
        for text in user_texts:
            for token in re.findall(r"[a-z0-9]+", text):
                if token in stopwords or len(token) < 3:
                    continue
                if token not in counts:
                    candidates.append(token)
                    counts[token] = 0
                counts[token] += 1
        ranked = sorted(candidates, key=lambda token: (-counts[token], candidates.index(token)))
        title_tokens = ranked[:6]
        title = " ".join(title_tokens)
        title = re.sub(r"\s+", " ", title).strip()
        title = re.sub(r"[^a-z0-9\- ]+", "", title)
        if len(title) > 40:
            title = title[:40].rstrip()
        return title or "chat"

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
