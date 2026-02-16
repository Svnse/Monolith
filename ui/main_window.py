from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFrame, QLabel, QStackedLayout
)
from PySide6.QtCore import Qt, QDateTime, QTimer
from PySide6.QtGui import QMouseEvent

from core.state import SystemStatus, AppState
from ui.bridge import UIBridge
import core.style as _style  # dynamic theme bridge — always read from _style.* for fresh values
from ui.addons.host import AddonHost
from ui.components.atoms import SidebarButton
from ui.components.complex import GradientLine, VitalsWindow, SplitControlBlock
from ui.components.module_strip import ModuleStrip

class MonolithUI(QMainWindow):
    def __init__(self, state: AppState, ui_bridge: UIBridge):
        super().__init__()
        self.state = state
        self.ui_bridge = ui_bridge
        self.vitals_win = None
        self._drag_pos = None
        self._chat_title = "Untitled Chat"
        self._terminal_titles: dict[str, tuple[str, str]] = {}

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(1100, 700)

        main_widget = QWidget()
        main_widget.setObjectName("MainFrame")
        main_widget.setStyleSheet(f"""
            QWidget {{ background: {_style.BG_MAIN}; }}
            QWidget#MainFrame {{ border: 1px solid {_style.BORDER_LIGHT}; }}
        """)
        self.setCentralWidget(main_widget)

        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(1,1,1,1)
        root_layout.setSpacing(0)

        # Top Gradient
        self.gradient_line = GradientLine()
        root_layout.addWidget(self.gradient_line)

        # Top Bar
        self.top_bar = self._build_top_bar()
        root_layout.addWidget(self.top_bar)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(0)

        # --- SIDEBAR ---
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(70)
        self.sidebar.setStyleSheet(f"background: {_style.BG_SIDEBAR}; border-right: 1px solid {_style.BORDER_SUBTLE};")
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(5, 15, 5, 15)
        sidebar_layout.setSpacing(10)

        self.module_strip = ModuleStrip()
        self.module_strip.sig_module_selected.connect(self.switch_to_module)
        self.module_strip.sig_module_closed.connect(self.close_module)

        self.btn_hub = SidebarButton("◉", "HOME")
        self.btn_hub.clicked.connect(lambda: self.set_page("hub"))

        self.btn_addons = SidebarButton("＋", "MODULES")
        self.btn_addons.clicked.connect(lambda: self.set_page("addons"))

        sidebar_layout.addWidget(self.module_strip)
        sidebar_layout.addStretch() 
        sidebar_layout.addWidget(self.btn_hub)
        sidebar_layout.addWidget(self.btn_addons)

        content_layout.addWidget(self.sidebar)

        # --- PAGE STACK ---
        self.stack = QStackedLayout()
        self.host: Optional[AddonHost] = None
        self.pages = {}

        self.empty_page = QWidget()
        self.stack.addWidget(self.empty_page)
        self.pages["empty"] = self.empty_page

        self.center_vbox = QVBoxLayout()
        self.center_vbox.addLayout(self.stack)
        content_layout.addLayout(self.center_vbox)

        root_layout.addLayout(content_layout)

        # --- Bottom status bar ---
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 8, 2)
        bottom_bar.addStretch()
        self.lbl_status = QLabel("READY")
        self.lbl_status.setStyleSheet(f"color: {_style.FG_PLACEHOLDER}; font-size: 8px; font-weight: bold; background: transparent;")
        bottom_bar.addWidget(self.lbl_status)
        root_layout.addLayout(bottom_bar)

        # --- Time update timer ---
        self._time_timer = QTimer(self)
        self._time_timer.timeout.connect(self._update_time_display)
        self._time_timer.start(60000)

        self.ui_bridge.sig_terminal_header.connect(self.update_terminal_header)

    def attach_host(self, host: AddonHost) -> None:
        self.host = host
        hub = host.mount_page("hub")
        addons = host.mount_page("addons")

        self.stack.addWidget(hub)
        self.pages["hub"] = hub

        self.stack.addWidget(addons)
        self.pages["addons"] = addons

        self.set_page("hub")

    # ---------------- WINDOW BEHAVIOR ----------------

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and event.position().y() < 40:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None

    # ---------------- MODULE SYSTEM ----------------

    def close_module(self, mod_id):
        current = self.stack.currentWidget()
        target_w = None
        for i in range(self.stack.count()):
            w = self.stack.widget(i)
            if getattr(w, '_mod_id', None) == mod_id:
                target_w = w
                break
        
        if target_w:
            self.stack.removeWidget(target_w)
            target_w.deleteLater()
            
        self.module_strip.remove_module(mod_id)

        if current == target_w:
            remaining = self.module_strip.get_order()
            if remaining:
                self.switch_to_module(remaining[-1])
            else:
                self.set_page("empty")

    def switch_to_module(self, mod_id):
        for i in range(self.stack.count()):
            w = self.stack.widget(i)
            if getattr(w, '_mod_id', None) == mod_id:
                self.stack.setCurrentWidget(w)
                self._update_sidebar_state(module_selection=True)
                self.module_strip.select_module(mod_id)
                self.lbl_monolith.setVisible(True)
                # Only show chat title for terminal modules
                if getattr(w, '_addon_id', None) == "terminal":
                    self.update_terminal_header(mod_id, *self._terminal_titles.get(mod_id, ("Untitled Chat", QDateTime.currentDateTime().toString("ddd • HH:mm"))))
                else:
                    self.lbl_chat_title.hide()
                    self.lbl_chat_time.hide()
                return

    def _update_sidebar_state(self, page_idx=None, module_selection=False):
        self.btn_hub.setChecked(page_idx == "hub" and not module_selection)
        self.btn_addons.setChecked(page_idx == "addons" and not module_selection)
        if not module_selection: self.module_strip.deselect_all()

    def update_status(self, engine_key: str, status: SystemStatus):
        if status == SystemStatus.ERROR:
            self.lbl_status.setStyleSheet(f"color: {_style.FG_ERROR}; font-size: 8px; font-weight: bold; background: transparent;")
        elif status == SystemStatus.LOADING:
            self.lbl_status.setStyleSheet(f"color: {_style.FG_WARN}; font-size: 8px; font-weight: bold; background: transparent;")
        else:
            self.lbl_status.setStyleSheet(f"color: {_style.FG_PLACEHOLDER}; font-size: 8px; font-weight: bold; background: transparent;")
        status_text = status.value if hasattr(status, "value") else str(status)
        if not engine_key.startswith("llm"):
            status_text = f"{engine_key.upper()}: {status_text}"
        self.lbl_status.setText(status_text)

    def update_ctx(self, used):
        self.state.ctx_used = used

    def update_terminal_header(self, mod_id, title, timestamp):
        if mod_id:
            self._terminal_titles[mod_id] = (title or "Untitled Chat", timestamp or QDateTime.currentDateTime().toString("ddd • HH:mm"))

        current = self.stack.currentWidget()
        current_mod = getattr(current, "_mod_id", None) if current is not None else None
        if not current_mod:
            self.lbl_chat_title.clear()
            self.lbl_chat_time.clear()
            self.lbl_chat_title.hide()
            self.lbl_chat_time.hide()
            return

        current_title, current_time = self._terminal_titles.get(
            current_mod,
            ("Untitled Chat", QDateTime.currentDateTime().toString("ddd • HH:mm")),
        )
        if current_mod == mod_id or mod_id == "":
            self.lbl_chat_title.setText(current_title)
            self.lbl_chat_time.setText(current_time)
            self.lbl_chat_title.show()
            self.lbl_chat_time.show()

    def set_page(self, page_id):
        target = self.pages.get(page_id)
        if target:
            self.stack.setCurrentWidget(target)
        self._update_sidebar_state(page_idx=page_id)
        self.lbl_monolith.setVisible(page_id != "hub")
        self.update_terminal_header("", "", "")

    def _update_time_display(self):
        current = self.stack.currentWidget()
        current_mod = getattr(current, "_mod_id", None) if current is not None else None
        if current_mod and getattr(current, "_addon_id", None) == "terminal":
            now = QDateTime.currentDateTime().toString("ddd • HH:mm")
            self.lbl_chat_time.setText(now)
            if current_mod in self._terminal_titles:
                title = self._terminal_titles[current_mod][0]
                self._terminal_titles[current_mod] = (title, now)

    def _build_top_bar(self):
        bar = QFrame()
        bar.setFixedHeight(35)
        bar.setStyleSheet(f"background: {_style.BG_SIDEBAR}; border-bottom: 1px solid {_style.BORDER_SUBTLE};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)

        self.lbl_monolith = QLabel("MONOLITH")
        self.lbl_monolith.setStyleSheet(
            f"color: {_style.ACCENT_PRIMARY_DARK}; font-size: 14px; font-weight: bold; "
            "letter-spacing: 3px; background: transparent;"
        )
        layout.addWidget(self.lbl_monolith)
        layout.addStretch()

        self.lbl_chat_title = QLabel(self._chat_title)
        self.lbl_chat_title.setStyleSheet(f"color: {_style.FG_TEXT}; font-size: 10px; font-weight: bold;")
        self.lbl_chat_time = QLabel(QDateTime.currentDateTime().toString("ddd • HH:mm"))
        self.lbl_chat_time.setStyleSheet(f"color: {_style.FG_DIM}; font-size: 10px;")
        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 8, 0)
        title_box.setSpacing(0)
        title_box.addWidget(self.lbl_chat_title, alignment=Qt.AlignRight)
        title_box.addWidget(self.lbl_chat_time, alignment=Qt.AlignRight)
        layout.addLayout(title_box)

        self.win_controls = SplitControlBlock()
        self.win_controls.minClicked.connect(self.showMinimized)
        self.win_controls.maxClicked.connect(self.toggle_maximize)
        self.win_controls.closeClicked.connect(self.close)
        layout.addWidget(self.win_controls)

        return bar

    def toggle_maximize(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def apply_theme_refresh(self):
        """Re-apply all stylesheets after theme change. Rebuilds the entire UI appearance."""
        # Main frame
        main_widget = self.centralWidget()
        if main_widget:
            main_widget.setStyleSheet(f"""
                QWidget {{ background: {_style.BG_MAIN}; }}
                QWidget#MainFrame {{ border: 1px solid {_style.BORDER_LIGHT}; }}
            """)
        # Sidebar
        self.sidebar.setStyleSheet(f"background: {_style.BG_SIDEBAR}; border-right: 1px solid {_style.BORDER_SUBTLE};")
        # Top bar
        self.top_bar.setStyleSheet(f"background: {_style.BG_SIDEBAR}; border-bottom: 1px solid {_style.BORDER_SUBTLE};")
        self.lbl_monolith.setStyleSheet(
            f"color: {_style.ACCENT_PRIMARY_DARK}; font-size: 14px; font-weight: bold; "
            f"letter-spacing: 3px; background: transparent;"
        )
        self.lbl_chat_title.setStyleSheet(f"color: {_style.FG_TEXT}; font-size: 10px; font-weight: bold;")
        self.lbl_chat_time.setStyleSheet(f"color: {_style.FG_DIM}; font-size: 10px;")
        # Status bar
        self.lbl_status.setStyleSheet(
            f"color: {_style.FG_PLACEHOLDER}; font-size: 8px; font-weight: bold; background: transparent;"
        )
        # Gradient line
        self.gradient_line.update()
        # Sidebar buttons — re-apply with current checked state
        self.btn_hub.update_style(self.btn_hub.isChecked())
        self.btn_addons.update_style(self.btn_addons.isChecked())
        # Window control block
        if hasattr(self.win_controls, 'refresh_style'):
            self.win_controls.refresh_style()

    def toggle_vitals(self):
        if not self.vitals_win:
            self.vitals_win = VitalsWindow(self.state, self)
        
        if not self.vitals_win.isVisible():
            self.vitals_win.show()
        else:
            self.vitals_win.close()
