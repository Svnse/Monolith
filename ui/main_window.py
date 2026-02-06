from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFrame, QLabel, QStackedLayout
)
from PySide6.QtCore import Qt, QDateTime
from PySide6.QtGui import QMouseEvent

from core.state import SystemStatus, AppState
from ui.bridge import UIBridge
from core.style import BG_MAIN, BG_SIDEBAR, FG_ACCENT, FG_ERROR, FG_WARN
from ui.addons.host import AddonHost
from ui.components.atoms import SidebarButton, SkeetButton
from ui.components.complex import GradientLine, VitalsWindow, SplitControlBlock, FlameLabel
from ui.components.module_strip import ModuleStrip

class MonolithUI(QMainWindow):
    def __init__(self, state: AppState, ui_bridge: UIBridge):
        super().__init__()
        self.state = state
        self.ui_bridge = ui_bridge
        self.vitals_win = None
        self._drag_pos = None
        self._chat_title = "Untitled Chat"

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(1100, 700)

        main_widget = QWidget()
        main_widget.setObjectName("MainFrame")
        main_widget.setStyleSheet(f"""
            QWidget {{ background: {BG_MAIN}; }}
            QWidget#MainFrame {{ border: 1px solid #333; }}
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
        self.sidebar.setStyleSheet(f"background: {BG_SIDEBAR}; border-right: 1px solid #1a1a1a;")
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(5, 15, 5, 15)
        sidebar_layout.setSpacing(10)

        self.module_strip = ModuleStrip()
        self.module_strip.sig_module_selected.connect(self.switch_to_module)
        self.module_strip.sig_module_closed.connect(self.close_module)

        self.btn_hub = SidebarButton("◉", "HUB")
        self.btn_hub.clicked.connect(lambda: self.set_page("hub"))

        self.btn_addons = SidebarButton("＋", "ADDONS")
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
                return

    def _update_sidebar_state(self, page_idx=None, module_selection=False):
        self.btn_hub.setChecked(page_idx == "hub" and not module_selection)
        self.btn_addons.setChecked(page_idx == "addons" and not module_selection)
        if not module_selection: self.module_strip.deselect_all()

    def update_status(self, engine_key, status=None):
        if status is None:
            status = engine_key
            engine_key = "llm"
        if status == SystemStatus.ERROR:
            self.lbl_status.setStyleSheet(f"color: {FG_ERROR}; font-size: 10px; font-weight: bold;")
        elif status == SystemStatus.LOADING:
            self.lbl_status.setStyleSheet(f"color: {FG_WARN}; font-size: 10px; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet(f"color: {FG_ACCENT}; font-size: 10px; font-weight: bold;")
        status_text = status.value if hasattr(status, "value") else str(status)
        if engine_key != "llm":
            status_text = f"{engine_key.upper()}: {status_text}"
        self.lbl_status.setText(status_text)

    def update_ctx(self, used):
        self.state.ctx_used = used

    def update_terminal_header(self, title, timestamp):
        self._chat_title = title or "Untitled Chat"
        self.lbl_chat_title.setText(self._chat_title)
        self.lbl_chat_time.setText(timestamp or QDateTime.currentDateTime().toString("ddd • HH:mm"))

    def set_page(self, page_id):
        target = self.pages.get(page_id)
        if target:
            self.stack.setCurrentWidget(target)
        self._update_sidebar_state(page_idx=page_id)

    def _build_top_bar(self):
        bar = QFrame()
        bar.setFixedHeight(35)
        bar.setStyleSheet("background: #111; border-bottom: 1px solid #222;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)

        self.lbl_status = QLabel("READY")
        self.lbl_status.setStyleSheet(f"color: {FG_ACCENT}; font-size: 10px; font-weight: bold;")

        btn_vitals = SkeetButton("VITALS")
        btn_vitals.setFixedSize(75, 26)
        btn_vitals.clicked.connect(self.toggle_vitals)

        btn_overseer = SkeetButton("OVERSEER")
        btn_overseer.setFixedSize(95, 26)
        btn_overseer.clicked.connect(self.ui_bridge.sig_open_overseer.emit)

        layout.addWidget(btn_vitals)
        layout.addWidget(btn_overseer)
        layout.addWidget(self.lbl_status)
        layout.addStretch()

        self.lbl_model = FlameLabel("MONOLITH")
        layout.addWidget(self.lbl_model)
        layout.addStretch()

        self.lbl_chat_title = QLabel(self._chat_title)
        self.lbl_chat_title.setStyleSheet("color: #dcdcdc; font-size: 10px; font-weight: bold;")
        self.lbl_chat_time = QLabel(QDateTime.currentDateTime().toString("ddd • HH:mm"))
        self.lbl_chat_time.setStyleSheet("color: #777; font-size: 10px;")
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

    def toggle_vitals(self):
        if not self.vitals_win:
            self.vitals_win = VitalsWindow(self.state, self)
        
        if not self.vitals_win.isVisible():
            self.vitals_win.show()
        else:
            self.vitals_win.close()
