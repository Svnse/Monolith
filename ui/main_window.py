from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFrame, QLabel, QStackedLayout
)
from PySide6.QtCore import Qt, QDateTime, QTimer
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent, QKeyEvent, QMouseEvent, QCursor

from core.state import SystemStatus, AppState
from ui.bridge import UIBridge
from ui.addons.host import AddonHost
from ui.components.atoms import SidebarButton
from ui.components.complex import GradientLine, VitalsWindow, SplitControlBlock
from ui.components.command_palette import CommandPalette
from ui.components.drop_zone import DropZoneOverlay
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
        self.setMouseTracking(True)
        self.resize(1100, 700)
        self.setMinimumSize(600, 400)

        # Resize grip state
        self._resize_edge = None   # None or combination of "top","bottom","left","right"
        self._resize_origin = None
        self._resize_geo = None
        self._GRIP = 4  # pixels from edge that trigger resize

        main_widget = QWidget()
        main_widget.setObjectName("MainFrame")
        main_widget.setMouseTracking(True)
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
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(70)
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 15, 0, 15)
        sidebar_layout.setSpacing(10)

        self.module_strip = ModuleStrip()
        self.module_strip.sig_module_selected.connect(self.switch_to_module)
        self.module_strip.sig_module_closed.connect(self.close_module)
        self.setAcceptDrops(True)

        self.btn_hub = SidebarButton("", "HOME")
        self.btn_hub.clicked.connect(lambda: self.set_page("hub"))

        self.btn_addons = SidebarButton("", "MODULES")
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
        self.lbl_status.setObjectName("lbl_status")
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

        self.palette = CommandPalette(self.centralWidget(), host.registry, host)
        self.drop_zone = DropZoneOverlay(self.centralWidget(), host.registry, host)

        self.set_page("hub")

    # ---------------- WINDOW BEHAVIOR ----------------

    def _edge_at(self, pos):
        """Return 'right+bottom' if pos is within the bottom-right grip zone, else None."""
        g = self._GRIP * 3
        r = self.rect()
        if pos.x() >= r.width() - g and pos.y() >= r.height() - g:
            return "right+bottom"
        return None

    _CURSOR_MAP = {
        "right+bottom": Qt.SizeFDiagCursor,
    }

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return
        pos = event.position().toPoint()
        edge = self._edge_at(pos)
        if edge:
            self._resize_edge = edge
            self._resize_origin = event.globalPosition().toPoint()
            self._resize_geo = self.geometry()
            event.accept()
            return
        if pos.y() < 40:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        # Active resize
        if self._resize_edge and event.buttons() == Qt.LeftButton:
            delta = event.globalPosition().toPoint() - self._resize_origin
            geo = self._resize_geo
            new_geo = geo.__class__(geo)
            if "right" in self._resize_edge:
                new_geo.setRight(geo.right() + delta.x())
            if "bottom" in self._resize_edge:
                new_geo.setBottom(geo.bottom() + delta.y())
            if "left" in self._resize_edge:
                new_geo.setLeft(geo.left() + delta.x())
            if "top" in self._resize_edge:
                new_geo.setTop(geo.top() + delta.y())
            if new_geo.width() >= self.minimumWidth() and new_geo.height() >= self.minimumHeight():
                self.setGeometry(new_geo)
            event.accept()
            return
        # Active drag
        if self._drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        # Hover cursor
        edge = self._edge_at(event.position().toPoint())
        if edge and edge in self._CURSOR_MAP:
            self.setCursor(self._CURSOR_MAP[edge])
        else:
            if self.cursor().shape() != Qt.ArrowCursor:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None
        self._resize_edge = None
        self._resize_origin = None
        self._resize_geo = None

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_K and hasattr(self, "palette"):
            self.palette.toggle()
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and hasattr(self, "drop_zone"):
            event.acceptProposedAction()
            self.drop_zone.activate(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        if hasattr(self, "drop_zone"):
            self.drop_zone.deactivate()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if hasattr(self, "drop_zone"):
            self.drop_zone.handle_drop(event)
        event.acceptProposedAction()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "palette") and self.palette.isVisible():
            self.palette.setGeometry(self.centralWidget().rect())
        if hasattr(self, "drop_zone") and self.drop_zone.isVisible():
            self.drop_zone.setGeometry(self.centralWidget().rect())

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
            self.lbl_status.setProperty("state", "error")
        elif status == SystemStatus.LOADING:
            self.lbl_status.setProperty("state", "loading")
        else:
            self.lbl_status.setProperty("state", "ready")
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)
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
        bar.setObjectName("top_bar")
        bar.setFixedHeight(35)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)

        self.lbl_monolith = QLabel("MONOLITH")
        self.lbl_monolith.setObjectName("lbl_monolith")
        layout.addWidget(self.lbl_monolith)
        layout.addStretch()

        self.lbl_chat_title = QLabel(self._chat_title)
        self.lbl_chat_title.setObjectName("lbl_chat_title")
        self.lbl_chat_time = QLabel(QDateTime.currentDateTime().toString("ddd • HH:mm"))
        self.lbl_chat_time.setObjectName("lbl_chat_time")
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
