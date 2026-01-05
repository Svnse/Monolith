# ==================================================
#   LEGACY.
#  COMBINED MONOLITH SOURCE FILE
#  Auto-generated — DO NOT EDIT DIRECTLY
# ==================================================



# --------------------------------------------------
# FILE: main.py
# --------------------------------------------------

import sys
from PySide6.QtWidgets import QApplication

from core.state import AppState
from engine.llm import LLMEngine
from ui.main_window import MonolithUI

def main():
    app = QApplication(sys.argv)
    
    # 1. Initialize State & Core
    state = AppState()
    engine = LLMEngine(state)
    
    # 2. Initialize UI
    ui = MonolithUI(state)

    # 3. Connect UI Commands -> Engine
    ui.page_settings.sig_load.connect(engine.load_model)
    ui.page_settings.sig_unload.connect(engine.unload_model)
    ui.page_chat.sig_generate.connect(engine.generate)

    # 4. Connect Engine Updates -> UI
    engine.sig_trace.connect(ui.page_chat.append_trace)
    engine.sig_token.connect(ui.page_chat.append_token)
    engine.sig_status.connect(ui.update_status)
    engine.sig_usage.connect(ui.update_ctx)

    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

# --------------------------------------------------
# FILE: core\state.py
# --------------------------------------------------

from enum import Enum

# System Status Enum
class SystemStatus(Enum):
    READY = "READY"
    LOADING = "LOADING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    UNLOADING = "UNLOADING"

# Shared Application State
class AppState:
    def __init__(self):
        # System
        self.gguf_path: str | None = None
        self.model_loaded: bool = False
        self.status: SystemStatus = SystemStatus.READY
        
        # Resources
        self.ctx_limit: int = 8192
        self.ctx_used: int = 0
        
        # AI Configuration
        self.temp: float = 0.7
        self.top_p: float = 0.9
        self.max_tokens: int = 2048
        self.system_prompt: str = "You are Monolith. Be precise."
        self.context_injection: str = ""

# --------------------------------------------------
# FILE: core\style.py
# --------------------------------------------------

# ======================
# THEME: SKEET / GAMESENSE
# ======================
BG_MAIN = "#0C0C0C"       # Deep dark background
BG_SIDEBAR = "#111111"    # Sidebar background
BG_PANEL = "#141414"      # Panel background
BG_GROUP = "#0e0e0e"      # Groupbox background
BG_INPUT = "#0f0f0f"      # Input field background

BORDER_DARK = "#2a2a2a"   # Subtle borders
BORDER_LIGHT = "#333333"  # Highlight borders

FG_TEXT = "#dcdcdc"       # Main text
FG_DIM = "#777777"        # Dim text / Labels
FG_ACCENT = "#96c93d"     # "Enabled" Green
FG_ERROR = "#d44e4e"      # Error Red
FG_WARN = "#e0b020"       # Warning Yellow

ACCENT_GOLD = "#D4AF37"   # Monolith Identity Gold

# --------------------------------------------------
# FILE: engine\llm.py
# --------------------------------------------------

from PySide6.QtCore import QObject, QThread, Signal
from core.state import AppState, SystemStatus

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

class ModelLoader(QThread):
    trace = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, path, n_ctx=8192, n_gpu_layers=-1):
        super().__init__()
        self.path = path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

    def run(self):
        if Llama is None:
            self.error.emit("CRITICAL: 'llama-cpp-python' library not found.")
            return

        try:
            self.trace.emit(f"→ init backend: {self.path}")
            llm_instance = Llama(
                model_path=self.path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            self.finished.emit(llm_instance)
        except Exception as e:
            self.error.emit(f"Load Failed: {str(e)}")

class GeneratorWorker(QThread):
    token = Signal(str)
    trace = Signal(str)
    done = Signal()
    usage = Signal(int)

    def __init__(self, llm, messages, temp, top_p, max_tokens):
        super().__init__()
        self.llm = llm
        self.messages = messages
        self.temp = temp
        self.top_p = top_p
        self.max_tokens = max_tokens

    def run(self):
        self.trace.emit("→ inference started")
        try:
            if self.isInterruptionRequested():
                return

            stream = self.llm.create_chat_completion(
                messages=self.messages,
                temperature=self.temp,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True
            )

            total_generated = 0
            for chunk in stream:
                if self.isInterruptionRequested():
                    self.trace.emit("→ inference aborted")
                    break

                if "content" in chunk["choices"][0]["delta"]:
                    text = chunk["choices"][0]["delta"]["content"]
                    self.token.emit(text)
                    total_generated += 1
                    self.usage.emit(total_generated)
            
            self.trace.emit("→ inference complete")
        except Exception as e:
            self.trace.emit(f"<span style='color:red'>ERROR: {e}</span>")
        finally:
            self.done.emit()

class LLMEngine(QObject):
    sig_token = Signal(str)
    sig_trace = Signal(str)
    sig_status = Signal(SystemStatus)
    sig_finished = Signal()
    sig_usage = Signal(int)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.llm = None
        self.loader = None
        self.worker = None

    def load_model(self):
        if self.state.status == SystemStatus.LOADING:
            return
        
        if not self.state.gguf_path:
            self.sig_trace.emit("ERROR: No GGUF selected.")
            return

        self.set_status(SystemStatus.LOADING)
        # Keep reference to loader to prevent GC
        self.loader = ModelLoader(self.state.gguf_path, self.state.ctx_limit)
        self.loader.trace.connect(self.sig_trace)
        self.loader.error.connect(self._on_load_error)
        self.loader.finished.connect(self._on_load_success)
        self.loader.start()

    def _on_load_success(self, llm_instance):
        self.llm = llm_instance
        self.state.model_loaded = True
        self.set_status(SystemStatus.READY)
        self.sig_trace.emit("→ system online")

    def _on_load_error(self, err_msg):
        self.sig_trace.emit(f"<span style='color:red'>{err_msg}</span>")
        self.set_status(SystemStatus.ERROR)

    def unload_model(self):
        if self.state.status == SystemStatus.RUNNING:
            self.sig_trace.emit("ERROR: Cannot unload while generating.")
            return

        if self.llm:
            self.set_status(SystemStatus.UNLOADING)
            del self.llm
            self.llm = None
        self.state.model_loaded = False
        self.set_status(SystemStatus.READY)
        self.sig_trace.emit("→ model unloaded")

    def generate(self, user_input):
        if not self.state.model_loaded:
            self.sig_trace.emit("ERROR: Model offline.")
            return

        if self.state.status == SystemStatus.RUNNING:
            self.sig_trace.emit("ERROR: Busy. Wait for completion.")
            return

        self.set_status(SystemStatus.RUNNING)
        
        messages = [{"role": "system", "content": self.state.system_prompt}]
        if self.state.context_injection:
            messages.append({"role": "system", "content": f"CONTEXT: {self.state.context_injection}"})
        messages.append({"role": "user", "content": user_input})

        self.worker = GeneratorWorker(
            self.llm, messages, self.state.temp, 
            self.state.top_p, self.state.max_tokens
        )
        self.worker.token.connect(self.sig_token)
        self.worker.trace.connect(self.sig_trace)
        self.worker.usage.connect(self._on_usage_update)
        self.worker.done.connect(self._on_gen_finish)
        self.worker.start()

    def stop_generation(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()

    def _on_usage_update(self, count):
        self.sig_usage.emit(count)

    def _on_gen_finish(self):
        self.sig_token.emit("\n")
        self.set_status(SystemStatus.READY)
        self.sig_finished.emit()

    def set_status(self, s):
        self.state.status = s
        self.sig_status.emit(s)

# --------------------------------------------------
# FILE: ui\main_window.py
# --------------------------------------------------

import uuid
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFrame, QLabel, QStackedLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from core.state import SystemStatus, AppState
from core.style import BG_MAIN, BG_SIDEBAR, FG_ACCENT, FG_ERROR, FG_WARN
from ui.components.atoms import SidebarButton, SkeetButton
from ui.components.complex import GradientLine, VitalsWindow, SplitControlBlock, FlameLabel
from ui.components.module_strip import ModuleStrip

# PAGES
from ui.pages.chat import PageChat
from ui.pages.settings import PageSettings
from ui.pages.databank import PageFiles

# MODULES
from ui.modules.manager import PageAddons
from ui.modules.injector import InjectorWidget

class MonolithUI(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.vitals_win = None
        self._drag_pos = None

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

        self.btn_chat = SidebarButton("⌖", "TERMINAL") 
        self.btn_chat.clicked.connect(lambda: self.set_page(0))

        self.btn_files = SidebarButton("▤", "DATABANK")
        self.btn_files.clicked.connect(lambda: self.set_page(1))

        self.module_strip = ModuleStrip()
        self.module_strip.sig_module_selected.connect(self.switch_to_module)
        self.module_strip.sig_module_closed.connect(self.close_module)

        self.btn_conf = SidebarButton("⚙", "CONFIG")
        self.btn_conf.clicked.connect(lambda: self.set_page(3))

        self.btn_addons = SidebarButton("＋", "ADDONS")
        self.btn_addons.clicked.connect(lambda: self.set_page(2))

        sidebar_layout.addWidget(self.btn_chat)
        sidebar_layout.addWidget(self.btn_files)
        sidebar_layout.addWidget(self.module_strip)
        sidebar_layout.addStretch() 
        sidebar_layout.addWidget(self.btn_conf)
        sidebar_layout.addWidget(self.btn_addons)

        content_layout.addWidget(self.sidebar)

        # --- PAGE STACK ---
        self.stack = QStackedLayout()
        self.page_chat = PageChat(state)
        self.page_files = PageFiles(state)
        self.page_addons = PageAddons(state)
        self.page_settings = PageSettings(state)
        
        self.stack.addWidget(self.page_chat)
        self.stack.addWidget(self.page_files)
        self.stack.addWidget(self.page_addons)
        self.stack.addWidget(self.page_settings)

        self.center_vbox = QVBoxLayout()
        self.center_vbox.addLayout(self.stack)
        content_layout.addLayout(self.center_vbox)

        root_layout.addLayout(content_layout)

        # --- SIGNALS ---
        self.page_addons.sig_launch_addon.connect(self.launch_addon)
        self.set_page(0)

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

    def launch_addon(self, addon_type: str):
        mod_id = str(uuid.uuid4())
        widget = None
        icon_char = "?"
        label_text = "MODULE"
        
        if addon_type == "injector":
            widget = InjectorWidget(self)
            icon_char = "💉" 
            label_text = "RUNTIME"
        
        if widget:
            widget._mod_id = mod_id
            self.stack.addWidget(widget)
            self.module_strip.add_module(mod_id, icon_char, label_text)
            
            if hasattr(widget, 'sig_closed'):
                widget.sig_closed.connect(lambda: self.close_module(mod_id))
            if hasattr(widget, 'sig_finished'):
                widget.sig_finished.connect(lambda: self.module_strip.flash_module(mod_id))
            
            self.switch_to_module(mod_id)

    def close_module(self, mod_id):
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

        if self.stack.currentWidget() == target_w:
             self.set_page(0)

    def switch_to_module(self, mod_id):
        for i in range(self.stack.count()):
            w = self.stack.widget(i)
            if getattr(w, '_mod_id', None) == mod_id:
                self.stack.setCurrentWidget(w)
                self._update_sidebar_state(module_selection=True)
                self.module_strip.select_module(mod_id)
                return

    def _update_sidebar_state(self, page_idx=None, module_selection=False):
        self.btn_chat.setChecked(page_idx == 0 and not module_selection)
        self.btn_files.setChecked(page_idx == 1 and not module_selection)
        self.btn_addons.setChecked(page_idx == 2 and not module_selection)
        self.btn_conf.setChecked(page_idx == 3 and not module_selection)
        if not module_selection: self.module_strip.deselect_all()

    def update_status(self, status):
        if status == SystemStatus.ERROR:
            self.lbl_status.setStyleSheet(f"color: {FG_ERROR}; font-size: 10px; font-weight: bold;")
        elif status == SystemStatus.LOADING:
            self.lbl_status.setStyleSheet(f"color: {FG_WARN}; font-size: 10px; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet(f"color: {FG_ACCENT}; font-size: 10px; font-weight: bold;")
        self.lbl_status.setText(status.value if hasattr(status, "value") else str(status))

        # Pass loading state to settings page to lock buttons
        self.page_settings.set_loading_state(status == SystemStatus.LOADING or status == SystemStatus.RUNNING)

    def update_ctx(self, used):
        self.state.ctx_used = used

    def set_page(self, idx):
        self.stack.setCurrentIndex(idx)
        self._update_sidebar_state(page_idx=idx)

    def _build_top_bar(self):
        bar = QFrame()
        bar.setFixedHeight(35)
        bar.setStyleSheet("background: #111; border-bottom: 1px solid #222;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)

        self.lbl_status = QLabel("READY")
        self.lbl_status.setStyleSheet(f"color: {FG_ACCENT}; font-size: 10px; font-weight: bold;")

        btn_vitals = SkeetButton("VITALS")
        btn_vitals.setFixedSize(60, 22)
        btn_vitals.clicked.connect(self.toggle_vitals)

        layout.addWidget(btn_vitals)
        layout.addWidget(self.lbl_status)
        layout.addStretch()

        self.lbl_model = FlameLabel("MONOLITH")
        layout.addWidget(self.lbl_model)
        layout.addStretch()

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

# --------------------------------------------------
# FILE: ui\components\atoms.py
# --------------------------------------------------

from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QSlider, QHBoxLayout, 
    QPushButton, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QDragEnterEvent

from core.style import BG_GROUP, BORDER_DARK, FG_TEXT, FG_DIM, FG_ACCENT, ACCENT_GOLD

# ======================
# HELPER
# ======================
def import_vbox(widget, l=15, t=25, r=15, b=15):
    from PySide6.QtWidgets import QVBoxLayout
    v = QVBoxLayout(widget)
    v.setContentsMargins(l, t, r, b)
    v.setSpacing(10)
    return v

# ======================
# BASIC UI PRIMITIVES
# ======================

class SkeetGroupBox(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            SkeetGroupBox {{
                background: {BG_GROUP};
                border: 1px solid {BORDER_DARK};
                margin-top: 10px; 
            }}
        """)
        self.layout_main = import_vbox(self)
        self.lbl_title = QLabel(title, self)
        self.lbl_title.setStyleSheet(f"""
            color: {FG_TEXT}; 
            font-weight: bold; font-size: 11px;
            background: {BG_GROUP}; padding: 0 4px;
        """)
        self.lbl_title.adjustSize()
        self.lbl_title.move(10, -3)

    def add_widget(self, widget):
        self.layout_main.addWidget(widget)

    def add_layout(self, layout):
        self.layout_main.addLayout(layout)

class SkeetButton(QPushButton):
    def __init__(self, text, accent=False):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        col = FG_ACCENT if accent else FG_TEXT
        self.setStyleSheet(f"""
            QPushButton {{ background: #181818; border: 1px solid #333; color: {FG_DIM}; padding: 6px 12px; font-size: 11px; font-weight: bold; border-radius: 2px; }}
            QPushButton:hover {{ background: #222; color: {col}; border: 1px solid {col}; }}
            QPushButton:disabled {{ background: #111; color: #333; border: 1px solid #222; }}
        """)

class SkeetSlider(QWidget):
    valueChanged = Signal(float)
    def __init__(self, label, min_v, max_v, init_v, is_int=False):
        super().__init__()
        self.is_int = is_int
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.lbl = QLabel(label)
        self.lbl.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")
        val_str = str(int(init_v) if is_int else f"{init_v:.2f}")
        self.val_lbl = QLabel(val_str)
        self.val_lbl.setStyleSheet(f"color: {FG_TEXT}; font-size: 11px; font-weight: bold;")
        self.val_lbl.setFixedWidth(40)
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = QSlider(Qt.Horizontal)
        if is_int:
            self.slider.setRange(int(min_v), int(max_v))
            self.slider.setValue(int(init_v))
        else:
            self.slider.setRange(int(min_v*100), int(max_v*100))
            self.slider.setValue(int(init_v*100))
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height: 4px; background: #222; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {FG_DIM}; width: 8px; margin: -2px 0; border-radius: 4px; }}
            QSlider::handle:horizontal:hover {{ background: {FG_ACCENT}; }}
            QSlider::sub-page:horizontal {{ background: {FG_ACCENT}; border-radius: 2px; }}
        """)
        self.slider.valueChanged.connect(self._on_change)
        layout.addWidget(self.lbl)
        layout.addWidget(self.slider)
        layout.addWidget(self.val_lbl)
    def _on_change(self, val):
        real_val = val if self.is_int else val / 100.0
        val_str = str(int(real_val) if self.is_int else f"{real_val:.2f}")
        self.val_lbl.setText(val_str)
        self.valueChanged.emit(float(real_val))

class SidebarButton(QPushButton):
    def __init__(self, icon_char, text, checkable=True):
        super().__init__()
        self.setCheckable(checkable)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(60, 45) # Shorter height
        if checkable: self.setAutoExclusive(False)
        self.setAcceptDrops(True) 
        
        # Text Only Layout (Skeet Style)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        self.lbl_text = QLabel(text)
        self.lbl_text.setAlignment(Qt.AlignCenter)
        self.lbl_text.setStyleSheet(f"color: {FG_DIM}; font-size: 9px; font-weight: bold;")
        
        layout.addWidget(self.lbl_text)
        self.update_style(False)

    def nextCheckState(self): pass 

    def setChecked(self, checked):
        super().setChecked(checked)
        self.update_style(checked)

    def update_style(self, checked):
        color = ACCENT_GOLD if checked else FG_DIM
        bg = "#1a1a1a" if checked else "transparent"
        # Pure text style, no icon char
        self.lbl_text.setStyleSheet(f"color: {color}; background: transparent; font-size: 10px; font-weight: bold;")
        # Add a left border indicator for active state
        border = f"border-left: 2px solid {ACCENT_GOLD};" if checked else "border: none;"
        self.setStyleSheet(f"background: {bg}; {border}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()

class CollapsibleSection(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.layout_main = import_vbox(self, 0, 0, 0, 0)
        self.layout_main.setSpacing(0)
        self.btn_toggle = QPushButton(title)
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setStyleSheet(f"""
            QPushButton {{
                color: {FG_DIM}; background: transparent; 
                border: none; text-align: left; font-weight: bold; font-size: 10px;
            }}
            QPushButton:checked {{ color: {ACCENT_GOLD}; }}
            QPushButton:hover {{ color: {FG_TEXT}; }}
        """)
        self.btn_toggle.clicked.connect(self.toggle_animation)
        self.layout_main.addWidget(self.btn_toggle)
        self.content_area = QScrollArea()
        self.content_area.setMaximumHeight(0) 
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setFrameShape(QFrame.NoFrame)
        self.content_area.setWidgetResizable(True)
        self.content_area.setStyleSheet("background: transparent;")
        self.layout_main.addWidget(self.content_area)
        self.anim = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.anim.setDuration(300)
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)

    def set_content_layout(self, layout):
        w = QWidget()
        w.setLayout(layout)
        self.content_area.setWidget(w)

    def toggle_animation(self):
        checked = self.btn_toggle.isChecked()
        content_height = self.content_area.widget().layout().sizeHint().height() if self.content_area.widget() else 100
        self.anim.setStartValue(0 if checked else content_height)
        self.anim.setEndValue(content_height if checked else 0)
        self.anim.start()

# --------------------------------------------------
# FILE: ui\components\complex.py
# --------------------------------------------------

import math
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QDialog, QHBoxLayout, QVBoxLayout, 
    QPushButton, QProgressBar, QGridLayout
)
from PySide6.QtCore import Qt, QTimer, Signal, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QColor, QLinearGradient, QFont, QPainterPath, QFontMetrics
)

from core.style import ACCENT_GOLD, FG_DIM, FG_TEXT, FG_ACCENT, FG_ERROR, BG_INPUT, BORDER_DARK

# ======================
# FLAME LABEL (FIXED)
# ======================
class FlameLabel(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self._text = text
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(50)
        # Use a thick, bold font for the mask to work well
        self.font_obj = QFont("Segoe UI", 14, QFont.Bold)
        self.setFixedHeight(30)
        self.setFixedWidth(120) 

    def _animate(self):
        # Move the gradient phase
        self.phase -= 0.08
        if self.phase < -1.0: self.phase = 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Setup Gradient (Fire Effect)
        # The gradient moves vertically based on self.phase
        h = self.height()
        grad = QLinearGradient(0, h + (h * self.phase), 0, -h + (h * self.phase))
        grad.setSpread(QLinearGradient.RepeatSpread)
        
        # Fire Colors: Dark Grey -> Gold -> White -> Dark Grey
        grad.setColorAt(0.0, QColor("#333"))
        grad.setColorAt(0.4, QColor(ACCENT_GOLD))
        grad.setColorAt(0.5, QColor("white"))
        grad.setColorAt(0.6, QColor(ACCENT_GOLD))
        grad.setColorAt(1.0, QColor("#333"))

        # 2. Create Text Path
        # We convert text to a shape so we can fill it with the gradient
        path = QPainterPath()
        # Center the text vertically
        fm = QFontMetrics(self.font_obj)
        text_w = fm.horizontalAdvance(self._text)
        text_h = fm.ascent()
        x = (self.width() - text_w) / 2
        y = (self.height() + text_h) / 2 - fm.descent()
        
        path.addText(x, y, self.font_obj, self._text)

        # 3. Draw
        painter.setBrush(grad)
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)

# ======================
# VITALS WINDOW (COMPACT)
# ======================
class VitalsWindow(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        
        self.frame = QFrame()
        # Glassmorphic + Ultra Compact
        self.frame.setStyleSheet(f"background: rgba(8, 8, 8, 230); border: 1px solid {ACCENT_GOLD}; border-radius: 4px;")
        
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setSpacing(2) 
        frame_layout.setContentsMargins(6, 6, 6, 6)
        
        # Header
        head = QHBoxLayout()
        lbl = QLabel("SYSTEM VITALS")
        lbl.setStyleSheet(f"color: {ACCENT_GOLD}; font-weight: 900; font-size: 9px; border:none; background: transparent;")
        btn_x = QPushButton("×")
        btn_x.setFixedSize(14, 14)
        btn_x.clicked.connect(self.close)
        btn_x.setStyleSheet("color: #666; border: none; font-weight: bold; background: transparent; padding:0; margin:0;")
        head.addWidget(lbl)
        head.addStretch()
        head.addWidget(btn_x)
        frame_layout.addLayout(head)
        
        # Bars
        self.bars = {}
        for key in ["VRAM", "CTX", "CPU", "GPU"]:
            row = QHBoxLayout()
            row.setSpacing(4)
            l = QLabel(key)
            l.setStyleSheet("color: #888; font-size: 8px; font-weight: bold; border:none; background: transparent;")
            l.setFixedWidth(22)
            
            bar = QProgressBar()
            bar.setFixedHeight(2) # Ultra thin
            bar.setTextVisible(False)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: #222; border: none; border-radius: 1px; }}
                QProgressBar::chunk {{ background: {FG_ACCENT}; border-radius: 1px; }}
            """)
            bar.setValue(0)
            self.bars[key] = bar
            
            row.addWidget(l)
            row.addWidget(bar)
            frame_layout.addLayout(row)
            
        layout.addWidget(self.frame)
        
        # Make the window itself small
        self.setFixedSize(140, 90)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)
        self.old_pos = None

    def update_stats(self):
        if self.state.ctx_limit > 0:
            ctx_p = int((self.state.ctx_used / self.state.ctx_limit) * 100)
            self.bars["CTX"].setValue(ctx_p)
        import random
        base_load = 10 if not self.state.model_loaded else 40
        self.bars["VRAM"].setValue(base_load + random.randint(0, 5))
        self.bars["CPU"].setValue(random.randint(5, 15))
        self.bars["GPU"].setValue(base_load + random.randint(0, 10))
        
    def mousePressEvent(self, e): self.old_pos = e.globalPosition().toPoint()
    def mouseReleaseEvent(self, e): self.old_pos = None
    def mouseMoveEvent(self, e):
        if self.old_pos:
            delta = e.globalPosition().toPoint() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = e.globalPosition().toPoint()

# ======================
# MODE SELECTOR (GOLD)
# ======================
class ModeSelector(QWidget):
    modeChanged = Signal(str) # "OPERATOR" or "OVERSEER"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 5, 20, 5)
        
        self.btn_op = self._make_box("OPERATOR", True)
        self.btn_ov = self._make_box("OVERSEER", False)
        
        layout.addStretch()
        layout.addWidget(self.btn_op)
        layout.addWidget(self.btn_ov)
        layout.addStretch()

    def _make_box(self, title, active):
        btn = QPushButton(title)
        btn.setFixedSize(120, 35)
        btn.setCheckable(True)
        btn.setChecked(active)
        btn.setCursor(Qt.PointingHandCursor)
        self._style_btn(btn, active)
        btn.clicked.connect(lambda: self._select(title))
        return btn

    def _style_btn(self, btn, active):
        # GOLD highlight when active
        border = ACCENT_GOLD if active else "#333"
        bg = "#1a1a1a" if active else BG_INPUT
        color = ACCENT_GOLD if active else FG_DIM
        weight = "bold" if active else "normal"
        
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {bg}; 
                border: 1px solid {border}; 
                color: {color};
                font-family: 'Segoe UI'; font-size: 10px; font-weight: {weight};
                border-radius: 2px;
            }}
            QPushButton:hover {{ border-color: {ACCENT_GOLD}; color: {FG_TEXT}; }}
        """)

    def _select(self, mode):
        is_op = (mode == "OPERATOR")
        self.btn_op.setChecked(is_op)
        self.btn_ov.setChecked(not is_op)
        
        self._style_btn(self.btn_op, is_op)
        self._style_btn(self.btn_ov, not is_op)
        
        self.modeChanged.emit(mode)

class GradientLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(2)
        self.offset = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._step)
        self.timer.start(33) 

    def _step(self):
        self.offset = (self.offset + 0.015) % 1.0
        self.repaint()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        grad = QLinearGradient(0, 0, self.width(), 0)
        c_gold = QColor(ACCENT_GOLD)
        c_dark = QColor("#111111")
        grad.setSpread(QLinearGradient.RepeatSpread)
        w = self.width()
        start_x = -self.offset * w
        grad.setStart(start_x, 0)
        grad.setFinalStop(start_x + w, 0)
        grad.setColorAt(0.0, c_dark)
        grad.setColorAt(0.5, c_gold)
        grad.setColorAt(1.0, c_dark)
        painter.fillRect(self.rect(), grad)

class SplitControlBlock(QWidget):
    minClicked = Signal()
    maxClicked = Signal()
    closeClicked = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedSize(45, 34)
        layout = QGridLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(1)
        base_style = f"""
            QPushButton {{
                background: #1a1a1a; border: none; color: {FG_DIM};
                font-family: 'Segoe UI'; font-size: 8px;
            }}
            QPushButton:hover {{ background: {ACCENT_GOLD}; color: black; }}
            QPushButton:pressed {{ background: #b08d2b; color: black; }}
        """
        self.btn_min = QPushButton("─")
        self.btn_min.setFixedSize(22, 16) 
        self.btn_min.setStyleSheet(base_style)
        self.btn_min.clicked.connect(self.minClicked)
        
        self.btn_max = QPushButton("□")
        self.btn_max.setFixedSize(22, 16)
        self.btn_max.setStyleSheet(base_style)
        self.btn_max.clicked.connect(self.maxClicked)
        
        self.btn_close = QPushButton("×")
        self.btn_close.setFixedHeight(16)
        self.btn_close.setStyleSheet(f"""
            QPushButton {{
                background: #1a1a1a; border: none; color: {FG_DIM};
                font-family: 'Segoe UI'; font-size: 12px;
            }}
            QPushButton:hover {{ background: {FG_ERROR}; color: white; }}
            QPushButton:pressed {{ background: #a00; color: white; }}
        """)
        self.btn_close.clicked.connect(self.closeClicked)
        layout.addWidget(self.btn_min, 0, 0)
        layout.addWidget(self.btn_max, 0, 1)
        layout.addWidget(self.btn_close, 1, 0, 1, 2)

# --------------------------------------------------
# FILE: ui\components\module_strip.py
# --------------------------------------------------

import math
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF

from core.style import ACCENT_GOLD, FG_DIM, FG_TEXT
from .atoms import SidebarButton

class OverflowArrow(QWidget):
    clicked = Signal()
    def __init__(self):
        super().__init__()
        self.setFixedHeight(15)
        self.setCursor(Qt.PointingHandCursor)
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._anim)
        self.timer.start(50)
        
    def _anim(self):
        self.phase += 0.2
        self.update()
        
    def mousePressEvent(self, e): self.clicked.emit()
        
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        y_off = math.sin(self.phase) * 2
        c = QColor(ACCENT_GOLD)
        c.setAlpha(150)
        p.setPen(QPen(c, 1.5))
        p.setBrush(Qt.NoBrush)
        cx = self.width() / 2
        cy = self.height() / 2 + y_off
        path = QPolygonF([QPoint(cx - 4, cy - 2), QPoint(cx, cy + 3), QPoint(cx + 4, cy - 2)])
        p.drawPolyline(path)

class ModuleIcon(SidebarButton):
    sig_close = Signal(str)
    sig_select = Signal(str)

    def __init__(self, mod_id, icon_char, label_text):
        super().__init__(icon_char, label_text, checkable=True)
        self.mod_id = mod_id
        self.code = icon_char
        self.is_pulsing = False
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda: self.sig_close.emit(self.mod_id))
        
        self.pulse_phase = 0.0
        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self._step_pulse)
        self.pulse_timer.start(50)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton: self.sig_close.emit(self.mod_id)
        elif e.button() == Qt.LeftButton: self.sig_select.emit(self.mod_id)

    def set_active(self, val):
        self.setChecked(val)
        if val: self.is_pulsing = False

    def flash(self):
        if not self.isChecked():
            self.is_pulsing = True
            QTimer.singleShot(2000, lambda: self.set_pulsing(False))

    def set_pulsing(self, val):
        self.is_pulsing = val
        if val: self.pulse_phase = 0.0
        self.update()

    def _step_pulse(self):
        if self.is_pulsing:
            self.pulse_phase += 0.15
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.is_pulsing and not self.isChecked():
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            glow = QColor(ACCENT_GOLD)
            alpha = int(((math.sin(self.pulse_phase) + 1) / 2) * 50)
            glow.setAlpha(alpha)
            p.fillRect(self.rect(), glow)

class ModuleGroup(QWidget):
    sig_item_select = Signal(str)
    sig_item_close = Signal(str)

    def __init__(self, code):
        super().__init__()
        self.code = code
        self.expanded = False
        self.items = {} 
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(2)
        
        self.header = QPushButton(f"{code} ▸")
        self.header.setFixedSize(50, 40)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setStyleSheet(f"""
            QPushButton {{
                color: {FG_DIM}; background: transparent; 
                border: 1px solid #222; font-weight: bold; font-size: 10px;
            }}
            QPushButton:hover {{ color: {FG_TEXT}; border-color: #444; }}
        """)
        self.header.clicked.connect(self.toggle)
        self.layout.addWidget(self.header)
        
        self.container = QWidget()
        self.cont_layout = QVBoxLayout(self.container)
        self.cont_layout.setContentsMargins(10, 0, 0, 5)
        self.cont_layout.setSpacing(2)
        self.container.setVisible(False)
        self.layout.addWidget(self.container)

    def add_item(self, mod_id, icon_char, label_text):
        icon = ModuleIcon(mod_id, icon_char, label_text)
        icon.sig_select.connect(self.sig_item_select)
        icon.sig_close.connect(self.sig_item_close)
        icon.setFixedSize(40, 40)
        icon.lbl_icon.setStyleSheet("font-size: 14px;")
        icon.lbl_text.setVisible(False)
        self.items[mod_id] = icon
        self.cont_layout.addWidget(icon)
        self.update_recent(mod_id)

    def remove_item(self, mod_id):
        if mod_id in self.items:
            icon = self.items.pop(mod_id)
            icon.deleteLater()

    def update_recent(self, mod_id):
        for mid, icon in self.items.items():
            icon.set_pulsing(mid == mod_id)

    def toggle(self):
        self.expanded = not self.expanded
        self.container.setVisible(self.expanded)
        self.header.setText(f"{self.code} {'▾' if self.expanded else '▸'}")

class ModuleStrip(QWidget):
    sig_module_selected = Signal(str)
    sig_module_closed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        self.content = QWidget()
        self.vbox = QVBoxLayout(self.content)
        self.vbox.setContentsMargins(5, 5, 5, 5)
        self.vbox.setSpacing(5)
        self.vbox.addStretch() 
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll)
        self.arrow = OverflowArrow()
        self.arrow.clicked.connect(self.scroll_down)
        self.arrow.setVisible(False)
        layout.addWidget(self.arrow)
        self.groups = {}
        self.singles = {}
        self.registry = {}

    def add_module(self, mod_id, icon_char, label_text):
        self.registry[mod_id] = (icon_char, label_text)
        existing = [mid for mid, (c, l) in self.registry.items() if c == icon_char and mid != mod_id]
        if not existing:
            icon = ModuleIcon(mod_id, icon_char, label_text)
            icon.sig_select.connect(self.sig_module_selected)
            icon.sig_close.connect(self.sig_module_closed)
            self.vbox.insertWidget(self.vbox.count()-1, icon)
            self.singles[mod_id] = icon
        else:
            if icon_char not in self.groups:
                prev_id = existing[0]
                prev_data = self.registry[prev_id]
                prev_icon = self.singles.pop(prev_id)
                self.vbox.removeWidget(prev_icon)
                prev_icon.deleteLater()
                group = ModuleGroup(icon_char)
                group.sig_item_select.connect(self.sig_module_selected)
                group.sig_item_close.connect(self.sig_module_closed)
                self.groups[icon_char] = group
                self.vbox.insertWidget(self.vbox.count()-1, group)
                group.add_item(prev_id, prev_data[0], prev_data[1])
                group.add_item(mod_id, icon_char, label_text)
                if not group.expanded: group.toggle()
            else:
                group = self.groups[icon_char]
                group.add_item(mod_id, icon_char, label_text)
                if not group.expanded: group.toggle()
                group.update_recent(mod_id)
        self._check_overflow()

    def remove_module(self, mod_id):
        if mod_id not in self.registry: return
        icon_char, _ = self.registry.pop(mod_id)
        if mod_id in self.singles:
            icon = self.singles.pop(mod_id)
            icon.deleteLater()
        elif icon_char in self.groups:
            group = self.groups[icon_char]
            group.remove_item(mod_id)
            remaining = [mid for mid, (c, l) in self.registry.items() if c == icon_char]
            if len(remaining) == 1:
                last_id = remaining[0]
                last_data = self.registry[last_id]
                self.vbox.removeWidget(group)
                del self.groups[icon_char]
                group.deleteLater()
                icon = ModuleIcon(last_id, last_data[0], last_data[1])
                icon.sig_select.connect(self.sig_module_selected)
                icon.sig_close.connect(self.sig_module_closed)
                self.vbox.insertWidget(self.vbox.count()-1, icon)
                self.singles[last_id] = icon
        self._check_overflow()

    def select_module(self, mod_id):
        self.deselect_all()
        if mod_id in self.singles:
            self.singles[mod_id].set_active(True)
        else:
            data = self.registry.get(mod_id)
            if data and data[0] in self.groups:
                grp = self.groups[data[0]]
                if mod_id in grp.items:
                    grp.items[mod_id].set_active(True)
    
    def flash_module(self, mod_id):
        if mod_id in self.singles:
            self.singles[mod_id].flash()
        else:
            data = self.registry.get(mod_id)
            if data and data[0] in self.groups:
                grp = self.groups[data[0]]
                if mod_id in grp.items:
                    grp.items[mod_id].flash()

    def deselect_all(self):
        for icon in self.singles.values(): icon.set_active(False)
        for grp in self.groups.values():
            for icon in grp.items.values(): icon.set_active(False)

    def _check_overflow(self):
        self.content.adjustSize()
        overflow = self.content.sizeHint().height() > self.scroll.height()
        self.arrow.setVisible(overflow)

    def scroll_down(self):
        vs = self.scroll.verticalScrollBar()
        vs.setValue(vs.value() + 50)

# --------------------------------------------------
# FILE: ui\modules\injector.py
# --------------------------------------------------

import sys
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QPlainTextEdit, QSplitter, QFrame
)
from PySide6.QtCore import Qt, Signal, QProcess, QUrl
from PySide6.QtGui import QTextCursor, QDragEnterEvent, QDropEvent

from core.style import BG_GROUP, BORDER_DARK, FG_DIM, FG_ACCENT, BG_INPUT, FG_ERROR, ACCENT_GOLD

class InjectorWidget(QWidget):
    sig_closed = Signal()
    sig_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("InjectorRoot")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        toolbar = QFrame()
        toolbar.setFixedHeight(35)
        toolbar.setStyleSheet(f"background: {BG_GROUP}; border-bottom: 1px solid {BORDER_DARK};")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 0, 10, 0)
        
        lbl_title = QLabel("RUNTIME")
        lbl_title.setStyleSheet(f"color: {ACCENT_GOLD}; font-weight: bold; font-size: 11px;")
        
        self.btn_run = QPushButton("▶ EXECUTE")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setStyleSheet(f"""
            QPushButton {{ background: #1a1a1a; color: {FG_ACCENT}; border: 1px solid #333; padding: 4px 10px; font-weight:bold; font-size: 10px;}}
            QPushButton:hover {{ background: #222; border-color: {FG_ACCENT}; }}
        """)
        self.btn_run.clicked.connect(self.run_code)

        btn_close = QPushButton("×")
        btn_close.setFixedSize(20, 20)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet("background: transparent; color: #555; border: none; font-weight: bold; font-size: 14px;")
        btn_close.clicked.connect(self.close_addon)
        
        tb_layout.addWidget(lbl_title)
        tb_layout.addStretch()
        tb_layout.addWidget(self.btn_run)
        tb_layout.addWidget(btn_close)
        
        layout.addWidget(toolbar)
        
        # Splitter (Code | Console)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER_DARK}; }}")
        
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("# Drag .py file here or write code...")
        self.editor.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {BG_INPUT}; color: #dcdcdc; 
                border: none; font-family: 'Consolas', monospace; font-size: 12px; padding: 10px;
            }}
        """)
        
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Output...")
        self.console.setStyleSheet(f"""
            QPlainTextEdit {{
                background: #080808; color: {FG_DIM}; 
                border: none; border-left: 1px solid {BORDER_DARK};
                font-family: 'Consolas', monospace; font-size: 11px; padding: 10px;
            }}
        """)
        
        splitter.addWidget(self.editor)
        splitter.addWidget(self.console)
        splitter.setSizes([400, 400])
        
        layout.addWidget(splitter)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)

    def dragEnterEvent(self, event: QDragEnterEvent):
        # Accept if files (Explorer) or Text (Qt Tree View default drag)
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        file_path = None
        
        # Case 1: Dragged from Explorer (MimeType: text/uri-list)
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()

        # Case 2: Dragged from Databank Tree (MimeType: text/plain usually)
        # The tree might just pass the path string
        elif event.mimeData().hasText():
            text = event.mimeData().text()
            # Clean up if it has file:/// prefix even in text mode
            if text.startswith("file:///"):
                file_path = QUrl(text).toLocalFile()
            elif os.path.exists(text):
                file_path = text

        if file_path and os.path.exists(file_path):
            self._load_file(file_path)
        else:
            self.console.appendHtml(f"<span style='color:{FG_ERROR}'>ERROR: Could not resolve file path.</span>")

    def _load_file(self, path):
        if os.path.isfile(path) and path.endswith(".py"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.editor.setPlainText(f.read())
                self.console.appendHtml(f"<span style='color:{FG_ACCENT}'>→ LOADED: {os.path.basename(path)}</span>")
            except Exception as e:
                self.console.appendHtml(f"<span style='color:{FG_ERROR}'>ERROR: {e}</span>")
        else:
             self.console.appendHtml(f"<span style='color:{FG_ERROR}'>ERROR: Not a .py file</span>")

    def run_code(self):
        code = self.editor.toPlainText()
        if not code.strip(): return
        
        if self.process.state() != QProcess.NotRunning:
            self.console.appendHtml(f"<span style='color:{FG_ERROR}'>BUSY: Process running...</span>")
            return

        self.console.clear()
        self.console.appendHtml(f"<span style='color:{FG_ACCENT}'>→ EXECUTING SCRIPT...</span>")
        self.process.start(sys.executable, ["-c", code])

    def _read_output(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(data)

    def _process_finished(self):
        self.console.appendHtml(f"<br><span style='color:{FG_DIM}'>→ PROCESS TERMINATED</span>")
        self.sig_finished.emit()

    def close_addon(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
        self.sig_closed.emit()
        self.deleteLater()

# --------------------------------------------------
# FILE: ui\modules\manager.py
# --------------------------------------------------

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Signal

from ui.components.atoms import SkeetGroupBox, SkeetButton
from core.style import FG_DIM

class PageAddons(QWidget):
    sig_launch_addon = Signal(str)

    def __init__(self, state):
        super().__init__()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        grp_modules = SkeetGroupBox("AVAILABLE MODULES")
        
        mod_layout = QVBoxLayout()
        mod_layout.setSpacing(15)
        
        lbl_info = QLabel("Select a runtime module to attach to the workspace.")
        lbl_info.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")
        
        btn_injector = SkeetButton("LAUNCH PY INJECTOR")
        btn_injector.clicked.connect(lambda: self.sig_launch_addon.emit("injector"))
        
        mod_layout.addWidget(lbl_info)
        mod_layout.addWidget(btn_injector)
        mod_layout.addStretch()
        
        grp_modules.add_layout(mod_layout)
        layout.addWidget(grp_modules)

# --------------------------------------------------
# FILE: ui\pages\chat.py
# --------------------------------------------------

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QLineEdit, QPushButton
)
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Signal, Qt

from core.style import BG_INPUT, FG_DIM, FG_ACCENT, ACCENT_GOLD
from ui.components.atoms import SkeetGroupBox, SkeetButton, CollapsibleSection

class PageChat(QWidget):
    sig_generate = Signal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state
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
        
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setStyleSheet(f"""
            background: {BG_INPUT}; color: #ccc; border: 1px solid #222; 
            font-family: 'Consolas', monospace; font-size: 12px;
        """)
        chat_layout.addWidget(self.chat)
        
        self.drawer = CollapsibleSection("▼ PRE-CONTEXT INJECTION")
        d_layout = QVBoxLayout()
        d_layout.setContentsMargins(5,5,5,5)
        self.txt_ctx = QLineEdit()
        self.txt_ctx.setPlaceholderText("Override system prompt...")
        self.txt_ctx.setStyleSheet(f"""
            background: {BG_INPUT}; color: #888; border: 1px solid #333; 
            font-family: 'Verdana'; font-size: 11px; padding: 4px;
        """)
        self.txt_ctx.textChanged.connect(lambda: setattr(self.state, 'context_injection', self.txt_ctx.text()))
        d_layout.addWidget(self.txt_ctx)
        self.drawer.set_content_layout(d_layout)
        chat_layout.addWidget(self.drawer)
        
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
        layout.addLayout(left_col, 3)

        right_col = QVBoxLayout()
        trace_group = SkeetGroupBox("REASONING TRACE")
        self.trace = QTextEdit()
        self.trace.setReadOnly(True)
        self.trace.setStyleSheet(f"""
            background: {BG_INPUT}; color: {FG_ACCENT}; border: 1px solid #222; 
            font-family: 'Consolas', monospace; font-size: 10px;
        """)
        trace_group.add_widget(self.trace)
        right_col.addWidget(trace_group)
        layout.addLayout(right_col, 2)

    def send(self):
        txt = self.input.text().strip()
        if not txt: return
        self.input.clear()
        self.chat.append(f"<span style='color:{ACCENT_GOLD}'><b>USER:</b></span> {txt}")
        self.chat.append(f"<span style='color:{FG_ACCENT}'><b>MONOLITH:</b></span>")
        self.chat.moveCursor(QTextCursor.End)
        self.sig_generate.emit(txt)

    def append_token(self, t): self.chat.insertPlainText(t)
    def append_trace(self, html): self.trace.append(html)
    def clear_chat(self):
        self.chat.clear()
        self.trace.clear()
        self.chat.append(f"<span style='color:{FG_DIM}'>--- SESSION RESET ---</span>")

# --------------------------------------------------
# FILE: ui\pages\databank.py
# --------------------------------------------------

import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
    QTreeView, QHeaderView, QFileSystemModel, QInputDialog, 
    QLabel, QMessageBox, QMenu
)
from PySide6.QtCore import QDir, Qt

from ui.components.atoms import SkeetGroupBox, SkeetButton
from core.style import BG_INPUT, BORDER_DARK, FG_DIM, ACCENT_GOLD, BG_MAIN, FG_TEXT

class TerminalFileTree(QTreeView):
    def __init__(self, start_path):
        super().__init__()
        self.model = QFileSystemModel()
        
        self.model.setReadOnly(False)
        self.model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        self.model.setNameFilterDisables(False)
        
        self.change_root(start_path)
        self.setModel(self.model)

        self.setDragEnabled(True) 
        self.setDragDropMode(QTreeView.DragOnly)

        self.setStyleSheet(f"""
            QTreeView {{
                background: {BG_INPUT};
                color: #ccc;
                border: 1px solid {BORDER_DARK};
                font-family: 'Consolas', monospace;
                font-size: 12px;
                outline: 0;
            }}
            QTreeView::item {{ padding: 4px; }}
            QTreeView::item:hover {{ background: #222; }}
            QTreeView::item:selected {{ background: {ACCENT_GOLD}; color: black; }}
            
            QHeaderView::section {{
                background: #111;
                color: {FG_DIM};
                border: none;
                padding: 4px;
                font-weight: bold;
            }}
        """)
        
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.setColumnWidth(1, 80)
        self.setColumnHidden(2, True) 
        self.setColumnHidden(3, True) 
        self.setAnimated(False)
        self.setIndentation(20)
        self.setSortingEnabled(False)
        self.setContextMenuPolicy(Qt.CustomContextMenu)

    def change_root(self, path):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            try:
                os.makedirs(abs_path)
            except OSError:
                pass 
                
        self.model.setRootPath(abs_path)
        self.setRootIndex(self.model.index(abs_path))

class PageFiles(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        
        base_dir = "C:\\Models\\knowledge_base"
        if not os.path.exists("C:\\Models"):
            base_dir = os.path.join(os.getcwd(), "knowledge_base")
            
        self.current_path = base_dir
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        grp = SkeetGroupBox("DATABANK")
        gl = QVBoxLayout()
        gl.setSpacing(10)
        
        # --- TOP EXPLORER BAR ---
        nav_bar = QHBoxLayout()
        
        self.inp_path = QLineEdit()
        self.inp_path.setText(self.current_path)
        self.inp_path.setPlaceholderText("Path...")
        self.inp_path.setStyleSheet(f"""
            QLineEdit {{
                background: {BG_INPUT}; color: {ACCENT_GOLD}; 
                border: 1px solid #333; padding: 6px; font-family: 'Consolas';
            }}
        """)
        self.inp_path.returnPressed.connect(self.navigate_to_path)
        
        self.inp_search = QLineEdit()
        self.inp_search.setPlaceholderText("Search files...")
        self.inp_search.setFixedWidth(200)
        self.inp_search.setStyleSheet(f"""
            QLineEdit {{
                background: {BG_INPUT}; color: white; 
                border: 1px solid #333; padding: 6px;
            }}
        """)
        self.inp_search.textChanged.connect(self.on_search)
        
        nav_bar.addWidget(QLabel("📂"))
        nav_bar.addWidget(self.inp_path)
        nav_bar.addSpacing(10)
        nav_bar.addWidget(QLabel("🔍"))
        nav_bar.addWidget(self.inp_search)
        gl.addLayout(nav_bar)
        
        # --- FILE TREE ---
        self.tree = TerminalFileTree(self.current_path)
        self.tree.customContextMenuRequested.connect(self.open_menu)
        self.tree.clicked.connect(self.on_click_item) 
        gl.addWidget(self.tree)
        
        # --- BOTTOM ACTION BAR ---
        actions = QHBoxLayout()
        
        btn_add = SkeetButton("+ MKDIR")
        btn_add.clicked.connect(self.new_folder)
        
        btn_del = SkeetButton("× DELETE")
        btn_del.clicked.connect(self.delete_item)
        
        btn_ref = SkeetButton("⟳ REFRESH")
        btn_ref.clicked.connect(self.refresh)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")
        
        actions.addWidget(btn_add)
        actions.addWidget(btn_del)
        actions.addWidget(btn_ref)
        actions.addStretch()
        actions.addWidget(self.lbl_status)
        
        gl.addLayout(actions)
        grp.add_layout(gl)
        layout.addWidget(grp)

    def navigate_to_path(self):
        new_path = self.inp_path.text()
        if os.path.exists(new_path):
            self.current_path = new_path
            self.tree.change_root(new_path)
            self.lbl_status.setText(f"Navigated to: {new_path}")
        else:
            self.lbl_status.setText("Error: Path does not exist")

    def on_click_item(self, index):
        path = self.tree.model.filePath(index)
        if os.path.isdir(path):
            self.inp_path.setText(path)

    def refresh(self):
        self.tree.change_root(self.current_path)
        self.lbl_status.setText("Refreshed")

    def on_search(self, text):
        if text:
            self.tree.model.setNameFilters([f"*{text}*"])
        else:
            self.tree.model.setNameFilters([])

    def get_selected_path(self):
        indexes = self.tree.selectedIndexes()
        if indexes:
            return self.tree.model.filePath(indexes[0])
        return self.current_path

    def new_folder(self):
        target_path = self.get_selected_path()
        if os.path.isfile(target_path):
            target_path = os.path.dirname(target_path)
            
        name, ok = self.ask_input("New Folder", "Folder Name:")
        if ok and name:
            new_dir = os.path.join(target_path, name)
            try:
                os.makedirs(new_dir, exist_ok=True)
                self.lbl_status.setText(f"Created: {name}")
            except Exception as e:
                self.lbl_status.setText(f"Error: {e}")

    def delete_item(self):
        target = self.get_selected_path()
        if target == self.current_path:
            self.lbl_status.setText("Cannot delete root folder")
            return
            
        if os.path.exists(target):
            msg = QMessageBox(self)
            msg.setWindowTitle("Confirm Delete")
            msg.setText(f"Are you sure you want to delete:\n{os.path.basename(target)}?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setStyleSheet(f"""
                QMessageBox {{ background: {BG_MAIN}; }}
                QLabel {{ color: {FG_TEXT}; }}
                QPushButton {{ background: #222; color: #ccc; border: 1px solid #444; padding: 5px; }}
            """)
            if msg.exec() == QMessageBox.Yes:
                try:
                    if os.path.isdir(target):
                        shutil.rmtree(target)
                    else:
                        os.remove(target)
                    self.lbl_status.setText("Item deleted")
                except Exception as e:
                    self.lbl_status.setText(f"Delete Error: {e}")

    def open_menu(self, position):
        menu = QMenu()
        menu.setStyleSheet(f"""
            QMenu {{ background: #111; color: {FG_TEXT}; border: 1px solid {ACCENT_GOLD}; }}
            QMenu::item:selected {{ background: {ACCENT_GOLD}; color: black; }}
        """)
        
        act_del = QAction("Delete", self)
        act_del.triggered.connect(self.delete_item)
        
        act_new = QAction("New Folder", self)
        act_new.triggered.connect(self.new_folder)
        
        menu.addAction(act_new)
        menu.addAction(act_del)
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def ask_input(self, title, label):
        dlg = QInputDialog(self)
        dlg.setWindowTitle(title)
        dlg.setLabelText(label)
        dlg.setStyleSheet(f"""
            QDialog {{ background: {BG_MAIN}; border: 1px solid {ACCENT_GOLD}; }}
            QLabel {{ color: {FG_TEXT}; }}
            QLineEdit {{ background: {BG_INPUT}; color: white; border: 1px solid #333; }}
            QPushButton {{ background: #222; color: #ccc; border: 1px solid #444; padding: 5px; }}
        """)
        ok = dlg.exec()
        return dlg.textValue(), (ok == 1)

# --------------------------------------------------
# FILE: ui\pages\settings.py
# --------------------------------------------------

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
        sys.stdout = self.sys_out
        sys.stderr = self.sys_out

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
