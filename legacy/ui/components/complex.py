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
