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
