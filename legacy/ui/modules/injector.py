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

    def closeEvent(self, event):
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.process.waitForFinished(300)
        event.accept()
