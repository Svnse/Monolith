import os
import sys

from PySide6.QtCore import QProcess, Qt, QUrl, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import core.style as _s


class InjectorWidget(QWidget):
    sig_closed = Signal()
    sig_finished = Signal()

    def __init__(self, parent=None, ui_bridge=None):
        super().__init__(parent)
        self._ui_bridge = ui_bridge
        self.setAcceptDrops(True)
        self.setObjectName("InjectorRoot")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar — no title label: the companion pane already titles the
        # tab as "RUNTIME". EXECUTE button stays on the left so it's the
        # primary action; the × close stays for top-level / pop-out usage.
        toolbar = QFrame()
        toolbar.setObjectName("injector_toolbar")
        toolbar.setFixedHeight(32)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 0, 8, 0)

        self.btn_run = QPushButton("▶ EXECUTE")
        self.btn_run.setObjectName("injector_run")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self.run_code)

        btn_close = QPushButton("×")
        btn_close.setObjectName("injector_close")
        btn_close.setFixedSize(20, 20)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.clicked.connect(self.close_addon)

        tb_layout.addWidget(self.btn_run)
        tb_layout.addStretch()
        tb_layout.addWidget(btn_close)

        layout.addWidget(toolbar)

        # Vertical splitter: editor on top, console on bottom. Handle is
        # wide + accent-colored so the user can clearly see where to drag
        # to resize the input vs. output split.
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(6)
        splitter.setObjectName("injector_splitter")
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {_s.BORDER_LIGHT}; }}"
            f"QSplitter::handle:hover {{ background: {_s.ACCENT_PRIMARY}; }}"
        )

        self.editor = QPlainTextEdit()
        self.editor.setObjectName("injector_editor")
        self.editor.setPlaceholderText("# Drag .py file here or write code...")

        self.console = QPlainTextEdit()
        self.console.setObjectName("injector_console")
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Output...")

        splitter.addWidget(self.editor)
        splitter.addWidget(self.console)
        splitter.setSizes([300, 200])
        layout.addWidget(splitter)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        file_path = None
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
        elif event.mimeData().hasText():
            text = event.mimeData().text()
            if text.startswith("file:///"):
                file_path = QUrl(text).toLocalFile()
            elif os.path.exists(text):
                file_path = text

        if file_path and os.path.exists(file_path):
            self._load_file(file_path)
        else:
            self.console.appendHtml(f"<span style='color:{_s.FG_ERROR}'>ERROR: Could not resolve file path.</span>")

    def _load_file(self, path):
        if os.path.isfile(path) and path.endswith(".py"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.editor.setPlainText(f.read())
                self.console.appendHtml(f"<span style='color:{_s.FG_INFO}'>→ LOADED: {os.path.basename(path)}</span>")
                self._log("INFO", f"[injector] loaded {os.path.basename(path)}")
            except Exception as e:
                self.console.appendHtml(f"<span style='color:{_s.FG_ERROR}'>ERROR: {e}</span>")
                self._log("ERROR", f"[injector] load failed: {e}")
        else:
            self.console.appendHtml(f"<span style='color:{_s.FG_ERROR}'>ERROR: Not a .py file</span>")
            self._log("WARNING", "[injector] not a .py file")

    def run_code(self):
        code = self.editor.toPlainText()
        if not code.strip():
            return
        if self.process.state() != QProcess.NotRunning:
            self.console.appendHtml(f"<span style='color:{_s.FG_ERROR}'>BUSY: Process running...</span>")
            self._log("WARNING", "[injector] busy: process running")
            return

        self.console.clear()
        self.console.appendHtml(f"<span style='color:{_s.FG_INFO}'>→ EXECUTING SCRIPT...</span>")
        self._log("INFO", "[injector] executing script")
        self.process.start(sys.executable, ["-c", code])

    def _read_output(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(data)

    def _process_finished(self):
        self.console.appendHtml(f"<br><span style='color:{_s.FG_DIM}'>→ PROCESS TERMINATED</span>")
        self._log("INFO", "[injector] process terminated")
        self.sig_finished.emit()

    def close_addon(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
        self.sig_closed.emit()
        self._log("INFO", "[injector] closed")
        self.deleteLater()

    def closeEvent(self, event):
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.process.waitForFinished(300)
        event.accept()

    def _log(self, severity: str, message: str) -> None:
        if self._ui_bridge is None:
            return
        self._ui_bridge.sig_monitor_log.emit(severity, message)
