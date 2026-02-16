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

        toolbar = QFrame()
        toolbar.setObjectName("injector_toolbar")
        toolbar.setFixedHeight(35)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 0, 10, 0)

        lbl_title = QLabel("RUNTIME")
        lbl_title.setObjectName("injector_title")

        self.btn_run = QPushButton("▶ EXECUTE")
        self.btn_run.setObjectName("injector_run")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self.run_code)

        btn_close = QPushButton("×")
        btn_close.setObjectName("injector_close")
        btn_close.setFixedSize(20, 20)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.clicked.connect(self.close_addon)

        tb_layout.addWidget(lbl_title)
        tb_layout.addStretch()
        tb_layout.addWidget(self.btn_run)
        tb_layout.addWidget(btn_close)

        layout.addWidget(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setObjectName("injector_splitter")

        self.editor = QPlainTextEdit()
        self.editor.setObjectName("injector_editor")
        self.editor.setPlaceholderText("# Drag .py file here or write code...")

        self.console = QPlainTextEdit()
        self.console.setObjectName("injector_console")
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Output...")

        splitter.addWidget(self.editor)
        splitter.addWidget(self.console)
        splitter.setSizes([400, 400])
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
            self.console.appendHtml("<span style='color:#ef4444'>ERROR: Could not resolve file path.</span>")

    def _load_file(self, path):
        if os.path.isfile(path) and path.endswith(".py"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.editor.setPlainText(f.read())
                self.console.appendHtml(f"<span style='color:#6d8cff'>→ LOADED: {os.path.basename(path)}</span>")
            except Exception as e:
                self.console.appendHtml(f"<span style='color:#ef4444'>ERROR: {e}</span>")
        else:
            self.console.appendHtml("<span style='color:#ef4444'>ERROR: Not a .py file</span>")

    def run_code(self):
        code = self.editor.toPlainText()
        if not code.strip():
            return
        if self.process.state() != QProcess.NotRunning:
            self.console.appendHtml("<span style='color:#ef4444'>BUSY: Process running...</span>")
            return

        self.console.clear()
        self.console.appendHtml("<span style='color:#6d8cff'>→ EXECUTING SCRIPT...</span>")
        self.process.start(sys.executable, ["-c", code])

    def _read_output(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(data)

    def _process_finished(self):
        self.console.appendHtml("<br><span style='color:#6b7280'>→ PROCESS TERMINATED</span>")
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
