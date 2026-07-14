from __future__ import annotations

import json
import os
import shlex
import sys

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QWidget

import core.style as s
from ui.components.atoms import MonoButton, MonoGroupBox


class ExternalProcessModule(QWidget):
    def __init__(
        self,
        name: str,
        entry_path: str,
        command: str | None = None,
        workdir: str | None = None,
        ui_bridge=None,
        state=None,
        guard=None,
    ):
        super().__init__()
        self._name = name
        self._entry_path = entry_path
        self._command = command
        self._workdir = workdir
        self._ui_bridge = ui_bridge
        self._state = state
        self._guard = guard
        self._last_output = ""
        self._rpc_buffer = ""
        self._rpc_id = 0
        self._registered_verbs: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(10)

        box = MonoGroupBox("EXTERNAL ADDON")
        box_layout = QVBoxLayout()
        box_layout.setSpacing(10)

        title = QLabel(self._name)
        title.setStyleSheet(f"color: {s.FG_TEXT}; font-size: 14px; font-weight: bold;")
        box_layout.addWidget(title)

        path_label = QLabel(self._entry_path)
        path_label.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        box_layout.addWidget(path_label)

        cmd_label = QLabel(self._command or f"{sys.executable} {self._entry_path}")
        cmd_label.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        cmd_label.setWordWrap(True)
        box_layout.addWidget(cmd_label)

        btn_row = QHBoxLayout()
        self.btn_start = MonoButton("START", accent=True)
        self.btn_stop = MonoButton("STOP")
        self.btn_restart = MonoButton("RESTART")
        for btn in (self.btn_start, self.btn_stop, self.btn_restart):
            btn.setFixedHeight(26)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_restart)
        btn_row.addStretch()
        box_layout.addLayout(btn_row)

        self.status_label = QLabel("Stopped")
        self.status_label.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        box_layout.addWidget(self.status_label)

        self.output_label = QLabel("")
        self.output_label.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        self.output_label.setWordWrap(True)
        box_layout.addWidget(self.output_label)

        box.add_layout(box_layout)
        root.addWidget(box)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.started.connect(lambda: self._set_status("Running"))
        self._process.started.connect(lambda: self._send_notification("monolith.ready", {}))
        self._process.finished.connect(lambda _code, _status: self._set_status("Stopped"))
        self._process.errorOccurred.connect(lambda _err: self._set_status("Error", warn=True))

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_restart.clicked.connect(self._restart)

        self.destroyed.connect(lambda *_args: self._stop(force=True))

    def on_engine_event(self, engine_key, event, payload) -> None:
        self._send_notification(
            "monolith.engine_event",
            {"engine_key": engine_key, "event": event, "payload": payload},
        )

    def _set_status(self, text: str, warn: bool = False) -> None:
        color = s.FG_WARN if warn else s.FG_DIM
        self.status_label.setStyleSheet(f"color: {color}; font-size: 10px;")
        self.status_label.setText(text)
        if self._ui_bridge is not None:
            sev = "WARNING" if warn else "INFO"
            self._ui_bridge.sig_monitor_log.emit(sev, f"[{self._name}] {text}")

    def _start(self) -> None:
        if self._process.state() != QProcess.NotRunning:
            return
        if self._ui_bridge is not None:
            self._ui_bridge.sig_monitor_log.emit("INFO", f"[{self._name}] start requested")
        workdir = self._workdir or (os.path.dirname(self._entry_path) if self._entry_path else ".")
        self._process.setWorkingDirectory(workdir)
        if self._command:
            parts = shlex.split(self._command, posix=False)
            if parts:
                program, args = parts[0], parts[1:]
                self._process.start(program, args)
                return
        if self._entry_path:
            self._process.start(sys.executable, [self._entry_path])

    def _stop(self, force: bool = False) -> None:
        if self._process.state() == QProcess.NotRunning:
            return
        if self._ui_bridge is not None:
            self._ui_bridge.sig_monitor_log.emit("INFO", f"[{self._name}] stop requested")
        self._process.terminate()
        if force:
            self._process.kill()
            return
        if not self._process.waitForFinished(2000):
            self._process.kill()

    def _restart(self) -> None:
        if self._ui_bridge is not None:
            self._ui_bridge.sig_monitor_log.emit("INFO", f"[{self._name}] Restart requested")
        self._stop()
        self._start()

    def _on_output(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode(errors="ignore")
        if not data:
            return
        self._last_output = data.strip().splitlines()[-1]
        if self._last_output:
            self.output_label.setText(f"Last output: {self._last_output}")
        self._consume_rpc_data(data)

    # ---- JSON-RPC (newline delimited) ----

    def _send_notification(self, method: str, params: dict) -> None:
        self._send_rpc({"jsonrpc": "2.0", "method": method, "params": params})

    def _send_response(self, rpc_id: object, result: object = None, error: object = None) -> None:
        payload = {"jsonrpc": "2.0", "id": rpc_id}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        self._send_rpc(payload)

    def _send_rpc(self, payload: dict) -> None:
        try:
            data = (json.dumps(payload) + "\n").encode("utf-8")
            self._process.write(data)
        except Exception:
            pass

    def _consume_rpc_data(self, data: str) -> None:
        if not data:
            return
        self._rpc_buffer += data
        while "\n" in self._rpc_buffer:
            line, self._rpc_buffer = self._rpc_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            self._handle_rpc_message(msg)

    def _handle_rpc_message(self, msg: dict) -> None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return
        method = msg.get("method")
        params = msg.get("params") or {}
        rpc_id = msg.get("id")

        if method == "monolith.state":
            result = self._snapshot_state()
            if rpc_id is not None:
                self._send_response(rpc_id, result=result)
            return

        if method == "monolith.emit":
            signal = str(params.get("signal", "monitor_log"))
            message = str(params.get("message", ""))
            level = str(params.get("level", "INFO")).upper()
            if self._ui_bridge is not None and signal == "monitor_log":
                self._ui_bridge.sig_monitor_log.emit(level, message)
            if self._guard is not None and signal == "trace":
                self._guard.sig_trace.emit("external", message)
            if rpc_id is not None:
                self._send_response(rpc_id, result={"ok": True})
            return

        if method == "monolith.register_verbs":
            verbs = params.get("verbs") or []
            if isinstance(verbs, list):
                self._registered_verbs = [str(v) for v in verbs]
                self.status_label.setText(f"Running • verbs={', '.join(self._registered_verbs)}")
            if rpc_id is not None:
                self._send_response(rpc_id, result={"ok": True})
            return

        if rpc_id is not None:
            self._send_response(rpc_id, error={"code": -32601, "message": "Method not found"})

    def _snapshot_state(self) -> dict:
        state = self._state
        if state is None:
            return {"status": "unknown"}
        return {
            "status": getattr(state, "status", None).value if getattr(state, "status", None) else None,
            "ctx_limit": getattr(state, "ctx_limit", None),
            "ctx_used": getattr(state, "ctx_used", None),
        }
