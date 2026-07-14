"""
ui/pages/connections.py — Connections module

Lets external agents (Claude Code via MCP, Kimi via HTTP, etc.)
join the active chat session as a participant.

The server is owned by this page and lives for as long as the tab is open.
When an agent sends a message it is attached as an `agent` message in the
active chat timeline. Dispatch to the local LLM is controlled by routing mode:
manual/advisory require explicit user approval, auto dispatches immediately.

Layout (companion pane addon):
  ┌─────────────────────────────┐
  │  CONNECTIONS                │
  │ ┌─ SERVER ────────────────┐ │
  │ │ Port [7821] ● running   │ │
  │ │          [START] [STOP] │ │
  │ └─────────────────────────┘ │
  │ ┌─ CONNECT ───────────────┐ │
  │ │ HTTP  POST …/chat  [⧉]  │ │
  │ │ MCP   python …    [⧉]  │ │
  │ └─────────────────────────┘ │
  │ ┌─ ACTIVITY ──────────────┐ │
  │ │  scrolling log          │ │
  │ └─────────────────────────┘ │
  └─────────────────────────────┘
"""

from __future__ import annotations

import copy
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QCheckBox,
    QVBoxLayout,
    QWidget,
)

import core.style as s
from core.channel_tag import strip_channel_tags_for_display
from core.message_interceptors import register_interceptor
from core.state import SystemStatus
from engine.agent_server import (
    AgentServer,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_GENERATION_START,
    EVENT_STATUS,
    EVENT_TOKEN,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    _set_active_server,
)
from engine.external_agents import (
    dispatch as dispatch_external,
    load_peers,
    add_peer,
    remove_peer,
    get_peers,
    peer_names,
    parse_mentions,
    strip_mention,
    ping_peer,
)
from ui.components.atoms import MonoButton, MonoGroupBox, CollapsibleSection


# ── thread-safe signal relay ───────────────────────────────────────────────

class _Relay(QObject):
    """Carries events from the server thread into the Qt main thread."""
    message_received = Signal(str, str)  # agent_name, text
    log_line = Signal(str)
    agent_reply = Signal(str, str)       # label, reply_text — from external agent thread
    participant_changed = Signal()       # participant joined or left — refresh connected agents UI
    reset_requested = Signal()           # /reset endpoint — start a fresh chat surface (New Chat)
    load_requested = Signal()            # /load_model endpoint — load the selected model


# ── page ───────────────────────────────────────────────────────────────────

class ConnectionsPage(QWidget):
    MODE_MANUAL = "manual"
    MODE_ADVISORY = "advisory"
    MODE_AUTO = "auto"

    def __init__(self, state, ui_bridge, guard=None, ctx=None, parent=None):
        super().__init__(parent)
        self._state = state
        self._ui_bridge = ui_bridge
        self._guard = guard
        self._ctx = ctx
        self._active_engine_key: str | None = None
        import os as _os
        self._autostart_enabled = _os.getenv("MONOLITH_AGENT_AUTOSTART", "").strip().lower() in ("1", "true", "yes", "on")
        self._active_chat_ref: QWidget | None = None
        self._awaiting_engine_completion: bool = False
        self._pending_agent_messages: list[dict[str, object]] = []
        self._completion_poll_timer = QTimer(self)
        self._completion_poll_timer.setSingleShot(True)
        self._completion_poll_timer.setInterval(100)
        self._completion_poll_timer.timeout.connect(self._poll_active_request_completion)
        self._state_snapshot_lock = threading.Lock()
        self._state_snapshot_cache: dict = {
            "chat": None,
            "model_status": "no_active_chat",
            "recent_messages": [],
            "message_count": 0,
            "routing_mode": self.MODE_AUTO,
            "pending_approvals": 0,
        }

        # Thread-safe relay from server callbacks → Qt slots
        self._relay = _Relay()
        self._relay.message_received.connect(self._on_agent_message)
        self._relay.log_line.connect(self._append_log)
        self._relay.agent_reply.connect(self._on_external_agent_reply)
        self._relay.participant_changed.connect(self._refresh_connected_agents)
        self._relay.reset_requested.connect(self._on_reset_requested)
        self._relay.load_requested.connect(self._on_load_requested)

        # Load peer registry (external agents reachable by URL)
        load_peers()

        # Server (no Qt deps — safe to call from any thread)
        self._server = AgentServer()
        self._server.on_message = lambda name, msg: self._relay.message_received.emit(name, msg)
        self._server.on_log = lambda text: self._relay.log_line.emit(text)
        self._server.on_state_request = self._build_state_snapshot
        self._server.on_participant_change = lambda: self._relay.participant_changed.emit()
        # Debug-read callbacks (added 2026-05-11). Worker-thread reads of the
        # active PageChat's session + interceptor state, backed by the same
        # 250ms cache used by _build_state_snapshot.
        self._server.on_session_messages = self._build_session_messages_snapshot
        self._server.on_interceptor_state = self._build_interceptor_state_snapshot
        # /reset endpoint — start a fresh/cold chat surface the way "New Chat"
        # does. Called from the server thread; emits a Qt signal to hop onto the
        # main thread (fire-and-forget, like on_message), so it returns a
        # *dispatch* receipt rather than a completion claim.
        self._server.on_reset = self._dispatch_reset
        # /load_model — load the selected model so /chat works after an autonomous
        # restart. Fire-and-forget hop onto the Qt thread (like on_reset/on_message).
        self._server.on_load_model = lambda: self._relay.load_requested.emit()

        self._build_ui()
        self._register_nudge_interceptor()

        if guard is not None:
            guard.sig_token.connect(self._on_token)
            guard.sig_status.connect(self._on_engine_status)

        # Keep a thread-safe copy for /state handlers running off the Qt thread.
        self._state_snapshot_timer = QTimer(self)
        self._state_snapshot_timer.setInterval(250)
        self._state_snapshot_timer.timeout.connect(self._refresh_state_snapshot_cache)
        self._state_snapshot_timer.start()
        self._refresh_state_snapshot_cache()

        if self._autostart_enabled:
            # Defer so the guard/engine + UI are fully ready before binding the port.
            QTimer.singleShot(300, self._autostart_server)

    # ── nudge interceptor ─────────────────────────────────────────────────

    def _register_nudge_interceptor(self) -> None:
        """Register a message interceptor that injects active nudges into the system prompt."""
        server = self._server

        def _nudge_interceptor(messages, config):
            nudges = server.consume_nudges()
            if not nudges:
                return None  # no change
            # Find the system message and append nudges
            for msg in messages:
                if msg.get("role") == "system":
                    nudge_block = "\n\n[AGENT NUDGES — ephemeral context from external agents]\n"
                    for n in nudges:
                        nudge_block += f"- {n}\n"
                    msg["content"] = msg["content"] + nudge_block
                    break
            return messages

        register_interceptor(_nudge_interceptor)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Outer layout: a single scroll area wrapping the body. The body holds
        # two always-visible groups (SERVER, ROUTING) and three collapsibles
        # for reference / observability sections (ENDPOINTS, AGENTS, ACTIVITY).
        # The companion-pane resize grip is what controls the actual width;
        # this just makes sure overflow scrolls instead of squeezing.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        body = QWidget()
        root = QVBoxLayout(body)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        scroll.setWidget(body)

        # No H1: the companion pane already titles the tab. Removing the
        # duplicate "CONNECTIONS" header reclaims a row at the top.

        # ── SERVER ───────────────────────────────────────────────────────
        # Compact: port + interfaces-checkbox on row 1, status + start/stop
        # on row 2. The two control rows replace what was previously a
        # 4-row stack (port label + spin + checkbox + button row).
        server_box = MonoGroupBox("SERVER")
        srv_layout = QVBoxLayout()
        srv_layout.setSpacing(6)

        port_row = QHBoxLayout()
        port_row.setSpacing(8)
        port_lbl = QLabel("Port")
        port_lbl.setStyleSheet(f"color: {s.FG_DIM}; font-size: 11px;")
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(7821)
        self._port_spin.setFixedWidth(72)
        self._port_spin.setStyleSheet(
            f"QSpinBox {{ background: {s.BG_INPUT}; color: {s.FG_TEXT};"
            f" border: 1px solid {s.BORDER_DARK}; border-radius: 3px;"
            f" padding: 2px 6px; font-size: 11px; }}"
        )
        self._all_interfaces_check = QCheckBox("0.0.0.0")
        self._all_interfaces_check.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        self._all_interfaces_check.setToolTip(
            "Bind to all interfaces so sandboxed agents (Codex, etc.) can\n"
            "connect via LAN IP. Requires MONOLITH_AGENT_TOKEN and exposes\n"
            "the port on the local network."
        )
        port_row.addWidget(port_lbl)
        port_row.addWidget(self._port_spin)
        port_row.addWidget(self._all_interfaces_check)
        port_row.addStretch()
        srv_layout.addLayout(port_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._status_dot = QLabel("● stopped")
        self._status_dot.setStyleSheet(f"color: {s.FG_DIM}; font-size: 11px;")
        self._btn_start = MonoButton("START", accent=True)
        self._btn_start.setFixedHeight(24)
        self._btn_stop = MonoButton("STOP")
        self._btn_stop.setFixedHeight(24)
        self._btn_stop.setEnabled(False)
        btn_row.addWidget(self._status_dot)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        srv_layout.addLayout(btn_row)

        server_box.add_layout(srv_layout)
        root.addWidget(server_box)

        # ── ROUTING ──────────────────────────────────────────────────────
        # Primary action surface (after starting the server, this is the
        # main thing the user touches). Keeps the pending controls visible
        # so the user notices queued approvals immediately.
        policy_box = MonoGroupBox("ROUTING")
        policy_layout = QVBoxLayout()
        policy_layout.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Mode")
        mode_lbl.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        self._mode_combo = QComboBox()
        self._mode_combo.setFixedHeight(24)
        self._mode_combo.addItem("manual", self.MODE_MANUAL)
        self._mode_combo.addItem("advisory", self.MODE_ADVISORY)
        self._mode_combo.addItem("auto (default)", self.MODE_AUTO)
        self._mode_combo.setCurrentIndex(2)  # auto — peer /chat auto-routes (autonomous default)
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self._mode_combo, 1)
        policy_layout.addLayout(mode_row)

        self._pending_label = QLabel("Pending approvals: 0")
        self._pending_label.setStyleSheet(
            f"color: {s.FG_DIM}; font-size: 10px; font-family: monospace;"
        )
        policy_layout.addWidget(self._pending_label)

        pending_btn_row = QHBoxLayout()
        pending_btn_row.setSpacing(6)
        self._btn_approve_pending = MonoButton("APPROVE NEXT")
        self._btn_approve_pending.setFixedHeight(24)
        self._btn_reject_pending = MonoButton("REJECT NEXT")
        self._btn_reject_pending.setFixedHeight(24)
        self._btn_approve_pending.setEnabled(False)
        self._btn_reject_pending.setEnabled(False)
        pending_btn_row.addWidget(self._btn_approve_pending)
        pending_btn_row.addWidget(self._btn_reject_pending)
        policy_layout.addLayout(pending_btn_row)

        policy_box.add_layout(policy_layout)
        root.addWidget(policy_box)

        # ── ENDPOINTS (collapsed) ────────────────────────────────────────
        # Reference data: copy-paste URLs for external agents to connect to.
        # Not used in steady state; collapsed by default.
        endpoints_section = CollapsibleSection("ENDPOINTS")
        endpoints_inner = QVBoxLayout()
        endpoints_inner.setContentsMargins(0, 4, 0, 4)
        endpoints_inner.setSpacing(4)
        self._http_row = _CopyRow("HTTP", "— not running")
        self._mcp_row = _CopyRow("MCP ", "— not running")
        self._sse_row = _CopyRow("SSE ", "— not running")
        self._state_row = _CopyRow("OBS ", "— not running")
        endpoints_inner.addWidget(self._http_row)
        endpoints_inner.addWidget(self._sse_row)
        endpoints_inner.addWidget(self._state_row)
        endpoints_inner.addWidget(self._mcp_row)
        endpoints_section.set_content_layout(endpoints_inner)
        root.addWidget(endpoints_section)

        # ── AGENTS (collapsed) ───────────────────────────────────────────
        # List of currently-joined external participants. Empty in most
        # sessions; collapse to avoid the "— none connected —" placeholder.
        agents_section = CollapsibleSection("AGENTS")
        agents_inner = QVBoxLayout()
        agents_inner.setContentsMargins(0, 4, 0, 4)
        agents_inner.setSpacing(4)
        self._agents_list_vbox = QVBoxLayout()
        self._agents_list_vbox.setSpacing(3)
        agents_inner.addLayout(self._agents_list_vbox)
        self._agents_empty_label = QLabel("— none connected —")
        self._agents_empty_label.setStyleSheet(
            f"color: {s.FG_DIM}; font-size: 10px; padding: 4px 0px;"
        )
        self._agents_list_vbox.addWidget(self._agents_empty_label)
        agents_section.set_content_layout(agents_inner)
        root.addWidget(agents_section)

        # ── ACTIVITY (collapsed) ─────────────────────────────────────────
        # Scrolling log of agent traffic; useful for debug but noisy.
        # The inner QScrollArea remains so the log itself scrolls.
        activity_section = CollapsibleSection("ACTIVITY")
        activity_inner = QVBoxLayout()
        activity_inner.setContentsMargins(0, 4, 0, 4)
        activity_inner.setSpacing(4)
        self._log_area = QScrollArea()
        self._log_area.setWidgetResizable(True)
        self._log_area.setFixedHeight(160)
        self._log_area.setStyleSheet("background: transparent; border: none;")
        self._log_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._log_container = QWidget()
        self._log_vbox = QVBoxLayout(self._log_container)
        self._log_vbox.setContentsMargins(0, 0, 0, 0)
        self._log_vbox.setSpacing(1)
        self._log_vbox.addStretch()
        self._log_area.setWidget(self._log_container)
        activity_inner.addWidget(self._log_area)
        activity_section.set_content_layout(activity_inner)
        root.addWidget(activity_section)

        root.addStretch()

        self._btn_start.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._stop)
        self._btn_approve_pending.clicked.connect(self._approve_next_pending)
        self._btn_reject_pending.clicked.connect(self._reject_next_pending)
        self._mode_combo.currentIndexChanged.connect(lambda _idx: self._update_pending_controls())
        self._update_pending_controls()

    # ── server control ────────────────────────────────────────────────────

    def _start(self) -> None:
        port = self._port_spin.value()
        host = "0.0.0.0" if self._all_interfaces_check.isChecked() else "127.0.0.1"
        try:
            self._server.start(port, host=host)
        except ValueError as exc:
            self._append_log(f"start blocked: {exc}")
            return
        _set_active_server(self._server)
        self._port_spin.setEnabled(False)
        self._all_interfaces_check.setEnabled(False)
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status_dot.setText("● running")
        self._status_dot.setStyleSheet(f"color: {s.FG_ACCENT}; font-size: 11px;")
        display_host = "0.0.0.0" if host == "0.0.0.0" else "localhost"
        base = f"http://{display_host}:{port}"
        self._http_row.set_value(f"POST {base}/chat")
        self._sse_row.set_value(f"POST {base}/chat/stream")
        self._state_row.set_value(f"GET  {base}/state")
        script = str(Path(__file__).resolve().parents[2] / "engine" / "agent_server.py")
        self._mcp_row.set_value(f'python "{script}" --stdio')
        self._refresh_state_snapshot_cache()

    def _stop(self) -> None:
        _set_active_server(None)
        self._server.stop()
        self._port_spin.setEnabled(True)
        self._all_interfaces_check.setEnabled(True)
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status_dot.setText("● stopped")
        self._status_dot.setStyleSheet(f"color: {s.FG_DIM}; font-size: 11px;")
        self._http_row.set_value("— not running")
        self._sse_row.set_value("— not running")
        self._state_row.set_value("— not running")
        self._mcp_row.set_value("— not running")
        self._refresh_state_snapshot_cache()

    def _autostart_server(self) -> None:
        """Start the agent server automatically when MONOLITH_AGENT_AUTOSTART=1, so
        Connect is live on launch without a button click (enables autonomous /chat
        + /load_model). Deferred via QTimer so the engine is ready first."""
        if getattr(self._server, "_running", False):
            return
        try:
            self._start()
            self._append_log("● autostart: agent server started (MONOLITH_AGENT_AUTOSTART=1)")
        except Exception as exc:
            self._append_log(f"✗ autostart failed: {exc}")

    def _on_load_requested(self) -> None:
        """/load_model dispatched here on the Qt thread. Load the model currently
        selected/configured in Monolith via the active chat (mirrors the LOAD
        MODEL button → PageChat.load_model())."""
        active = self._active_chat_widget()
        if active is None:
            self._append_log("  ✗ load_model: no active chat")
            return
        loader = getattr(active, "load_model", None)
        if callable(loader):
            try:
                loader()
                self._append_log("● load_model: loading the selected model")
            except Exception as exc:
                self._append_log(f"  ✗ load_model failed: {exc}")
        else:
            self._append_log("  ✗ load_model: active chat has no load_model()")

    # ── signal handlers ───────────────────────────────────────────────────

    def _on_agent_message(self, agent_name: str, message: str) -> None:
        """Received from server thread via queued signal. Now in Qt main thread."""
        # Log line is display-only — strip the [CHANNEL: ...] header the
        # agent server prepends so the raw scaffold doesn't show. The tagged
        # `message` still flows on to the model below.
        self._append_log(f"[{agent_name}] → {strip_channel_tags_for_display(message)[:72]}")

        # Find the active terminal and route by policy mode.
        active = self._active_chat_widget()
        if active is None:
            self._append_log("  ✗ no active chat — message dropped")
            self._server.push_done()  # unblock the waiting HTTP request
            return

        engine_key = getattr(active, "_engine_key", None)
        if engine_key is None:
            self._append_log("  ✗ chat has no engine key — message dropped")
            self._server.push_done()
            return

        mode_data = str(self._mode_combo.currentData() or "").strip().lower()
        mode_text = str(self._mode_combo.currentText() or "").strip().lower()
        mode = mode_data or self.MODE_MANUAL
        if mode_text.startswith("auto"):
            mode = self.MODE_AUTO
        elif mode_text.startswith("advisory"):
            mode = self.MODE_ADVISORY
        elif mode_text.startswith("manual"):
            mode = self.MODE_MANUAL
        agent = str(agent_name or "Agent").strip() or "Agent"
        text = str(message or "").strip()

        submit_agent = getattr(active, "_submit_agent_message", None)
        if not callable(submit_agent):
            self._append_log("  ✗ chat missing _submit_agent_message — message dropped")
            self._server.push_done()
            return

        # ── @mention routing ──────────────────────────────────────────────
        # If the message targets an external agent (@gemini, @codex, etc.),
        # add it to the timeline but suppress Monolith's generation.
        # The external agent's reply comes back via _on_external_agent_reply.
        mentions = parse_mentions(text)
        if mentions:
            # Add the message to timeline without dispatching to Monolith
            submit_agent(
                agent,
                text,
                auto_dispatch=False,
                add_to_timeline=True,
                approved=False,
            )
            self._server.push_token("Routing to @" + ", @".join(mentions) + "...")
            self._server.push_done()

            for target_name in mentions:
                clean_text = strip_mention(text, target_name)
                self._append_log(f"  → @{target_name}: {clean_text[:60]}")

                def _on_reply(label, reply, _name=target_name):
                    self._relay.agent_reply.emit(label, reply)

                def _on_err(label, err, _name=target_name):
                    self._relay.agent_reply.emit(label, err)

                dispatch_external(
                    target_name,
                    clean_text,
                    on_reply=_on_reply,
                    on_error=_on_err,
                )
            return
        # ── end @mention routing ──────────────────────────────────────────

        if mode == self.MODE_AUTO:
            self._begin_active_request(engine_key, active)
            submit_agent(
                agent,
                text,
                auto_dispatch=True,
                add_to_timeline=True,
                approved=True,
            )
            self._append_log("  → auto-dispatched (mode=auto)")
            return

        # Manual/advisory mode: queue until the user explicitly approves.
        message_index = submit_agent(
            agent,
            text,
            auto_dispatch=False,
            add_to_timeline=True,
            approved=False,
        )
        self._pending_agent_messages.append(
            {
                "agent_name": agent,
                "message": text,
                "chat": active,
                "message_index": message_index,
                "engine_key": engine_key,
            }
        )
        self._append_log(
            f"  … queued for approval ({len(self._pending_agent_messages)} pending, mode={mode})"
        )
        self._update_pending_controls()
        self._server.push_token("Queued for USER approval.")
        self._server.push_done()

    def _on_external_agent_reply(self, label: str, reply: str) -> None:
        """External agent (Gemini, Codex, etc.) finished — inject reply into timeline."""
        self._append_log(f"  [{label}] replied: {reply[:60]}")
        active = self._active_chat_widget()
        if active is None:
            return
        submit_agent = getattr(active, "_submit_agent_message", None)
        if not callable(submit_agent):
            return
        engine_key = getattr(active, "_engine_key", None)
        # Post the external agent's reply as an agent message, no Monolith generation
        submit_agent(
            label,
            reply,
            auto_dispatch=False,
            add_to_timeline=True,
            approved=False,
        )


    def _on_token(self, engine_key: str, token: str) -> None:
        if engine_key == self._active_engine_key:
            self._server.push_token(token)
        # Broadcast to SSE/webhooks regardless of active request
        self._server.broadcast_event(EVENT_TOKEN, {"text": token, "engine": engine_key})

    def _on_engine_status(self, engine_key: str, status: SystemStatus) -> None:
        # Broadcast status change to all observers
        status_name = status.name if hasattr(status, "name") else str(status)
        self._server.broadcast_event(EVENT_STATUS, {
            "status": status_name, "engine": engine_key,
        })
        self._refresh_state_snapshot_cache()

        if engine_key != self._active_engine_key:
            return
        if status == SystemStatus.ERROR:
            self._server.broadcast_event(EVENT_ERROR, {
                "message": "generation failed", "engine": engine_key,
            })
            self._finish_active_request("[error: generation failed]")
            return
        if status == SystemStatus.READY:
            # READY can fire between tool-loop hops; only complete once the chat is truly idle.
            self._schedule_completion_poll()

    def _update_pending_controls(self) -> None:
        count = len(self._pending_agent_messages)
        mode = str(self._mode_combo.currentData() or self.MODE_MANUAL)
        self._pending_label.setText(f"Pending approvals: {count}")
        enabled = count > 0 and mode != self.MODE_AUTO
        self._btn_approve_pending.setEnabled(enabled)
        self._btn_reject_pending.setEnabled(enabled)
        self._refresh_state_snapshot_cache()

    def _approve_next_pending(self) -> None:
        if not self._pending_agent_messages:
            self._update_pending_controls()
            return

        item = self._pending_agent_messages.pop(0)
        self._update_pending_controls()

        active = item.get("chat")
        if active is None:
            self._append_log("  ✗ queued message lost (chat closed)")
            return
        submit_agent = getattr(active, "_submit_agent_message", None)
        if not callable(submit_agent):
            self._append_log("  ✗ cannot approve, chat missing agent submit method")
            return

        agent = str(item.get("agent_name", "Agent")).strip() or "Agent"
        text = str(item.get("message", ""))
        msg_index = item.get("message_index")
        engine_key = item.get("engine_key")
        if isinstance(engine_key, str):
            self._begin_active_request(engine_key, active)

        submit_agent(
            agent,
            text,
            auto_dispatch=True,
            add_to_timeline=False,
            approved=True,
            message_index=msg_index if isinstance(msg_index, int) else None,
        )
        self._append_log(f"  ✓ approved [{agent}]")

    def _reject_next_pending(self) -> None:
        if not self._pending_agent_messages:
            self._update_pending_controls()
            return

        item = self._pending_agent_messages.pop(0)
        self._update_pending_controls()
        agent = str(item.get("agent_name", "Agent")).strip() or "Agent"
        self._append_log(f"  ✕ rejected [{agent}]")

    # ── helpers ───────────────────────────────────────────────────────────

    def _begin_active_request(self, engine_key: str, chat_widget: QWidget | None) -> None:
        self._active_engine_key = str(engine_key or "")
        self._active_chat_ref = chat_widget
        self._awaiting_engine_completion = True
        self._request_start_time = time.monotonic()
        if self._completion_poll_timer.isActive():
            self._completion_poll_timer.stop()
        self._server.broadcast_event(EVENT_GENERATION_START, {
            "engine": self._active_engine_key,
        })
        self._refresh_state_snapshot_cache()

    def _finish_active_request(self, final_token: str | None = None) -> None:
        if final_token:
            self._server.push_token(str(final_token))
        self._server.push_done()
        # Broadcast done with the full accumulated response
        self._server.broadcast_event(EVENT_DONE, {
            "engine": self._active_engine_key or "",
        })
        self._active_engine_key = None
        self._active_chat_ref = None
        self._awaiting_engine_completion = False
        if self._completion_poll_timer.isActive():
            self._completion_poll_timer.stop()
        self._refresh_state_snapshot_cache()

    def _schedule_completion_poll(self) -> None:
        if not self._awaiting_engine_completion:
            return
        if not self._completion_poll_timer.isActive():
            self._completion_poll_timer.start()

    _POLL_TIMEOUT = 115.0  # seconds — just under the HTTP handler's 120s timeout

    def _poll_active_request_completion(self) -> None:
        if not self._awaiting_engine_completion or not self._active_engine_key:
            return

        # Safety timeout — never let the poll loop run longer than the HTTP timeout
        elapsed = time.monotonic() - getattr(self, "_request_start_time", 0)
        if elapsed > self._POLL_TIMEOUT:
            self._append_log("  ⚠ poll timeout — forcing completion")
            self._finish_active_request("[error: tool loop timed out]")
            return

        active = self._active_chat_ref or self._active_chat_widget()
        if active is None:
            self._finish_active_request("[error: active chat unavailable]")
            return

        is_running = bool(getattr(active, "_is_running", False))
        pending_tool_results = bool(getattr(active, "_pending_tool_results", []))
        awaiting_update = bool(getattr(active, "_awaiting_update_restart", False))
        pending_tool_parse = self._chat_has_unprocessed_tool_commands(active)
        tool_loop_active = bool(getattr(active, "_tool_loop_active", False))
        if is_running or pending_tool_results or awaiting_update or pending_tool_parse or tool_loop_active:
            self._completion_poll_timer.start()
            return

        self._finish_active_request()

    def _chat_has_unprocessed_tool_commands(self, chat_widget: QWidget) -> bool:
        session = getattr(chat_widget, "_current_session", None)
        if not isinstance(session, dict):
            return False
        messages = session.get("messages", [])
        if not isinstance(messages, list):
            return False
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).strip().lower() != "assistant":
                continue
            text = str(message.get("text", "") or "")
            return "<tool_call>" in text
        return False

    def _build_state_snapshot(self) -> dict:
        """
        Build a state dict for the /state endpoint.
        Called from HTTP thread — returns a cached snapshot built on the Qt thread.
        """
        with self._state_snapshot_lock:
            return copy.deepcopy(self._state_snapshot_cache)

    def _build_session_messages_snapshot(self, n: int) -> list[dict]:
        """Worker-thread snapshot of the most-recent N session messages.

        Reads from the cached snapshot refreshed every 250ms; deep-copies the
        slice so the caller can mutate freely. The cache holds up to 200
        messages; callers asking for more get capped.
        """
        with self._state_snapshot_lock:
            cached = list(self._state_snapshot_cache.get("session_messages_full") or [])
        if not cached:
            return []
        slice_n = max(1, min(int(n), len(cached)))
        return copy.deepcopy(cached[-slice_n:])

    def _build_interceptor_state_snapshot(self) -> dict:
        """Worker-thread snapshot of interceptor-visible state."""
        with self._state_snapshot_lock:
            return copy.deepcopy(self._state_snapshot_cache.get("interceptor_state") or {})

    def _dispatch_reset(self) -> dict:
        """Server-thread entrypoint for /reset.

        Marshals the actual reset onto the Qt main thread via a queued signal —
        the SAME fire-and-forget pattern as on_message/on_participant_change. We
        must NOT touch Qt widgets here (we're on the HTTP thread). The reset runs
        later on the Qt event loop, so we can only confirm dispatch, not
        completion; the receipt says so honestly.
        """
        self._relay.reset_requested.emit()
        return {"dispatched": True, "surface": "new_chat"}

    def _on_reset_requested(self) -> None:
        """Qt-main-thread slot: start a fresh chat surface like "New Chat".

        Invokes the active PageChat's _start_new_session() — the exact faithful
        path the UI uses (chat.py). _active_chat_ref is the in-flight chat (set
        during a request); fall back to the currently-displayed conversation.
        """
        active = self._active_chat_ref or self._active_chat_widget()
        if active is None:
            self._append_log("  ⚠ /reset: no active chat surface")
            return
        starter = getattr(active, "_start_new_session", None)
        if not callable(starter):
            self._append_log("  ⚠ /reset: active chat has no _start_new_session")
            return
        try:
            starter()
            self._append_log("  ↻ /reset: fresh chat surface started")
        except Exception as exc:
            self._append_log(f"  ⚠ /reset failed: {exc}")

    def _refresh_state_snapshot_cache(self) -> None:
        snap = self._collect_state_snapshot()
        with self._state_snapshot_lock:
            self._state_snapshot_cache = snap

    def _collect_state_snapshot(self) -> dict:
        """Build a state snapshot on the Qt main thread."""
        snap: dict = {}
        active = self._active_chat_ref or self._active_chat_widget()
        if active is None:
            snap["chat"] = None
            snap["model_status"] = "no_active_chat"
            return snap

        # Model activity state
        is_running = bool(getattr(active, "_is_running", False))
        tool_loop = bool(getattr(active, "_tool_loop_active", False))
        pending_tools = bool(getattr(active, "_pending_tool_results", []))
        awaiting_update = bool(getattr(active, "_awaiting_update_restart", False))

        if is_running:
            status = "generating"
        elif tool_loop or pending_tools:
            status = "tool_loop"
        elif awaiting_update:
            status = "awaiting_update"
        else:
            status = "idle"
        snap["model_status"] = status

        # Tool loop depth
        snap["tool_followup_depth"] = int(
            getattr(active, "_tool_followup_depth", 0)
        )
        snap["tool_followup_retries"] = int(
            getattr(active, "_tool_followup_retries", 0)
        )

        # Recent messages (last 10)
        session = getattr(active, "_current_session", None)
        if isinstance(session, dict):
            messages = session.get("messages", [])
            recent = []
            for msg in messages[-10:]:
                if not isinstance(msg, dict):
                    continue
                recent.append({
                    "role": str(msg.get("role", "")),
                    "text": str(msg.get("text", ""))[:200],
                    "agent_name": msg.get("agent_name", ""),
                })
            snap["recent_messages"] = recent
            snap["message_count"] = len(messages)
        else:
            snap["recent_messages"] = []
            snap["message_count"] = 0

        # Routing mode
        mode_data = str(self._mode_combo.currentData() or "manual")
        snap["routing_mode"] = mode_data
        snap["pending_approvals"] = len(self._pending_agent_messages)

        # Debug-read data: full session messages (up to 200) and interceptor
        # state. Captured here on the Qt thread so the worker-thread callbacks
        # can return a copy without touching live state.
        try:
            if hasattr(active, "build_session_messages_snapshot"):
                snap["session_messages_full"] = active.build_session_messages_snapshot(200)
            else:
                snap["session_messages_full"] = []
        except Exception:
            snap["session_messages_full"] = []
        try:
            if hasattr(active, "build_interceptor_state_snapshot"):
                snap["interceptor_state"] = active.build_interceptor_state_snapshot()
            else:
                snap["interceptor_state"] = {}
        except Exception:
            snap["interceptor_state"] = {}

        return snap

    def _active_chat_widget(self) -> QWidget | None:
        ui = getattr(self._ctx, "ui", None) if self._ctx else None
        if ui is None:
            return None
        stack = getattr(ui, "_conversation_stack", None)
        if stack is None:
            return None
        return stack.currentWidget()

    def _append_log(self, text: str) -> None:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {s.FG_DIM}; font-size: 10px; font-family: monospace;"
        )
        lbl.setWordWrap(True)
        # Insert before the trailing stretch
        self._log_vbox.insertWidget(self._log_vbox.count() - 1, lbl)
        # Cap at 60 entries (60 labels + 1 stretch = 61 items)
        while self._log_vbox.count() > 62:
            item = self._log_vbox.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        # Scroll to bottom
        sb = self._log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── peer management ───────────────────────────────────────────────────

    def _refresh_connected_agents(self) -> None:
        """Rebuild the Connected Agents block from the live participant registry."""
        # Clear existing rows
        while self._agents_list_vbox.count():
            item = self._agents_list_vbox.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        srv = self._server
        with srv._participants_lock:
            participants = list(srv._participants.values())

        if not participants:
            self._agents_empty_label = QLabel("— none connected —")
            self._agents_empty_label.setStyleSheet(
                f"color: {s.FG_DIM}; font-size: 10px; padding: 4px 0px;"
            )
            self._agents_list_vbox.addWidget(self._agents_empty_label)
            return

        for p in participants:
            name = p.get("name", "?")
            url = p.get("url") or ""

            row = QHBoxLayout()
            row.setSpacing(6)

            dot = QLabel("●")
            dot.setStyleSheet(f"color: {s.FG_ACCENT}; font-size: 9px;")
            dot.setFixedWidth(10)

            name_lbl = QLabel(f"@{name.lower()}")
            name_lbl.setStyleSheet(
                f"color: {s.ACCENT_PRIMARY}; font-size: 10px; font-family: monospace;"
            )
            name_lbl.setFixedWidth(90)

            url_lbl = QLabel(url or "no /chat url")
            url_lbl.setStyleSheet(
                f"color: {s.FG_DIM if url else '#e05555'}; font-size: 9px; font-family: monospace;"
            )

            row.addWidget(dot)
            row.addWidget(name_lbl)
            row.addWidget(url_lbl, stretch=1)

            container = QWidget()
            container.setLayout(row)
            self._agents_list_vbox.addWidget(container)

    def _on_add_peer(self) -> None:
        name = self._peer_name_input.text().strip().lower()
        url = self._peer_url_input.text().strip()
        if not name or not url:
            self._append_log("  ✗ peer name and URL required")
            return
        result = add_peer(name, name.capitalize(), url.rstrip("/"))
        if result.get("ok"):
            self._peer_name_input.clear()
            self._peer_url_input.clear()
            self._refresh_peer_list()
            self._append_log(f"  + peer added: @{name} → {url}")
        else:
            self._append_log(f"  ✗ {result.get('error', 'failed')}")

    def _refresh_peer_list(self) -> None:
        # Clear existing rows
        while self._peer_list_vbox.count():
            item = self._peer_list_vbox.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        peers = get_peers()
        if not peers:
            empty = QLabel("no peers configured")
            empty.setStyleSheet(f"color: {s.FG_DIM}; font-size: 9px;")
            self._peer_list_vbox.addWidget(empty)
            return

        for peer in peers.values():
            row = QHBoxLayout()
            row.setSpacing(4)

            name_lbl = QLabel(f"@{peer['name']}")
            name_lbl.setStyleSheet(
                f"color: {s.ACCENT_PRIMARY}; font-size: 10px; font-family: monospace;"
            )
            name_lbl.setFixedWidth(80)

            url_lbl = QLabel(peer["url"])
            url_lbl.setStyleSheet(f"color: {s.FG_DIM}; font-size: 9px; font-family: monospace;")

            btn_ping = MonoButton("ping")
            btn_ping.setFixedHeight(20)
            btn_ping.setFixedWidth(36)
            btn_ping.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {s.FG_DIM};"
                f" border: 1px solid {s.BORDER_DARK}; border-radius: 2px; font-size: 9px; }}"
                f"QPushButton:hover {{ color: {s.FG_TEXT}; border-color: {s.BORDER_LIGHT}; }}"
            )

            btn_remove = MonoButton("x")
            btn_remove.setFixedHeight(20)
            btn_remove.setFixedWidth(20)
            btn_remove.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {s.FG_DIM};"
                f" border: 1px solid {s.BORDER_DARK}; border-radius: 2px; font-size: 9px; }}"
                f"QPushButton:hover {{ color: #e05555; border-color: #e05555; }}"
            )

            peer_name = peer["name"]
            btn_ping.clicked.connect(lambda _, n=peer_name: self._ping_peer(n))
            btn_remove.clicked.connect(lambda _, n=peer_name: self._remove_peer(n))

            row.addWidget(name_lbl)
            row.addWidget(url_lbl, stretch=1)
            row.addWidget(btn_ping)
            row.addWidget(btn_remove)

            container = QWidget()
            container.setLayout(row)
            self._peer_list_vbox.addWidget(container)

    def _ping_peer(self, name: str) -> None:
        peers = get_peers()
        peer = peers.get(name)
        if not peer:
            return
        self._append_log(f"  pinging @{name}...")

        def _do_ping():
            ok = ping_peer(name)
            self._relay.log_line.emit(
                f"  @{name} {'● online' if ok else '✗ unreachable'}"
            )

        import threading
        threading.Thread(target=_do_ping, daemon=True).start()

    def _remove_peer(self, name: str) -> None:
        remove_peer(name)
        self._refresh_peer_list()
        self._append_log(f"  - peer removed: @{name}")

    # ── cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._state_snapshot_timer.isActive():
            self._state_snapshot_timer.stop()
        self._server.stop()
        super().closeEvent(event)


# ── reusable copy-row widget ───────────────────────────────────────────────

class _CopyRow(QWidget):
    """Label prefix + monospace value + copy button on one line."""

    def __init__(self, prefix: str, value: str, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl_prefix = QLabel(prefix)
        lbl_prefix.setFixedWidth(36)
        lbl_prefix.setStyleSheet(
            f"color: {s.FG_DIM}; font-size: 10px; font-weight: bold;"
        )

        self._value_lbl = QLabel(value)
        self._value_lbl.setStyleSheet(
            f"color: {s.FG_TEXT}; font-size: 10px; font-family: monospace;"
        )
        self._value_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._copy_btn = MonoButton("⧉")
        self._copy_btn.setFixedSize(22, 22)
        self._copy_btn.setToolTip("Copy to clipboard")
        self._copy_btn.clicked.connect(self._copy)

        layout.addWidget(lbl_prefix)
        layout.addWidget(self._value_lbl, 1)
        layout.addWidget(self._copy_btn)

    def set_value(self, text: str) -> None:
        self._value_lbl.setText(text)

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._value_lbl.text())
