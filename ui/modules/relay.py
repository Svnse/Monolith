"""Monolith Relay — native Qt multi-agent chat module.

Layout
──────
QHBoxLayout
├── Left  (~75%)  QSplitter
│     ├── Message list  (QListWidget + _MessageCard widgets, stretch=1)
│     ├── @ toggle bar  (QHBoxLayout — one toggle per participant)
│     └── Input row     (QTextEdit + SEND button)
└── Right (~25%)
      └── PARTICIPANTS  (MonoGroupBox → QListWidget)
"""

from __future__ import annotations

import json
import time
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal, QSize
from PySide6.QtGui import QColor, QPainter, QPen, QKeyEvent
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QScrollArea, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

import core.style as _s
from ui.components.atoms import MonoButton, MonoGroupBox


# ─────────────────────────────────────────────────────────────────────────────
# Participant colors for known agents (fallback palette handled in router)
# ─────────────────────────────────────────────────────────────────────────────

_KNOWN_COLORS: dict[str, str] = {
    "claude":  "#a78bfa",
    "codex":   "#facc15",
    "kimi":    "#38bdf8",
    "gemini":  "#4285f4",
    "you":     "#e2e8f0",
}

_PALETTE = ["#f472b6", "#34d399", "#fb923c", "#818cf8", "#a3e635", "#22d3ee"]
_palette_idx = 0


def _color_for(name: str, provided: str = "") -> str:
    if provided:
        return provided
    return _KNOWN_COLORS.get(name.lower(), _PALETTE[hash(name) % len(_PALETTE)])


# ─────────────────────────────────────────────────────────────────────────────
# _StatusDot — colored circle widget
# ─────────────────────────────────────────────────────────────────────────────

class _StatusDot(QWidget):
    def __init__(self, color: str, filled: bool = True, parent=None):
        super().__init__(parent)
        self.setFixedSize(8, 8)
        self._color = QColor(color)
        self._filled = filled

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def set_filled(self, filled: bool) -> None:
        self._filled = filled
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._filled:
            p.setBrush(self._color)
            p.setPen(Qt.NoPen)
        else:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(self._color, 1.5))
        p.drawEllipse(0, 0, 7, 7)


# ─────────────────────────────────────────────────────────────────────────────
# _MessageCard — one message in the list
# ─────────────────────────────────────────────────────────────────────────────

_KIND_ACCENT: dict[str, str] = {
    "user":    "",          # filled by sender color
    "chat":    "",
    "agent":   "#7eb89f",
    "loop":    "#6b8cb0",
    "tool":    "#d4a76a",
    "system":  "#555566",
    "join":    "#7eb89f",
    "leave":   "#d46a6a",
    "approval":"#d4a76a",
    "guard":   "#d4a76a",
}


class _MessageCard(QFrame):
    def __init__(
        self,
        sender: str,
        text: str,
        msg_type: str = "chat",
        time_str: str = "",
        sender_color: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)

        accent = sender_color or _KIND_ACCENT.get(msg_type, _s.ACCENT_PRIMARY)

        self.setStyleSheet(
            f"QFrame {{ background: {_s.BG_BUTTON}; "
            f"border: 1px solid {_s.BORDER_LIGHT}; "
            f"border-left: 3px solid {accent}; "
            f"border-radius: 3px; }}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(3)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(8)

        role_lbl = QLabel(sender.upper())
        role_lbl.setStyleSheet(
            f"color: {accent}; font-size: 8px; font-weight: bold; "
            f"letter-spacing: 1px; background: transparent; border: none;"
        )
        header.addWidget(role_lbl)

        if time_str:
            time_lbl = QLabel(time_str)
            time_lbl.setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 8px; "
                f"font-family: Consolas; background: transparent; border: none;"
            )
            header.addWidget(time_lbl)

        header.addStretch()
        lay.addLayout(header)

        # Body
        if text:
            body = QLabel(text)
            body.setWordWrap(True)
            body.setTextInteractionFlags(Qt.TextSelectableByMouse)
            body.setStyleSheet(
                f"color: {_s.FG_TEXT}; font-size: 10px; "
                f"font-family: Consolas; background: transparent; border: none;"
            )
            lay.addWidget(body)

    def sizeHint(self) -> QSize:
        return QSize(400, 48)


def _make_list_item(card: _MessageCard) -> QListWidgetItem:
    item = QListWidgetItem()
    item.setSizeHint(card.sizeHint())
    return item


# ─────────────────────────────────────────────────────────────────────────────
# _ParticipantRow — one row in the participants panel
# ─────────────────────────────────────────────────────────────────────────────

class _ParticipantRow(QWidget):
    def __init__(self, name: str, label: str, color: str, kind: str, parent=None):
        super().__init__(parent)
        self.name = name
        self._color = color

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(8)

        filled = kind in ("human", "loop")
        self._dot = _StatusDot(color, filled=filled)
        lay.addWidget(self._dot)

        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 10px; "
            f"font-family: Consolas; background: transparent; border: none;"
        )
        lay.addWidget(lbl, stretch=1)

        kind_lbl = QLabel("LOOP" if kind == "loop" else "CLI" if kind == "external" else "")
        kind_lbl.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 8px; letter-spacing: 1px; "
            f"background: transparent; border: none;"
        )
        lay.addWidget(kind_lbl)

    def set_online(self, online: bool) -> None:
        self._dot.set_filled(online)


# ─────────────────────────────────────────────────────────────────────────────
# PageRelay
# ─────────────────────────────────────────────────────────────────────────────

class PageRelay(QWidget):
    # Signals up to the addon factory for engine dispatch
    sig_load    = Signal()
    sig_send    = Signal(dict)      # {action, sender, text, ...}
    sig_action  = Signal(dict)      # generic room action {action, ...}

    def __init__(self, state, ui_bridge, parent=None):
        super().__init__(parent)
        self.state = state
        self.ui_bridge = ui_bridge

        self._participants: dict[str, dict] = {}     # name → {color, label, kind}
        self._toggles: dict[str, MonoButton] = {}    # name → toggle button
        self._username = "you"
        self._server_ready = False

        self._build_ui()
        self._refresh_styles()

        # Wire theme changes
        if hasattr(ui_bridge, "sig_theme_changed"):
            ui_bridge.sig_theme_changed.connect(self._on_theme_changed)

        # Request participants on startup once server is ready
        QTimer.singleShot(1200, self._request_who)

    # ─────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter)

        # ── Left column ────────────────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(8)

        # Message list
        self._msg_list = QListWidget()
        self._msg_list.setWordWrap(True)
        self._msg_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._msg_list.setSpacing(2)
        left_lay.addWidget(self._msg_list, stretch=1)

        # @ toggle bar
        self._toggle_frame = QFrame()
        self._toggle_frame.setFixedHeight(36)
        self._toggle_layout = QHBoxLayout(self._toggle_frame)
        self._toggle_layout.setContentsMargins(0, 0, 0, 0)
        self._toggle_layout.setSpacing(6)
        self._toggle_layout.addStretch()
        left_lay.addWidget(self._toggle_frame)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self._input = QTextEdit()
        self._input.setPlaceholderText("Type a message or @mention...")
        self._input.setFixedHeight(70)
        self._input.installEventFilter(self)
        input_row.addWidget(self._input, stretch=1)

        self._btn_send = MonoButton("SEND", accent=True)
        self._btn_send.setFixedWidth(72)
        self._btn_send.clicked.connect(self._on_send)
        input_row.addWidget(self._btn_send, alignment=Qt.AlignBottom)

        left_lay.addLayout(input_row)
        splitter.addWidget(left)

        # ── Right column ───────────────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        self._grp_participants = MonoGroupBox("PARTICIPANTS")
        self._participant_list = QListWidget()
        self._participant_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._grp_participants.add_widget(self._participant_list)
        right_lay.addWidget(self._grp_participants)

        # Status bar
        self._status_lbl = QLabel("● OFFLINE")
        self._status_lbl.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 8px; font-weight: bold; "
            f"letter-spacing: 1px; padding: 4px;"
        )
        right_lay.addWidget(self._status_lbl)

        # Invite button
        self._btn_invite = MonoButton("+ INVITE")
        self._btn_invite.clicked.connect(self._on_invite)
        right_lay.addWidget(self._btn_invite)

        right_lay.addStretch()
        splitter.addWidget(right)

        # Splitter ratio 75/25
        splitter.setSizes([750, 250])

        # Add "You" as permanent participant
        self._add_participant("you", "You", _KNOWN_COLORS["you"], "human")

    # ─────────────────────────────────────────────────────────────────
    # Keyboard: Enter to send (Shift+Enter = newline)
    # ─────────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        if obj is self._input and isinstance(event, QKeyEvent):
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if not (event.modifiers() & Qt.ShiftModifier):
                    self._on_send()
                    return True
        return super().eventFilter(obj, event)

    # ─────────────────────────────────────────────────────────────────
    # Engine event handler (called by relay_factory on sig_token)
    # ─────────────────────────────────────────────────────────────────

    def on_relay_event(self, json_str: str) -> None:
        try:
            event = json.loads(json_str)
        except Exception:
            return

        kind = str(event.get("event") or "")

        if kind == "message":
            self._handle_message(event.get("data") or {})

        elif kind == "joined":
            p = event.get("participant") or {}
            name  = str(p.get("name") or "")
            color = str(p.get("color") or "")
            label = str(p.get("label") or name)
            ekind = str(p.get("kind") or "external")
            if name:
                self._add_participant(name, label, color, ekind)

        elif kind == "left":
            self._remove_participant(str(event.get("name") or ""))

        elif kind == "participants":
            for p in (event.get("data") or []):
                name  = str(p.get("name") or "")
                color = str(p.get("color") or "")
                label = str(p.get("label") or name)
                ekind = str(p.get("kind") or "external")
                if name:
                    self._add_participant(name, label, color, ekind)

        elif kind == "wake":
            # Wake notification — just show in UI, actual wake happens via queue file
            name = str(event.get("name") or "")
            self._append_system(f"Waking @{name}...")

        elif kind == "guard":
            hops = event.get("max_hops", 4)
            self._append_system(
                f"Loop guard: {hops} agent hops reached. "
                "Type /continue to resume."
            )

    def on_status(self, status) -> None:
        from core.state import SystemStatus
        if status == SystemStatus.READY:
            if not self._server_ready:
                self._server_ready = True
                self._append_system("Relay server ready. MCP :8200 (HTTP)  :8201 (SSE)")
                self._request_who()
            self._status_lbl.setText("● ONLINE")
            self._status_lbl.setStyleSheet(
                f"color: #7eb89f; font-size: 8px; font-weight: bold; "
                f"letter-spacing: 1px; padding: 4px;"
            )
        elif status == SystemStatus.LOADING:
            self._status_lbl.setText("◌ STARTING...")
            self._status_lbl.setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 8px; font-weight: bold; "
                f"letter-spacing: 1px; padding: 4px;"
            )
        elif status == SystemStatus.ERROR:
            self._server_ready = False
            self._status_lbl.setText("✕ ERROR")
            self._status_lbl.setStyleSheet(
                f"color: {_s.FG_ERROR}; font-size: 8px; font-weight: bold; "
                f"letter-spacing: 1px; padding: 4px;"
            )

    # ─────────────────────────────────────────────────────────────────
    # Message rendering
    # ─────────────────────────────────────────────────────────────────

    def _handle_message(self, msg: dict) -> None:
        if msg.get("type") == "clear":
            self._msg_list.clear()
            return

        sender   = str(msg.get("sender") or "")
        text     = str(msg.get("text") or "")
        msg_type = str(msg.get("type") or "chat")
        time_str = str(msg.get("time") or "")

        color = self._participants.get(sender.lower(), {}).get("color") or _color_for(sender)
        card  = _MessageCard(sender, text, msg_type, time_str, sender_color=color)
        item  = _make_list_item(card)

        # Recalculate preferred height from content
        card.adjustSize()
        item.setSizeHint(card.sizeHint())

        self._msg_list.addItem(item)
        self._msg_list.setItemWidget(item, card)
        self._msg_list.scrollToBottom()

    def _append_system(self, text: str) -> None:
        t = time.strftime("%H:%M:%S")
        card = _MessageCard("system", text, "system", t)
        item = _make_list_item(card)
        card.adjustSize()
        item.setSizeHint(card.sizeHint())
        self._msg_list.addItem(item)
        self._msg_list.setItemWidget(item, card)
        self._msg_list.scrollToBottom()

    # ─────────────────────────────────────────────────────────────────
    # Participants
    # ─────────────────────────────────────────────────────────────────

    def _add_participant(self, name: str, label: str, color: str, kind: str) -> None:
        key = name.lower()
        color = _color_for(name, color)

        if key in self._participants:
            # Already tracked — just update color if different
            self._participants[key]["color"] = color
            return

        self._participants[key] = {"color": color, "label": label, "kind": kind}

        # Participant list row
        row = _ParticipantRow(name, label, color, kind)
        item = QListWidgetItem()
        item.setSizeHint(QSize(200, 28))
        item.setData(Qt.UserRole, key)
        self._participant_list.addItem(item)
        self._participant_list.setItemWidget(item, row)

        # @ toggle button (skip "you" — no need to @mention yourself)
        if key != "you":
            self._add_toggle(name, label, color)

    def _remove_participant(self, name: str) -> None:
        key = name.lower()
        self._participants.pop(key, None)

        # Remove from list
        for i in range(self._participant_list.count()):
            item = self._participant_list.item(i)
            if item and item.data(Qt.UserRole) == key:
                self._participant_list.takeItem(i)
                break

        # Remove toggle
        btn = self._toggles.pop(key, None)
        if btn:
            btn.setParent(None)
            btn.deleteLater()

    def _add_toggle(self, name: str, label: str, color: str) -> None:
        key = name.lower()
        if key in self._toggles:
            return

        btn = MonoButton(f"@{label}")
        btn.setCheckable(True)
        btn.setFixedHeight(24)
        btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: 1px solid {color}40; "
            f"color: {color}; padding: 2px 8px; font-size: 9px; "
            f"font-weight: bold; border-radius: 2px; }}"
            f"QPushButton:checked {{ background: {color}25; border-color: {color}; }}"
            f"QPushButton:hover {{ background: {color}15; }}"
        )
        # Insert before the stretch
        count = self._toggle_layout.count()
        self._toggle_layout.insertWidget(count - 1, btn)
        self._toggles[key] = btn

    # ─────────────────────────────────────────────────────────────────
    # Send
    # ─────────────────────────────────────────────────────────────────

    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()

        # Prepend toggled @mentions
        toggled = [
            f"@{self._participants[k]['label']}"
            for k, btn in self._toggles.items()
            if btn.isChecked()
        ]
        if toggled:
            prefix = " ".join(toggled) + " "
            text = prefix + text

        self.sig_send.emit({
            "action": "send",
            "sender": self._username,
            "text":   text,
        })

    def _request_who(self) -> None:
        self.sig_action.emit({"action": "who"})

    # ─────────────────────────────────────────────────────────────────
    # Invite button — clipboard helper
    # ─────────────────────────────────────────────────────────────────

    def _on_invite(self) -> None:
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QCursor, QClipboard
        from PySide6.QtWidgets import QApplication

        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: {_s.BG_BUTTON}; color: {_s.FG_TEXT}; "
            f"border: 1px solid {_s.BORDER_LIGHT}; font-size: 10px; }}"
            f"QMenu::item:selected {{ background: {_s.BG_BUTTON_HOVER}; }}"
        )

        agents = {
            "Claude Code": "claude mcp add relay --transport http http://127.0.0.1:8200/mcp",
            "Codex / Kimi (.mcp.json)":
                '{"mcpServers":{"relay":{"type":"http","url":"http://127.0.0.1:8200/mcp"}}}',
            "Gemini (.gemini/settings.json)":
                '{"mcpServers":{"relay":{"type":"sse","url":"http://127.0.0.1:8201/sse"}}}',
        }
        for label, cmd in agents.items():
            act = menu.addAction(label)
            act.setData(cmd)

        chosen = menu.exec(QCursor.pos())
        if chosen and chosen.data():
            QApplication.clipboard().setText(chosen.data())
            self._append_system(f"Copied MCP registration command for {chosen.text()}")

    # ─────────────────────────────────────────────────────────────────
    # Theme
    # ─────────────────────────────────────────────────────────────────

    def _on_theme_changed(self) -> None:
        self._refresh_styles()

    def _refresh_styles(self) -> None:
        self._msg_list.setStyleSheet(
            f"QListWidget {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_SUBTLE}; "
            f"padding: 4px; }}"
            f"QListWidget::item {{ border: none; margin: 2px 0; }}"
            f"{_s.SCROLLBAR_STYLE}"
        )
        self._participant_list.setStyleSheet(
            f"QListWidget {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_SUBTLE}; "
            f"padding: 2px; }}"
            f"QListWidget::item {{ border: none; margin: 1px 0; }}"
            f"{_s.SCROLLBAR_STYLE}"
        )
        self._input.setStyleSheet(
            f"QTextEdit {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT}; "
            f"border: 1px solid {_s.BORDER_LIGHT}; padding: 8px; "
            f"font-family: Verdana; font-size: 11px; }}"
            f"QTextEdit:focus {{ border: 1px solid {_s.ACCENT_PRIMARY}; }}"
            f"{_s.SCROLLBAR_STYLE}"
        )
        self._toggle_frame.setStyleSheet(
            f"QFrame {{ background: transparent; border: none; }}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Internal loop opt-in (called by Code page when toggle flipped)
    # ─────────────────────────────────────────────────────────────────

    def join_loop(self, name: str, run_id: str, color: str = "", label: str = "") -> None:
        """Register an internal Loop run as a relay participant."""
        color = color or _PALETTE[hash(name) % len(_PALETTE)]
        label = label or name
        self._add_participant(name, label, color, "loop")
        self.sig_action.emit({
            "action":  "join_loop",
            "name":    name,
            "run_id":  run_id,
            "color":   color,
            "label":   label,
        })

    def leave_loop(self, name: str) -> None:
        """Deregister an internal Loop run."""
        self._remove_participant(name)
        self.sig_action.emit({"action": "leave_loop", "name": name})
