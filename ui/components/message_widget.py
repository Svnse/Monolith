import json as _json
import time as _time
from pathlib import Path as _Path

import core.style as _s
from core.skill_registry import canonical_tool_name
from core.cmd_parser import normalize_tool_call_to_cmd
from ui.components.markdown_renderer import render_message_html

from PySide6.QtCore import Qt, QSize, Signal, QTimer, QPropertyAnimation, QEasingCurve, QVariantAnimation
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget, QTextEdit,
)
from PySide6.QtGui import QCursor, QPainter, QPen, QColor, QTextCursor, QTextOption


class _AutoTextView(QTextEdit):
    """Read-only text display that auto-sizes to its content height."""
    heightChanged = Signal(int)

    def __init__(self, auto_resize: bool = True):
        super().__init__()
        self._auto_resize = auto_resize
        self.setReadOnly(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.setCursor(Qt.IBeamCursor)
        self.document().setDocumentMargin(0)
        self.document().contentsChanged.connect(self._schedule_update)
        self.setStyleSheet("background: transparent; border: none; padding: 0px; margin: 0px;")
        self._last_h = 18
        self.setFixedHeight(18)
        self._pending = False
        self._suspend_auto_resize = False
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(0)
        self._update_timer.timeout.connect(self._do_update_height)

    def _schedule_update(self):
        if not self._auto_resize or self._suspend_auto_resize:
            return
        if not self._pending:
            self._pending = True
            if not self._update_timer.isActive():
                self._update_timer.start()

    def _do_update_height(self):
        self._pending = False
        try:
            self._update_height()
        except RuntimeError:
            # Widget may already be destroyed while a queued timer callback drains.
            return

    def _update_height(self):
        if not self._auto_resize:
            return
        try:
            vw = self.viewport().width()
            doc = self.document()
        except RuntimeError:
            return
        if vw < 10:
            vw = self.width() - 4
        doc.setTextWidth(max(vw, 20))
        h = int(doc.size().height()) + 6
        h = max(h, 18)
        if h != self._last_h:
            self._last_h = h
            self.setFixedHeight(h)
            self.heightChanged.emit(h)

    def wheelEvent(self, event):
        # Don't scroll internally — let the parent list handle it.
        event.ignore()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_resize:
            self._update_height()

    def ideal_height(self, available_width: int) -> int:
        doc = self.document().clone()
        doc.setDocumentMargin(0)
        doc.setTextWidth(max(available_width, 20))
        h = int(doc.size().height()) + 6
        return max(h, 18)

    def append_plain_text(self, text: str) -> None:
        if not text:
            return
        self._suspend_auto_resize = True
        try:
            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(text)
        finally:
            self._suspend_auto_resize = False


class _HoverAction(QPushButton):
    """Floating hover action used above a message bubble."""

    def __init__(self, label: str, tooltip: str = ""):
        super().__init__(label)
        if tooltip:
            self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(18)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                color: {_s.FG_DIM};
                border: none;
                border-radius: 3px;
                padding: 0 6px;
                font-size: 9px;
                font-family: Consolas;
            }}
            QPushButton:hover {{
                background: {_s.BG_BUTTON_HOVER};
                color: {_s.FG_TEXT};
            }}
            """
        )


_DOT_FRAMES = ["", ".", "..", "..."]


class _ThinkBlock(QFrame):
    """
    Collapsible thinking block.

    While streaming: shows  ▶ Thinking...  (animated dots)
    On close tag:    auto-collapses, label becomes  ▶ Thought (Xs)
    Click arrow to expand / collapse the raw think content.
    """

    sig_layout_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("think_block")
        self.setStyleSheet("QFrame#think_block { background: transparent; border: none; }")

        self._compact = False
        self._expanded = True   # start expanded so streaming content is visible
        self._start_ts = _time.monotonic()
        self._elapsed = 0.0
        self._dot_idx = 0
        self._done = False
        self._anim: QPropertyAnimation | None = None
        self._pending_text = ""
        self._last_text_width = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── header row ──────────────────────────────────────────
        self._header = QWidget()
        header = QHBoxLayout(self._header)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        self._btn = QPushButton("▼")
        self._btn.setFixedSize(18, 18)
        self._btn.setCursor(Qt.PointingHandCursor)
        self._btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            f"color: {_s.FG_DIM}; font-size: 9px; padding: 0px; }}"
            f"QPushButton:hover {{ color: {_s.FG_TEXT}; }}"
        )
        self._btn.clicked.connect(self._toggle)
        header.addWidget(self._btn)

        self._lbl = QLabel("Thinking")
        self._lbl.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; background: transparent;"
        )
        header.addWidget(self._lbl)
        header.addStretch()
        outer.addWidget(self._header)

        # ── body (collapsible) ──────────────────────────────────
        self._body = QFrame()
        self._body.setVisible(True)
        self._body.setStyleSheet(
            f"background: transparent; border-left: 1px solid {_s.BORDER_SUBTLE}; margin-left: 8px;"
        )
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 4, 0, 4)
        body_layout.setSpacing(0)

        self._text = _AutoTextView()
        self._text.setObjectName("think_content")
        self._text.setStyleSheet(
            "background: transparent; border: none; "
            f"color: {_s.FG_TEXT}; font-size: 9px; font-family: Consolas;"
        )
        self._text.heightChanged.connect(lambda _: self.sig_layout_changed.emit())
        body_layout.addWidget(self._text)
        outer.addWidget(self._body)

        # animated dots timer (stops when done)
        self._dot_timer = QTimer(self)
        self._dot_timer.setInterval(400)
        self._dot_timer.timeout.connect(self._tick_dots)
        self._dot_timer.start()
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(33)
        self._flush_timer.timeout.connect(self._flush_pending_text)

    # ── public API ──────────────────────────────────────────────

    def append_think_text(self, text: str) -> None:
        if not text:
            return
        self._pending_text += text
        if len(self._pending_text) >= 256:
            self._flush_pending_text()
            return
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def _flush_pending_text(self) -> None:
        if not self._pending_text:
            return
        text = self._pending_text
        self._pending_text = ""
        self._text.append_plain_text(text)
        self._text._update_height()
        if self._expanded and self._body.isVisible():
            if self._anim:
                self._anim.stop()
                self._anim = None
            self._body.setMaximumHeight(16777215)
        self.sig_layout_changed.emit()

    def close_block(self) -> None:
        """Called when </think> is received. Auto-collapses, shows duration."""
        self._flush_pending_text()
        self._done = True
        self._dot_timer.stop()
        self._elapsed = _time.monotonic() - self._start_ts
        self._lbl.setText(f"Thought ({self._elapsed:.1f}s)")
        self._lbl.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; background: transparent;"
        )
        if self._compact:
            self._body.setVisible(self._expanded)
            self.sig_layout_changed.emit()
            return
        # Collapse
        if self._expanded:
            self._expanded = False
            self._btn.setText("▶")
            self._set_body_visible(False, animate=True)
            self.sig_layout_changed.emit()

    # ── internal ────────────────────────────────────────────────

    def duration_seconds(self) -> float:
        if self._done:
            return self._elapsed
        return max(0.0, _time.monotonic() - self._start_ts)

    def has_content(self) -> bool:
        return bool(self._text.toPlainText().strip() or self._pending_text.strip())

    def set_compact(self, compact: bool) -> None:
        self._compact = bool(compact)
        self._header.setVisible(not self._compact)
        if self._compact:
            self._body.setVisible(self._expanded)
            self._body.setMaximumHeight(16777215)
        self.sig_layout_changed.emit()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        if self._compact:
            self._body.setVisible(self._expanded)
            self.sig_layout_changed.emit()
            return
        self._btn.setText("▼" if self._expanded else "▶")
        self._set_body_visible(self._expanded, animate=False)
        self.sig_layout_changed.emit()

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._btn.setText("▼" if self._expanded else "▶")
        self._set_body_visible(self._expanded, animate=True)
        self.sig_layout_changed.emit()

    def _tick_dots(self) -> None:
        self._dot_idx = (self._dot_idx + 1) % len(_DOT_FRAMES)
        dots = _DOT_FRAMES[self._dot_idx]
        self._lbl.setText(f"Thinking{dots}")

    def _set_body_visible(self, visible: bool, animate: bool = True) -> None:
        if self._anim:
            self._anim.stop()
            self._anim = None
        if not animate:
            self._body.setVisible(visible)
            return
        if visible:
            self._body.setVisible(True)
            self._body.setMaximumHeight(0)
            self._text._update_height()
            lay = self._body.layout()
            if lay:
                lay.activate()
            target = self._body.sizeHint().height()
            self._anim = QPropertyAnimation(self._body, b"maximumHeight", self)
            self._anim.setDuration(160)
            self._anim.setStartValue(0)
            self._anim.setEndValue(target)
            self._anim.setEasingCurve(QEasingCurve.OutCubic)
            def _finish():
                self._body.setMaximumHeight(16777215)
                self._anim = None
                self.sig_layout_changed.emit()
            self._anim.finished.connect(_finish)
            self._anim.start()
        else:
            start = self._body.height()
            self._anim = QPropertyAnimation(self._body, b"maximumHeight", self)
            self._anim.setDuration(120)
            self._anim.setStartValue(start)
            self._anim.setEndValue(0)
            self._anim.setEasingCurve(QEasingCurve.InCubic)
            def _finish():
                self._body.setVisible(False)
                self._body.setMaximumHeight(0)
                self._anim = None
                self.sig_layout_changed.emit()
            self._anim.finished.connect(_finish)
            self._anim.start()

    def sizeHint(self):
        header_h = self._header.sizeHint().height() if self._header.isVisible() else 0
        body_h = 0
        if self._body.isVisible():
            bh = self._body.sizeHint().height()
            max_h = self._body.maximumHeight()
            if max_h > 0:
                bh = min(bh, max_h)
            body_h = bh
        return QSize(self.width() if self.width() > 0 else 200, header_h + body_h)


def _normalize_tool_call_for_summary(parsed) -> dict:
    """Coerce a parsed <tool_call> payload into the flat-shape dict the
    summary helpers below expect (`{"tool":"...", **args}` or
    `{"calls":[...], "mode":...}`).

    Hermes-shape inputs `{"name":"X","arguments":{...}}` are normalized via
    the canonical parser. Anything else passes through unchanged so legacy
    archived JSON still renders.
    """
    if not isinstance(parsed, dict):
        return {}
    cmd = normalize_tool_call_to_cmd(parsed)
    return cmd if isinstance(cmd, dict) else dict(parsed)


def _skill_summary(json_str: str) -> str:
    """Build a one-line label from a tool_call JSON string."""
    try:
        parsed = _json.loads(json_str)
    except Exception:
        return "⟡  tool call"
    cmd = _normalize_tool_call_for_summary(parsed)
    calls = cmd.get("calls")
    if isinstance(calls, list):
        return f"tool batch -> {len(calls)} call{'s' if len(calls) != 1 else ''}"
    op = canonical_tool_name(cmd.get("tool", cmd.get("skill", cmd.get("op", "tool"))))
    if op == "read_file":
        name = _Path(cmd.get("path", "")).name or "?"
        return f"⟡  read_file  →  {name}"
    if op == "list_files":
        name = _Path(cmd.get("path", "")).name or "?"
        pat = cmd.get("pattern", "*")
        return f"⟡  list_files  →  {name}  [{pat}]"
    if op == "grep":
        return f"⟡  grep  →  {str(cmd.get('pattern', '?'))[:48]}"
    if op == "save_note":
        return f"⟡  save_note  →  {cmd.get('title', '?')}"
    if op == "calculate":
        return f"⟡  calculate  →  {str(cmd.get('expr', '?'))[:48]}"
    if op == "search_history":
        return f"⟡  search_history  →  {str(cmd.get('query', '?'))[:48]}"
    if op == "open_addon":
        return f"⟡  open_addon  →  {cmd.get('addon', '?')}"
    return f"⟡  {op}"


def _tool_summary(json_str: str) -> str:
    """ASCII-safe one-line label from a tool_call JSON string."""
    try:
        parsed = _json.loads(json_str)
    except Exception:
        return "tool call"
    cmd = _normalize_tool_call_for_summary(parsed)
    calls = cmd.get("calls")
    if isinstance(calls, list):
        return f"tool batch -> {len(calls)} call{'s' if len(calls) != 1 else ''}"
    op = canonical_tool_name(cmd.get("tool", cmd.get("skill", cmd.get("op", "tool"))))
    if op == "read_file":
        name = _Path(cmd.get("path", "")).name or "?"
        return f"read_file -> {name}"
    if op == "list_files":
        name = _Path(cmd.get("path", "")).name or "?"
        pat = cmd.get("pattern", "*")
        return f"list_files -> {name} [{pat}]"
    if op == "grep":
        return f"grep -> {str(cmd.get('pattern', '?'))[:48]}"
    if op == "save_note":
        return f"save_note -> {cmd.get('title', '?')}"
    if op == "calculate":
        return f"calculate -> {str(cmd.get('expr', '?'))[:48]}"
    if op == "search_history":
        return f"search_history -> {str(cmd.get('query', '?'))[:48]}"
    if op == "open_addon":
        return f"open_addon -> {cmd.get('addon', '?')}"
    return op


def _tool_preview_lines(json_str: str) -> list[str]:
    try:
        parsed = _json.loads(json_str)
    except Exception:
        return ["tool call"]
    cmd = _normalize_tool_call_for_summary(parsed)
    calls = cmd.get("calls")
    if isinstance(calls, list):
        lines = []
        for call in calls[:3]:
            if not isinstance(call, dict):
                continue
            tool = canonical_tool_name(call.get("tool", call.get("skill", call.get("op", "tool")))) or "tool"
            args = [f"{key}={value!r}" for key, value in call.items() if key not in {"id", "tool", "skill", "op"}]
            lines.append(f"{tool}: {', '.join(args[:3]) or 'no parameters'}")
        if len(calls) > 3:
            lines.append(f"... {len(calls) - 3} more call(s)")
        return lines or ["tool batch"]
    if not isinstance(cmd, dict):
        return ["tool call"]
    args = [f"{key}={value!r}" for key, value in cmd.items() if key not in {"id", "tool", "skill", "op"}]
    return args[:3] or ["no parameters"]


# Tag parsing lives entirely in ui.pages.assistant_turn_box.AssistantStreamNormalizer.
# This widget receives pre-routed updates via apply_stream_update() and must not
# re-parse tag text. Adding parser logic here would re-introduce the split-brain
# state-machine bug where `<tool_call>` leaked as plain text.


class _SkillBlock(QFrame):
    """
    Inline tool-call indicator shown when the model emits a <tool_call> block.

    While streaming: shows a compact Calling tool label.
    On close tag: parses JSON and updates the label with the tool summary.
    Starts collapsed; user can expand to see raw JSON params.
    """

    sig_layout_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("skill_block")
        self.setStyleSheet("QFrame#skill_block { background: transparent; border: none; }")

        self._json_buf = ""
        self._expanded = False
        self._done = False
        self._anim: QPropertyAnimation | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(0)

        # ── header row ──────────────────────────────────────────
        self._header = QWidget()
        header = QHBoxLayout(self._header)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        self._btn = QPushButton("▶")
        self._btn.setFixedSize(18, 18)
        self._btn.setCursor(Qt.PointingHandCursor)
        self._btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            f"color: {_s.FG_DIM}; font-size: 9px; padding: 0px; }}"
            f"QPushButton:hover {{ color: {_s.ACCENT_PRIMARY}; }}"
        )
        self._btn.clicked.connect(self._toggle)
        header.addWidget(self._btn)

        # Status dot: pulses while streaming, becomes a check on close.
        self._status_dot = QLabel("●")
        self._status_dot.setFixedHeight(18)
        self._status_dot.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 8px; "
            "font-family: Consolas; background: transparent;"
        )
        header.addWidget(self._status_dot)
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(450)
        self._status_timer.timeout.connect(self._tick_status_dot)
        self._status_phase = False
        self._status_timer.start()

        # Tool-name pill: subtle background pill that displays just the tool
        # name once parsed (or "calling tool..." while streaming).
        self._lbl = QLabel("calling tool…")
        self._lbl.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 9px; "
            "font-family: Consolas; font-weight: 600; "
            f"background: {_s.BG_SURFACE_2}; "
            "padding: 1px 8px; border-radius: 8px; "
            f"border: 1px solid {_s.BORDER_SUBTLE};"
        )
        header.addWidget(self._lbl)
        header.addStretch()
        outer.addWidget(self._header)

        # ── body (collapsed by default) ──────────────────────────
        self._body = QFrame()
        self._body.setVisible(False)
        self._body.setMaximumHeight(0)
        self._body.setStyleSheet(
            f"background: transparent; "
            f"border-left: 1px solid {_s.ACCENT_PRIMARY}; margin-left: 8px;"
        )
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 4, 0, 4)
        body_layout.setSpacing(0)

        self._text = _AutoTextView()
        self._text.setObjectName("skill_content")
        self._text.setStyleSheet(
            "background: transparent; border: none; "
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas;"
        )
        self._text.heightChanged.connect(lambda _: self.sig_layout_changed.emit())
        body_layout.addWidget(self._text)
        outer.addWidget(self._body)

    # ── public API ──────────────────────────────────────────────

    def append_content(self, text: str) -> None:
        self._json_buf += text

    def _tick_status_dot(self) -> None:
        self._status_phase = not self._status_phase
        col = _s.ACCENT_PRIMARY if self._status_phase else _s.FG_DIM
        self._status_dot.setStyleSheet(
            f"color: {col}; font-size: 8px; "
            "font-family: Consolas; background: transparent;"
        )

    def close_block(self) -> None:
        """Called when </tool_call> is received. Parses JSON, updates label."""
        self._done = True
        # Stop the pulsing status dot and turn it into a solid check mark.
        if self._status_timer.isActive():
            self._status_timer.stop()
        self._status_dot.setText("✓")
        self._status_dot.setStyleSheet(
            f"color: {_s.FG_ACCENT}; font-size: 9px; "
            "font-family: Consolas; background: transparent;"
        )
        label = _tool_summary(self._json_buf)
        self._lbl.setText(label)
        # Pretty-print JSON into the body for when user expands
        try:
            parsed = _json.loads(self._json_buf)
            pretty = _json.dumps(parsed, indent=2)
        except Exception:
            pretty = self._json_buf
        self._text.setPlainText(pretty)
        self._text._update_height()
        self.sig_layout_changed.emit()

    # ── internal ────────────────────────────────────────────────

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._btn.setText("▼" if self._expanded else "▶")
        self._set_body_visible(self._expanded, animate=True)
        self.sig_layout_changed.emit()

    def _set_body_visible(self, visible: bool, animate: bool = True) -> None:
        if self._anim:
            self._anim.stop()
            self._anim = None
        if not animate:
            self._body.setVisible(visible)
            return
        if visible:
            self._body.setVisible(True)
            self._body.setMaximumHeight(0)
            self._text._update_height()
            lay = self._body.layout()
            if lay:
                lay.activate()
            target = max(self._body.sizeHint().height(), self._text._last_h + 12)
            self._anim = QPropertyAnimation(self._body, b"maximumHeight", self)
            self._anim.setDuration(150)
            self._anim.setStartValue(0)
            self._anim.setEndValue(target)
            self._anim.setEasingCurve(QEasingCurve.OutCubic)
            def _finish_open():
                self._body.setMaximumHeight(16777215)
                self._anim = None
                self.sig_layout_changed.emit()
                self.updateGeometry()
            self._anim.finished.connect(_finish_open)
            self._anim.start()
        else:
            start = self._body.height()
            self._anim = QPropertyAnimation(self._body, b"maximumHeight", self)
            self._anim.setDuration(120)
            self._anim.setStartValue(start)
            self._anim.setEndValue(0)
            self._anim.setEasingCurve(QEasingCurve.InCubic)
            def _finish_close():
                self._body.setVisible(False)
                self._body.setMaximumHeight(0)
                self._anim = None
                self.sig_layout_changed.emit()
            self._anim.finished.connect(_finish_close)
            self._anim.start()

    def sizeHint(self):
        header_h = self._header.sizeHint().height()
        body_h = 0
        if self._body.isVisible():
            bh = self._body.sizeHint().height()
            max_h = self._body.maximumHeight()
            if 0 < max_h < 16777215:
                bh = min(bh, max_h)
            body_h = bh
        return QSize(self.width() if self.width() > 0 else 200, header_h + body_h + 4)


class _SkillCardBlock(QFrame):
    """Inline streaming tool-call card matching ToolGroupCard aesthetic.

    Accent strip + pulsing dot + compact label. Minimal and consistent
    with the post-completion ToolGroupCard rendered in tool_bubbles.py.
    """

    sig_layout_changed = Signal()

    _DOT_SIZE = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._json_buf = ""
        self._done = False
        self._expanded = False
        self._anim: QPropertyAnimation | None = None

        self.setStyleSheet("QFrame { background: transparent; border: none; }")

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(0)

        self._accent = QFrame()
        self._accent.setFixedWidth(3)
        self._accent.setStyleSheet(
            f"background: {_s.ACCENT_PRIMARY}; border: none;"
            " border-top-left-radius: 4px; border-bottom-left-radius: 4px;"
        )
        root.addWidget(self._accent)

        self._card = QFrame()
        self._card.setMinimumWidth(0)
        self._card.setStyleSheet(
            f"background: {_s.BG_GROUP}; border: none;"
            " border-top-right-radius: 4px; border-bottom-right-radius: 4px;"
        )
        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        header = QWidget()
        header.setCursor(Qt.PointingHandCursor)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 10, 4)
        header_layout.setSpacing(8)

        self._dot = QWidget()
        self._dot.setFixedSize(14, 14)
        self._dot_color = QColor(_s.ACCENT_PRIMARY)
        self._dot_opacity = 1.0
        self._dot.paintEvent = self._paint_dot
        self._pulse_anim = QVariantAnimation(self)
        self._pulse_anim.setStartValue(0.35)
        self._pulse_anim.setKeyValueAt(0.5, 1.0)
        self._pulse_anim.setEndValue(0.35)
        self._pulse_anim.setDuration(1200)
        self._pulse_anim.setLoopCount(-1)
        self._pulse_anim.setEasingCurve(QEasingCurve.InOutSine)
        self._pulse_anim.valueChanged.connect(self._on_pulse)
        self._pulse_anim.start()
        header_layout.addWidget(self._dot)

        self._lbl = QLabel("calling tool…")
        self._lbl.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 11px; font-family: Consolas, monospace;"
            f" font-weight: bold; background: transparent;"
        )
        self._lbl.setMinimumWidth(0)
        self._lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        header_layout.addWidget(self._lbl, 1)

        self._chevron = QLabel("☰")
        self._chevron.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; background: transparent;"
        )
        header_layout.addWidget(self._chevron)

        card_layout.addWidget(header)
        self._header_widget = header
        header.mousePressEvent = lambda _e: self._toggle()

        self._body = QFrame()
        self._body.setVisible(False)
        self._body.setMaximumHeight(0)
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(30, 0, 10, 4)
        body_layout.setSpacing(0)

        self._text = _AutoTextView()
        self._text.setStyleSheet(
            "background: transparent; border: none; "
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas;"
        )
        self._text.heightChanged.connect(lambda _: self.sig_layout_changed.emit())
        body_layout.addWidget(self._text)
        card_layout.addWidget(self._body)

        root.addWidget(self._card, 1)

    def _paint_dot(self, _event) -> None:
        p = QPainter(self._dot)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setOpacity(self._dot_opacity)
        d = self._DOT_SIZE
        x = (self._dot.width() - d) // 2
        y = (self._dot.height() - d) // 2
        p.setBrush(self._dot_color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(x, y, d, d)
        p.end()

    def _on_pulse(self, value) -> None:
        try:
            self._dot_opacity = float(value)
        except (TypeError, ValueError):
            return
        self._dot.update()

    def append_content(self, text: str) -> None:
        self._json_buf += text
        label = _tool_summary(self._json_buf)
        if label and label != "tool call":
            self._lbl.setText(f"running {label}…")

    def close_block(self) -> None:
        self._done = True
        self._pulse_anim.stop()
        self._dot_opacity = 1.0
        self._dot_color = QColor(_s.FG_ACCENT)
        self._dot.update()
        label = _tool_summary(self._json_buf)
        self._lbl.setText(f"{label} ✓" if label else "tool ✓")
        try:
            parsed = _json.loads(self._json_buf)
            pretty = _json.dumps(parsed, indent=2)
        except Exception:
            pretty = self._json_buf
        self._text.setPlainText(pretty)
        self._text._update_height()
        self.sig_layout_changed.emit()

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._chevron.setText("☷" if self._expanded else "☰")
        self._set_body_visible(self._expanded)
        self.sig_layout_changed.emit()

    def _set_body_visible(self, visible: bool) -> None:
        if self._anim:
            self._anim.stop()
            self._anim = None
        if visible:
            self._body.setVisible(True)
            self._body.setMaximumHeight(0)
            self._text._update_height()
            lay = self._body.layout()
            if lay:
                lay.activate()
            target = max(self._body.sizeHint().height(), self._text._last_h + 12)
            self._anim = QPropertyAnimation(self._body, b"maximumHeight", self)
            self._anim.setDuration(150)
            self._anim.setStartValue(0)
            self._anim.setEndValue(target)
            self._anim.setEasingCurve(QEasingCurve.OutCubic)
            def _finish():
                self._body.setMaximumHeight(16777215)
                self._anim = None
                self.sig_layout_changed.emit()
            self._anim.finished.connect(_finish)
            self._anim.start()
        else:
            start = self._body.height()
            self._anim = QPropertyAnimation(self._body, b"maximumHeight", self)
            self._anim.setDuration(120)
            self._anim.setStartValue(start)
            self._anim.setEndValue(0)
            self._anim.setEasingCurve(QEasingCurve.InCubic)
            def _finish():
                self._body.setVisible(False)
                self._body.setMaximumHeight(0)
                self._anim = None
                self.sig_layout_changed.emit()
            self._anim.finished.connect(_finish)
            self._anim.start()

    def sizeHint(self):
        h = self._header_widget.sizeHint().height() + 4
        if self._body.isVisible():
            bh = self._body.sizeHint().height()
            mx = self._body.maximumHeight()
            if 0 < mx < 16777215:
                bh = min(bh, mx)
            h += bh
        return QSize(self.width() if self.width() > 0 else 200, h)


class MessageWidget(QFrame):
    sig_action = Signal(str, int)
    sig_delete = Signal(int)
    sig_edit = Signal(int)
    sig_regen = Signal(int)
    sig_switch_take = Signal(int, int)  # (index, direction) — in-chat ‹k/n› take switcher
    sig_height_changed = Signal()  # emitted when internal height changes
    sig_open_attachment = Signal(object)  # emits the clicked Attachment

    _HEADER_H = 16
    _MARGINS = (12, 2, 6, 2)  # left, top, right, bottom
    _SPACING = 0
    _DEBUG_GAP = False

    def __init__(self, index: int, role: str, text: str, timestamp: str, attachments=None):
        super().__init__()
        self._index = index
        self._role = role
        self._content = text or ""
        self._is_streaming = False
        self._attachments = list(attachments or [])
        self._attach_row = None
        self._attach_label = None

        self.setAttribute(Qt.WA_Hover, True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("class", "MessageWidget")
        self.setProperty("role", role)
        self.setProperty("editing", False)

        is_assistant = role == "assistant"
        is_system = role == "system"

        # Subtle role-tinted bubble bg + accent line on the leading edge.
        # Keeps the dark/minimal aesthetic but makes user vs assistant
        # visually distinct at a glance without needing avatars.
        if is_assistant:
            self.setStyleSheet(
                "QFrame[class=\"MessageWidget\"][role=\"assistant\"] {"
                f"  background: {_s.BG_SURFACE_1};"
                f"  border-left: 2px solid {_s.ACCENT_PRIMARY};"
                "  border-radius: 3px;"
                "}"
            )
        elif is_system:
            self.setStyleSheet(
                "QFrame[class=\"MessageWidget\"][role=\"system\"] {"
                f"  background: transparent;"
                f"  border-left: 2px solid {_s.FG_DIM};"
                "  border-radius: 3px;"
                "}"
            )
        else:
            self.setStyleSheet(
                "QFrame[class=\"MessageWidget\"][role=\"user\"] {"
                f"  background: transparent;"
                f"  border-left: 2px solid {_s.BORDER_SUBTLE};"
                "  border-radius: 3px;"
                "}"
            )

        root = QVBoxLayout(self)
        root.setContentsMargins(*self._MARGINS)
        root.setSpacing(self._SPACING)

        head = QHBoxLayout()
        head.setSpacing(6)

        self.lbl_role = QLabel((role or "").upper())
        self.lbl_role.setObjectName("msg_role")
        self.lbl_role.setProperty("role", role)
        self.lbl_role.setFixedHeight(self._HEADER_H)
        # Theme the role label per role so the eye picks up speaker quickly.
        _role_color = _s.ACCENT_PRIMARY if is_assistant else _s.FG_SECONDARY
        if is_system:
            _role_color = _s.FG_DIM
        self.lbl_role.setStyleSheet(
            f"color: {_role_color}; font-size: 9px; font-family: Consolas;"
            "font-weight: 600; letter-spacing: 1px; background: transparent;"
        )
        head.addWidget(self.lbl_role)

        # Streaming pulse: a single small dot that fades in/out while tokens
        # are arriving. Hidden for non-assistant messages and after finalize.
        self._stream_pulse = QLabel("●")
        self._stream_pulse.setFixedHeight(self._HEADER_H)
        self._stream_pulse.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 8px; "
            "font-family: Consolas; background: transparent;"
        )
        self._stream_pulse.setVisible(False)
        head.addWidget(self._stream_pulse)
        self._stream_pulse_timer = QTimer(self)
        self._stream_pulse_timer.setInterval(450)
        self._stream_pulse_timer.timeout.connect(self._tick_stream_pulse)
        self._stream_pulse_phase = False

        pretty_ts = (timestamp or "")
        if "T" in pretty_ts and len(pretty_ts) >= 16:
            pretty_ts = pretty_ts[11:16]
        self.lbl_time = QLabel(pretty_ts)
        self.lbl_time.setObjectName("msg_time")
        self.lbl_time.setFixedHeight(self._HEADER_H)
        head.addWidget(self.lbl_time)
        self.lbl_tokens = QLabel("")
        self.lbl_tokens.setObjectName("msg_tokens")
        self.lbl_tokens.setFixedHeight(self._HEADER_H)
        self.lbl_tokens.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; background: transparent;"
        )
        self.lbl_tokens.hide()
        head.addWidget(self.lbl_tokens)

        self._think_badge = QFrame()
        self._think_badge.setObjectName("msg_think_badge")
        self._think_badge.setFixedHeight(self._HEADER_H)
        self._think_badge.setCursor(Qt.PointingHandCursor)
        self._think_badge.setVisible(False)
        badge_layout = QHBoxLayout(self._think_badge)
        badge_layout.setContentsMargins(4, 0, 0, 0)
        badge_layout.setSpacing(3)
        self._think_dot = QLabel("•")
        self._think_dot.setFixedWidth(8)
        self._think_dot.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 6px; font-family: Consolas; background: transparent;"
        )
        badge_layout.addWidget(self._think_dot)
        self._think_duration = QLabel("")
        self._think_duration.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; background: transparent;"
        )
        badge_layout.addWidget(self._think_duration)
        self._think_badge.mousePressEvent = self._on_think_badge_mouse_press
        head.addWidget(self._think_badge)
        head.addStretch()

        root.addLayout(head)

        # Attachment chips: render [ATTACHED] blobs as clickable chips instead of
        # dumping the raw block text into the bubble. Click routes to the databank.
        if self._attachments:
            from html import escape as _esc
            self._attach_row = QWidget()
            _arow = QHBoxLayout(self._attach_row)
            _arow.setContentsMargins(0, 2, 0, 2)
            _arow.setSpacing(0)
            _chips = []
            for _i, _att in enumerate(self._attachments):
                _meta = _att.size or _att.type or ""
                _cap = f"{_att.label or 'attachment'} · {_meta}" if _meta else (_att.label or "attachment")
                _chips.append(
                    f'📎 <a href="attach:{_i}" style="color:{_s.ACCENT_PRIMARY};'
                    f' text-decoration:none;">{_esc(_cap)}</a>'
                )
            self._attach_label = QLabel("&nbsp;&nbsp;&nbsp;".join(_chips))
            self._attach_label.setTextFormat(Qt.RichText)
            self._attach_label.setWordWrap(True)
            self._attach_label.setOpenExternalLinks(False)
            self._attach_label.setCursor(Qt.PointingHandCursor)
            self._attach_label.setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas;"
                f" background: {_s.BG_SURFACE_2}; border: 1px solid {_s.BORDER_SUBTLE};"
                " border-radius: 6px; padding: 3px 8px;"
            )
            self._attach_label.linkActivated.connect(self._on_attach_link)
            _arow.addWidget(self._attach_label)
            _arow.addStretch()
            root.addWidget(self._attach_row)

        # think blocks container (hidden until first <think>)
        if self._DEBUG_GAP:
            self._gap_marker = QLabel("•")
            self._gap_marker.setStyleSheet("color: red; font-size: 10px; background: transparent;")
            self._gap_marker.setFixedHeight(8)
            self._gap_marker.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            root.addWidget(self._gap_marker)

        self._think_blocks: list[_ThinkBlock] = []
        self._think_container = QWidget()
        self._think_container.setVisible(False)
        think_cont_layout = QVBoxLayout(self._think_container)
        think_cont_layout.setContentsMargins(0, 4, 0, 4)
        think_cont_layout.setSpacing(2)
        self._think_cont_layout = think_cont_layout
        root.addWidget(self._think_container)

        self.text_view = _AutoTextView()
        self.text_view.setObjectName("msg_content")
        self.text_view.setProperty("role", role)
        self._set_text_view(self._content)
        self.text_view.heightChanged.connect(self._on_text_height_changed)
        root.addWidget(self.text_view)

        self._skill_blocks: list[_SkillBlock] = []
        self._skill_container = QWidget()
        self._skill_container.setVisible(False)
        skill_cont_layout = QVBoxLayout(self._skill_container)
        skill_cont_layout.setContentsMargins(0, 6, 0, 6)
        skill_cont_layout.setSpacing(4)
        self._skill_cont_layout = skill_cont_layout
        root.addWidget(self._skill_container)

        self._output_tokens = 0
        if text:
            self._update_token_label()

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._height_pending = False
        self._layout_update_timer = QTimer(self)
        self._layout_update_timer.setSingleShot(True)
        self._layout_update_timer.setInterval(0)
        self._layout_update_timer.timeout.connect(self._do_update_height)
        self._last_content_width = 0
        self._last_total_height = 28
        self._stream_chars_since_layout = 0
        self._stream_layout_timer = QTimer(self)
        self._stream_layout_timer.setSingleShot(True)
        self._stream_layout_timer.setInterval(33)
        self._stream_layout_timer.timeout.connect(self._flush_stream_layout)
        self._hover_hide_timer = QTimer(self)
        self._hover_hide_timer.setSingleShot(True)
        self._hover_hide_timer.setInterval(80)
        self._hover_hide_timer.timeout.connect(self._maybe_hide_hover_bar)
        self._think_expanded = False

        # Block lifecycle state. Tags are parsed upstream; these just track the
        # currently open think/skill block so apply_stream_update() can append
        # to the right one.
        self._current_think: _ThinkBlock | None = None
        self._current_skill: _SkillBlock | None = None
        # Bearing-update card (parallel to _current_skill but for
        # <bearing_update> envelopes — Bearing V0). Reuses the SkillCardBlock
        # widget infrastructure with relabeled icon/title; opens on
        # bearing_update_opened, closes on bearing_update_closed.
        self._current_bearing: _SkillBlock | None = None
        self._suppress_leading_newline = False
        self._hover_bar = QFrame(self)
        self._hover_bar.setObjectName("msg_hover_bar")
        self._hover_bar.setStyleSheet(
            f"""
            QFrame#msg_hover_bar {{
                background: {_s.BG_PANEL};
                border: 1px solid {_s.BORDER_LIGHT};
                border-radius: 5px;
            }}
            """
        )
        hover_layout = QHBoxLayout(self._hover_bar)
        hover_layout.setContentsMargins(2, 2, 2, 2)
        hover_layout.setSpacing(1)
        if role == "user":
            self.btn_edit = _HoverAction("Edit")
            self.btn_edit.clicked.connect(lambda: self._emit_action("edit"))
            hover_layout.addWidget(self.btn_edit)
        if is_assistant:
            self.btn_thumbs_up = _HoverAction("↑")
            self.btn_thumbs_up.setToolTip("Mark response as good")
            self.btn_thumbs_up.clicked.connect(lambda: self._emit_action("thumbs_up"))
            hover_layout.addWidget(self.btn_thumbs_up)
            self.btn_thumbs_down = _HoverAction("↓")
            self.btn_thumbs_down.setToolTip("Mark response as bad")
            self.btn_thumbs_down.clicked.connect(lambda: self._emit_action("thumbs_down"))
            hover_layout.addWidget(self.btn_thumbs_down)
            # ‹ k/n › take switcher — same _HoverAction style as the regen
            # control, added immediately before it; hidden until set_take_info
            # gives it data.
            self._take_prev = _HoverAction("‹")
            self._take_prev.setToolTip("Previous take")
            self._take_label = QLabel("")
            self._take_label.setStyleSheet(
                f"color: {_s.FG_DIM}; font-family: Consolas; font-size: 9px;")
            self._take_next = _HoverAction("›")
            self._take_next.setToolTip("Next take")
            self._take_prev.clicked.connect(
                lambda: self.sig_switch_take.emit(self._index, -1))
            self._take_next.clicked.connect(
                lambda: self.sig_switch_take.emit(self._index, +1))
            for w in (self._take_prev, self._take_label, self._take_next):
                w.setVisible(False)
                hover_layout.addWidget(w)
            self.btn_regen = _HoverAction("Regen")
            self.btn_regen.clicked.connect(lambda: self._emit_action("regen"))
            hover_layout.addWidget(self.btn_regen)
            self.btn_copy = _HoverAction("Copy")
            # Copy emits an outcome signal AND copies — the chat handler
            # records the outcome; the local helper does the clipboard work.
            self.btn_copy.clicked.connect(self._copy_with_outcome)
            hover_layout.addWidget(self.btn_copy)
        if not is_system:
            self.btn_delete = _HoverAction("Delete")
            self.btn_delete.clicked.connect(lambda: self._emit_action("delete"))
            hover_layout.addWidget(self.btn_delete)
        self._hover_bar.adjustSize()
        self._hover_bar.hide()

    def _on_attach_link(self, href: str) -> None:
        """linkActivated handler for attachment chips. href = 'attach:<index>'."""
        if not isinstance(href, str) or not href.startswith("attach:"):
            return
        try:
            idx = int(href.split(":", 1)[1])
        except (ValueError, IndexError):
            return
        if 0 <= idx < len(self._attachments):
            self.sig_open_attachment.emit(self._attachments[idx])

    def _on_text_height_changed(self, text_h=None):
        if not self._height_pending:
            self._height_pending = True
            if not self._layout_update_timer.isActive():
                self._layout_update_timer.start()

    def _do_update_height(self):
        self._height_pending = False
        try:
            w = self.width() if self.width() > 50 else 600
            m = self._MARGINS
            content_w = w - m[0] - m[2] - 2  # 2px for border-left
            if content_w > 50:
                if content_w != self._last_content_width:
                    self._last_content_width = content_w
                    self._think_container.setFixedWidth(content_w)
                    self._skill_container.setFixedWidth(content_w)
                    for block in self._think_blocks + self._skill_blocks:
                        block.setFixedWidth(content_w)
                        if hasattr(block, "_body"):
                            block._body.setFixedWidth(content_w)
                        if hasattr(block, "_text"):
                            text_w = max(content_w - 12, 20)
                            if getattr(block, "_last_text_width", 0) != text_w:
                                block._last_text_width = text_w
                                block._text.setFixedWidth(text_w)
                                block._text._schedule_update()
                    if self._DEBUG_GAP:
                        self._gap_marker.setFixedWidth(content_w)
                    self.text_view.setFixedWidth(content_w)
                    self.text_view._schedule_update()
                    if self._attach_row is not None:
                        self._attach_row.setFixedWidth(content_w)
            if self._think_container.layout():
                self._think_container.layout().activate()
            self._think_container.adjustSize()
            if self._skill_container.layout():
                self._skill_container.layout().activate()
            self._skill_container.adjustSize()
            text_h = self.text_view._last_h
            think_h = 0
            if not self._think_container.isHidden():
                lay = self._think_container.layout()
                if lay:
                    lay.activate()
                think_h = self._think_container.sizeHint().height()
            skill_h = 0
            if not self._skill_container.isHidden():
                lay = self._skill_container.layout()
                if lay:
                    lay.activate()
                skill_h = self._skill_container.sizeHint().height()
            attach_h = self._attach_row.sizeHint().height() if self._attach_row is not None else 0
            total = m[1] + self._HEADER_H + attach_h + self._SPACING + text_h + think_h + skill_h + m[3]
            total = max(total, 28)
            if total != self._last_total_height:
                self._last_total_height = total
                self.setFixedHeight(total)
                self.sig_height_changed.emit()
        except RuntimeError:
            # Widget/layout may already be deleted while queued callbacks drain.
            return

    def sizeHint(self):
        w = self.width() if self.width() > 50 else 600
        m = self._MARGINS
        content_w = w - m[0] - m[2] - 2  # 2px for border-left
        text_h = self.text_view.ideal_height(content_w)
        think_h = 0
        if not self._think_container.isHidden():
            lay = self._think_container.layout()
            if lay:
                lay.activate()
            think_h = self._think_container.sizeHint().height()
        skill_h = 0
        if not self._skill_container.isHidden():
            lay = self._skill_container.layout()
            if lay:
                lay.activate()
            skill_h = self._skill_container.sizeHint().height()
        attach_h = self._attach_row.sizeHint().height() if self._attach_row is not None else 0
        total = m[1] + self._HEADER_H + attach_h + self._SPACING + text_h + think_h + skill_h + m[3]
        return QSize(w, max(total, 28))

    def enterEvent(self, event):
        super().enterEvent(event)
        self._show_hover_bar()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._schedule_hide_hover_bar()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            old_w = event.oldSize().width()
            new_w = event.size().width()
        except Exception:
            old_w = -1
            new_w = self.width()
        if old_w != new_w:
            self._on_text_height_changed()
        if self._hover_bar.isVisible():
            self._position_hover_bar()

    def request_layout_update(self) -> None:
        self._on_text_height_changed()

    def get_selectable_text_views(self) -> list:
        """Return selectable QTextEdit views in document order.

        Used by ChatSelectionManager to drive cross-widget drag-select.
        Per the UX choice, think and skill blocks contribute only when
        the user has them visibly expanded — selecting over collapsed
        sections won't sweep their hidden text into the copy.
        """
        views = []
        # Think blocks render above the main body when expanded.
        if self._think_container.isVisible() and self._think_expanded:
            for block in self._think_blocks:
                text = getattr(block, "_text", None)
                if text is not None and text.isVisible():
                    views.append(text)
        # Main message body.
        if self.text_view.isVisible():
            views.append(self.text_view)
        # Skill (tool-call) cards rendered below the body; include only
        # when the user has expanded the card's body.
        if self._skill_container.isVisible():
            for block in self._skill_blocks:
                body = getattr(block, "_body", None)
                text = getattr(block, "_text", None)
                if text is None or body is None:
                    continue
                if body.isVisible() and body.maximumHeight() != 0:
                    views.append(text)
        return views

    def _position_hover_bar(self) -> None:
        bar = self._hover_bar
        bar.adjustSize()
        margin_left, margin_top, margin_right, _margin_bottom = self._MARGINS
        x = max(margin_left, self.width() - bar.width() - margin_right)
        y = max(margin_top, 2)
        bar.move(x, y)

    def _show_hover_bar(self) -> None:
        if self._hover_bar.layout() is not None and self._hover_bar.layout().count() == 0:
            return
        if self._hover_hide_timer.isActive():
            self._hover_hide_timer.stop()
        self._position_hover_bar()
        self._hover_bar.show()
        self._hover_bar.raise_()

    def _schedule_hide_hover_bar(self) -> None:
        if self._hover_bar.isVisible():
            self._hover_hide_timer.start()

    def _maybe_hide_hover_bar(self) -> None:
        local_pos = self.mapFromGlobal(QCursor.pos())
        if self.rect().contains(local_pos) or self._hover_bar.geometry().contains(local_pos):
            return
        self._hover_bar.hide()

    def _copy_to_clipboard(self) -> None:
        from PySide6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._content)

    def _copy_with_outcome(self) -> None:
        """Copy to clipboard and emit a `copy` outcome action.

        The chat handler records the outcome to the turn trace; the local
        clipboard write is unconditional (we want the copy to succeed even
        if outcome recording fails).
        """
        self._copy_to_clipboard()
        self._emit_action("copy")

    def _clear_embedded_text_selection(self) -> None:
        views = [self.text_view]
        views.extend(getattr(block, "_text", None) for block in self._think_blocks)
        views.extend(getattr(block, "_text", None) for block in self._skill_blocks)
        for view in views:
            if view is None:
                continue
            try:
                cursor = view.textCursor()
                if cursor.hasSelection():
                    cursor.clearSelection()
                    view.setTextCursor(cursor)
                view.moveCursor(QTextCursor.End)
            except RuntimeError:
                continue

    def _on_think_badge_mouse_press(self, event) -> None:
        self._clear_embedded_text_selection()
        self._toggle_think_expand()
        if event is not None:
            event.accept()

    def _toggle_think_expand(self) -> None:
        if not self._think_blocks:
            return
        self._think_expanded = not self._think_expanded
        self._think_container.setVisible(self._think_expanded)
        for block in self._think_blocks:
            block.set_expanded(self._think_expanded)
        self._on_text_height_changed()

    def _update_think_badge(self, thinking_text: str = "", thinking_done: bool = False) -> None:
        cached_text = ""
        if self._think_blocks:
            cached_text = self._think_blocks[-1]._text.toPlainText()
        sample_text = thinking_text or cached_text
        has_thinking = bool(sample_text.strip()) or any(block.has_content() for block in self._think_blocks)
        if not has_thinking:
            self._think_badge.hide()
            return
        if thinking_done and self._think_blocks:
            duration = self._think_blocks[-1].duration_seconds()
            if duration >= 0.2:
                label = f"{duration:.1f}s"
            else:
                label = f"~{max(1, len(sample_text) // 180)}s"
        else:
            label = "thinking"
        self._think_duration.setText(label)
        self._think_badge.show()

    def _update_token_label(self) -> None:
        if self._role != "assistant":
            return
        think_len = sum(len(b._text.toPlainText()) for b in self._think_blocks)
        n = max(1, (len(self._content) + think_len) // 4)
        self.lbl_tokens.setText(f"↓{n}t")
        self.lbl_tokens.show()

    def _emit_action(self, action: str) -> None:
        self.sig_action.emit(action, self._index)
        if action == "edit":
            self.sig_edit.emit(self._index)
            return
        if action == "regen":
            self.sig_regen.emit(self._index)
            return
        if action == "delete":
            self.sig_delete.emit(self._index)

    def set_editing(self, active: bool) -> None:
        if bool(self.property("editing")) == bool(active):
            return
        self.setProperty("editing", bool(active))
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
        self.update()

    def _tick_stream_pulse(self) -> None:
        # Two-state opacity flip via stylesheet alpha; cheap and avoids a
        # QGraphicsEffect for a single-character label.
        self._stream_pulse_phase = not self._stream_pulse_phase
        col = _s.ACCENT_PRIMARY if self._stream_pulse_phase else _s.FG_DIM
        self._stream_pulse.setStyleSheet(
            f"color: {col}; font-size: 8px; "
            "font-family: Consolas; background: transparent;"
        )

    def begin_stream(self) -> None:
        """Mark this widget as actively receiving tokens. Pulses the indicator."""
        if self._role != "assistant":
            return
        self._is_streaming = True
        self._stream_pulse.setVisible(True)
        if not self._stream_pulse_timer.isActive():
            self._stream_pulse_timer.start()

    def end_stream(self) -> None:
        """Mark stream complete. Hides the pulse and stops the timer."""
        self._is_streaming = False
        if self._stream_pulse_timer.isActive():
            self._stream_pulse_timer.stop()
        self._stream_pulse.setVisible(False)

    def append_token(self, token: str):
        if not token:
            return
        # First token from this stream auto-engages the pulse so callers
        # don't need to remember to call begin_stream() explicitly.
        if self._role == "assistant" and not self._is_streaming:
            self.begin_stream()
        self._content += token
        self.text_view.append_plain_text(token)
        self._stream_chars_since_layout += len(token)
        if "\n" in token or self._stream_chars_since_layout >= 64:
            self._flush_stream_layout()
            return
        if not self._stream_layout_timer.isActive():
            self._stream_layout_timer.start()

    def streaming_think_active(self) -> bool:
        return self._current_think is not None

    def _reset_structured_content(self) -> None:
        self._content = ""
        self._current_think = None
        self._current_skill = None
        self._current_bearing = None
        self._suppress_leading_newline = False
        self._stream_chars_since_layout = 0
        if self._stream_layout_timer.isActive():
            self._stream_layout_timer.stop()
        while self._think_cont_layout.count():
            item = self._think_cont_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        while self._skill_cont_layout.count():
            item = self._skill_cont_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._think_blocks.clear()
        self._skill_blocks.clear()
        self._think_expanded = False
        self._think_badge.hide()
        self._think_container.setVisible(False)
        self._skill_container.setVisible(False)
        self.text_view.clear()

    def _set_text_view(self, text: str) -> None:
        """Render text into the text view.

        Assistant messages go through the themed markdown renderer
        (syntax-highlighted code blocks, themed tables/blockquotes/inline-code).
        User and system messages stay plain to avoid surprising the sender's
        own input with HTML interpretation.
        """
        if self._role == "assistant" and text:
            try:
                self.text_view.setHtml(render_message_html(text))
            except Exception:
                # Fall back to plain text if the renderer or pygments hits
                # something unexpected -- never block the message from showing.
                self.text_view.setPlainText(text)
        else:
            self.text_view.setPlainText(text)

    def _render_structured_text(self, text: str) -> None:
        self._reset_structured_content()
        self._content = text or ""
        self._set_text_view(self._content)
        self._on_text_height_changed()

    def _open_think_block(self) -> None:
        block = _ThinkBlock()
        block.set_compact(True)
        block.set_expanded(self._think_expanded)
        block.sig_layout_changed.connect(self._on_text_height_changed_no_arg)
        self._think_blocks.append(block)
        self._think_cont_layout.addWidget(block)
        self._think_container.setVisible(self._think_expanded)
        self._current_think = block
        self._update_think_badge(thinking_done=False)
        self.sig_height_changed.emit()

    def _close_think_block(self) -> None:
        if self._current_think:
            self._current_think.close_block()
            self._current_think = None
        self._update_think_badge(thinking_done=True)
        self._update_token_label()
        self._suppress_leading_newline = True
        self.sig_height_changed.emit()

    def _open_skill_block(self) -> None:
        block = _SkillCardBlock()
        block.sig_layout_changed.connect(self._on_text_height_changed_no_arg)
        self._skill_blocks.append(block)
        self._skill_cont_layout.addWidget(block)
        self._skill_container.setVisible(True)
        self._current_skill = block
        self._sync_structured_layout_now()

    def _close_skill_block(self) -> None:
        if self._current_skill:
            self._current_skill.close_block()
            self._current_skill = None
        self._suppress_leading_newline = True
        self._sync_structured_layout_now()

    def _open_bearing_block(self) -> None:
        """Open a bearing-update card. Uses BearingUpdateCard (modern
        ToolGroupCard-style aesthetic) — pulsing dot during stream, settled
        label when closed, expandable JSON body. Distinct from tool-call
        cards via the SUBSTRATE badge.
        """
        from ui.components.tool_bubbles import BearingUpdateCard
        block = BearingUpdateCard()
        block.sig_height_changed.connect(self._on_text_height_changed_no_arg)
        self._skill_blocks.append(block)
        self._skill_cont_layout.addWidget(block)
        self._skill_container.setVisible(True)
        self._current_bearing = block
        self._sync_structured_layout_now()

    def _close_bearing_block(self) -> None:
        if self._current_bearing:
            self._current_bearing.close_block()
            self._current_bearing = None
        self._suppress_leading_newline = True
        self._sync_structured_layout_now()

    def _on_text_height_changed_no_arg(self) -> None:
        self._on_text_height_changed()

    def _sync_structured_layout_now(self) -> None:
        if self._layout_update_timer.isActive():
            self._layout_update_timer.stop()
        previous_height = self._last_total_height
        self._do_update_height()
        if self._last_total_height == previous_height:
            self.sig_height_changed.emit()

    def _flush_stream_layout(self) -> None:
        if self._stream_layout_timer.isActive():
            self._stream_layout_timer.stop()
        self._stream_chars_since_layout = 0
        self.text_view._update_height()
        self._do_update_height()
        if self._content:
            self._update_token_label()

    def _ensure_think_block(self) -> _ThinkBlock:
        if self._current_think is None:
            self._open_think_block()
        assert self._current_think is not None
        return self._current_think

    def append_thinking_token(self, text: str) -> None:
        if not text:
            return
        block = self._ensure_think_block()
        block.append_think_text(text)
        self._update_think_badge(text, thinking_done=False)

    def close_thinking_block(self) -> None:
        if self._current_think is not None:
            self._close_think_block()

    def apply_stream_update(self, update) -> None:
        if update is None:
            return

        # Open blocks BEFORE pushing content so the first character lands in
        # the right container.
        if getattr(update, "thinking_opened", False) and self._current_think is None:
            self._update_think_badge(thinking_done=False)
        if getattr(update, "tool_call_opened", False) and self._current_skill is None:
            self._open_skill_block()
        if getattr(update, "bearing_update_opened", False) and self._current_bearing is None:
            self._open_bearing_block()

        thinking_text = str(getattr(update, "thinking_text", "") or "")
        if thinking_text:
            self.append_thinking_token(thinking_text)

        tool_call_text = str(getattr(update, "tool_call_text", "") or "")
        if tool_call_text:
            if self._current_skill is None:
                self._open_skill_block()
            if self._current_skill is not None:
                self._current_skill.append_content(tool_call_text)

        bearing_update_text = str(getattr(update, "bearing_update_text", "") or "")
        if bearing_update_text:
            if self._current_bearing is None:
                self._open_bearing_block()
            if self._current_bearing is not None:
                self._current_bearing.append_content(bearing_update_text)

        answer_text = str(getattr(update, "answer_text", "") or "")
        if answer_text:
            self.append_token(answer_text)

        if getattr(update, "thinking_closed", False):
            self.close_thinking_block()
        if getattr(update, "tool_call_closed", False):
            self._close_skill_block()
        if getattr(update, "bearing_update_closed", False):
            self._close_bearing_block()

    def apply_assistant_display(
        self,
        answer_text: str,
        thinking_text: str = "",
        *,
        thinking_done: bool = True,
    ) -> None:
        self._reset_structured_content()
        self._content = answer_text or ""
        self._set_text_view(self._content)
        if thinking_text:
            block = self._ensure_think_block()
            block.append_think_text(thinking_text)
            block._flush_pending_text()
            if thinking_done:
                self.close_thinking_block()
            else:
                self._update_think_badge(thinking_text, thinking_done=False)
        self._flush_stream_layout()

    def finalize(self):
        if self._current_think is not None:
            self.close_thinking_block()
        # Stop the streaming pulse before rerendering so the document doesn't
        # repaint with the indicator still visible.
        self.end_stream()
        # Re-render streamed plaintext through the themed markdown pipeline
        # now that the message is complete (syntax highlighting, tables, etc).
        if self._role == "assistant" and self._content:
            try:
                self.text_view.setHtml(render_message_html(self._content))
            except Exception:
                self.text_view.setPlainText(self._content)
        self._flush_stream_layout()
        self._on_text_height_changed()

    def update_main_text(self, text: str) -> None:
        """Update the visible message body with raw text."""
        self._content = text or ""
        self._set_text_view(self._content)
        self._on_text_height_changed()
        if self._content:
            self._update_token_label()

    def set_text(self, text: str) -> None:
        self._render_structured_text(text or "")
        if self._content:
            self._update_token_label()

    def set_index(self, idx: int):
        self._index = idx

    def set_take_info(self, info) -> None:
        """Show/hide the ‹k/n› take switcher and set its label. ``info`` is a
        (k, n) tuple or None (hidden). No-op for non-assistant widgets that
        never built the controls."""
        if not hasattr(self, "_take_prev"):
            return
        visible = info is not None
        for w in (self._take_prev, self._take_label, self._take_next):
            w.setVisible(visible)
        if info:
            self._take_label.setText(f"{info[0]}/{info[1]}")
