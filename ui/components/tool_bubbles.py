from __future__ import annotations

import html
import json
from pathlib import Path

from PySide6.QtCore import (
    Qt, Signal, QPropertyAnimation, QEasingCurve, QSize, QTimer, QUrl, QVariantAnimation,
)
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget, QTextEdit,
)
from PySide6.QtGui import QTextOption, QPixmap, QDesktopServices, QColor, QPainter

import core.style as _s
from core.skill_registry import canonical_tool_name


_THUMB_PX = 96  # max thumbnail edge in chat strip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pretty_ts(timestamp: str) -> str:
    pretty = timestamp or ""
    if "T" in pretty and len(pretty) >= 16:
        pretty = pretty[11:16]
    return pretty


def _tool_name(payload: dict) -> str:
    token = payload.get("tool", payload.get("skill", payload.get("op", "")))
    return canonical_tool_name(token) or str(token or "tool")


def _strip_call_prefix(text: str) -> tuple[str | None, str]:
    body = str(text or "")
    if body.startswith("[call:"):
        line, _, rest = body.partition("\n")
        if line.endswith("]"):
            return line[6:-1], rest
    return None, body


def _call_summary(payload: dict) -> str:
    """One-line summary for a tool call payload."""
    calls = payload.get("calls")
    if isinstance(calls, list):
        names = []
        for c in calls[:4]:
            if isinstance(c, dict):
                names.append(_tool_name(c))
        suffix = f" +{len(calls) - 4}" if len(calls) > 4 else ""
        return f"batch: {', '.join(names)}{suffix}"
    tool = _tool_name(payload)
    if tool == "spawn_subagent":
        # Compact "Agent" line on the command — replaces the verbose subagent
        # bubble. The active-agents spine (core.active_agents) carries the live
        # view to the model; here we just label the call.
        frame = str(payload.get("frame") or "").strip() \
            or f"L{payload.get('_validated_child_level', payload.get('level', 2))}"
        return f"● Agent: {frame} (running…)"
    path = payload.get("path", "")
    if path:
        name = Path(path).name
        if tool == "read_file":
            return f"read_file -> {name}"
        if tool == "list_files":
            pat = payload.get("pattern", "*")
            return f"list_files -> {name} [{pat}]"
        if tool == "write_file":
            return f"write_file -> {name}"
        if tool == "edit_file":
            return f"edit_file -> {name}"
        if tool == "grep":
            pat = str(payload.get("pattern", ""))[:40]
            return f"grep \"{pat}\" -> {name}"
    if tool == "grep":
        return f"grep \"{str(payload.get('pattern', ''))[:40]}\""
    if tool == "calculate":
        return f"calculate -> {str(payload.get('expr', ''))[:48]}"
    if tool == "save_note":
        return f"save_note -> {payload.get('title', '?')}"
    if tool == "search_history":
        return f"search_history -> {str(payload.get('query', ''))[:40]}"
    if tool == "open_addon":
        return f"open_addon -> {payload.get('addon', '?')}"
    if tool == "run_command":
        return f"run_command -> {str(payload.get('command', ''))[:40]}"
    if tool == "generate_image":
        return f"generate_image -> {str(payload.get('prompt', ''))[:40]}"
    if tool == "generate_audio":
        return f"generate_audio -> {str(payload.get('prompt', ''))[:40]}"
    if tool == "soundtrap":
        return f"soundtrap -> {str(payload.get('op') or payload.get('verb') or 'state')}"
    if tool == "web_search":
        return f"web_search -> {str(payload.get('query', ''))[:40]}"
    # fallback: show first 2 params
    args = [f"{k}={v!r}" for k, v in payload.items() if k not in {"id", "tool", "skill", "op"}]
    if args:
        return f"{tool}({', '.join(args[:2])})"
    return tool


def _result_summary(payload: dict) -> str:
    """One-line summary for a tool result payload."""
    tool = canonical_tool_name(payload.get("tool", "")) or str(payload.get("tool", "tool"))
    _, body = _strip_call_prefix(str(payload.get("result", "")))
    body = body.strip()
    if tool == "spawn_subagent" or body.startswith("[SUBAGENT_RESULT"):
        # The verbose subagent output is the model's work product (it folds back
        # to the model via the followup loop, untouched); the human sees a
        # one-liner that flips running -> done.
        return "● Agent (running…)" if "PENDING" in body else "● Agent (done)"
    first_line = ""
    for line in body.splitlines():
        if line.strip():
            first_line = line.strip()
            break
    total = len(body)
    lines_count = len([l for l in body.splitlines() if l.strip()])
    if not body:
        return f"{tool} -> (empty)"
    if tool in ("calculate", "save_note", "open_addon") and first_line:
        return f"{tool} -> {first_line[:80]}"
    if tool in ("read_file", "list_files", "grep", "search_history"):
        return f"{tool} -> {first_line[:60]}  ({lines_count} lines)"
    if tool == "generate_image":
        # generate_image is async; surface the artifact status alongside the
        # text body so the user can see at a glance "still cooking" vs "done".
        status = str(payload.get("status") or "")
        image_paths = payload.get("image_paths") or []
        if image_paths:
            return f"{tool} -> {first_line[:60]}  [{len(image_paths)} image(s) ready]"
        if status == "pending":
            return f"{tool} -> {first_line[:60]}  (pending...)"
        return f"{tool} -> {first_line[:80]}"
    if tool in ("write_file", "edit_file"):
        # Short status line carrying the full path — show it all ("where");
        # the row wraps, so don't clip mid-path.
        return f"{tool} -> {first_line}"
    if tool in ("generate_audio", "soundtrap", "web_search", "run_command"):
        return f"{tool} -> {first_line[:80]}"
    if total < 100:
        return f"{tool} -> {first_line[:80]}"
    return f"{tool} -> {first_line[:60]}  ({total} chars)"


# Tools whose *result* body is worth previewing inline (read-ish output). Write
# tools preview the *call* content instead (what was written), not the result
# status line.
_RESULT_PREVIEW_TOOLS = frozenset({
    "read_file", "grep", "run_command", "list_files", "search_history", "calculate",
})


def _preview_source(kind: str, payload: dict | None) -> str | None:
    """The text to preview inline for a tool-card entry, or None.

    Call entries preview their ``content`` (what was written/saved); result
    entries of read-ish tools preview the result body. Pure — unit-testable.
    """
    payload = payload or {}
    if kind == "call":
        content = payload.get("content")
        return str(content) if content else None
    if kind == "result":
        tool = canonical_tool_name(payload.get("tool", "")) or str(payload.get("tool", "") or "")
        if tool in _RESULT_PREVIEW_TOOLS:
            _, body = _strip_call_prefix(str(payload.get("result", "")))
            body = body.strip()
            return body or None
    return None


def _clip_preview(text: str, *, max_lines: int = 3, max_chars: int = 200) -> tuple[str, bool]:
    """Clip *text* to a compact preview. Returns ``(clipped, has_more)`` where
    has_more is True when the original was longer (so a 'show more' affordance
    is warranted). Pure — unit-testable."""
    text = str(text or "")
    if not text.strip():
        return ("", False)
    lines = text.splitlines()
    has_more = len(lines) > max_lines or len(text) > max_chars
    clipped = "\n".join(lines[:max_lines])
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rstrip()
        has_more = True
    return (clipped, has_more)


# ---------------------------------------------------------------------------
# Shared auto-sizing text view (matches _SkillBlock / _ThinkBlock pattern)
# ---------------------------------------------------------------------------

class _AutoTextView(QTextEdit):
    heightChanged = Signal(int)

    def __init__(self):
        super().__init__()
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
        self.setStyleSheet("background: transparent; border: none; padding: 0; margin: 0;")
        self._last_h = 18
        self.setFixedHeight(18)
        self._pending = False
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(0)
        self._update_timer.timeout.connect(self._do_update_height)

    def _schedule_update(self):
        if not self._pending:
            self._pending = True
            if not self._update_timer.isActive():
                self._update_timer.start()

    def _do_update_height(self):
        self._pending = False
        try:
            self._update_height()
        except RuntimeError:
            return

    def _update_height(self):
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_update()


# ---------------------------------------------------------------------------
# Compact base — single header line + collapsible body
# ---------------------------------------------------------------------------

class _CompactBubble(QFrame):
    sig_delete = Signal(int)
    sig_edit = Signal(int)
    sig_regen = Signal(int)
    sig_height_changed = Signal()
    sig_expand_in_companion = Signal(str, object)

    def __init__(self, index: int, role: str, timestamp: str, parent=None):
        super().__init__(parent)
        self._index = index
        self._role = role
        self._expanded = False
        self._anim: QPropertyAnimation | None = None
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setProperty("class", "MessageWidget")
        self.setProperty("role", role)
        self.setStyleSheet("QFrame { background: transparent; border: none; }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 1, 6, 1)
        outer.setSpacing(0)

        # ── header row ─────────────────────────────────────────
        self._header = QWidget()
        hdr = QHBoxLayout(self._header)
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(6)

        self._btn = QPushButton("\u25b6")
        self._btn.setFixedSize(16, 16)
        self._btn.setCursor(Qt.PointingHandCursor)
        self._btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            f"color: {_s.FG_DIM}; font-size: 8px; padding: 0; }}"
            f"QPushButton:hover {{ color: {_s.ACCENT_PRIMARY}; }}"
        )
        self._btn.clicked.connect(self._toggle)
        hdr.addWidget(self._btn)

        self._lbl = QLabel("")
        self._lbl.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; "
            f"font-family: Consolas; background: transparent;"
        )
        hdr.addWidget(self._lbl, 1)

        ts_lbl = QLabel(_pretty_ts(timestamp))
        ts_lbl.setStyleSheet(f"color: {_s.FG_INFO}; font-size: 8px; font-family: Consolas;")
        hdr.addWidget(ts_lbl)

        btn_del = QPushButton("x")
        btn_del.setFixedSize(16, 16)
        btn_del.setCursor(Qt.PointingHandCursor)
        btn_del.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; padding: 0; }}"
            f"QPushButton:hover {{ color: {_s.FG_ERROR}; }}"
        )
        btn_del.clicked.connect(lambda: self.sig_delete.emit(self._index))
        hdr.addWidget(btn_del)

        outer.addWidget(self._header)

        # ── body (collapsed by default) ────────────────────────
        self._body = QFrame()
        self._body.setVisible(False)
        self._body.setMaximumHeight(0)
        self._body.setStyleSheet(
            f"background: transparent; "
            f"border-left: 1px solid {_s.BORDER_SUBTLE}; margin-left: 8px;"
        )
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 4, 0, 4)
        self._body_layout.setSpacing(2)

        self._text = _AutoTextView()
        self._text.setStyleSheet(
            "background: transparent; border: none; "
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas;"
        )
        self._text.heightChanged.connect(lambda _: self.sig_height_changed.emit())
        self._body_layout.addWidget(self._text)
        outer.addWidget(self._body)

    # ── expand / collapse ──────────────────────────────────────

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._btn.setText("\u25bc" if self._expanded else "\u25b6")
        self._set_body_visible(self._expanded)
        self.sig_height_changed.emit()

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
            def _finish_open():
                self._body.setMaximumHeight(16777215)
                self._anim = None
                self.sig_height_changed.emit()
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
                self.sig_height_changed.emit()
            self._anim.finished.connect(_finish_close)
            self._anim.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sig_height_changed.emit()

    def sizeHint(self):
        h = self._header.sizeHint().height() + 2
        if self._body.isVisible():
            bh = self._body.sizeHint().height()
            mx = self._body.maximumHeight()
            if mx > 0:
                bh = min(bh, mx)
            h += bh
        return QSize(self.width() if self.width() > 0 else 200, h)

    def get_selectable_text_views(self) -> list:
        """Expose the body text view to ChatSelectionManager only when the
        bubble is visibly expanded — collapsed bubbles don't contribute
        text to a cross-widget drag-select.
        """
        if not self._expanded:
            return []
        if not self._body.isVisible() or self._body.maximumHeight() == 0:
            return []
        return [self._text]


# ---------------------------------------------------------------------------
# ToolCallBubble — compact
# ---------------------------------------------------------------------------

class ToolCallBubble(_CompactBubble):
    sig_approved = Signal(dict)
    sig_rejected = Signal(dict)

    def __init__(self, index: int, payload_text: str, timestamp: str, parent=None):
        super().__init__(index, "tool_call", timestamp, parent=parent)
        self._payload_text = payload_text
        self._payload = self._parse_payload(payload_text)

        summary = _call_summary(self._payload)
        self._lbl.setText(summary)
        self._lbl.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 9px; "
            f"font-family: Consolas; font-weight: bold; background: transparent;"
        )

        # body: pretty-printed JSON
        try:
            pretty = json.dumps(json.loads(payload_text), indent=2)
        except Exception:
            pretty = payload_text
        self._text.setPlainText(pretty)
        self._text._update_height()

    def _parse_payload(self, payload_text: str) -> dict:
        try:
            payload = json.loads(payload_text)
        except Exception:
            return {"tool": "tool", "raw": payload_text}
        return payload if isinstance(payload, dict) else {"tool": "tool", "raw": payload}


# ---------------------------------------------------------------------------
# ToolResultBubble — compact
# ---------------------------------------------------------------------------

class ToolResultBubble(_CompactBubble):
    def __init__(self, index: int, payload_text: str, timestamp: str, parent=None):
        super().__init__(index, "tool_result", timestamp, parent=parent)
        self._payload_text = payload_text
        self._payload = self._parse_payload(payload_text)

        summary = _result_summary(self._payload)
        self._lbl.setText(summary)

        # body: full result text
        _, body = _strip_call_prefix(str(self._payload.get("result", "")))
        body = body.strip() or "(empty result)"
        self._text.setPlainText(body)
        self._text._update_height()

        # companion action (e.g. "FULL FILE" for read_file)
        companion = self._companion_label()
        if companion:
            btn = QPushButton(companion)
            btn.setFixedHeight(20)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton {{ background: transparent; border: 1px solid {_s.BORDER_SUBTLE}; "
                f"border-radius: 3px; color: {_s.FG_DIM}; font-size: 9px; "
                f"font-family: Consolas; padding: 0 6px; }}"
                f"QPushButton:hover {{ color: {_s.ACCENT_PRIMARY}; border-color: {_s.ACCENT_PRIMARY}; }}"
            )
            btn.clicked.connect(
                lambda: self.sig_expand_in_companion.emit(
                    canonical_tool_name(self._payload.get("tool", "")) or str(self._payload.get("tool", "tool")),
                    self._payload,
                )
            )
            self._body_layout.addWidget(btn, alignment=Qt.AlignLeft)

        # generate_image artifact rendering: when sig_artifact_ready has fired
        # and the chat session has mutated this message to include image_paths,
        # build a thumb strip inside the bubble body. The body stays collapsed
        # by default, but once expanded the strip is visible alongside the text.
        # Auto-expand on first render if we already have images so the user
        # doesn't have to click to see what came out.
        tool_name = canonical_tool_name(self._payload.get("tool", "")) or ""
        if tool_name == "generate_image":
            image_paths = list(self._payload.get("image_paths") or [])
            if image_paths:
                self._build_image_strip(image_paths)
                # Auto-expand the body so the strip is visible. Deferred via
                # QTimer so the toggle fires AFTER the widget is attached to a
                # QListWidgetItem and has a viewport width — otherwise
                # sig_height_changed emits to nobody and the strip stays
                # collapsed until the user manually clicks the disclosure arrow.
                if not self._expanded:
                    QTimer.singleShot(0, self._toggle)

    def _build_image_strip(self, image_paths: list[str]) -> None:
        """Render N thumbnails side-by-side inside the bubble body. Click a
        thumb to open the full-resolution PNG in the OS image viewer. v2
        will add 'Pin to VISION' once SDModule.load_artifact() exists."""
        strip = QWidget()
        strip_layout = QHBoxLayout(strip)
        strip_layout.setContentsMargins(0, 6, 0, 2)
        strip_layout.setSpacing(6)
        for path_str in image_paths:
            p = Path(path_str)
            if not p.exists():
                continue
            pix = QPixmap(str(p))
            if pix.isNull():
                continue
            thumb_lbl = QLabel()
            thumb_lbl.setCursor(Qt.PointingHandCursor)
            thumb_lbl.setFixedSize(_THUMB_PX, _THUMB_PX)
            thumb_lbl.setAlignment(Qt.AlignCenter)
            thumb_lbl.setStyleSheet(
                f"QLabel {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_SUBTLE}; "
                f"border-radius: 4px; }}"
                f"QLabel:hover {{ border-color: {_s.ACCENT_PRIMARY}; }}"
            )
            scaled = pix.scaled(
                _THUMB_PX - 4, _THUMB_PX - 4,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            thumb_lbl.setPixmap(scaled)
            thumb_lbl.setToolTip(str(p))
            # Mouse-release on the QLabel triggers viewer-open.
            thumb_lbl.mousePressEvent = (
                lambda _ev, target=str(p): QDesktopServices.openUrl(QUrl.fromLocalFile(target))
            )
            strip_layout.addWidget(thumb_lbl)
        strip_layout.addStretch()
        self._body_layout.addWidget(strip)

    def _parse_payload(self, payload_text: str) -> dict:
        try:
            payload = json.loads(payload_text)
        except Exception:
            return {"tool": "tool", "result": payload_text}
        if not isinstance(payload, dict):
            return {"tool": "tool", "result": payload}
        call = payload.get("call")
        call_id = None
        if isinstance(call, dict):
            call_id = call.get("id")
        if call_id and "call_id" not in payload:
            payload["call_id"] = call_id
        return payload

    def _companion_label(self) -> str:
        tool = canonical_tool_name(self._payload.get("tool", "")) or str(self._payload.get("tool", "tool"))
        if tool == "read_file":
            return "FULL FILE"
        if tool == "search_history":
            return "SHOW ALL"
        if tool in ("list_files", "grep"):
            return "OPEN FILES"
        return ""


# ---------------------------------------------------------------------------
# Animated indicators (ported from Monolith)
# ---------------------------------------------------------------------------

class _PulsingDot(QWidget):
    """Soft pulsating accent dot used in ToolGroupCard while tools execute."""

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._opacity = 1.0
        self.setFixedSize(14, 14)
        self._pulse = QVariantAnimation(self)
        self._pulse.setStartValue(0.35)
        self._pulse.setKeyValueAt(0.5, 1.0)
        self._pulse.setEndValue(0.35)
        self._pulse.setDuration(1200)
        self._pulse.setLoopCount(-1)
        self._pulse.setEasingCurve(QEasingCurve.InOutSine)
        self._pulse.valueChanged.connect(self._on_pulse)
        self._fade: QVariantAnimation | None = None

    def _on_pulse(self, value):
        try:
            self._opacity = float(value)
        except (TypeError, ValueError):
            return
        self.update()

    def start(self):
        if self._fade is not None:
            self._fade.stop()
            self._fade = None
        self.setVisible(True)
        self._opacity = 1.0
        self._pulse.start()

    def stop(self):
        self._pulse.stop()

    def fade_out(self, duration_ms: int = 300):
        self._pulse.stop()
        if self._fade is not None:
            self._fade.stop()
        anim = QVariantAnimation(self)
        anim.setStartValue(float(self._opacity))
        anim.setEndValue(0.0)
        anim.setDuration(duration_ms)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.valueChanged.connect(self._on_pulse)

        def _done():
            self.setVisible(False)
            self._opacity = 1.0
            self._fade = None

        anim.finished.connect(_done)
        self._fade = anim
        anim.start()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setOpacity(self._opacity)
        diameter = 10
        x = (self.width() - diameter) // 2
        y = (self.height() - diameter) // 2
        p.setBrush(self._color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(x, y, diameter, diameter)
        p.end()


class ToolGroupCard(QWidget):
    """Persistent collapsible card that groups consecutive tool_call /
    tool_result messages into one UI element. Replaces the per-message
    ToolCallBubble + ToolResultBubble pair with a single card showing a
    running count and pulsing dot while tools execute.

    Each entry in the expanded body is click-to-open in the companion pane.
    """

    sig_height_changed = Signal()
    sig_expand_in_companion = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._index = -1
        self._row_hidden = False
        self._expanded = False
        self._animating = False
        self._entries: list[dict] = []
        self._latest_tool_name: str | None = None
        self.setObjectName("tool_group_card")

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
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

        self._anim = _PulsingDot(_s.ACCENT_PRIMARY)
        self._anim.hide()
        header_layout.addWidget(self._anim)

        self._label = QLabel("")
        self._label.setStyleSheet(
            f"color: {_s.FG_ACCENT}; font-size: 11px; font-family: Consolas, monospace;"
            f" font-weight: bold; background: transparent;"
        )
        self._label.setMinimumWidth(0)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        header_layout.addWidget(self._label, 1)

        self._chevron = QLabel("☰")
        self._chevron.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; background: transparent;"
        )
        header_layout.addWidget(self._chevron)

        card_layout.addWidget(header)
        self._header_widget = header
        header.mousePressEvent = lambda _e: self._toggle_expand()

        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(30, 0, 10, 4)
        body_layout.setSpacing(1)
        self._body.hide()
        self._body_layout = body_layout
        card_layout.addWidget(self._body)

        root.addWidget(self._card, 1)

    def _last_call_summary(self) -> str:
        for entry in reversed(self._entries):
            if entry["kind"] == "call":
                s = entry.get("summary", "")
                return s[:48] + "…" if len(s) > 48 else s
        return ""

    def add_call(self, summary: str, payload: dict | None = None) -> None:
        payload = payload or {}
        self._entries.append({"kind": "call", "summary": summary, "payload": payload})
        tool_name = canonical_tool_name(payload.get("tool", "")) or str(payload.get("tool", "") or "")
        if tool_name:
            self._latest_tool_name = tool_name
        lbl = self._make_entry_label(
            f"▸  {summary}", _s.FG_TEXT, kind="call", payload=payload
        )
        self._body_layout.addWidget(lbl)
        self._add_preview("call", payload)
        self._update_header()

    def add_result(self, summary: str, payload: dict | None = None) -> None:
        self._entries.append({"kind": "result", "summary": summary, "payload": payload or {}})
        lbl = self._make_entry_label(
            f"◂  {summary}", _s.FG_DIM, kind="result", payload=payload or {}
        )
        self._body_layout.addWidget(lbl)
        self._add_preview("result", payload or {})
        self._update_header()

    def _make_entry_label(self, text: str, color: str, *, kind: str, payload: dict) -> QLabel:
        # Wrap (don't hard-cut at 72) so the full line — including the complete
        # path — is readable when the card is expanded.
        display = text if len(text) <= 400 else text[:397] + "…"
        lbl = QLabel(display)
        lbl.setToolTip(text)
        lbl.setWordWrap(True)
        lbl.setMinimumWidth(0)
        tool_name = canonical_tool_name(payload.get("tool", "")) or str(payload.get("tool", ""))
        clickable = bool(tool_name) and not isinstance(payload.get("calls"), list)
        if clickable:
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.setStyleSheet(
                f"QLabel {{ color: {color}; font-size: 10px; font-family: Consolas, monospace;"
                f" background: transparent; }}"
                f"QLabel:hover {{ color: {_s.ACCENT_PRIMARY}; }}"
            )
            if kind == "call":
                expand_payload = {"tool": tool_name, "call": payload, "result": ""}
            else:
                expand_payload = payload
            lbl.mousePressEvent = (
                lambda _e, n=tool_name, p=expand_payload: self.sig_expand_in_companion.emit(n, p)
            )
        else:
            lbl.setStyleSheet(
                f"color: {color}; font-size: 10px; font-family: Consolas, monospace;"
                f" background: transparent;"
            )
        return lbl

    def _add_preview(self, kind: str, payload: dict) -> None:
        """Add a compact (~3-line) content preview under an entry, with a
        'show more' toggle when the content is longer. Lives in the expandable
        body, so it only shows when the card is expanded. Shows 'what it wrote'
        (call content) / 'what it found' (read-ish result body)."""
        src = _preview_source(kind, payload)
        if not src:
            return
        clipped, has_more = _clip_preview(src)
        if not clipped:
            return
        prev = QLabel(clipped)
        prev.setWordWrap(True)
        prev.setMinimumWidth(0)
        prev.setTextInteractionFlags(Qt.TextSelectableByMouse)
        prev.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas, monospace;"
            f" background: {_s.BG_INPUT}; border-left: 1px solid {_s.BORDER_SUBTLE};"
            " padding: 3px 6px; margin-left: 10px;"
        )
        self._body_layout.addWidget(prev)
        if not has_more:
            return
        toggle = QLabel("▸ show more")
        toggle.setCursor(Qt.PointingHandCursor)
        toggle.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 9px; font-family: Consolas, monospace;"
            " background: transparent; margin-left: 12px;"
        )
        full = src if len(src) <= 4000 else src[:4000] + "\n…(truncated — click the row to open the full content)"
        state = {"open": False}

        def _toggle(_e=None):
            state["open"] = not state["open"]
            prev.setText(full if state["open"] else clipped)
            toggle.setText("▾ show less" if state["open"] else "▸ show more")
            self.sig_height_changed.emit()

        toggle.mousePressEvent = _toggle
        self._body_layout.addWidget(toggle)

    def set_animating(self, animating: bool) -> None:
        was_animating = self._animating
        self._animating = animating
        if animating:
            if not was_animating:
                self._anim.setVisible(True)
                self._anim.start()
            self._update_header()
        else:
            if was_animating:
                self._anim.fade_out()
            self._update_header()

    def _running_text(self) -> str:
        n_calls = sum(1 for e in self._entries if e["kind"] == "call")
        n_results = sum(1 for e in self._entries if e["kind"] == "result")
        latest = self._latest_tool_name
        if not self._animating:
            completed = n_results or n_calls or len(self._entries)
            if completed <= 0:
                return "Tools"
            last_call = self._last_call_summary()
            if n_calls == 1 and latest:
                return f"Tool · {latest} complete"
            if last_call:
                return f"Tools · {completed} complete · {last_call}"
            return f"Tools · {completed} complete"
        suffix = "..."
        if n_calls == 0:
            return "preparing" + suffix
        if n_calls == 1:
            return f"running {latest}{suffix}" if latest else f"running tool{suffix}"
        if latest:
            return f"{n_calls} tools · {latest}{suffix}"
        return f"running {n_calls} tools{suffix}"

    def _update_header(self) -> None:
        self._label.setText(self._running_text())
        self._label.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 11px; font-family: Consolas, monospace;"
            f" font-weight: bold; background: transparent;"
        )

    def _toggle_expand(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._chevron.setText("☷" if self._expanded else "☰")
        self.sig_height_changed.emit()

    def sizeHint(self):
        w = self.width() if self.width() > 0 else 200
        h = 30
        if self._expanded and self._body.isVisible():
            m = self._body_layout.contentsMargins()
            root = self.layout()
            root_m = root.contentsMargins() if root is not None else None
            side = (root_m.left() + root_m.right()) if root_m is not None else 8
            body_content_w = max(w - side - 3 - m.left() - m.right(), 40)
            body_h = m.top() + m.bottom()
            count = self._body_layout.count()
            entries = 0
            for i in range(count):
                item = self._body_layout.itemAt(i)
                if item is None:
                    continue
                widget = item.widget()
                if widget is None:
                    continue
                if widget.hasHeightForWidth():
                    body_h += widget.heightForWidth(body_content_w)
                else:
                    body_h += widget.sizeHint().height()
                entries += 1
            if entries > 1:
                body_h += self._body_layout.spacing() * (entries - 1)
            h += body_h
        return QSize(w, h)

    def get_selectable_text_views(self) -> list:
        """ToolGroupCard uses QLabels (not QTextEdits) for its body entries,
        so it doesn't contribute to cross-widget drag-select. Return empty
        for ChatSelectionManager compatibility.
        """
        return []


# ---------------------------------------------------------------------------
# BearingUpdateCard — modern single-envelope card for Bearing V0
# ---------------------------------------------------------------------------


class BearingUpdateCard(QWidget):
    """Collapsible card surfaced when the model emits a <bearing_update>
    envelope. Mirrors ToolGroupCard's aesthetic (accent strip, pulsing dot
    header, chevron-toggled body) but is single-envelope, NOT a tool-call
    group accumulator.

    Lifecycle:
      - open: pulsing dot, label "Bearing · updating"
      - streaming: envelope text accumulates in body
      - close: pulse fades, label flips to "Bearing · updated"

    Body shows the pretty-printed JSON envelope on expand. SUBSTRATE badge
    distinguishes from tool-call cards.
    """

    sig_height_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._streaming = True
        self._envelope_text = ""
        self._expanded = False
        self.setObjectName("bearing_update_card")

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
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

        # Pulsing dot — animation runs while envelope is streaming.
        self._anim = _PulsingDot(_s.ACCENT_PRIMARY)
        self._anim.start()
        header_layout.addWidget(self._anim)

        # Header label.
        self._label = QLabel("Bearing · updating")
        self._label.setStyleSheet(
            f"color: {_s.FG_ACCENT}; font-size: 11px; font-family: Consolas, monospace;"
            f" font-weight: bold; background: transparent;"
        )
        header_layout.addWidget(self._label, 1)

        # SUBSTRATE badge — visual distinguisher from tool-call cards.
        self._badge = QLabel("SUBSTRATE")
        self._badge.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas, monospace;"
            f" font-weight: bold; background: transparent;"
        )
        header_layout.addWidget(self._badge)

        # Expand/collapse chevron.
        self._chevron = QLabel("☰")
        self._chevron.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; background: transparent;"
        )
        header_layout.addWidget(self._chevron)
        card_layout.addWidget(header)
        header.mousePressEvent = lambda _e: self._toggle_expand()

        # Body — JSON envelope, hidden until expand.
        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(30, 0, 10, 4)
        body_layout.setSpacing(1)
        self._body_text = QLabel("")
        self._body_text.setWordWrap(True)
        self._body_text.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self._body_text.setStyleSheet(
            f"color: {_s.FG_SECONDARY}; font-size: 10px; font-family: Consolas, monospace;"
            f" background: transparent;"
        )
        body_layout.addWidget(self._body_text)
        self._body.hide()
        card_layout.addWidget(self._body)

        root.addWidget(self._card, 1)

    def append_content(self, text: str) -> None:
        """Stream envelope text into the body (kept hidden until expand)."""
        if not text:
            return
        self._envelope_text += text
        # Pretty-print if it parses; otherwise show raw.
        try:
            parsed = json.loads(self._envelope_text)
            pretty = json.dumps(parsed, indent=2)
        except Exception:
            pretty = self._envelope_text
        self._body_text.setText(pretty)
        if self._expanded:
            self.sig_height_changed.emit()

    def close_block(self) -> None:
        """Mark envelope complete. Pulse stops, label flips to 'updated'."""
        self._streaming = False
        self._anim.fade_out()
        self._label.setText("Bearing · updated")
        # Settle the accent to the success-tinted color (matches ToolGroupCard
        # done-state).
        self._accent.setStyleSheet(
            f"background: {_s.FG_ACCENT}; border: none;"
            " border-top-left-radius: 4px; border-bottom-left-radius: 4px;"
        )

    def _toggle_expand(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._chevron.setText("☷" if self._expanded else "☰")
        # Mark our own size hint dirty AND invalidate the parent layout chain.
        # This card lives nested inside MessageWidget._skill_container, whose
        # sizeHint() is cached by its QVBoxLayout. A plain visibility toggle
        # does NOT bubble an invalidation up to that layout on every platform
        # (Windows honors the cache; offscreen re-activates regardless), so the
        # host's height recompute reads a STALE container hint, total never
        # changes, and the collapse is swallowed — leaving the row stuck at the
        # expanded height until an unrelated width change forces the refresh.
        # updateGeometry() replicates what un-maximizing the window does for
        # free: it notifies the parent layout that this widget's hint changed.
        self.updateGeometry()
        parent = self.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().invalidate()
        self.sig_height_changed.emit()

    def sizeHint(self):
        w = self.width() if self.width() > 0 else 200
        h = 30
        if self._expanded and self._body.isVisible():
            try:
                body_sh = self._body_text.sizeHint()
                h += body_sh.height() + 8
            except Exception:
                h += 60
        return QSize(w, h)

    def get_selectable_text_views(self) -> list:
        # Body is a QLabel — cross-widget drag-select not supported here.
        return []
