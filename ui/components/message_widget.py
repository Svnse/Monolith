import html
import re

from PySide6.QtCore import Qt, QSize, Signal, QTimer
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget, QTextEdit,
)
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QTextOption

import core.style as _s


class _AutoTextView(QTextEdit):
    """Read-only text display that auto-sizes to its content height."""
    heightChanged = Signal(int)

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.setCursor(Qt.IBeamCursor)
        self.document().setDocumentMargin(0)
        self.document().contentsChanged.connect(self._schedule_update)
        self.setStyleSheet("""
            QTextEdit {
                background: transparent; border: none; padding: 0px; margin: 0px;
                selection-background-color: rgba(255, 255, 255, 0.15);
                selection-color: inherit;
            }
        """)
        font = QFont("Consolas, Segoe UI Emoji, Noto Color Emoji, Apple Color Emoji", 12)
        self.setFont(font)
        self._last_h = 18
        self.setFixedHeight(18)
        self._pending = False

    def _schedule_update(self):
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._do_update_height)

    def _do_update_height(self):
        self._pending = False
        self._update_height()

    def _update_height(self):
        vw = self.viewport().width()
        if vw < 10:
            vw = self.width() - 4
        if vw < 10:
            # Widget not laid out yet (hidden or newly created).
            # Skip — resizeEvent will re-run this once it has real dimensions.
            return
        doc = self.document()
        doc.setTextWidth(max(vw, 20))
        h = int(doc.size().height()) + 6
        h = max(h, 18)
        if h != self._last_h:
            self._last_h = h
            self.setFixedHeight(h)
            self.heightChanged.emit(h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_height()

    def ideal_height(self, available_width: int) -> int:
        doc = self.document().clone()
        doc.setDocumentMargin(0)
        doc.setTextWidth(max(available_width, 20))
        h = int(doc.size().height()) + 6
        return max(h, 18)


class _IconAction(QPushButton):
    """Tiny icon-only action button for message hover bar."""

    def __init__(self, icon_char: str, tooltip: str):
        super().__init__(icon_char)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(24, 24)
        self.setProperty("class", "msg_icon_action")


class _InlineTraceEntry(QFrame):
    """Compact inline execution trace row with expandable details and pulse state."""

    heightChanged = Signal()

    def __init__(self, summary: str, detail: str = "", parent=None):
        super().__init__(parent)
        self._expanded = False
        self._active = False
        self._pulse_on = False
        self._summary = summary or "trace"
        self._done_state: str = "running"  # "running" | "done" | "failed"

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("inline_trace_entry")
        self._apply_frame_style("running")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 5, 8, 5)
        root.setSpacing(3)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        self.btn_toggle = QPushButton("▸")
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.setFixedSize(18, 18)
        self.btn_toggle.clicked.connect(self._toggle)

        self.lbl_summary = QLabel(self._summary)
        self.lbl_summary.setWordWrap(True)
        self._apply_summary_style(active=False)

        head.addWidget(self.btn_toggle, 0, Qt.AlignVCenter)
        head.addWidget(self.lbl_summary, 1)
        root.addLayout(head)

        self._detail_wrap = QWidget()
        self._detail_wrap.setVisible(False)
        detail_layout = QVBoxLayout(self._detail_wrap)
        detail_layout.setContentsMargins(20, 0, 0, 0)
        detail_layout.setSpacing(0)

        self.detail = _AutoTextView()
        self.detail.setStyleSheet(
            f"QTextEdit {{"
            f"  background: transparent; border: none;"
            f"  border-left: 2px solid {_s.ACCENT_PRIMARY};"
            f"  color: {_s.FG_SECONDARY}; padding: 4px 8px;"
            f"  font-size: 10px; font-family: Consolas;"
            f"  selection-background-color: rgba(255, 255, 255, 0.15);"
            f"  selection-color: inherit;"
            f"}}"
        )
        self.detail.heightChanged.connect(lambda _h: self.heightChanged.emit())
        self.detail.setPlainText(detail or "")
        detail_layout.addWidget(self.detail)
        root.addWidget(self._detail_wrap)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(420)
        self._pulse_timer.timeout.connect(self._on_pulse)

        self._apply_arrow_style(active=False)

    def _apply_frame_style(self, state: str):
        """Update frame border to reflect state: running | done | failed."""
        if state == "failed":
            left_color = _s.FG_ERROR
        else:
            left_color = _s.BORDER_LIGHT
        self.setStyleSheet(
            f"QFrame#inline_trace_entry {{"
            f"  background: transparent;"
            f"  border: none;"
            f"  border-left: 2px solid {left_color};"
            f"}}"
        )

    def _toggle(self):
        self._expanded = not self._expanded
        self._detail_wrap.setVisible(self._expanded)
        self.btn_toggle.setText("▾" if self._expanded else "▸")
        self.heightChanged.emit()

    def _apply_arrow_style(self, *, active: bool):
        if active:
            color = self._rgba(_s.ACCENT_PRIMARY, 1.0 if self._pulse_on else 0.62)
        else:
            color = _s.FG_DIM
        self.btn_toggle.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent; border: none;"
            f"  color: {color}; font-family: Consolas; font-size: 11px; padding: 0px;"
            f"}}"
            f"QPushButton:hover {{ color: {_s.ACCENT_PRIMARY}; }}"
        )

    @staticmethod
    def _rgba(color_hex: str, alpha: float) -> str:
        c = QColor(color_hex if isinstance(color_hex, str) else "#ffffff")
        a = max(0.0, min(1.0, float(alpha)))
        return f"rgba({c.red()}, {c.green()}, {c.blue()}, {a:.3f})"

    def _apply_summary_style(self, *, active: bool):
        if active:
            color = _s.FG_TEXT if self._pulse_on else _s.FG_SECONDARY
        else:
            color = _s.FG_TEXT
        self.lbl_summary.setStyleSheet(
            f"color: {color}; font-family: Consolas; font-size: 11px; background: transparent;"
        )

    def _on_pulse(self):
        self._pulse_on = not self._pulse_on
        self._apply_arrow_style(active=self._active)
        self._apply_summary_style(active=self._active)

    def set_summary(self, summary: str):
        self._summary = summary or "trace"
        self.lbl_summary.setText(self._summary)
        self.heightChanged.emit()

    def append_detail(self, text: str):
        if not text:
            return
        current = self.detail.toPlainText().rstrip()
        updated = f"{current}\n{text}".strip() if current else text.strip()
        self.detail.setPlainText(updated)
        self.heightChanged.emit()

    def set_status(self, status: str):
        norm = str(status or "").strip().lower()
        if norm in {"running", "pending", "active"}:
            self._done_state = "running"
            self._apply_frame_style("running")
            self.set_active(True)
            return
        if norm in {"ok", "done", "success"}:
            self._done_state = "done"
            self._apply_frame_style("done")
            self.set_active(False)
            return
        self._done_state = "failed"
        self._apply_frame_style("failed")
        self.set_active(False)

    def set_active(self, active: bool):
        if self._active == bool(active):
            return
        self._active = bool(active)
        if self._active:
            self._pulse_on = True
            self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
            self._pulse_on = False
        self._apply_arrow_style(active=self._active)
        self._apply_summary_style(active=self._active)
        self.heightChanged.emit()


class MessageWidget(QFrame):
    sig_delete = Signal(int)
    sig_edit = Signal(int)
    sig_regen = Signal(int)
    sig_height_changed = Signal()  # emitted when internal height changes

    _HEADER_H = 16
    _MARGINS = (8, 4, 8, 4)  # left, top, right, bottom
    _SPACING = 1

    @staticmethod
    def _rgba(color_hex: str, alpha: float) -> str:
        c = QColor(color_hex if isinstance(color_hex, str) else "#ffffff")
        a = max(0.0, min(1.0, float(alpha)))
        return f"rgba({c.red()}, {c.green()}, {c.blue()}, {a:.3f})"

    def __init__(self, index: int, role: str, text: str, timestamp: str):
        super().__init__()
        self._index = index
        self._role = role
        self._content = text or ""
        self._supports_rich = role in ("assistant", "system")
        self._loading = False
        self._loading_step = 0
        self._trace_entries: dict[int, _InlineTraceEntry] = {}
        self._trace_entry_states: dict[int, str] = {}
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(260)
        self._loading_timer.timeout.connect(self._on_loading_tick)

        self.setAttribute(Qt.WA_Hover, True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("class", "MessageWidget")
        self.setProperty("role", role)
        self.setStyleSheet(
            f"MessageWidget {{ border-left: 2px solid {_s.BORDER_SUBTLE}; }}"
            f"MessageWidget QLabel {{ border: none; }}"
            f"MessageWidget QWidget {{ border: none; }}"
        )

        is_assistant = role == "assistant"
        is_system = role == "system"

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(*self._MARGINS)
        self._root.setSpacing(self._SPACING)

        head = QHBoxLayout()
        head.setSpacing(6)

        self.lbl_role = QLabel((role or "").upper())
        self.lbl_role.setObjectName("msg_role")
        self.lbl_role.setProperty("role", role)
        self.lbl_role.setFixedHeight(self._HEADER_H)
        head.addWidget(self.lbl_role)

        pretty_ts = (timestamp or "")
        if "T" in pretty_ts and len(pretty_ts) >= 16:
            pretty_ts = pretty_ts[11:16]
        self.lbl_time = QLabel(pretty_ts)
        self.lbl_time.setObjectName("msg_time")
        self.lbl_time.setFixedHeight(self._HEADER_H)
        head.addWidget(self.lbl_time)
        head.addStretch()

        self.actions = QWidget()
        self.actions.setObjectName("msg_actions")
        actions_layout = QHBoxLayout(self.actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)

        if not is_system:
            if role == "user":
                self.btn_edit = _IconAction("\u270e", "Edit")
                self.btn_edit.clicked.connect(lambda: self.sig_edit.emit(self._index))
                actions_layout.addWidget(self.btn_edit)

            if is_assistant:
                self.btn_regen = _IconAction("\u27f2", "Regenerate")
                self.btn_regen.clicked.connect(lambda: self.sig_regen.emit(self._index))
                actions_layout.addWidget(self.btn_regen)

            self.btn_delete = _IconAction("\u2715", "Delete")
            self.btn_delete.clicked.connect(lambda: self.sig_delete.emit(self._index))
            actions_layout.addWidget(self.btn_delete)

        self.actions.setVisible(False)
        head.addWidget(self.actions)
        self._root.addLayout(head)

        # Single flow area — narration blocks and tool rows interleave in order
        self._flow_wrap: QWidget | None = None
        self._flow_layout_inner: QVBoxLayout | None = None
        self._active_narration_block: _AutoTextView | None = None
        if is_assistant:
            self._flow_wrap = QWidget()
            self._flow_wrap.setVisible(False)
            self._flow_layout_inner = QVBoxLayout(self._flow_wrap)
            self._flow_layout_inner.setContentsMargins(0, 2, 0, 4)
            self._flow_layout_inner.setSpacing(4)
            self._root.addWidget(self._flow_wrap)

        self.text_view = _AutoTextView()
        self.text_view.setObjectName("msg_content")
        self.text_view.setProperty("role", role)
        self._render_content(force_rich=(role == "system"))
        self.text_view.heightChanged.connect(self._on_text_height_changed)
        self._root.addWidget(self.text_view)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._on_text_height_changed(self.text_view.height())

    def _on_text_height_changed(self, text_h):
        self._recompute_height()

    def _recompute_height(self):
        hint_h = int(self.layout().sizeHint().height()) if self.layout() is not None else 28
        self.setFixedHeight(max(hint_h + 2, 28))
        self.sig_height_changed.emit()

    def sizeHint(self):
        w = self.width() if self.width() > 50 else 600
        if self.layout() is None:
            return QSize(w, 28)
        return QSize(w, max(int(self.layout().sizeHint().height()) + 2, 28))

    def enterEvent(self, event):
        self.actions.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.actions.setVisible(False)
        super().leaveEvent(event)

    def append_token(self, token: str):
        if not token:
            return
        if self._loading:
            self.set_loading(False)
        self._content += token
        if self._role == "system" and self._supports_rich:
            self._render_content(force_rich=True)
            self._on_text_height_changed(self.text_view.height())
            return
        cursor = self.text_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)

    def finalize(self):
        self.set_loading(False)
        self._render_content(force_rich=self._supports_rich)
        self._recompute_height()

    def set_index(self, idx: int):
        self._index = idx

    def _on_loading_tick(self):
        self._loading_step = (self._loading_step + 1) % 3
        dots = "." * (self._loading_step + 1)
        self.text_view.setPlainText(dots)
        self._recompute_height()

    def set_loading(self, enabled: bool):
        if enabled:
            if self._loading:
                return
            self._loading = True
            self._loading_step = 0
            self.text_view.setPlainText(".")
            self._loading_timer.start()
            self._recompute_height()
            return

        if not self._loading:
            return
        self._loading = False
        self._loading_timer.stop()
        self._render_content(force_rich=False)
        self._recompute_height()

    def _render_content(self, force_rich: bool):
        if force_rich and self._supports_rich:
            self.text_view.setHtml(self._render_markdown_like(self._content))
            return
        self.text_view.setPlainText(self._content)

    def append_narration(self, text: str):
        if self._flow_layout_inner is None or self._flow_wrap is None:
            return
        raw = str(text or "")
        if not raw.strip():
            return
        if self._active_narration_block is None:
            nb = _AutoTextView()
            nb.setStyleSheet(
                f"QTextEdit {{"
                f"  background: transparent; border: none;"
                f"  color: {_s.FG_SECONDARY}; padding: 2px 4px 2px 4px;"
                f"  font-size: 12px;"
                f"  selection-background-color: rgba(255, 255, 255, 0.15);"
                f"  selection-color: inherit;"
                f"}}"
            )
            nb.heightChanged.connect(lambda _h: self._recompute_height())
            self._flow_layout_inner.addWidget(nb)
            self._active_narration_block = nb
            self._flow_wrap.setVisible(True)
        current = self._active_narration_block.toPlainText()
        self._active_narration_block.setPlainText(current + raw)
        self._recompute_height()

    def _refresh_trace_section(self):
        if self._flow_wrap is not None and self._flow_layout_inner is not None:
            self._flow_wrap.setVisible(self._flow_layout_inner.count() > 0)
        self._recompute_height()

    def upsert_trace_entry(self, trace_id: int, *, summary: str, detail: str = "", status: str = "running"):
        if self._flow_layout_inner is None or self._flow_wrap is None:
            return
        entry = self._trace_entries.get(int(trace_id))
        if entry is None:
            # Freeze current narration block — next narration will appear below this entry
            self._active_narration_block = None
            entry = _InlineTraceEntry(summary=summary, detail=detail)
            entry.heightChanged.connect(self._recompute_height)
            self._flow_layout_inner.addWidget(entry)
            self._flow_wrap.setVisible(True)
            self._trace_entries[int(trace_id)] = entry
        else:
            entry.set_summary(summary)
            if detail:
                entry.append_detail(detail)
        entry.set_status(status)
        self._trace_entry_states[int(trace_id)] = str(status or "").strip().lower()
        self._recompute_height()

    def append_trace_detail(self, trace_id: int, text: str):
        entry = self._trace_entries.get(int(trace_id))
        if entry is None:
            self.upsert_trace_entry(int(trace_id), summary=f"trace #{trace_id}", detail=text, status="running")
            return
        entry.append_detail(text)
        self._refresh_trace_section()

    def set_trace_status(self, trace_id: int, status: str):
        entry = self._trace_entries.get(int(trace_id))
        if entry is None:
            return
        entry.set_status(status)
        self._trace_entry_states[int(trace_id)] = str(status or "").strip().lower()
        self._refresh_trace_section()

    def _render_markdown_like(self, raw_text: str) -> str:
        text = (raw_text or "").replace("\r\n", "\n")
        code_pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

        blocks: list[str] = []
        cursor = 0
        for match in code_pattern.finditer(text):
            before = text[cursor:match.start()]
            if before:
                blocks.append(self._render_plain_segment(before))

            lang = (match.group(1) or "").strip() or "text"
            code = (match.group(2) or "").rstrip("\n")
            blocks.append(self._render_code_segment(code, lang))
            cursor = match.end()

        tail = text[cursor:]
        if tail:
            blocks.append(self._render_plain_segment(tail))

        if not blocks:
            blocks.append(self._render_plain_segment(text))

        return (
            "<div style='margin:0;padding:0;'>"
            + "".join(blocks)
            + "</div>"
        )

    def _render_plain_segment(self, segment: str) -> str:
        parts = re.split(r"(`[^`\n]+`)", segment)
        out: list[str] = []
        for part in parts:
            if not part:
                continue
            if part.startswith("`") and part.endswith("`") and len(part) >= 2:
                inline = html.escape(part[1:-1])
                out.append(
                    f"<code style='background:{_s.BG_INPUT}; border:1px solid {_s.BORDER_SUBTLE}; "
                    f"padding:1px 4px; border-radius:3px; font-family:Consolas; color:{_s.FG_TEXT};'>{inline}</code>"
                )
            else:
                out.append(html.escape(part).replace("\n", "<br>"))
        return f"<div style='margin:0 0 6px 0; line-height:1.35;'>{''.join(out)}</div>"

    def _render_code_segment(self, code: str, lang: str) -> str:
        code_escaped = html.escape(code)
        lang_escaped = html.escape(lang)
        return (
            f"<div style='margin:4px 0 8px 0;'>"
            f"<div style='font-size:9px; color:{_s.FG_DIM}; margin:0 0 2px 0; font-family:Consolas;'>{lang_escaped}</div>"
            f"<pre style='margin:0; padding:8px 10px; border-radius:6px; "
            f"background:{_s.BG_INPUT}; border:1px solid {_s.BORDER_SUBTLE}; "
            f"color:{_s.FG_TEXT}; font-family:Consolas; font-size:11px; white-space:pre-wrap;'>{code_escaped}</pre>"
            f"</div>"
        )
