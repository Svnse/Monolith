from __future__ import annotations

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    Qt,
    Signal,
    QSize,
    QTimer,
)
from PySide6.QtGui import QKeySequence, QTextCharFormat, QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import core.style as _s
from core.channel_tag import strip_channel_tags_for_display
from core.attached_blocks import split_attached
from ui.chat_selection import ChatSelectionManager
from ui.components.message_widget import MessageWidget
from ui.components.tool_bubbles import (
    ToolCallBubble,
    ToolGroupCard,
    ToolResultBubble,
    _call_summary,
    _result_summary,
)
import json as _json


class SmoothMessageList(QListWidget):
    """Chat list with bounded, animated wheel movement."""

    _NOTCH_PIXELS = 72.0
    _MAX_WHEEL_FRACTION = 0.32
    _MIN_WHEEL_CAP = 96.0
    _EDGE_EPSILON_PIXELS = 3
    _EDGE_JITTER_PIXELS = 8.0

    def __init__(self) -> None:
        super().__init__()
        self.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        bar = self.verticalScrollBar()
        bar.setSingleStep(18)
        self._smooth_scroll_target = bar.value()
        self._last_scroll_maximum = bar.maximum()
        bar.rangeChanged.connect(self._on_scroll_range_changed)
        self._scroll_anim = QPropertyAnimation(bar, b"value", self)
        self._scroll_anim.setEasingCurve(QEasingCurve.OutCubic)

    @classmethod
    def _wheel_delta_pixels(cls, angle_y: int, pixel_y: int, viewport_height: int) -> float:
        if pixel_y:
            raw = float(pixel_y)
        else:
            raw = (float(angle_y) / 120.0) * cls._NOTCH_PIXELS
        cap = max(cls._MIN_WHEEL_CAP, float(viewport_height) * cls._MAX_WHEEL_FRACTION)
        return max(-cap, min(cap, raw))

    @classmethod
    def _edge_target_for_delta(
        cls,
        value: int,
        minimum: int,
        maximum: int,
        delta: float,
        has_pixel_delta: bool,
    ) -> int | None:
        near_bottom = value >= maximum - cls._EDGE_EPSILON_PIXELS
        near_top = value <= minimum + cls._EDGE_EPSILON_PIXELS
        if near_bottom and (
            delta < 0
            or (has_pixel_delta and 0 < delta <= cls._EDGE_JITTER_PIXELS)
        ):
            return maximum
        if near_top and (
            delta > 0
            or (has_pixel_delta and -cls._EDGE_JITTER_PIXELS <= delta < 0)
        ):
            return minimum
        return None

    def _on_scroll_range_changed(self, _minimum: int, maximum: int) -> None:
        bar = self.verticalScrollBar()
        old_maximum = self._last_scroll_maximum
        was_at_bottom = bar.value() >= old_maximum - self._EDGE_EPSILON_PIXELS
        self._last_scroll_maximum = maximum
        if not was_at_bottom:
            self._sync_smooth_scroll_target()
            return

        if self._scroll_anim.state() == QAbstractAnimation.State.Running:
            self._scroll_anim.stop()
        bar.setValue(maximum)
        self._smooth_scroll_target = maximum

    def _sync_smooth_scroll_target(self) -> None:
        bar = self.verticalScrollBar()
        self._smooth_scroll_target = max(bar.minimum(), min(bar.maximum(), bar.value()))

    def scrollToBottom(self) -> None:  # noqa: N802 - Qt override
        if hasattr(self, "_scroll_anim"):
            self._scroll_anim.stop()
        super().scrollToBottom()
        self._sync_smooth_scroll_target()

    def wheelEvent(self, event) -> None:
        bar = self.verticalScrollBar()
        if bar.maximum() <= bar.minimum():
            event.accept()
            return

        pixel_y = event.pixelDelta().y()
        angle_y = event.angleDelta().y()
        delta = self._wheel_delta_pixels(angle_y, pixel_y, self.viewport().height())
        if not delta:
            super().wheelEvent(event)
            return

        edge_target = self._edge_target_for_delta(
            bar.value(),
            bar.minimum(),
            bar.maximum(),
            delta,
            bool(pixel_y),
        )
        if edge_target is not None:
            self._scroll_anim.stop()
            bar.setValue(edge_target)
            self._smooth_scroll_target = edge_target
            event.accept()
            return

        if self._scroll_anim.targetObject() is not bar:
            self._scroll_anim.stop()
            self._scroll_anim.setTargetObject(bar)

        if self._scroll_anim.state() != QAbstractAnimation.State.Running:
            self._smooth_scroll_target = bar.value()

        target = int(round(self._smooth_scroll_target - delta))
        target = max(bar.minimum(), min(bar.maximum(), target))
        self._smooth_scroll_target = target

        self._scroll_anim.stop()
        self._scroll_anim.setDuration(95 if pixel_y else 140)
        self._scroll_anim.setStartValue(bar.value())
        self._scroll_anim.setEndValue(target)
        self._scroll_anim.start()
        event.accept()


class ChatInput(QPlainTextEdit):
    """Plain-text chat composer without QLineEdit's 32k character ceiling."""

    returnPressed = Signal()
    contentChanged = Signal(str)

    _MIN_HEIGHT = 44
    _MAX_HEIGHT = 126

    def __init__(self) -> None:
        super().__init__()
        self._key_interceptor = None
        self.setAcceptDrops(True)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setTabChangesFocus(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(self._MIN_HEIGHT)
        self.setMaximumHeight(self._MAX_HEIGHT)
        self.setFixedHeight(self._MIN_HEIGHT)
        self.textChanged.connect(self._on_document_text_changed)

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: object) -> None:  # noqa: N802 - QLineEdit compatibility
        self.setPlainText(str(text or ""))
        self._restore_char_format()
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def clear(self) -> None:
        super().clear()
        self._restore_char_format()
        self.setFixedHeight(self._MIN_HEIGHT)

    def _restore_char_format(self) -> None:
        fmt = QTextCharFormat()
        fmt.clearForeground()
        self.setCurrentCharFormat(fmt)

    def _on_document_text_changed(self) -> None:
        if not self.toPlainText():
            self._restore_char_format()
        self._sync_height_to_document()
        self.contentChanged.emit(self.text())

    def _sync_height_to_document(self) -> None:
        margins = self.contentsMargins()
        frame_padding = margins.top() + margins.bottom()
        doc_height = int(self.document().size().height())
        target = max(self._MIN_HEIGHT, min(self._MAX_HEIGHT, doc_height + frame_padding + 18))
        if target != self.height():
            self.setFixedHeight(target)

    def event(self, event) -> bool:
        if event.type() == event.Type.KeyPress and self._key_interceptor is not None:
            if self._key_interceptor(event):
                return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            self.returnPressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        mime = event.mimeData()
        if not mime.hasUrls():
            super().dropEvent(event)
            return
        tray = getattr(self, "_blob_tray", None)
        if tray is None:
            super().dropEvent(event)
            return
        from ui.components.blob_tray import is_zip_file
        for url in mime.urls():
            path = url.toLocalFile()
            if not path:
                continue
            if is_zip_file(path):
                tray.add_zip(path)
            else:
                tray.add_file(path)
        event.acceptProposedAction()

    def insertFromMimeData(self, source) -> None:
        text = source.text() if source else ""
        from ui.components.blob_tray import _PASTE_BLOB_THRESHOLD
        tray = getattr(self, "_blob_tray", None)
        if tray is not None and len(text) >= _PASTE_BLOB_THRESHOLD:
            tray.add_paste(text)
            return
        super().insertFromMimeData(source)


class ConversationSurface(QWidget):
    """Extracted chat-stream and input surface used by PageChat."""

    DEFAULT_INPUT_PLACEHOLDER = "Enter command... (try /agent)"
    EDIT_INPUT_PLACEHOLDER = "Edit message and press SAVE to regenerate"

    sig_send_requested = Signal(object)
    sig_mutation_requested = Signal(str, object)
    sig_agent_command = Signal(str, object)

    def __init__(self, controller, parent=None):
        super().__init__(parent or controller)
        self._controller = controller
        self.setObjectName("conversation_surface")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        chat_layout = QVBoxLayout()
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(10)

        self.message_list = SmoothMessageList()
        self.message_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.message_list.setSelectionMode(QListWidget.NoSelection)
        self.message_list.setFocusPolicy(Qt.NoFocus)
        self.message_list.setObjectName("message_list")
        self.message_list.setAttribute(Qt.WA_StyledBackground, True)
        self.message_list.viewport().setAttribute(Qt.WA_StyledBackground, True)
        self.message_list.setAutoFillBackground(True)
        self.message_list.viewport().setAutoFillBackground(True)
        self.message_list.setSpacing(2)
        chat_layout.addWidget(self.message_list)

        self._agent_status_label = QLabel("")
        self._agent_status_label.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 9px; font-weight: bold; background: transparent;"
        )
        self._agent_status_label.hide()
        chat_layout.addWidget(self._agent_status_label)

        self._agent_popup = QFrame(self)
        self._agent_popup.setFrameShape(QFrame.StyledPanel)
        self._agent_popup.setStyleSheet(
            f"""
            QFrame {{
                background: {_s.BG_PANEL};
                border: 1px solid {_s.ACCENT_PRIMARY};
                border-radius: 4px;
                padding: 6px;
            }}
            """
        )
        popup_layout = QVBoxLayout(self._agent_popup)
        popup_layout.setContentsMargins(8, 6, 8, 6)
        popup_layout.setSpacing(3)
        self._agent_popup_label = QLabel("")
        self._agent_popup_label.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 10px; font-family: Consolas, monospace; background: transparent;"
        )
        self._agent_popup_label.setWordWrap(True)
        popup_layout.addWidget(self._agent_popup_label)
        self._agent_popup.hide()

        from ui.components.command_picker import CommandPicker
        self._command_picker = CommandPicker(self)
        self._command_picker.sig_command_selected.connect(self._on_picker_selected)
        self._command_picker.hide()

        from ui.components.blob_tray import BlobTray
        self._blob_tray = BlobTray(self)
        chat_layout.addWidget(self._blob_tray)

        input_row = QHBoxLayout()
        self.input = ChatInput()
        self.input._key_interceptor = self._handle_picker_key
        self.input._blob_tray = self._blob_tray
        self._blob_tray.sig_blob_to_input.connect(self._on_blob_to_input)
        self.input.setPlaceholderText(self.DEFAULT_INPUT_PLACEHOLDER)
        self.input.returnPressed.connect(self.handle_send_click)
        self.input.contentChanged.connect(self._on_input_changed)
        self.input.setObjectName("chat_input")

        self.btn_send = QPushButton("SEND")
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.setFixedWidth(80)
        self.btn_send.clicked.connect(self.handle_send_click)

        input_row.addWidget(self.input)
        input_row.addWidget(self.btn_send)
        chat_layout.addLayout(input_row)

        root.addLayout(chat_layout)

        self._rebuild_send_template()
        self._set_send_button_state(is_running=False)
        self.message_list.viewport().installEventFilter(self)

        # Cross-widget drag-select. Each turn is a separate widget so
        # native QTextEdit selection only spans one message; the manager
        # paints programmatic selections on every text view in the drag
        # range so the visual feels like one continuous selection.
        self._selection = ChatSelectionManager(self._collect_message_widgets, self)
        self._drag_active = False

        # Tool-group folding state. Consecutive tool_call/tool_result rows
        # collapse into a single ToolGroupCard; tracked here across calls
        # to _append_message_widget.
        self._current_tool_group: ToolGroupCard | None = None
        self._current_tool_group_item: QListWidgetItem | None = None
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def focus_input(self) -> None:
        self.input.setFocus(Qt.OtherFocusReason)

    def _collect_message_widgets(self) -> list:
        widgets: list = []
        for row in range(self.message_list.count()):
            item = self.message_list.item(row)
            w = self.message_list.itemWidget(item)
            if w is not None:
                widgets.append(w)
        return widgets

    def _point_in_widget(self, global_pos, widget) -> bool:
        if widget is None or not widget.isVisible():
            return False
        top_left = widget.mapToGlobal(widget.rect().topLeft())
        rect = widget.rect().translated(top_left)
        return rect.contains(global_pos)

    def eventFilter(self, obj, event):
        if obj is self.message_list.viewport() and event.type() == QEvent.Resize:
            self._resize_all_message_items()
            return False

        etype = event.type()
        if etype == QEvent.MouseButtonPress:
            if self._on_global_mouse_press(event):
                return True
        elif etype == QEvent.MouseMove:
            if self._on_global_mouse_move(event):
                return True
        elif etype == QEvent.MouseButtonRelease:
            if self._on_global_mouse_release(event):
                return True
        elif etype == QEvent.KeyPress:
            if self._on_global_key_press(event):
                return True

        return super().eventFilter(obj, event)

    def _event_global_pos(self, event):
        # QMouseEvent in Qt 6: prefer globalPosition() (QPointF). Fall back
        # to globalPos() for compatibility with synthesized events.
        if hasattr(event, "globalPosition"):
            try:
                return event.globalPosition().toPoint()
            except Exception:
                pass
        return event.globalPos()

    def _on_global_mouse_press(self, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        global_pos = self._event_global_pos(event)
        if not self._point_in_widget(global_pos, self.message_list):
            # Click outside the chat list: dismiss any existing selection
            # but don't consume the event — other widgets need it.
            if self._selection.has_selection():
                self._selection.clear()
            return False
        hit = ChatSelectionManager.hit_test(global_pos)
        if hit is None:
            # Inside the list but not on a text view (e.g. between rows
            # or on a hover-action button) — clear and let through.
            self._selection.clear()
            return False
        view, offset = hit
        if self._selection.begin(view, offset):
            self._drag_active = True
            # Consume so the underlying QTextEdit doesn't kick off its
            # own single-widget native selection in parallel.
            return True
        return False

    def _on_global_mouse_move(self, event) -> bool:
        if not self._drag_active:
            return False
        global_pos = self._event_global_pos(event)
        hit = ChatSelectionManager.hit_test(global_pos)
        if hit is not None:
            view, offset = hit
            self._selection.update(view, offset)
        return True

    def _on_global_mouse_release(self, event) -> bool:
        if not self._drag_active:
            return False
        if event.button() != Qt.LeftButton:
            return False
        self._drag_active = False
        self._selection.end()
        return True

    def _on_global_key_press(self, event) -> bool:
        if event.matches(QKeySequence.Copy) and self._selection.has_selection():
            return self._selection.copy_to_clipboard()
        return False

    def _rebuild_send_template(self):
        self._btn_style_template = f"""
            QPushButton {{{{
                background: {{bg}};
                border: 1px solid {{color}};
                color: {{color}};
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
                border-radius: 2px;
            }}}}
            QPushButton:hover {{{{ background: {{color}}; color: black; }}}}
            QPushButton:pressed {{{{ background: {_s.ACCENT_PRIMARY_DARK}; }}}}
        """

    def _on_theme_changed(self, _theme_name: str = "") -> None:
        import core.style as _s_live
        from ui.components.markdown_renderer import reset_renderer_cache

        _s_live.refresh_styles()
        # Invalidate cached pygments + document CSS so the next assistant
        # render picks up the new theme's surface/border/text colors and
        # the dark/light-aware pygments style.
        reset_renderer_cache()
        self._rebuild_send_template()
        self._set_send_button_state(self._controller._is_running)

    def _resize_all_message_items(self):
        vw = self.message_list.viewport().width()
        if vw <= 50:
            return
        for row in range(self.message_list.count()):
            item = self.message_list.item(row)
            widget = self.message_list.itemWidget(item)
            if widget is not None:
                if not widget.isVisible():
                    item.setSizeHint(QSize(0, 0))
                    continue
                widget.setFixedWidth(vw - 2)
                if hasattr(widget, "request_layout_update"):
                    widget.request_layout_update()
                item.setSizeHint(widget.sizeHint())

    def sync_widget_geometry(self, widget: QWidget | None) -> None:
        """Force a single row to recompute/collapse its size hint.

        Used when a widget's visibility or rendered text changes outside the
        normal streaming token path (for example TOOL_LOOP_DONE stripping).
        """
        if widget is None:
            return
        try:
            vw = self.message_list.viewport().width()
        except RuntimeError:
            return
        for row in range(self.message_list.count()):
            item = self.message_list.item(row)
            row_widget = self.message_list.itemWidget(item)
            if row_widget is not widget:
                continue
            try:
                if not widget.isVisible():
                    item.setSizeHint(QSize(0, 0))
                    return
                if vw > 50:
                    widget.setFixedWidth(vw - 2)
                if hasattr(widget, "request_layout_update"):
                    widget.request_layout_update()
                item.setSizeHint(widget.sizeHint())
            except RuntimeError:
                return
            return

    def accept_external_text(
        self,
        text: str,
        *,
        label: str = "external text",
        force_attachment: bool = False,
    ) -> str:
        """Accept text from another workspace into the chat composer.

        Small text appends inline. Large text, or explicit attachment requests,
        becomes a BlobTray paste chip and will be wrapped by send().
        """
        payload = str(text or "")
        if not payload:
            return "ignored"

        from ui.components.blob_tray import _PASTE_BLOB_THRESHOLD

        if force_attachment or len(payload) >= _PASTE_BLOB_THRESHOLD:
            self._blob_tray.add_paste(payload, label=label)
            self.focus_input()
            return "attached"

        current = self.input.text()
        sep = "\n" if current and not current.endswith("\n") else ""
        self.input.setText(current + sep + payload)
        self.focus_input()
        return "inline"

    def send(self):
        ctrl = self._controller
        txt = self.input.text().strip()
        if not txt and not self._blob_tray.has_blobs():
            return
        if ctrl._handle_world_commands(txt):
            self.input.clear()
            self._blob_tray.clear()
            return
        if self._blob_tray.has_blobs():
            from ui.components.blob_tray import format_attached_blocks
            composed = format_attached_blocks(self._blob_tray.blobs(), txt)
            self._blob_tray.clear()
        else:
            composed = txt
        self.input.clear()
        self.sig_send_requested.emit(composed)

    def handle_send_click(self):
        ctrl = self._controller
        txt = self.input.text().strip()

        def _request_stop() -> None:
            if hasattr(ctrl, "request_stop_generation"):
                ctrl.request_stop_generation()
            else:
                ctrl.sig_stop.emit()

        if not ctrl._is_running:
            if ctrl._apply_agent_command(txt):
                self.input.clear()
                return
            self.send()
            return

        if not txt:
            self._set_send_button_state(is_running=True, stopping=True)
            _request_stop()
            return

        ctrl._clear_tool_followup_state()
        ctrl._pending_update_text = txt
        ctrl._awaiting_update_restart = True
        self.btn_send.setEnabled(False)
        ctrl._begin_update_trace(txt)
        _request_stop()

    def _set_send_button_state(self, is_running: bool, stopping: bool = False):
        ctrl = self._controller
        if is_running:
            if ctrl._pending_update_text and not ctrl._awaiting_update_restart:
                self.btn_send.setText("UPDATE")
                bg = _s.BG_INPUT
            else:
                self.btn_send.setText("■")
                bg = _s.BG_INPUT
            self.btn_send.setStyleSheet(
                self._btn_style_template.format(
                    bg=bg,
                    color=_s.ACCENT_PRIMARY,
                )
            )
            self.btn_send.setEnabled(not stopping)
        else:
            self.btn_send.setText("SAVE" if ctrl._is_editing_message() else "SEND")
            self.btn_send.setStyleSheet(
                self._btn_style_template.format(
                    bg=_s.BG_INPUT,
                    color=_s.ACCENT_PRIMARY,
                )
            )
            self.btn_send.setEnabled(True)

    def _on_input_changed(self, text):
        self._controller._update_agent_popup(text)
        if not self._controller._is_running:
            return
        self._set_send_button_state(is_running=True)

    def _on_blob_to_input(self, text: str) -> None:
        """A pasted-text chip's "Move to chat input" button: append its content to the
        composer (the tray already removed the chip) so it becomes inline message text
        instead of an [ATTACHED] block."""
        if not text:
            return
        current = self.input.text()
        sep = "\n" if current and not current.endswith("\n") else ""
        self.input.setText(current + sep + text)
        self.input.setFocus()

    def _handle_picker_key(self, event) -> bool:
        if not self._command_picker.isVisible():
            return False
        key = event.key()
        if key == Qt.Key_Tab:
            result = self._command_picker.accept_selected()
            if result is not None:
                prefix = getattr(self._command_picker, "_prompt_completion_prefix", None)
                if prefix is not None:
                    self.input.setText(prefix + result + " ")
                    self._command_picker._prompt_completion_prefix = None
                else:
                    self.input.setText(result + " ")
            self._command_picker.hide()
            return True
        if key in (Qt.Key_Return, Qt.Key_Enter):
            result = self._command_picker.accept_selected()
            if result is not None:
                prefix = getattr(self._command_picker, "_prompt_completion_prefix", None)
                if prefix is not None:
                    self.input.setText(prefix + result + " ")
                    self._command_picker._prompt_completion_prefix = None
                else:
                    self.input.setText(result + " ")
                self._command_picker.hide()
                return True
            return False
        if key == Qt.Key_Down:
            self._command_picker.select_next()
            return True
        if key == Qt.Key_Up:
            self._command_picker.select_prev()
            return True
        if key == Qt.Key_Escape:
            self._command_picker.hide()
            return True
        return False

    def _on_picker_selected(self, cmd_name: str) -> None:
        self.input.setText(cmd_name + " ")
        self._command_picker.hide()
        self.input.setFocus()

    def _position_command_picker(self) -> None:
        self._command_picker.adjustSize()
        popup_h = min(self._command_picker.sizeHint().height(), 220)
        self._command_picker.setGeometry(
            self.input.x(),
            self.input.y() - popup_h - 4,
            self.input.width() + self.btn_send.width() + 4,
            popup_h,
        )

    def append_token(self, token: str) -> None:
        self._controller.append_token(token)

    def load_session(self, session) -> None:
        self._render_session(session=session, show_reset=False)

    def finalize_response(self) -> None:
        self._resize_all_message_items()
        self.message_list.scrollToBottom()

    def add_message(self, role: str, text: str) -> None:
        idx = self._controller._add_message(role, text)
        self._append_message_widget(idx)

    def get_history(self) -> list[dict]:
        return self._controller._build_engine_history_from_session()

    @staticmethod
    def _parse_tool_payload(text: str) -> dict:
        try:
            payload = _json.loads(text or "")
        except Exception:
            return {"tool": "tool", "raw": text or ""}
        return payload if isinstance(payload, dict) else {"tool": "tool", "raw": payload}

    def _get_or_create_tool_group(self) -> ToolGroupCard:
        if self._current_tool_group is not None:
            return self._current_tool_group
        group = ToolGroupCard()
        ctrl = self._controller
        if hasattr(group, "sig_expand_in_companion") and hasattr(ctrl, "_expand_tool_artifact"):
            group.sig_expand_in_companion.connect(ctrl._expand_tool_artifact)
        item = QListWidgetItem()
        vw = self.message_list.viewport().width()
        if vw > 50:
            group.setFixedWidth(vw - 2)
        item.setSizeHint(group.sizeHint())
        self.message_list.addItem(item)
        self.message_list.setItemWidget(item, group)
        # Bind THIS group + item into the connection. Without the capture the
        # handler would fall back to self._current_tool_group, which any later
        # non-tool message resets to None (see _append_message_widget) — so
        # toggling a tool card from a completed turn would no-op and leave the
        # expanded body overflowing its stale collapsed row (squished/overlap)
        # until an unrelated viewport resize ran _resize_all_message_items.
        # Mirrors the per-widget closure used for MessageWidget rows.
        group.sig_height_changed.connect(
            lambda _g=group, _it=item: self._sync_tool_group_item_size(_g, _it)
        )
        self._current_tool_group = group
        self._current_tool_group_item = item
        return group

    def _sync_tool_group_item_size(self, group: ToolGroupCard | None = None,
                                   item: QListWidgetItem | None = None) -> None:
        # Streaming-time callers pass nothing and target the current group;
        # the sig_height_changed connection passes the emitting group+item
        # explicitly so a toggle on ANY tool card (including ones from earlier,
        # already-closed turns) resizes the correct row.
        if group is None:
            group = self._current_tool_group
        if item is None:
            item = self._current_tool_group_item
        if item is None or group is None:
            return
        try:
            if self.message_list.row(item) < 0:
                return
            vw = self.message_list.viewport().width()
            if vw > 50:
                group.setFixedWidth(vw - 2)
            item.setSizeHint(group.sizeHint())
        except RuntimeError:
            return

    def _append_card_widget(self, card):
        """Append an arbitrary card widget (e.g. the Workshop RunView) as one list row,
        wiring its size-sync with a PER-CARD closure (the row-sizing hazard fix)."""
        from PySide6.QtWidgets import QListWidgetItem
        item = QListWidgetItem(self.message_list)
        card.setFixedWidth(max(10, self.message_list.viewport().width() - 2))
        self.message_list.setItemWidget(item, card)
        item.setSizeHint(card.sizeHint())
        sig = getattr(card, "sig_height_changed", None)
        if sig is not None:
            sig.connect(lambda _c=card, _it=item: self._sync_card_item_size(_c, _it))
        self._current_tool_group = None  # a non-tool row breaks the tool-group run
        return item

    def _sync_card_item_size(self, card, item) -> None:
        try:
            if self.message_list.row(item) < 0:
                return
            item.setSizeHint(card.sizeHint())
        except RuntimeError:
            pass

    def _render_session(self, session=None, show_reset=False):
        ctrl = self._controller
        if session is None:
            session = ctrl._current_session
        ctrl.sig_debug.emit(f"[CHAT] _render_session: msgs={len(session['messages'])}, show_reset={show_reset}")
        self.message_list.clear()
        ctrl._active_widget = None
        # Reset tool-group state so a session re-render starts fresh.
        self._current_tool_group = None
        self._current_tool_group_item = None
        if not session["messages"]:
            return
        for idx, _msg in enumerate(session["messages"]):
            self._append_message_widget(idx)
        self._resize_all_message_items()
        self.message_list.scrollToBottom()

    def _on_open_attachment(self, att) -> None:
        """A chat attachment chip was clicked — route to the databank viewer."""
        ub = getattr(self._controller, "ui_bridge", None)
        if ub is not None and hasattr(ub, "sig_reveal_attachment"):
            ub.sig_reveal_attachment.emit(att)

    def _append_message_widget(self, idx: int, role=None, text=None, timestamp=None):
        ctrl = self._controller
        if idx >= 0:
            msg = ctrl._current_session["messages"][idx]
            role = msg.get("role", "")
            text = msg.get("text", "")
            timestamp = msg.get("time", "")
        role_name = role or ""

        # Fold consecutive tool_call/tool_result messages into one
        # ToolGroupCard rather than rendering each as its own bubble.
        # The card persists across the run of tool messages; any other role
        # ends the group so the next tool_call starts a fresh card.
        if role_name in ("tool_call", "tool_result"):
            payload = self._parse_tool_payload(text or "")
            group = self._get_or_create_tool_group()
            if role_name == "tool_call":
                group.add_call(_call_summary(payload), payload)
            else:
                group.add_result(_result_summary(payload), payload)
            self._sync_tool_group_item_size()
            return group

        # Any non-tool message closes the current group so the next tool
        # call starts a new card.
        self._current_tool_group = None
        self._current_tool_group_item = None

        # Strip the runtime-injected [CHANNEL: ...] header for DISPLAY only.
        # Peer/connect turns are stored with the tag embedded (the model reads
        # it from the stored text via history-build), so we clean the rendered
        # copy here without mutating the session message.
        text = strip_channel_tags_for_display(text or "")
        # Hide raw [ATTACHED]...[/ATTACHED] blocks from the bubble — they stay in
        # the stored/model text. Parsed attachments render as clickable chips.
        text, _attachments = split_attached(text or "")

        item = QListWidgetItem()
        if role_name == "assistant":
            widget = MessageWidget(idx, role_name, "", timestamp or "")
            snapshot = ctrl._assistant_box.build_display_snapshot(text or "", close_open=True)
            widget.apply_assistant_display(
                snapshot.answer_text,
                snapshot.thinking_text,
                thinking_done=not snapshot.thinking_active,
            )
            # Hide assistant rows that carry no public answer. Intermediate
            # tool-loop turns (a tool call + maybe <think>, no answer) were
            # rendering as empty "ASSISTANT" bubbles — the "excessive spaces"
            # (E, 2026-06-18). The reasoning is still in the reasoning-tree pane;
            # the chat shows only turns that actually said something to the user.
            has_answer = bool((snapshot.answer_text or "").strip())
            if not has_answer:
                widget.setVisible(False)
                item.setSizeHint(QSize(0, 0))
        else:
            widget = MessageWidget(idx, role_name, text or "", timestamp or "", attachments=_attachments)
        if hasattr(widget, "sig_open_attachment"):
            widget.sig_open_attachment.connect(self._on_open_attachment)
        if hasattr(widget, "sig_switch_take"):
            widget.sig_switch_take.connect(
                lambda target, direction: self.sig_mutation_requested.emit(
                    "switch_take", {"index": target, "direction": direction})
            )
            try:
                from ui.pages import session_tree
                if session_tree.active() and hasattr(widget, "set_take_info"):
                    widget.set_take_info(
                        session_tree.take_info_for_index(ctrl.current_session_data(), idx))
            except Exception:
                pass
        if hasattr(widget, "sig_action"):
            widget.sig_action.connect(
                lambda action, target: self.sig_mutation_requested.emit(action, {"index": target})
            )
        else:
            widget.sig_delete.connect(
                lambda target=idx: self.sig_mutation_requested.emit("delete", {"index": target})
            )
            if hasattr(widget, "sig_edit"):
                widget.sig_edit.connect(
                    lambda target=idx: self.sig_mutation_requested.emit("edit", {"index": target})
                )
            if hasattr(widget, "sig_regen"):
                widget.sig_regen.connect(
                    lambda target=idx: self.sig_mutation_requested.emit("regen", {"index": target})
                )
        if hasattr(widget, "sig_expand_in_companion"):
            widget.sig_expand_in_companion.connect(ctrl._expand_tool_artifact)

        def _sync_item_size():
            # Guard against queued height-change callbacks arriving after the
            # list has been cleared and the underlying C++ item deleted.
            try:
                if self.message_list.row(item) < 0:
                    return
            except RuntimeError:
                return

            try:
                vw = self.message_list.viewport().width()
            except RuntimeError:
                return

            try:
                if vw > 50:
                    widget.setFixedWidth(vw - 2)
                if not widget.isVisible():
                    item.setSizeHint(QSize(0, 0))
                else:
                    sh = widget.sizeHint()
                    item.setSizeHint(QSize(max(sh.width(), 1), max(sh.height(), 1)))
            except RuntimeError:
                return
            active_idx = getattr(ctrl, "_active_assistant_index", None)
            widget_idx = getattr(widget, "_index", None)
            think_streaming = getattr(widget, "streaming_think_active", lambda: False)()
            think_expanded = bool(getattr(widget, "_think_expanded", False))
            # Suppress autoscroll for think growth ONLY when the block is
            # collapsed (hidden). If the user has expanded it, follow new
            # lines like normal stream content — but only when already at
            # bottom (gated upstream by _auto_scroll_on_height_change).
            should_autoscroll = (
                bool(getattr(ctrl, "_is_running", False))
                and bool(getattr(ctrl, "_auto_scroll_on_height_change", False))
                and widget_idx == active_idx
                and (not think_streaming or think_expanded)
            )
            if should_autoscroll:
                # Defer scroll to next event loop tick so Qt finishes the
                # layout pass from setSizeHint before we reposition.
                # This prevents the visible "jump up" glitch during streaming.
                QTimer.singleShot(0, self.message_list.scrollToBottom)

        widget.sig_height_changed.connect(_sync_item_size)
        vw = self.message_list.viewport().width()
        if vw > 50:
            widget.setFixedWidth(vw - 2)
        item.setSizeHint(widget.sizeHint())
        self.message_list.addItem(item)
        self.message_list.setItemWidget(item, widget)
        return widget

    def _widget_for_index(self, idx: int):
        for row in range(self.message_list.count()):
            item = self.message_list.item(row)
            widget = self.message_list.itemWidget(item)
            if widget is not None and getattr(widget, "_index", None) == idx:
                return widget
        return None

    def rerender_row(self, idx: int) -> None:
        """Rebuild the widget at row `idx` from the (possibly updated) session
        message text. Used by PageChat._on_artifact_ready when generate_image
        artifacts arrive — the bubble needs to swap from pending to ready
        without disturbing surrounding rows.

        Only handles roles whose widget reads directly from message text
        (tool_call, tool_result). Other roles (assistant, user) are no-op so
        in-flight streaming state isn't accidentally clobbered."""
        ctrl = self._controller
        session = getattr(ctrl, "_current_session", None)
        if session is None:
            return
        messages = session.get("messages", [])
        if not (0 <= idx < len(messages)):
            return
        msg = messages[idx]
        role = msg.get("role", "")
        if role not in ("tool_call", "tool_result"):
            return
        for row in range(self.message_list.count()):
            item = self.message_list.item(row)
            widget = self.message_list.itemWidget(item)
            if widget is None or getattr(widget, "_index", None) != idx:
                continue
            text = msg.get("text", "") or ""
            timestamp = msg.get("time", "") or ""
            if role == "tool_call":
                new_widget = ToolCallBubble(idx, text, timestamp)
            else:
                new_widget = ToolResultBubble(idx, text, timestamp)
            new_widget.sig_delete.connect(
                lambda target=idx: self.sig_mutation_requested.emit("delete", {"index": target})
            )
            if hasattr(new_widget, "sig_expand_in_companion"):
                new_widget.sig_expand_in_companion.connect(ctrl._expand_tool_artifact)
            try:
                vw = self.message_list.viewport().width()
                if vw > 50:
                    new_widget.setFixedWidth(vw - 2)
            except Exception:
                pass
            new_widget.sig_height_changed.connect(
                lambda _item=item, _w=new_widget: _item.setSizeHint(QSize(max(_w.sizeHint().width(), 1), max(_w.sizeHint().height(), 1)))
            )
            item.setSizeHint(new_widget.sizeHint())
            self.message_list.setItemWidget(item, new_widget)
            return

    def set_editing_message(self, idx: int | None) -> None:
        for row in range(self.message_list.count()):
            item = self.message_list.item(row)
            widget = self.message_list.itemWidget(item)
            if widget is None or not hasattr(widget, "set_editing"):
                continue
            widget.set_editing(getattr(widget, "_index", None) == idx)
