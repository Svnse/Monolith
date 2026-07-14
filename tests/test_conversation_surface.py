from __future__ import annotations

from PySide6.QtCore import QAbstractAnimation, QEvent, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QAbstractItemView, QWidget

from ui.conversation_surface import ChatInput, ConversationSurface, SmoothMessageList
from ui.pages.assistant_turn_box import AssistantDisplaySnapshot


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _DummyAssistantBox:
    def build_display_snapshot(self, text: str, close_open: bool = True) -> AssistantDisplaySnapshot:
        return AssistantDisplaySnapshot(answer_text=text, thinking_active=not close_open)


class _DummyController(QWidget):
    sig_stop = Signal()

    def __init__(self, messages: list[dict]):
        super().__init__()
        self.ui_bridge = object()
        self._assistant_box = _DummyAssistantBox()
        self._is_running = False
        self._auto_scroll_on_height_change = False
        self._active_assistant_index = None
        self._editing = False
        self._current_session = {"messages": messages}

    def _build_action_review_panel(self) -> QWidget:
        return QWidget()

    def _handle_world_commands(self, _text: str) -> bool:
        return False

    def _apply_agent_command(self, _text: str) -> bool:
        return False

    def _clear_tool_followup_state(self) -> None:
        return None

    def _update_agent_popup(self, _text: str) -> None:
        return None

    def append_token(self, _token: str) -> None:
        return None

    def _add_message(self, role: str, text: str) -> int:
        self._current_session["messages"].append({"role": role, "text": text, "time": ""})
        return len(self._current_session["messages"]) - 1

    def _build_engine_history_from_session(self) -> list[dict]:
        return []

    def _expand_tool_artifact(self, _tool_name: str, _payload: object) -> None:
        return None

    def _is_editing_message(self) -> bool:
        return self._editing

    def request_stop_generation(self) -> None:
        self.sig_stop.emit()


class _WheelDelta:
    def __init__(self, y: int):
        self._y = y

    def y(self) -> int:
        return self._y


class _WheelEvent:
    def __init__(self, *, pixel_y: int = 0, angle_y: int = 0):
        self._pixel = _WheelDelta(pixel_y)
        self._angle = _WheelDelta(angle_y)
        self.accepted = False

    def pixelDelta(self) -> _WheelDelta:
        return self._pixel

    def angleDelta(self) -> _WheelDelta:
        return self._angle

    def accept(self) -> None:
        self.accepted = True


def test_conversation_surface_forwards_clicked_assistant_regen_index() -> None:
    app = _app()
    controller = _DummyController(
        [{"i": 1, "role": "assistant", "text": "Draft", "time": "2026-01-01T00:00:00+00:00"}]
    )
    surface = ConversationSurface(controller)
    events: list[tuple[str, object]] = []
    surface.sig_mutation_requested.connect(lambda action, payload: events.append((action, payload)))

    surface._append_message_widget(0)
    widget = surface._widget_for_index(0)
    assert widget is not None

    widget.sig_action.emit("regen", 0)
    app.processEvents()

    assert events == [("regen", {"index": 0})]


def test_conversation_surface_disables_message_list_selection() -> None:
    _app()
    controller = _DummyController([])
    surface = ConversationSurface(controller)

    assert surface.message_list.selectionMode() == QAbstractItemView.NoSelection


def test_chat_input_accepts_text_beyond_qlineedit_limit() -> None:
    _app()
    composer = ChatInput()
    long_text = "x" * 40000

    composer.setText(long_text)

    assert len(composer.text()) == len(long_text)


def test_paste_chip_move_to_input_appends_to_empty_composer() -> None:
    _app()
    controller = _DummyController([])
    surface = ConversationSurface(controller)
    surface._blob_tray.add_paste("the pasted body")

    surface._blob_tray._chips[0]._on_to_input_clicked()

    assert surface.input.text() == "the pasted body"   # moved inline into the composer
    assert not surface._blob_tray.has_blobs()           # no longer an attachment


def test_paste_chip_move_to_input_appends_after_existing_text_with_newline() -> None:
    _app()
    controller = _DummyController([])
    surface = ConversationSurface(controller)
    surface.input.setText("draft so far")
    surface._blob_tray.add_paste("pasted tail")

    surface._blob_tray._chips[0]._on_to_input_clicked()

    assert surface.input.text() == "draft so far\npasted tail"
    assert not surface._blob_tray.has_blobs()


def test_accept_external_text_small_payload_goes_inline() -> None:
    _app()
    controller = _DummyController([])
    surface = ConversationSurface(controller)

    result = surface.accept_external_text("from note", label="MonoNote")

    assert result == "inline"
    assert surface.input.text() == "from note"
    assert not surface._blob_tray.has_blobs()


def test_accept_external_text_large_payload_becomes_attachment() -> None:
    _app()
    controller = _DummyController([])
    surface = ConversationSurface(controller)
    body = "x" * 1200

    result = surface.accept_external_text(body, label="MonoNote")

    assert result == "attached"
    assert surface.input.text() == ""
    blobs = surface._blob_tray.blobs()
    assert len(blobs) == 1
    assert blobs[0]["label"] == "MonoNote"
    assert blobs[0]["content"] == body


def test_accept_external_text_can_force_attachment_for_small_payload() -> None:
    _app()
    controller = _DummyController([])
    surface = ConversationSurface(controller)

    result = surface.accept_external_text("short", label="MonoNote", force_attachment=True)

    assert result == "attached"
    assert surface.input.text() == ""
    assert surface._blob_tray.blobs()[0]["content"] == "short"


def test_conversation_surface_sends_long_prompt_without_truncation() -> None:
    _app()
    controller = _DummyController([])
    surface = ConversationSurface(controller)
    long_text = "prompt-" + ("x" * 40000)
    sent: list[str] = []
    surface.sig_send_requested.connect(sent.append)

    surface.input.setText(long_text)
    surface.send()

    assert sent == [long_text]


def test_chat_input_enter_sends_and_shift_enter_keeps_newline() -> None:
    _app()
    controller = _DummyController([])
    surface = ConversationSurface(controller)
    sent: list[str] = []
    surface.sig_send_requested.connect(sent.append)

    surface.input.setText("send me")
    surface.input.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier)
    )
    assert sent == ["send me"]

    surface.input.setText("keep")
    surface.input.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.ShiftModifier)
    )
    assert surface.input.text() == "keep\n"


def test_smooth_message_list_bounds_large_wheel_notches() -> None:
    assert SmoothMessageList._wheel_delta_pixels(120, 0, 600) == 72
    assert SmoothMessageList._wheel_delta_pixels(120 * 8, 0, 600) == 192
    assert SmoothMessageList._wheel_delta_pixels(-(120 * 8), 0, 600) == -192


def test_smooth_message_list_clamps_bottom_edge_jitter() -> None:
    assert SmoothMessageList._edge_target_for_delta(100, 0, 100, -24, False) == 100
    assert SmoothMessageList._edge_target_for_delta(100, 0, 100, 1, True) == 100
    assert SmoothMessageList._edge_target_for_delta(99, 0, 100, 1, True) == 100
    assert SmoothMessageList._edge_target_for_delta(98, 0, 100, 1, True) == 100
    assert SmoothMessageList._edge_target_for_delta(97, 0, 100, 6, True) == 100
    assert SmoothMessageList._edge_target_for_delta(96, 0, 100, 1, True) is None
    assert SmoothMessageList._edge_target_for_delta(100, 0, 100, 9, True) is None
    assert SmoothMessageList._edge_target_for_delta(100, 0, 100, 24, True) is None


def test_smooth_message_list_ignores_bottom_pixel_rebound() -> None:
    _app()
    message_list = SmoothMessageList()
    bar = message_list.verticalScrollBar()
    bar.setRange(0, 100)
    bar.setValue(100)
    event = _WheelEvent(pixel_y=1)

    message_list.wheelEvent(event)

    assert event.accepted is True
    assert bar.value() == 100
    assert message_list._smooth_scroll_target == 100
    assert message_list._scroll_anim.state() != QAbstractAnimation.State.Running


def test_smooth_message_list_stays_bottom_when_range_grows() -> None:
    _app()
    message_list = SmoothMessageList()
    bar = message_list.verticalScrollBar()
    bar.setRange(0, 100)
    bar.setValue(100)

    bar.setRange(0, 101)

    assert bar.value() == 101
    assert message_list._smooth_scroll_target == 101


def test_smooth_message_list_does_not_snap_range_growth_when_not_at_bottom() -> None:
    _app()
    message_list = SmoothMessageList()
    bar = message_list.verticalScrollBar()
    bar.setRange(0, 100)
    bar.setValue(50)

    bar.setRange(0, 101)

    assert bar.value() == 50
    assert message_list._smooth_scroll_target == 50


def test_smooth_message_list_scroll_to_bottom_stops_stale_animation() -> None:
    app = _app()
    message_list = SmoothMessageList()
    bar = message_list.verticalScrollBar()
    bar.setRange(0, 100)
    bar.setValue(60)
    message_list._smooth_scroll_target = 50
    message_list._scroll_anim.setStartValue(60)
    message_list._scroll_anim.setEndValue(50)
    message_list._scroll_anim.start()
    assert message_list._scroll_anim.state() == QAbstractAnimation.State.Running

    message_list.scrollToBottom()
    app.processEvents()

    assert message_list._scroll_anim.state() != QAbstractAnimation.State.Running
    assert bar.value() == bar.maximum()
    assert message_list._smooth_scroll_target == bar.maximum()


def test_conversation_surface_autoscroll_only_while_running() -> None:
    app = _app()
    controller = _DummyController(
        [{"i": 1, "role": "assistant", "text": "Draft", "time": "2026-01-01T00:00:00+00:00"}]
    )
    controller._auto_scroll_on_height_change = True
    controller._active_assistant_index = 0
    surface = ConversationSurface(controller)
    surface._append_message_widget(0)
    widget = surface._widget_for_index(0)
    assert widget is not None

    called = {"count": 0}
    original_scroll = surface.message_list.scrollToBottom

    def _scroll_probe() -> None:
        called["count"] += 1
        original_scroll()

    surface.message_list.scrollToBottom = _scroll_probe

    controller._is_running = False
    widget.sig_height_changed.emit()
    app.processEvents()
    assert called["count"] == 0

    controller._is_running = True
    widget.sig_height_changed.emit()
    app.processEvents()
    assert called["count"] >= 1


def test_conversation_surface_hides_assistant_row_when_only_think_blocks() -> None:
    _app()
    controller = _DummyController(
        [{"i": 1, "role": "assistant", "text": "<think>planning</think>", "time": "2026-01-01T00:00:00+00:00"}]
    )
    surface = ConversationSurface(controller)
    widget = surface._append_message_widget(0)
    assert widget is not None
    assert widget.isVisible() is False


def test_conversation_surface_ignores_stale_height_signal_after_clear() -> None:
    app = _app()
    controller = _DummyController(
        [{"i": 1, "role": "assistant", "text": "Draft", "time": "2026-01-01T00:00:00+00:00"}]
    )
    surface = ConversationSurface(controller)
    widget = surface._append_message_widget(0)
    assert widget is not None

    surface.message_list.clear()
    widget.sig_height_changed.emit()
    app.processEvents()


def test_channel_tag_hidden_from_display_but_kept_in_storage() -> None:
    """A connect/peer turn is stored as "[Codex] [CHANNEL: ...]\\n\\nbody" so the
    model still receives its channel context; the rendered widget must show the
    body without the raw tag."""
    _app()
    stored = "[Codex] [CHANNEL: connect/Codex, mcp send_message]\n\nHello peer"
    controller = _DummyController(
        [{"i": 1, "role": "agent", "text": stored, "time": "2026-01-01T00:00:00+00:00",
          "agent_name": "Codex", "agent_approved": True}]
    )
    surface = ConversationSurface(controller)
    widget = surface._append_message_widget(0)
    assert widget is not None

    # Display is clean...
    assert "[CHANNEL:" not in widget._content
    assert widget._content == "[Codex] Hello peer"
    # ...but the stored session message is untouched (the model reads it).
    assert "[CHANNEL: connect/Codex" in controller._current_session["messages"][0]["text"]
