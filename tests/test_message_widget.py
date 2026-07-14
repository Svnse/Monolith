from __future__ import annotations

from PySide6.QtWidgets import QApplication

from ui.components.message_widget import MessageWidget
from ui.pages.assistant_turn_box import (
    AssistantDisplayUpdate,
    AssistantStreamNormalizer,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _stream(widget: MessageWidget, normalizer: AssistantStreamNormalizer, text: str) -> None:
    """Drive a chunk through the production stream path."""
    widget.apply_stream_update(normalizer.consume(text))


def test_streamed_think_tags_never_appear_in_main_text() -> None:
    """Production stream path strips <think>...</think> from the answer body.

    The visible message must never briefly contain raw `<think>` characters --
    that flicker is the bug that the unified parser fixes.
    """
    _app()
    widget = MessageWidget(0, "assistant", "", "")
    norm = AssistantStreamNormalizer()

    _stream(widget, norm, "<think>")
    _stream(widget, norm, "reasoning")
    _stream(widget, norm, "</think>hello")
    widget.apply_stream_update(norm.finalize())
    widget.finalize()

    assert widget.text_view.toPlainText().strip() == "hello"
    assert widget.streaming_think_active() is False
    assert len(widget._think_blocks) == 1
    assert widget._think_blocks[0]._text.toPlainText() == "reasoning"


def test_streamed_tool_call_routes_into_skill_block_not_main_text() -> None:
    """<tool_call>...</tool_call> must reach a _SkillBlock, not the answer body."""
    _app()
    widget = MessageWidget(0, "assistant", "", "")
    norm = AssistantStreamNormalizer()

    _stream(widget, norm, "Sure, ")
    _stream(widget, norm, '<tool_call>{"name":"calc","args":{"x":1}}')
    _stream(widget, norm, "</tool_call>")
    _stream(widget, norm, " done.")
    widget.apply_stream_update(norm.finalize())
    widget.finalize()

    main_text = widget.text_view.toPlainText()
    assert "<tool_call>" not in main_text
    assert "</tool_call>" not in main_text
    assert '"name"' not in main_text
    assert len(widget._skill_blocks) == 1


def test_streamed_tool_card_reserves_height_while_open() -> None:
    app = _app()
    widget = MessageWidget(0, "assistant", "", "")
    widget.setFixedWidth(520)
    widget.adjustSize()
    app.processEvents()
    norm = AssistantStreamNormalizer()

    _stream(widget, norm, "<tool_call>")
    app.processEvents()

    assert len(widget._skill_blocks) == 1
    assert widget._skill_container.isHidden() is False
    assert widget.height() >= widget.sizeHint().height()
    assert widget.height() > 42
    widget.deleteLater()
    app.processEvents()


def test_internal_sentinels_never_reach_widget() -> None:
    """[TOOL_LOOP_DONE] is engine-internal and must be invisible to the user."""
    _app()
    widget = MessageWidget(0, "assistant", "", "")
    norm = AssistantStreamNormalizer()

    _stream(widget, norm, "[TOOL_LOOP_DONE]Final answer.")
    widget.apply_stream_update(norm.finalize())
    widget.finalize()

    plain = widget.text_view.toPlainText()
    assert "[TOOL_LOOP_DONE]" not in plain
    assert "Final answer." in plain


def test_partial_tag_buffer_does_not_stall_parser() -> None:
    """A bare `<` followed by a long non-tag stream must flush as plain text."""
    _app()
    widget = MessageWidget(0, "assistant", "", "")
    norm = AssistantStreamNormalizer()

    long_run = "<" + ("x" * 200) + " end"
    _stream(widget, norm, long_run)
    widget.apply_stream_update(norm.finalize())
    widget.finalize()

    plain = widget.text_view.toPlainText()
    assert "end" in plain
    assert "<" in plain


def test_message_widget_expands_height_while_streaming_wrapped_text() -> None:
    app = _app()
    widget = MessageWidget(0, "assistant", "", "")
    widget.setFixedWidth(220)

    widget.append_token("A" * 160)
    app.processEvents()

    assert widget.text_view.toPlainText() == "A" * 160
    assert widget.height() > 28
    widget.deleteLater()
    app.processEvents()


def test_message_widget_renders_thinking_block_without_raw_tags() -> None:
    app = _app()
    widget = MessageWidget(0, "assistant", "", "")
    widget.setFixedWidth(320)

    widget.apply_assistant_display("Final answer", "step one\nstep two", thinking_done=True)
    app.processEvents()

    assert widget.text_view.toPlainText() == "Final answer"
    assert len(widget._think_blocks) == 1
    assert widget._think_blocks[0]._text.toPlainText() == "step one\nstep two"
    assert widget._think_badge.isHidden() is False
    assert widget._think_container.isVisible() is False
    assert widget.streaming_think_active() is False
    widget.deleteLater()
    app.processEvents()


def test_message_widget_does_not_spawn_empty_thinking_block() -> None:
    app = _app()
    widget = MessageWidget(0, "assistant", "", "")

    widget.apply_stream_update(AssistantDisplayUpdate(thinking_opened=True))
    widget.apply_stream_update(AssistantDisplayUpdate(thinking_closed=True))
    app.processEvents()

    assert len(widget._think_blocks) == 0
    assert widget.streaming_think_active() is False
    widget.deleteLater()
    app.processEvents()


def test_message_widget_uses_hover_actions() -> None:
    _app()
    user_widget = MessageWidget(0, "user", "Prompt", "")
    assistant_widget = MessageWidget(1, "assistant", "Reply", "")

    assert user_widget.btn_edit.text() == "Edit"
    assert user_widget.btn_delete.text() == "Delete"
    assert assistant_widget.btn_regen.text() == "Regen"
    assert assistant_widget.btn_copy.text() == "Copy"
    assert assistant_widget.btn_delete.text() == "Delete"


def test_message_widget_reflows_height_when_width_shrinks() -> None:
    app = _app()
    widget = MessageWidget(0, "assistant", "", "")
    widget.setFixedWidth(520)
    widget.append_token("word " * 220)
    app.processEvents()
    initial_height = widget.height()
    assert initial_height > 28

    widget.setFixedWidth(220)
    widget.request_layout_update()
    app.processEvents()
    app.processEvents()
    app.processEvents()

    assert widget.height() > initial_height
    widget.deleteLater()
    app.processEvents()
