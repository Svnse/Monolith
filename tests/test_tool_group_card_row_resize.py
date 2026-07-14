"""Regression: expanding a ToolGroupCard from an already-completed turn must
resize ITS row, not no-op.

The card folds tool_call/tool_result messages into one row; once a later
non-tool message arrives, ConversationSurface resets _current_tool_group to
None. The bug: the card's sig_height_changed handler targeted
_current_tool_group_item, so a toggle on a closed group resized nothing and the
expanded body overflowed its stale collapsed row (squished / overlapping) until
an unrelated viewport resize ran _resize_all_message_items. Reproduced as the
maximized-window tool-block squish that only fixed itself on un-maximize.
"""
from __future__ import annotations

import os

# Qt needs a platform; offscreen lets show() flip real isVisible() (which
# ToolGroupCard.sizeHint() gates the expanded-height branch on) without a
# display. setdefault so we never clobber a runner-provided platform.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QVBoxLayout

from ui.conversation_surface import ConversationSurface
from ui.components.tool_bubbles import ToolGroupCard
from tests.test_conversation_surface import _DummyController


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _find_tool_group(surface):
    for row in range(surface.message_list.count()):
        item = surface.message_list.item(row)
        widget = surface.message_list.itemWidget(item)
        if isinstance(widget, ToolGroupCard):
            return widget, item
    return None, None


def test_expanding_closed_tool_group_resizes_its_own_row() -> None:
    app = _app()
    controller = _DummyController(
        [
            {"role": "tool_call", "text": '{"tool": "write_file", "path": "hello.py"}', "time": ""},
            {"role": "tool_result", "text": '{"tool": "write_file", "result": "[tool: invalid arguments for write_file]"}', "time": ""},
            {"role": "assistant", "text": "Done.", "time": ""},
        ]
    )
    surface = ConversationSurface(controller)
    # Give the surface a real, wide viewport — mirrors a maximized window, the
    # condition under which the bug was visible.
    layout = QVBoxLayout(controller)
    layout.addWidget(surface)
    controller.resize(1200, 700)
    controller.show()
    app.processEvents()

    # Render the three messages: the two tool rows fold into one ToolGroupCard,
    # then the assistant message closes the group (current -> None).
    for idx in range(3):
        surface._append_message_widget(idx)
    app.processEvents()

    assert surface._current_tool_group is None, "assistant message should close the group"

    group, item = _find_tool_group(surface)
    assert group is not None and item is not None
    assert group._expanded is False
    collapsed_h = item.sizeHint().height()

    # User expands the completed tool card to inspect it.
    group._toggle_expand()
    app.processEvents()

    expanded_row_h = item.sizeHint().height()
    content_h = group.sizeHint().height()

    # The expanded body must actually have height (guards against a no-op test
    # where visibility gating collapses content_h back to the header height).
    assert content_h > collapsed_h, "expanded content should be taller than collapsed header"
    # The fix: the row tracks the card it belongs to, even after the group was
    # closed. Pre-fix this stayed at collapsed_h and the body overlapped.
    assert expanded_row_h == content_h, (
        f"row ({expanded_row_h}px) must match expanded content ({content_h}px); "
        "stale row height causes the squished/overlapping tool block"
    )
