"""Tests for the attachment chip rendered in a user MessageWidget and the
databank file viewer. The raw [ATTACHED] block is hidden at render; a clickable
chip stands in for it and opening it routes to the databank."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication

from core.attached_blocks import Attachment
from ui.components.message_widget import MessageWidget


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_user_message_with_attachments_renders_chip_labels():
    _app()
    atts = [
        Attachment("notes.md", "1.2KB", "text", path=None, content="# hi"),
        Attachment("data.csv", "3KB", "text", path="C:/x/data.csv", content=None),
    ]
    w = MessageWidget(0, "user", "what do you think?", "", attachments=atts)
    assert w._attach_row is not None
    text = w._attach_label.text()
    assert "notes.md" in text
    assert "data.csv" in text


def test_clicking_chip_emits_open_attachment_with_right_item():
    _app()
    atts = [
        Attachment("a.md", "1B", "text", path=None, content="x"),
        Attachment("b.csv", "1B", "text", path="C:/x/b.csv", content=None),
    ]
    w = MessageWidget(0, "user", "hi", "", attachments=atts)
    got = []
    w.sig_open_attachment.connect(lambda a: got.append(a))
    w._on_attach_link("attach:1")  # simulate linkActivated for index 1
    assert len(got) == 1
    assert got[0].label == "b.csv"
    assert got[0].path == "C:/x/b.csv"


def test_message_without_attachments_has_no_chip_row():
    _app()
    w = MessageWidget(0, "user", "hello", "")
    assert getattr(w, "_attach_row", None) is None
