"""Tests for the databank file viewer's pure content loader."""
from __future__ import annotations

from ui.pages.databank import _load_file_view


def test_load_text_file(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Title\nbody line", encoding="utf-8")
    mode, text = _load_file_view(str(p))
    assert mode == "text"
    assert "# Title" in text


def test_load_binary_file(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x00\x01\x02\x03")
    mode, text = _load_file_view(str(p))
    assert mode == "binary"


def test_load_missing_file(tmp_path):
    mode, text = _load_file_view(str(tmp_path / "nope.txt"))
    assert mode == "missing"


def test_text_file_truncates_over_cap(tmp_path):
    p = tmp_path / "big.txt"
    p.write_text("x" * 100, encoding="utf-8")
    mode, text = _load_file_view(str(p), size_cap=10)
    assert mode == "text"
    assert "truncated" in text.lower()


def _page():
    from PySide6.QtWidgets import QApplication
    from ui.pages.databank import PageFiles
    QApplication.instance() or QApplication([])
    return PageFiles(None, None)


def test_open_attachment_inline_shows_content():
    from core.attached_blocks import Attachment
    page = _page()
    page.open_attachment(Attachment("note.md", "1B", "paste", path=None, content="hello world"))
    assert page.viewer_box.isVisibleTo(page)  # isVisible() is False until a top-level shows
    assert "hello world" in page.viewer.toPlainText()


def test_open_attachment_file_shows_content(tmp_path):
    from core.attached_blocks import Attachment
    f = tmp_path / "doc.txt"
    f.write_text("file body here", encoding="utf-8")
    page = _page()
    page.open_attachment(Attachment("doc.txt", "14B", "text", path=str(f), content=None))
    assert page.viewer_box.isVisibleTo(page)
    assert "file body here" in page.viewer.toPlainText()
