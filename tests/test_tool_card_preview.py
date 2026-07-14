"""Tests for the tool-card readability helpers: what content to preview and
how to clip it to a compact ~3-line preview with a 'show more' affordance."""
from __future__ import annotations

from ui.components.tool_bubbles import _preview_source, _clip_preview


# ── _preview_source: what gets previewed ─────────────────────────────────────


def test_write_file_call_previews_the_written_content():
    src = _preview_source("call", {"tool": "write_file", "path": "x.md", "content": "# Title\nbody"})
    assert src == "# Title\nbody"


def test_write_file_result_has_no_preview_content():
    # The result is just a status line; the content preview comes from the call.
    src = _preview_source("result", {"tool": "write_file", "result": "[write_file: written 5 chars to x.md]"})
    assert src is None


def test_read_file_result_previews_the_body():
    src = _preview_source("result", {"tool": "read_file", "result": "line one\nline two"})
    assert src is not None
    assert "line one" in src


def test_call_without_content_has_no_preview():
    assert _preview_source("call", {"tool": "grep", "pattern": "x"}) is None


# ── _clip_preview: compact clipping with has_more ────────────────────────────


def test_short_text_is_not_clipped():
    clipped, has_more = _clip_preview("a\nb")
    assert clipped == "a\nb"
    assert has_more is False


def test_more_than_max_lines_sets_has_more():
    clipped, has_more = _clip_preview("\n".join(str(i) for i in range(10)), max_lines=3)
    assert has_more is True
    assert clipped.count("\n") == 2  # exactly 3 lines kept


def test_long_single_line_clips_to_max_chars():
    clipped, has_more = _clip_preview("x" * 500, max_chars=200)
    assert has_more is True
    assert len(clipped) <= 200


# ── card rendering ───────────────────────────────────────────────────────────


def _label_texts(card):
    from PySide6.QtWidgets import QLabel
    lay = card._body_layout
    out = []
    for i in range(lay.count()):
        w = lay.itemAt(i).widget()
        if isinstance(w, QLabel):
            out.append(w.text())
    return out


def test_write_file_card_shows_content_preview_and_toggle():
    from PySide6.QtWidgets import QApplication
    from ui.components.tool_bubbles import ToolGroupCard
    QApplication.instance() or QApplication([])
    card = ToolGroupCard()
    card.add_call(
        "write_file -> spec.md",
        {"tool": "write_file", "path": "C:/x/spec.md",
         "content": "# Heading\nline one\nline two\nline three\nline four"},
    )
    texts = _label_texts(card)
    joined = "\n".join(texts)
    assert "# Heading" in joined           # content preview rendered
    assert any("show more" in t for t in texts)  # toggle present (5 lines > 3)


def test_short_content_has_no_toggle():
    from PySide6.QtWidgets import QApplication
    from ui.components.tool_bubbles import ToolGroupCard
    QApplication.instance() or QApplication([])
    card = ToolGroupCard()
    card.add_call("write_file -> a.md", {"tool": "write_file", "path": "a.md", "content": "one line"})
    texts = _label_texts(card)
    assert any("one line" in t for t in texts)
    assert not any("show more" in t for t in texts)
