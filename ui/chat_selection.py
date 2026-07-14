"""Cross-widget drag-select for the chat message list.

The conversation surface renders each turn as its own widget (MessageWidget,
ToolCallBubble, ToolResultBubble). Native QTextEdit drag-select only works
within a single text widget; this manager orchestrates programmatic native
selections across multiple widgets so a single mouse drag *visually* spans
the conversation while clicking Ctrl+C produces one concatenated copy.

Architecture:
  - `_FlatView` is one selectable text view in document order, tagged with
    its owner widget and a sub-index inside that owner (think block 0,
    main body 1, tool body 2, ...).
  - On mouse press inside a text view we record an anchor (view + cursor
    offset). On mouse move we record an active end. The manager paints
    selections on every view between anchor and end using QTextCursor.
  - Ctrl+C walks the included views in order and joins their selected
    text with newlines.

Skipped from selection:
  - Hidden widgets (collapsed think blocks, hidden assistant rows).
  - Tool/think bodies that the user has not expanded — per the chosen
    UX, only what the user can visibly see is included.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, QPoint, Qt
from PySide6.QtGui import QGuiApplication, QTextCursor
from PySide6.QtWidgets import QApplication, QTextEdit, QWidget


@dataclass(frozen=True)
class _Anchor:
    view: QTextEdit
    offset: int
    row: int          # position in the QListWidget
    sub: int          # text-view index within the owning widget


@dataclass(frozen=True)
class _FlatView:
    view: QTextEdit
    row: int
    sub: int
    owner: QWidget


class ChatSelectionManager(QObject):
    """Drives a single logical selection across multiple message widgets."""

    def __init__(self, get_widgets_in_order: Callable[[], list[QWidget]], parent=None):
        super().__init__(parent)
        self._get_widgets = get_widgets_in_order
        self._anchor: _Anchor | None = None
        self._active: _Anchor | None = None
        self._is_dragging = False

    # ── public API ──────────────────────────────────────────────────

    def begin(self, view: QTextEdit, offset: int) -> bool:
        located = self._locate(view)
        if located is None:
            return False
        self.clear()
        anchor = _Anchor(view, offset, located.row, located.sub)
        self._anchor = anchor
        self._active = anchor
        self._is_dragging = True
        self._render()
        return True

    def update(self, view: QTextEdit, offset: int) -> None:
        if not self._is_dragging or self._anchor is None:
            return
        located = self._locate(view)
        if located is None:
            return
        self._active = _Anchor(view, offset, located.row, located.sub)
        self._render()

    def end(self) -> None:
        if not self._is_dragging:
            return
        self._is_dragging = False
        # Pure click (no movement) — treat as clear so the previous
        # selection doesn't stay visible after a stray click.
        if self._anchor is not None and self._active is not None:
            if (self._anchor.view is self._active.view
                    and self._anchor.offset == self._active.offset):
                self.clear()

    def clear(self) -> None:
        for fv in self._enumerate_views():
            self._clear_selection(fv.view)
        self._anchor = None
        self._active = None
        self._is_dragging = False

    def has_selection(self) -> bool:
        if self._anchor is None or self._active is None:
            return False
        return (self._anchor.view is not self._active.view
                or self._anchor.offset != self._active.offset)

    def copy_to_clipboard(self) -> bool:
        text = self._collect_text()
        if not text:
            return False
        clip = QGuiApplication.clipboard()
        if clip is None:
            return False
        clip.setText(text)
        return True

    @staticmethod
    def hit_test(global_pos: QPoint) -> tuple[QTextEdit, int] | None:
        """Find the QTextEdit under a global screen point and the cursor
        offset within its document. Returns None if no text view is hit.
        """
        widget = QApplication.widgetAt(global_pos)
        node: QWidget | None = widget
        while node is not None:
            if isinstance(node, QTextEdit):
                local = node.viewport().mapFromGlobal(global_pos)
                cursor = node.cursorForPosition(local)
                return node, cursor.position()
            node = node.parentWidget()
        return None

    # ── internals ───────────────────────────────────────────────────

    def _enumerate_views(self) -> list[_FlatView]:
        flat: list[_FlatView] = []
        widgets = self._get_widgets() or []
        for row, owner in enumerate(widgets):
            if not getattr(owner, "isVisible", lambda: True)():
                continue
            getter = getattr(owner, "get_selectable_text_views", None)
            if not callable(getter):
                continue
            try:
                pairs = getter() or []
            except Exception:
                continue
            for sub, view in enumerate(pairs):
                if view is None:
                    continue
                flat.append(_FlatView(view=view, row=row, sub=sub, owner=owner))
        return flat

    def _locate(self, view: QTextEdit) -> _FlatView | None:
        for fv in self._enumerate_views():
            if fv.view is view:
                return fv
        return None

    def _render(self) -> None:
        if self._anchor is None or self._active is None:
            return
        a_key = (self._anchor.row, self._anchor.sub, self._anchor.offset)
        b_key = (self._active.row, self._active.sub, self._active.offset)
        if a_key <= b_key:
            start, end = self._anchor, self._active
        else:
            start, end = self._active, self._anchor

        start_key = (start.row, start.sub)
        end_key = (end.row, end.sub)

        for fv in self._enumerate_views():
            cur_key = (fv.row, fv.sub)
            if cur_key < start_key or cur_key > end_key:
                self._clear_selection(fv.view)
                continue
            if cur_key == start_key and cur_key == end_key:
                self._apply_selection(fv.view, start.offset, end.offset)
            elif cur_key == start_key:
                doc_end = len(fv.view.toPlainText())
                self._apply_selection(fv.view, start.offset, doc_end)
            elif cur_key == end_key:
                self._apply_selection(fv.view, 0, end.offset)
            else:
                doc_end = len(fv.view.toPlainText())
                self._apply_selection(fv.view, 0, doc_end)

    def _collect_text(self) -> str:
        if self._anchor is None or self._active is None:
            return ""
        chunks: list[str] = []
        for fv in self._enumerate_views():
            cursor = fv.view.textCursor()
            if not cursor.hasSelection():
                continue
            chunks.append(cursor.selectedText())
        # Qt uses U+2029 (paragraph separator) inside selectedText() to
        # represent line breaks — normalize to "\n" so the clipboard
        # content is a plain multi-line string consumers will expect.
        joined = "\n".join(chunks)
        return joined.replace(" ", "\n").replace(" ", "\n")

    @staticmethod
    def _apply_selection(view: QTextEdit, start: int, end: int) -> None:
        if start == end:
            ChatSelectionManager._clear_selection(view)
            return
        if start > end:
            start, end = end, start
        cursor = view.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        view.setTextCursor(cursor)

    @staticmethod
    def _clear_selection(view: QTextEdit) -> None:
        cursor = view.textCursor()
        if cursor.hasSelection():
            cursor.clearSelection()
            view.setTextCursor(cursor)
