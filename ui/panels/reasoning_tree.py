"""ReasoningTreePanel — companion-pane navigator + inspector for the reasoning
branch tree (spec: docs/superpowers/specs/2026-06-10-reasoning-tree-pane-design.md).

Pure data layer first (build_rows / extract_think — no Qt, fully testable), the
widget below it. House idiom: monothink-ledger skeleton, glyph rows, fixed row
heights, refresh-on-demand, the viewer never crashes.
"""
from __future__ import annotations

from dataclasses import dataclass

from core import branch_tree
from core.thinkpad import _THINK_RE   # the one think-envelope regex (do not duplicate)

_PREVIEW_CHARS = 40


@dataclass(frozen=True)
class Row:
    node_id: str
    label: str          # glyph-free text + chips
    glyph: str          # structure prefix, e.g. "● " / "├○ " / "└○ " / "  ├● "
    on_path: bool
    role: str
    take: tuple[int, int] | None   # (k, n) among same-parent siblings, None if n == 1
    has_think: bool


def extract_think(text: str) -> str:
    """All think-envelope bodies in ``text``, joined — same regex thinkpad uses."""
    return "\n".join(m.group(2).strip() for m in _THINK_RE.finditer(text or "")).strip()


def _chips(node: dict, take: tuple[int, int] | None) -> str:
    bits: list[str] = []
    origin = node.get("origin", "")
    if take is not None:
        bits.append(f"[take {take[0]}/{take[1]}]")
    if origin == "regen" or origin.startswith("thinkpad:"):
        bits.append(f"[{origin}]")
    if node.get("branch_kind") == "divergent":
        d = node.get("divergence")
        bits.append(f"✎ [divergent Δ{d:.1f}]" if isinstance(d, float) else "✎ [divergent]")
    elif origin == "edit":
        bits.append("✎ [edit]")
    msg = node.get("msg", {})
    if msg.get("role") == "assistant" and not (msg.get("text") or "").strip():
        bits.append("⟳")
    return "  ".join(bits)


def build_rows(tree: dict) -> list[Row]:
    """Depth-first rows for every node. Glyphs: ``●``/``○`` = on/off the active
    path; ``├``/``└`` open a sibling group; two spaces of indent per fork level.

    Iterative DFS via an explicit stack — avoids RecursionError on long linear
    sessions (Python's default recursion limit is ~1000 frames).

    Stack entries are ``(parent_node_id, child_index, depth)`` so each pop
    emits exactly one child row and then pushes: first that child's own expansion
    frame (to descend into its subtree), then the *next* sibling frame (to
    continue after the subtree).  Siblings are pushed last-first so earlier
    siblings are processed before later ones, preserving the original left-to-
    right DFS emission order and therefore all glyph/order tests.
    """
    on_path = set(branch_tree.active_path(tree))
    rows: list[Row] = []

    nodes = tree["nodes"]

    def _push_children(parent_id: str, depth: int, stack: list) -> None:
        """Push all children of ``parent_id`` onto the stack in reverse order."""
        kids = nodes[parent_id]["children"]
        n = len(kids)
        for i in range(n - 1, -1, -1):
            stack.append((parent_id, i, depth))

    # Stack entries: (parent_node_id, child_index_in_parent, depth_at_which_to_render)
    stack: list[tuple[str, int, int]] = []
    _push_children(tree["root_id"], 0, stack)

    while stack:
        parent_id, idx, depth = stack.pop()
        kids = nodes[parent_id]["children"]
        n = len(kids)
        cid = kids[idx]
        node = nodes[cid]
        msg = node["msg"]
        take = (idx + 1, n) if n > 1 else None
        dot = "●" if cid in on_path else "○"
        if n > 1:
            branch_glyph = "├" if idx < n - 1 else "└"
            glyph = "  " * depth + branch_glyph + dot + " "
        else:
            glyph = "  " * depth + dot + " "
        role = str(msg.get("role", ""))
        text = str(msg.get("text", "") or "")
        preview = text.strip().replace("\n", " ")[:_PREVIEW_CHARS]
        if role == "user":
            label = f'"{preview}"' if preview else "(empty)"
        else:
            label = ""
        chips = _chips(node, take)
        label = f"{label}  {chips}".strip() if chips else (label or role)
        rows.append(Row(node_id=cid, label=label, glyph=glyph, on_path=cid in on_path,
                        role=role, take=take,
                        has_think=_THINK_RE.search(text) is not None))
        # Descend into this child's subtree next.
        _push_children(cid, depth + (1 if n > 1 else 0), stack)

    return rows


# ── Qt widget layer ───────────────────────────────────────────────────────────
# Everything below depends on PySide6; the pure layer above is import-clean.

from PySide6.QtCore import Qt, QSize, QTimer, QPoint  # noqa: E402
from PySide6.QtGui import QColor, QCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QScrollArea, QVBoxLayout, QWidget,
)

import core.style as _s  # noqa: E402
from ui.components.atoms import MonoButton  # noqa: E402

_ROW_HEIGHT = 22
_HOVER_DELAY_MS = 350
_POPUP_MAX = QSize(600, 420)
_ON_PATH_COLOR = QColor(_s.FG_OK)   # status green token (UI_CONTRACT §3)


class _ThinkPopup(QFrame):
    """Frameless hover card showing a node's full think block. Display-only.

    Callbacks ``on_enter`` / ``on_leave`` are injected by the panel so the panel
    can cancel or restart its ``_hide_timer`` when the cursor moves into/out of
    the popup.  ``WA_Hover`` ensures Qt delivers synthetic enter/leave even with
    the ToolTip window flag (which can swallow them on some platforms); the
    geometry check in ``_maybe_hide_popup`` is the load-bearing truth guard.
    """

    def __init__(self) -> None:
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_Hover, True)
        self._on_enter_cb = None   # injected by ReasoningTreePanel
        self._on_leave_cb = None
        self.setStyleSheet(
            f"QFrame {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_SUBTLE}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        self._head = QLabel("")
        self._head.setStyleSheet(
            f"color: {_s.FG_DIM}; font-family: Consolas; font-size: 9px;")
        lay.addWidget(self._head)
        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._body.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-family: Consolas; font-size: 9px;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._body)
        lay.addWidget(scroll)
        self.setMaximumSize(_POPUP_MAX)

    def enterEvent(self, event) -> None:
        if callable(self._on_enter_cb):
            self._on_enter_cb()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if callable(self._on_leave_cb):
            self._on_leave_cb()
        super().leaveEvent(event)

    def present(self, head: str, text: str, at: QPoint) -> None:
        self._head.setText(head)
        self._body.setText(text)
        self.adjustSize()
        self.move(at)
        self.show()


class ReasoningTreePanel(QWidget):
    NODE_ROLE = Qt.ItemDataRole.UserRole

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._controller = None
        self._fingerprint = None
        self._collapse_old = False
        self._popup = _ThinkPopup()
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(_HOVER_DELAY_MS)
        self._hover_timer.timeout.connect(self._show_hover_popup)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(300)
        self._hide_timer.timeout.connect(self._maybe_hide_popup)
        # Inject enter/leave callbacks so the popup can cancel/restart _hide_timer.
        self._popup._on_enter_cb = self._hide_timer.stop
        self._popup._on_leave_cb = self._hide_timer.start
        self._hover_node: str | None = None
        self._build_ui()
        try:
            from ui.pages import session_tree
            session_tree.subscribe(self._on_tree_changed)
            self._session_tree_cb = self._on_tree_changed
        except Exception:
            self._session_tree_cb = None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Status chip only — identity lives in the pane header (UI_CONTRACT §2).
        head = QHBoxLayout()
        head.addStretch()
        self._chip = QLabel("—")
        self._chip.setStyleSheet(
            f"color: {_s.FG_DIM}; font-family: Consolas; font-size: 10px;")
        head.addWidget(self._chip)
        root.addLayout(head)

        self._hint = QLabel("click = switch path · dbl-click = trace pane · hover = think")
        self._hint.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas;")
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        self._list = QListWidget()
        self._list.setFrameShape(QListWidget.NoFrame)
        self._list.setMouseTracking(True)
        self._list.setProperty("panelInset", True)      # shared inset style (theme QSS)
        self._list.setStyleSheet("font-family: Consolas; font-size: 10px;")
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._list.itemEntered.connect(self._on_item_entered)
        self._list.viewport().installEventFilter(self)
        root.addWidget(self._list, stretch=1)

        bottom = QHBoxLayout()
        for label, slot in (("⟳ refresh", self.refresh),
                            ("⌂ go to active", self._scroll_to_active),
                            ("⊟ collapse old", self._toggle_collapse)):
            btn = MonoButton(label)
            btn.setFixedHeight(24)
            btn.clicked.connect(slot)
            bottom.addWidget(btn)
        bottom.addStretch(1)
        root.addLayout(bottom)

    # ── companion contract ────────────────────────────────────────────────

    def bind_controller(self, controller) -> None:
        self._controller = controller

    def closeEvent(self, event) -> None:
        """Unsubscribe from session_tree notifications on close (house insurance)."""
        if self._session_tree_cb is not None:
            try:
                from ui.pages import session_tree
                session_tree.unsubscribe(self._session_tree_cb)
            except Exception:
                pass
            self._session_tree_cb = None
        self._hover_timer.stop()
        self._hide_timer.stop()
        if self._popup is not None:
            self._popup.hide()
            self._popup.deleteLater()
            self._popup = None
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()

    def hideEvent(self, event) -> None:
        self._hover_timer.stop()
        self._hide_timer.stop()
        if self._popup is not None:
            self._popup.hide()
        super().hideEvent(event)

    # ── data plumbing ─────────────────────────────────────────────────────

    def _session(self) -> dict | None:
        provider = getattr(self._controller, "current_session_data", None)
        try:
            return provider() if callable(provider) else None
        except Exception:
            return None

    def _on_tree_changed(self, _op=None) -> None:
        # All tree mutations arrive on the Qt main thread today — engine events
        # are queued-signal marshalled before reaching here.  A future off-thread
        # writer must trampoline via QTimer.singleShot(0, self.refresh) instead
        # of calling refresh() directly.
        if self.isVisible():
            self.refresh()

    def refresh(self) -> None:
        try:
            self._refresh_inner()
        except Exception as exc:  # noqa: BLE001 — the viewer never crashes
            self._list.clear()
            self._list.addItem(QListWidgetItem(f"[error: {type(exc).__name__}: {exc}]"))

    def _refresh_inner(self) -> None:
        from ui.pages import session_tree
        if not session_tree.active():
            self._list.clear()
            self._chip.setText("—")
            self._hint.setText(
                "branch tree disabled — set MONOLITH_BRANCH_TREE_V1=1")
            return
        session = self._session()
        tree = (session or {}).get("tree")
        if not isinstance(tree, dict):
            self._list.clear()
            self._chip.setText("● 0 nodes")
            return
        fp = session_tree.fingerprint(session)
        if fp == self._fingerprint:
            return                          # rendered-state guard (MonoBase lesson)
        self._fingerprint = fp
        prev_scroll = self._list.verticalScrollBar().value()
        self._list.clear()
        rows = build_rows(tree)
        forks = sum(1 for r in rows if r.take and r.take[0] == 1)
        self._chip.setText(f"● {len(rows)} nodes · {forks} forks")
        hidden = 0
        for row in rows:
            if self._collapse_old and not row.on_path and row.take is None:
                hidden += 1
                continue
            item = QListWidgetItem(f"{row.glyph}{row.label}")
            item.setData(self.NODE_ROLE, row.node_id)
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            if row.on_path:
                item.setForeground(_ON_PATH_COLOR)
            self._list.addItem(item)
        if hidden:
            tail = QListWidgetItem(f"… {hidden} off-path rows hidden (⊟)")
            tail.setSizeHint(QSize(0, _ROW_HEIGHT))
            self._list.addItem(tail)
        bar = self._list.verticalScrollBar()
        bar.setValue(max(0, min(prev_scroll, bar.maximum())))

    # ── interactions ──────────────────────────────────────────────────────

    def _on_item_clicked(self, item) -> None:
        node_id = item.data(self.NODE_ROLE)
        if not node_id:
            return
        switch = getattr(self._controller, "switch_reasoning_path", None)
        if callable(switch):
            try:
                switch(node_id)
            except Exception:
                pass
        self.invalidate()

    def invalidate(self) -> None:
        """Public force-refresh: bust the fingerprint guard and rebuild. The seam
        external callers (session switch in chat.py) use instead of poking
        ``_fingerprint`` — matches the AUDIT-panel public-refresh precedent."""
        self._fingerprint = None
        self.refresh()

    def _on_item_double_clicked(self, item) -> None:
        node_id = item.data(self.NODE_ROLE)
        session = self._session()
        node = ((session or {}).get("tree", {}).get("nodes", {}) or {}).get(node_id)
        task_id = (node or {}).get("msg", {}).get("task_id", "")
        opener = getattr(self._controller, "open_reasoning_trace", None)
        if task_id and callable(opener):
            try:
                opener(task_id)
            except Exception:
                pass

    def _on_item_entered(self, item) -> None:
        self._hide_timer.stop()
        if self._popup is not None:
            self._popup.hide()
        self._hover_node = item.data(self.NODE_ROLE)
        self._hover_timer.start()

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        if event.type() in (QEvent.Leave, QEvent.MouseButtonPress):
            # Grace period: don't hide immediately — cursor may be moving INTO
            # the popup (positioned at cursor+12px).  _maybe_hide_popup checks
            # geometry before acting.
            self._hover_timer.stop()
            self._hide_timer.start()
        elif event.type() == QEvent.Wheel:
            # Wheel over the list = user is browsing rows, not reading the popup.
            # Cancel any pending show and schedule a graceful hide.
            self._hover_timer.stop()
            self._hide_timer.start()
        return super().eventFilter(obj, event)

    def _maybe_hide_popup(self, cursor_pos=None) -> None:
        """Hide the popup unless the cursor is currently inside its frame.

        ``cursor_pos`` can be supplied explicitly (useful in tests where
        QCursor.setPos may be unreliable in headless mode); defaults to the
        live cursor position.
        """
        if self._popup is None or not self._popup.isVisible():
            return
        pos = cursor_pos if cursor_pos is not None else QCursor.pos()
        if self._popup.frameGeometry().contains(pos):
            return          # cursor is inside the popup — leave it open
        self._popup.hide()

    def _popup_text_for(self, node_id: str) -> str | None:
        """Think text for the hover popup; None = popup suppressed (non-assistant)."""
        session = self._session()
        node = ((session or {}).get("tree", {}).get("nodes", {}) or {}).get(node_id)
        if not node or node.get("msg", {}).get("role") != "assistant":
            return None
        think = extract_think(str(node["msg"].get("text", "") or ""))
        return think if think else "(no think recorded)"

    def _show_hover_popup(self) -> None:
        if self._popup is None:
            return
        node_id = self._hover_node
        if not node_id:
            return
        text = self._popup_text_for(node_id)
        if text is None:
            return
        session = self._session()
        node = ((session or {}).get("tree", {}).get("nodes", {}) or {}).get(node_id, {})
        head = f"think · {node.get('origin', '?')} · {len(text)} chars"
        self._popup.present(head, text, QCursor.pos() + QPoint(12, 12))

    def _scroll_to_active(self) -> None:
        session = self._session()
        leaf = ((session or {}).get("tree") or {}).get("active_leaf")
        for r in range(self._list.count()):
            if self._list.item(r).data(self.NODE_ROLE) == leaf:
                self._list.scrollToItem(self._list.item(r))
                return

    def _toggle_collapse(self) -> None:
        self._collapse_old = not self._collapse_old
        self._fingerprint = None
        self.refresh()


__all__ = ("Row", "build_rows", "extract_think", "ReasoningTreePanel")
