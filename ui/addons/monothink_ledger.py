"""Monothink Ledger companion panel — the human's git-like undo surface for the
self-evolving reasoning scaffold.

A read-mostly viewer over ``core.monothink.list_ledger()``: each evolution shows as
a row (tag · rating · diff size · state), and every APPLIED version carries a
"Revert" button that restores the scaffold to that exact snapshot via
``core.monothink.revert_to_version()``. The revert is append-only — it is itself
recorded in the ledger, never rewriting history — so the panel is a safe net for
the autonomous training loop: any change the frontier rater drove can be undone
with one click.

Hosted in the companion pane (so it never calls self.resize()). Refresh is
on-demand (a Revert action or the ⟳ button) — no aggressive QTimer, which the
MonoBase dev panel taught us churns scroll/selection. Every backend call is
wrapped so a failure renders an error row and never crashes the panel.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

import core.style as _s
from ui.components.atoms import MonoButton

__all__ = ("MonothinkLedgerPanel",)

_ROW_HEIGHT = 30
_DEFAULT_LIMIT = 50


class MonothinkLedgerPanel(QWidget):
    """Scrollable evolution-ledger with per-version Revert buttons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Monothink Ledger")
        self._build_ui()
        self._reload()

    # ── construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        title = QLabel("Monothink Ledger")
        title.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 11px; font-weight: bold;"
            " font-family: Consolas; padding: 2px 4px;"
        )
        root.addWidget(title)

        self._hint = QLabel("")
        self._hint.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; padding: 0 4px;"
        )
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        self._list = QListWidget()
        self._list.setFrameShape(QListWidget.NoFrame)
        self._list.setStyleSheet(
            f"QListWidget {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_SUBTLE};"
            f" border-radius: 3px; color: {_s.FG_TEXT}; font-family: Consolas; font-size: 10px; }}"
            f"QListWidget::item {{ border-bottom: 1px solid {_s.BORDER_SUBTLE}; }}"
        )
        root.addWidget(self._list, stretch=1)

        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        refresh_btn = MonoButton("⟳ refresh")
        refresh_btn.setFixedHeight(24)
        refresh_btn.clicked.connect(self._reload)
        bottom.addWidget(refresh_btn)

        undo_btn = MonoButton("↩ undo last")
        undo_btn.setFixedHeight(24)
        undo_btn.clicked.connect(self._on_undo_last)
        bottom.addWidget(undo_btn)
        bottom.addStretch(1)
        root.addLayout(bottom)

    # ── data ──────────────────────────────────────────────────────────────

    def _reload(self) -> None:
        """Fetch the ledger and rebuild the list. Preserves scroll position."""
        prev_scroll = self._list.verticalScrollBar().value()
        try:
            from core.monothink import list_ledger
            rows = list_ledger(limit=_DEFAULT_LIMIT)
        except Exception as exc:  # noqa: BLE001 — a viewer must never crash
            self._list.clear()
            self._list.addItem(QListWidgetItem(f"[error: {type(exc).__name__}: {exc}]"))
            return

        self._list.clear()
        if not rows:
            self._hint.setText("No evolutions yet. Monothink writes a row here each time a "
                               "rating drives a scaffold edit.")
            self._list.addItem(QListWidgetItem("(ledger empty)"))
            return

        applied = sum(1 for r in rows if r.get("applied"))
        self._hint.setText(f"{len(rows)} entries · {applied} applied · click Revert to restore "
                           "any version (append-only — revert is itself recorded).")
        for row in rows:
            self._add_row(row)

        bar = self._list.verticalScrollBar()
        bar.setValue(max(0, min(prev_scroll, bar.maximum())))

    def _add_row(self, row: dict) -> None:
        turn_id = str(row.get("turn_id") or "?")
        tag = row.get("tag") or "—"
        rating = row.get("rating_value")
        diff = row.get("diff_chars")
        kind = row.get("kind") or "evolution"
        is_current = bool(row.get("is_current"))
        revertable = bool(row.get("revertable"))

        bits = [f"[{tag}]"]
        if isinstance(rating, int):
            bits.append(f"★{rating}")
        if isinstance(diff, int):
            bits.append(f"Δ{diff}")
        if kind == "rollback":
            bits.append(f"↩ revert→{str(row.get('reverted_to') or '')[:10]}")
        elif kind == "bootstrap":
            bits.append("· origin")
        elif not row.get("applied"):
            reason = str(row.get("reject_reason") or "rejected").split(":", 1)[0]
            bits.append(f"✕ {reason}")
        if is_current:
            bits.append("● CURRENT")
        label_text = "  ".join(bits)

        roww = QWidget()
        lay = QHBoxLayout(roww)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(6)
        lbl = QLabel(label_text)
        color = _s.ACCENT_PRIMARY if is_current else _s.FG_TEXT
        if not row.get("applied"):
            color = _s.FG_DIM
        lbl.setStyleSheet(f"color: {color}; font-family: Consolas; font-size: 10px;")
        lay.addWidget(lbl, stretch=1)

        if revertable and not is_current:
            btn = MonoButton("Revert")
            btn.setFixedHeight(20)
            # Default-arg capture: bind turn_id at loop time, not late-bound closure.
            btn.clicked.connect(lambda _checked=False, tid=turn_id: self._on_revert(tid))
            lay.addWidget(btn)

        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, turn_id)
        item.setSizeHint(QSize(0, _ROW_HEIGHT))  # fixed height — avoids QListWidget row-sizing churn
        self._list.addItem(item)
        self._list.setItemWidget(item, roww)

    # ── actions ───────────────────────────────────────────────────────────

    def _on_revert(self, turn_id: str) -> None:
        if (
            QMessageBox.question(
                self,
                "Revert scaffold",
                f"Restore the monothink scaffold to version {turn_id[:16]}?\n\n"
                "This is append-only — the revert is recorded in the ledger and can "
                "itself be undone.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._apply(lambda: __import__("core.monothink", fromlist=["revert_to_version"])
                    .revert_to_version(turn_id, reason="omnibar"), f"reverted to {turn_id[:12]}")

    def _on_undo_last(self) -> None:
        self._apply(lambda: __import__("core.monothink", fromlist=["rollback_last_apply"])
                    .rollback_last_apply(reason="omnibar_undo_last"), "undid last applied evolution")

    def _apply(self, action, ok_msg: str) -> None:
        try:
            result = action()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Revert failed", f"{type(exc).__name__}: {exc}")
            return
        if result is None:
            QMessageBox.information(self, "Nothing to revert",
                                    "No matching applied version to restore.")
            return
        self._reload()
