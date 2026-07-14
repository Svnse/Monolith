"""MonoSearch companion panel — the human's read-only window into the
MonoSearch self-knowledge tool.

This panel is a thin VIEWER over the in-process ``core.monosearch.service``.
It owns no data and writes nothing — it renders whatever the service returns
for the self-directed dashboard modes (Failing / Recurring / Pulling /
Unresolved) and for free-text Search. Refresh is on-demand (a mode click or
the ⟳ button); there is deliberately NO aggressive QTimer auto-refresh — the
MonoBase dev panel taught us that a 2s tick churns scroll position and
selection out from under the reader.

Hosted inside the companion pane (typical width 360-600px), so it never calls
self.resize(). Every service call is wrapped in try/except: a failure renders a
single "[error: ...]" row and never crashes the panel.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

import core.style as _s
from ui.components.atoms import MonoButton

__all__ = ("MonoSearchPanel",)


# Dashboard modes (mode-button label -> internal key). Search is handled
# separately because it reveals the query row.
_MODES = ("Failing", "Recurring", "Pulling", "Unresolved")

# The Search source dropdown. These are the friendly aliases the router's
# resolve_source() already understands (SOURCE_ALIASES) — "all" resolves to
# None (fan out to every adapter), the rest scope to a single store.
_SOURCES = (
    "all",
    "faults",
    "knowledge",
    "conversation",
    "turns",
    "memory",
    "bearing",
    "identity",
    "curiosity",
)

_DEFAULT_LIMIT = 10


class MonoSearchPanel(QWidget):
    """Read-only dashboard + search surface for MonoSearch.

    A row of mode buttons (Failing / Recurring / Pulling / Unresolved /
    Search) runs the corresponding query and lists the results. Search mode
    additionally reveals a query row (text + source combo + Go). A bottom row
    carries a ⟳ refresh button (re-runs the current mode) and a "<N> sources"
    label sourced from the live adapter registry.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MonoSearch")
        # NOTE: do NOT call self.resize() — this widget is hosted inside the
        # companion pane, which owns its width. (Matches MonoBaseDevPanel.)

        # Current dashboard mode key. "Search" is a distinct mode that shows
        # the query row; the four dashboard modes hide it.
        self._mode: str = "Failing"
        self._limit: int = self._resolve_limit()

        self._build_ui()
        # Default mode on open = Failing.
        self._run_mode("Failing")

    # ── limit resolution ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_limit() -> int:
        """Per-profile result count, with a hard fallback. Wrapped so a rename
        or import error degrades to a sane default instead of breaking
        construction."""
        try:
            from core.context_profiles import active_context_profile
            value = int(active_context_profile().monosearch_result_count)
            return value if value > 0 else _DEFAULT_LIMIT
        except Exception:
            return _DEFAULT_LIMIT

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── MODE BUTTONS ────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self._mode_buttons: dict[str, MonoButton] = {}
        for label in (*_MODES, "Search"):
            btn = MonoButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.clicked.connect(lambda _checked=False, m=label: self._on_mode_clicked(m))
            mode_row.addWidget(btn)
            self._mode_buttons[label] = btn
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        # ── SEARCH ROW (hidden unless Search mode is active) ────────────
        self._search_row = QWidget()
        search_layout = QHBoxLayout(self._search_row)
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(4)

        self._query_edit = QLineEdit()
        self._query_edit.setPlaceholderText("search query…")
        self._query_edit.setStyleSheet(
            f"QLineEdit {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT};"
            f" border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 3px;"
            f" padding: 3px 6px; font-family: Consolas; font-size: 10px; }}"
        )
        self._query_edit.returnPressed.connect(self._run_search)
        search_layout.addWidget(self._query_edit, 1)

        self._source_combo = QComboBox()
        self._source_combo.addItems(list(_SOURCES))
        self._source_combo.setStyleSheet(
            f"QComboBox {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT};"
            f" border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 3px;"
            f" padding: 3px 6px; font-family: Consolas; font-size: 10px; }}"
            f"QComboBox QAbstractItemView {{ background: {_s.BG_INPUT};"
            f" color: {_s.FG_TEXT}; selection-background-color: {_s.ACCENT_PRIMARY};"
            f" selection-color: {_s.BG_MAIN}; }}"
        )
        self._source_combo.setToolTip("Scope the search to one store, or 'all'.")
        search_layout.addWidget(self._source_combo)

        self._go_btn = MonoButton("Go", accent=True)
        self._go_btn.setFixedHeight(24)
        self._go_btn.clicked.connect(self._run_search)
        search_layout.addWidget(self._go_btn)

        self._search_row.setVisible(False)
        root.addWidget(self._search_row)

        # ── RESULTS LIST ────────────────────────────────────────────────
        self._results = QListWidget()
        self._results.setFrameShape(QListWidget.NoFrame)
        self._results.setStyleSheet(
            f"QListWidget {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_SUBTLE};"
            f" border-radius: 3px; color: {_s.FG_TEXT}; font-family: Consolas;"
            f" font-size: 10px; }}"
            f"QListWidget::item {{ padding: 4px 6px; }}"
            f"QListWidget::item:selected {{ background: {_s.ACCENT_PRIMARY}; color: {_s.BG_MAIN}; }}"
        )
        root.addWidget(self._results, stretch=1)

        # ── BOTTOM ROW: refresh + sources count ─────────────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        self._refresh_btn = MonoButton("⟳ refresh")
        self._refresh_btn.setFixedHeight(24)
        self._refresh_btn.setToolTip("Re-run the current mode.")
        self._refresh_btn.clicked.connect(self._refresh)
        bottom.addWidget(self._refresh_btn)
        bottom.addStretch(1)

        self._sources_label = QLabel(self._sources_text())
        self._sources_label.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; padding: 2px 4px;"
        )
        bottom.addWidget(self._sources_label)
        root.addLayout(bottom)

    def _sources_text(self) -> str:
        try:
            from core.monosearch import registry
            n = len(registry.all_adapters())
        except Exception:
            n = 0
        return f"{n} sources"

    # ── mode dispatch ────────────────────────────────────────────────────

    def _on_mode_clicked(self, mode: str) -> None:
        if mode == "Search":
            self._enter_search_mode()
        else:
            self._run_mode(mode)

    def _set_checked(self, active: str) -> None:
        for label, btn in self._mode_buttons.items():
            btn.setChecked(label == active)

    def _enter_search_mode(self) -> None:
        """Reveal the query row and run a search with the current query (if
        any). Empty query is fine — it lists recent rows for the chosen
        source."""
        self._mode = "Search"
        self._set_checked("Search")
        self._search_row.setVisible(True)
        self._run_search()

    def _run_mode(self, mode: str) -> None:
        """Run one of the four dashboard modes and render its rows."""
        self._mode = mode
        self._set_checked(mode)
        self._search_row.setVisible(False)
        if mode == "Failing":
            self._render_salience(self._call("failing"))
        elif mode == "Recurring":
            self._render_salience(self._call("recurring"))
        elif mode == "Pulling":
            self._render_records(self._call("pulling"), fmt=self._fmt_claim)
        elif mode == "Unresolved":
            self._render_records(self._call("unresolved"), fmt=self._fmt_claim)

    def _refresh(self) -> None:
        """Re-run whatever mode is active. Also refreshes the sources count."""
        self._sources_label.setText(self._sources_text())
        if self._mode == "Search":
            self._run_search()
        else:
            self._run_mode(self._mode)

    # ── service calls (every call isolated) ──────────────────────────────

    def _call(self, method: str):
        """Invoke a zero/limit-arg service dashboard method, returning its list
        or an error sentinel string the renderer turns into a single row."""
        try:
            from core.monosearch import service
            fn = getattr(service, method)
            return fn(self._limit)
        except Exception as exc:
            return f"[error: {type(exc).__name__}: {exc}]"

    def _run_search(self) -> None:
        self._set_checked("Search")
        query = self._query_edit.text().strip()
        source = self._source_combo.currentText()
        # "all" -> no source filter (router fans out to every adapter).
        filters = {} if source == "all" else {"source": source}
        try:
            from core.monosearch import service
            rows = service.search(query, filters, self._limit)
        except Exception as exc:
            self._render_error(f"[error: {type(exc).__name__}: {exc}]")
            return
        self._render_records(rows, fmt=self._fmt_search)

    # ── rendering ────────────────────────────────────────────────────────

    def _render_error(self, message: str) -> None:
        self._results.clear()
        self._results.addItem(QListWidgetItem(message))

    def _render_salience(self, rows) -> None:
        """Render failing/recurring dicts. Each dict: recurrence_key, source,
        count, salience."""
        self._results.clear()
        if isinstance(rows, str):  # error sentinel from _call
            self._results.addItem(QListWidgetItem(rows))
            return
        if not rows:
            self._results.addItem(QListWidgetItem("(no results)"))
            return
        for d in rows:
            try:
                key = str(d.get("recurrence_key", "?"))
                count = int(d.get("count", 0))
                source = str(d.get("source", "?"))
                salience = float(d.get("salience", 0.0))
                bar = self._salience_bar(salience)
                line = f"{key:<32} ×{count}  [{source}]  {bar}"
            except Exception as exc:
                line = f"[error: {type(exc).__name__}: {exc}]"
            self._results.addItem(QListWidgetItem(line))

    def _render_records(self, rows, *, fmt) -> None:
        """Render a list[Record] using the given per-row formatter."""
        self._results.clear()
        if isinstance(rows, str):  # error sentinel from _call
            self._results.addItem(QListWidgetItem(rows))
            return
        if not rows:
            self._results.addItem(QListWidgetItem("(no results)"))
            return
        for rec in rows:
            try:
                line = fmt(rec)
            except Exception as exc:
                line = f"[error: {type(exc).__name__}: {exc}]"
            item = QListWidgetItem(line)
            self._results.addItem(item)

    @staticmethod
    def _fmt_claim(rec) -> str:
        """Pulling / Unresolved row: the claim text, with tier + provenance
        prefixed for context."""
        tier = rec.evidence_tier.name
        prov = rec.provenance.value
        return f"[{tier}/{prov}] {rec.text}"

    @staticmethod
    def _fmt_search(rec) -> str:
        return f"{rec.namespaced_id}  [{rec.source}]  {rec.text[:60]}"

    @staticmethod
    def _salience_bar(salience: float, width: int = 10) -> str:
        """A tiny ▓ bar scaled to salience. salience is count×decay, so it can
        exceed 1 — clamp the bar to `width`."""
        try:
            n = int(max(0.0, min(float(salience), float(width))))
        except Exception:
            n = 0
        return "▓" * n
