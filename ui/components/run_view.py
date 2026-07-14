"""RunView — the single renderer for a Monoline run (Workshop Pane v2, spec 2026-06-14).

Renders a core.run_model.RunModel: header + one expandable row per block (status, kind,
timing, and on expand: derived inputs / outputs / verdict / detectors). Mounted live inline
in chat AND in the companion run browser — the ONLY Qt boundary over the pure model. It
subscribes to the model's plain observer callback and re-renders in place (no churn; expansion
survives updates). Emits sig_height_changed so a host chat row re-lays-out (row-sizing lesson).

Renders on the GUI thread: live events are marshalled to the UI thread before the builder
folds them, and the browser rehydrates on the UI thread.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

import core.style as _s

_DOT = {"pending": "◇", "running": "▶", "done": "✓", "error": "✕", "stopped": "⦸"}
_CLIP = 400   # per-field clip in the expanded detail
_PREVIEW_CLIP = 160


def _tok(name: str, fallback: str) -> str:
    return getattr(_s, name, fallback)


class _BlockRow(QFrame):
    sig_toggled = Signal()

    def __init__(self, block, derived_inputs: dict):
        super().__init__()
        self._block = block
        self._inputs = derived_inputs or {}
        self._expanded = False
        self.setObjectName("run_block_row")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(3)
        self._head = QLabel()
        self._head.setTextFormat(Qt.TextFormat.PlainText)
        lay.addWidget(self._head)
        self._preview = QLabel()
        self._preview.setTextFormat(Qt.TextFormat.PlainText)
        self._preview.setStyleSheet(f"color:{_s.FG_DIM}; font-size:10px;")
        self._preview.setWordWrap(False)
        lay.addWidget(self._preview)
        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._detail.setTextFormat(Qt.TextFormat.PlainText)
        self._detail.setVisible(False)
        self._detail.setStyleSheet(f"color:{_s.FG_DIM}; font-size:10px; font-family:Consolas;")
        lay.addWidget(self._detail)
        self._render()

    def update_block(self, block, derived_inputs: dict) -> None:
        self._block = block
        self._inputs = derived_inputs or {}
        self._render()   # keeps _expanded

    def _render(self) -> None:
        b = self._block
        dur = b.duration_ms()
        timing = f"  {dur:.0f}ms" if dur is not None else ""
        self._head.setText(f"{_DOT.get(b.status, '◇')} {b.label} [{b.kind}]  {b.status}{timing}")
        color = "#ff6b6b" if b.status == "error" else _s.FG_TEXT
        self._head.setStyleSheet(f"color:{color}; font-size:11px;")
        preview = self._preview_text()
        self._preview.setText(preview)
        self._preview.setVisible(bool(preview))
        self._detail.setText(self._detail_text())
        accent = self._status_color(b.status)
        bg = "#221717" if b.status == "error" else _tok("BG_SURFACE_2", _s.BG_PANEL)
        self.setStyleSheet(
            f"#run_block_row {{ background:{bg}; border:1px solid {_s.BORDER_SUBTLE}; "
            f"border-left:3px solid {accent}; border-radius:5px; }}"
        )

    def _preview_text(self) -> str:
        b = self._block
        if b.error:
            return f"error: {str(b.error)[:_PREVIEW_CLIP]}"
        if b.outputs:
            key, value = next(iter(b.outputs.items()))
            return f"out.{key}: {str(value)[:_PREVIEW_CLIP]}"
        if self._inputs:
            key, value = next(iter(self._inputs.items()))
            return f"in.{key}: {str(value)[:_PREVIEW_CLIP]}"
        return ""

    def _detail_text(self) -> str:
        b = self._block
        parts: list[str] = []
        for k, v in (self._inputs or {}).items():
            parts.append(f"in.{k}: {str(v)[:_CLIP]}")
        for k, v in (b.outputs or {}).items():
            parts.append(f"out.{k}: {str(v)[:_CLIP]}")
        if b.error:
            parts.append(f"error: {b.error}")
        if b.verdict:
            parts.append(f"verdict: {b.verdict.get('verdict', '')}")
        if b.detectors:
            parts.append("detectors: " + ",".join(str(d.get("kind", "")) for d in b.detectors))
        return "\n".join(parts)

    def status(self) -> str:
        return self._block.status

    def detail_text(self) -> str:
        return self._detail_text()

    def preview_text(self) -> str:
        return self._preview_text()

    def _status_color(self, status: str) -> str:
        if status == "done":
            return "#66b87a"
        if status == "running":
            return _s.ACCENT_PRIMARY
        if status == "error":
            return "#ff6b6b"
        if status == "stopped":
            return "#d4a657"   # neutral amber — a clean halt, not an error
        return _s.FG_DIM

    def set_expanded(self, on: bool) -> None:
        self._expanded = bool(on)
        self._detail.setVisible(self._expanded)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.set_expanded(not self._expanded)
        self.sig_toggled.emit()
        super().mouseReleaseEvent(event)


class _ClickFrame(QFrame):
    """A header bar that toggles the collapsible run card on click."""
    sig_clicked = Signal()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.sig_clicked.emit()
        super().mouseReleaseEvent(event)


class RunView(QWidget):
    sig_height_changed = Signal()

    def __init__(self, model=None, parent=None, *, show_final: bool = True,
                 collapsible: bool = False):
        super().__init__(parent)
        self.setObjectName("run_view")
        self._model = None
        self._show_final = show_final   # inline-in-chat passes False: the bubble carries the answer
        # collapsible: the whole run renders as ONE "tool block" (a clickable header that
        # encompasses the workflow); small by default, toggle to reveal the inner block rows.
        # The browser mount stays non-collapsible (it IS the inspector).
        self._collapsible = collapsible
        self._collapsed = collapsible
        self._rows: dict = {}   # block_id -> _BlockRow
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(3)
        self._header = QLabel()
        self._header.setStyleSheet(f"color:{_s.FG_TEXT}; font-weight:600;")
        if collapsible:
            self._header_host = _ClickFrame()
            self._header_host.setObjectName("run_header")
            self._header_host.setCursor(Qt.CursorShape.PointingHandCursor)
            hh = QHBoxLayout(self._header_host)
            hh.setContentsMargins(8, 6, 8, 6)
            hh.addWidget(self._header, 1)
            self._header_host.sig_clicked.connect(self.toggle_collapsed)
            self._header_host.setStyleSheet(
                f"#run_header {{ background:{_tok('BG_SURFACE_2', _s.BG_PANEL)}; "
                f"border:1px solid {_s.BORDER_SUBTLE}; border-radius:5px; }}")
            root.addWidget(self._header_host)
        else:
            root.addWidget(self._header)
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        root.addWidget(self._rows_host)
        self._rows_host.setVisible(not self._collapsed)
        self._final = QLabel()
        self._final.setWordWrap(True)
        self._final.setVisible(False)
        self._final.setStyleSheet(f"color:{_s.FG_TEXT}; font-size:11px;")
        root.addWidget(self._final)
        if model is not None:
            self.bind(model)

    def bind(self, model) -> None:
        """Bind (or re-bind) to a RunModel. Re-binding to the same model just re-renders;
        switching models moves the subscription (the browser re-binds as the user picks runs)."""
        if model is self._model:
            self._render()
            return
        if self._model is not None:
            self._model.unsubscribe(self._on_model_changed)
        self._model = model
        model.subscribe(self._on_model_changed)
        self._render()

    def _on_model_changed(self, m) -> None:
        if m is not self._model:
            return  # stale subscription from a model we've since re-bound away from
        self._render()

    def _render(self) -> None:
        m = self._model
        if m is None:
            return
        for blk in m.block_list():
            row = self._rows.get(blk.id)
            if row is None:
                row = _BlockRow(blk, m.inputs_for(blk.id))
                row.sig_toggled.connect(self._emit_height)
                self._rows[blk.id] = row
                self._rows_layout.addWidget(row)
            else:
                row.update_block(blk, m.inputs_for(blk.id))
        n = len(m.block_list())
        summary = f"{m.name} · {m.status} · {n} blocks"
        if self._collapsible:
            summary = ("▾ " if not self._collapsed else "▸ ") + summary
        self._header.setText(summary)
        if m.final_output and self._show_final:
            self._final.setText(f"→ {m.final_output[:1000]}")
            self._final.setVisible(True)
        elif m.error:   # errors always show (no chat bubble carries them)
            self._final.setText(f"error: {m.error}")
            self._final.setVisible(True)
        self.updateGeometry()
        self.sig_height_changed.emit()

    def _emit_height(self) -> None:
        self.updateGeometry()
        self.sig_height_changed.emit()

    # -- collapse (the whole run as one "tool block") --
    def toggle_collapsed(self) -> None:
        if not self._collapsible:
            return
        self._collapsed = not self._collapsed
        self._rows_host.setVisible(not self._collapsed)
        if self._model is not None:
            self._render()        # refresh the chevron + summary
        self._emit_height()

    def is_collapsed(self) -> bool:
        return self._collapsed

    def rows_visible(self) -> bool:
        return self._rows_host.isVisible()

    # -- introspection --
    def block_row_count(self) -> int:
        return len(self._rows)

    def header_text(self) -> str:
        return self._header.text()

    def row_status(self, block_id: str) -> str:
        row = self._rows.get(block_id)
        return row.status() if row is not None else ""

    def expand_row(self, block_id: str) -> None:
        row = self._rows.get(block_id)
        if row is not None:
            row.set_expanded(True)
            self._emit_height()

    def row_detail_text(self, block_id: str) -> str:
        row = self._rows.get(block_id)
        return row.detail_text() if row is not None else ""

    def row_preview_text(self, block_id: str) -> str:
        row = self._rows.get(block_id)
        return row.preview_text() if row is not None else ""

    def row_spacing(self) -> int:
        return self._rows_layout.spacing()

    def final_output_text(self) -> str:
        return self._final.text()
