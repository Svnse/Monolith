"""QuestionPanel — UI for the ask_user clarifying-question tool.

Renders a question with 2-4 option buttons (single-select) or 2-4 checkboxes
+ SUBMIT button (multi_select). When the user picks, emits sig_answered with
the question_id and the chosen label(s). Mirror of ActionReviewPanel pattern
used for /approve, intentionally similar so future maintainers recognize the
shape.

Lifecycle:
  * show_question(payload) — render and show the panel for a pending question
  * clear() — reset state and hide (called after answer is delivered)
  * sig_answered(question_id, answers) — emitted on button-click or SUBMIT

`answers` is always a list[str] of chosen labels:
  * single-select: list of exactly one label
  * multi_select: list of zero or more labels (zero allowed only when the
    user explicitly clicks SUBMIT with nothing checked — caller decides
    whether to accept or reject empty)
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import core.style as _s


class QuestionPanel(QWidget):
    sig_answered = Signal(str, list)  # (question_id, [chosen labels])
    sig_dismissed = Signal(str)        # (question_id) — user closed without answering

    def __init__(self, parent=None):
        super().__init__(parent)
        self._question_id: str | None = None
        self._multi_select: bool = False
        self._checkboxes: list[QCheckBox] = []
        self._buttons_layout: QVBoxLayout | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        panel = QFrame()
        panel.setObjectName("question_panel")
        panel.setStyleSheet(
            f"""
            QFrame#question_panel {{
                background: {_s.BG_SURFACE_1};
                border: none;
                border-radius: 8px;
            }}
            """
        )
        self._panel_layout = QVBoxLayout(panel)
        self._panel_layout.setContentsMargins(12, 12, 12, 12)
        self._panel_layout.setSpacing(8)

        # Header row: title + optional header chip + close
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self._lbl_title = QLabel("CLARIFICATION NEEDED")
        self._lbl_title.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 10px; font-family: Consolas; font-weight: bold;"
        )
        header.addWidget(self._lbl_title)

        self._lbl_chip = QLabel("")
        self._lbl_chip.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; "
            f"border: 1px solid {_s.FG_DIM}; border-radius: 3px; padding: 1px 6px;"
        )
        self._lbl_chip.hide()
        header.addWidget(self._lbl_chip)

        header.addStretch()

        self._close = QPushButton("DISMISS")
        self._close.setCursor(Qt.PointingHandCursor)
        self._close.setFlat(True)
        self._close.setStyleSheet(
            f"QPushButton {{ color: {_s.FG_DIM}; border: none; background: transparent; font-size: 9px; }}"
            f"QPushButton:hover {{ color: {_s.FG_TEXT}; }}"
        )
        self._close.clicked.connect(self._on_dismiss)
        header.addWidget(self._close)
        self._panel_layout.addLayout(header)

        # Question text
        self._lbl_question = QLabel("")
        self._lbl_question.setWordWrap(True)
        self._lbl_question.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 11px; font-family: Consolas;"
        )
        self._panel_layout.addWidget(self._lbl_question)

        # Container for option widgets (rebuilt each time)
        self._options_container = QWidget()
        self._options_layout = QVBoxLayout(self._options_container)
        self._options_layout.setContentsMargins(0, 4, 0, 0)
        self._options_layout.setSpacing(6)
        self._panel_layout.addWidget(self._options_container)

        # Submit row (used only for multi_select)
        self._submit_row = QHBoxLayout()
        self._submit_row.setContentsMargins(0, 4, 0, 0)
        self._submit_row.setSpacing(8)
        self._submit_row.addStretch()
        self._submit_btn = QPushButton("SUBMIT")
        self._submit_btn.setCursor(Qt.PointingHandCursor)
        self._submit_btn.setFixedHeight(24)
        self._submit_btn.setStyleSheet(self._primary_button_style())
        self._submit_btn.clicked.connect(self._on_submit_multi)
        self._submit_row.addWidget(self._submit_btn)
        self._submit_row_widget = QWidget()
        self._submit_row_widget.setLayout(self._submit_row)
        self._panel_layout.addWidget(self._submit_row_widget)
        self._submit_row_widget.hide()

        root.addWidget(panel)
        self.hide()

    # ── styling helpers ─────────────────────────────────────────────

    def _primary_button_style(self) -> str:
        return (
            f"QPushButton {{ background: transparent; color: {_s.ACCENT_PRIMARY}; "
            f"border: 1px solid {_s.ACCENT_PRIMARY}; border-radius: 6px; "
            f"padding: 6px 12px; font-size: 10px; font-family: Consolas; "
            f"text-align: left; }} "
            f"QPushButton:hover {{ background: {_s.ACCENT_PRIMARY}22; }}"
        )

    def _checkbox_style(self) -> str:
        return (
            f"QCheckBox {{ color: {_s.FG_TEXT}; font-size: 10px; font-family: Consolas; "
            f"spacing: 8px; padding: 2px 0; }}"
        )

    # ── public API ──────────────────────────────────────────────────

    def show_question(self, payload: dict) -> None:
        """Render the question panel from a payload from ask_user executor."""
        self._clear_options()
        self._question_id = str(payload.get("question_id") or "")
        question = str(payload.get("question") or "")
        header = payload.get("header")
        options = payload.get("options") or []
        self._multi_select = bool(payload.get("multi_select", False))

        self._lbl_question.setText(question)
        if header:
            self._lbl_chip.setText(str(header).upper())
            self._lbl_chip.show()
        else:
            self._lbl_chip.hide()

        if self._multi_select:
            self._build_multi_select(options)
            self._submit_row_widget.show()
        else:
            self._build_single_select(options)
            self._submit_row_widget.hide()

        self.show()

    def clear(self) -> None:
        self._question_id = None
        self._multi_select = False
        self._lbl_question.clear()
        self._lbl_chip.hide()
        self._clear_options()
        self._submit_row_widget.hide()
        self.hide()

    # ── option rendering ────────────────────────────────────────────

    def _build_single_select(self, options: list[dict]) -> None:
        """One button per option; click → emit answered + clear."""
        for opt in options:
            label = str(opt.get("label") or "").strip()
            description = str(opt.get("description") or "").strip()
            if not label:
                continue
            btn = self._make_option_button(label, description)
            btn.clicked.connect(lambda _checked=False, lab=label: self._on_single_pick(lab))
            self._options_layout.addWidget(btn)

    def _build_multi_select(self, options: list[dict]) -> None:
        """Checkboxes; SUBMIT collects all checked + emits answered."""
        self._checkboxes = []
        for opt in options:
            label = str(opt.get("label") or "").strip()
            description = str(opt.get("description") or "").strip()
            if not label:
                continue
            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            cb = QCheckBox(label)
            cb.setStyleSheet(self._checkbox_style())
            cb.setCursor(Qt.PointingHandCursor)
            self._checkboxes.append(cb)
            v.addWidget(cb)
            if description:
                desc_lbl = QLabel(f"    {description}")
                desc_lbl.setWordWrap(True)
                desc_lbl.setStyleSheet(
                    f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; padding-left: 22px;"
                )
                v.addWidget(desc_lbl)
            self._options_layout.addWidget(container)

    def _make_option_button(self, label: str, description: str) -> QPushButton:
        text = label if not description else f"{label}\n  {description}"
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(self._primary_button_style())
        btn.setMinimumHeight(32)
        return btn

    def _clear_options(self) -> None:
        # Remove all widgets currently in the options layout
        while self._options_layout.count():
            item = self._options_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._checkboxes = []

    # ── click handlers ──────────────────────────────────────────────

    def _on_single_pick(self, label: str) -> None:
        if self._question_id is None:
            return
        qid = self._question_id
        self.clear()
        self.sig_answered.emit(qid, [label])

    def _on_submit_multi(self) -> None:
        if self._question_id is None:
            return
        chosen = [cb.text() for cb in self._checkboxes if cb.isChecked()]
        qid = self._question_id
        self.clear()
        self.sig_answered.emit(qid, chosen)

    def _on_dismiss(self) -> None:
        if self._question_id is None:
            self.clear()
            return
        qid = self._question_id
        self.clear()
        self.sig_dismissed.emit(qid)
