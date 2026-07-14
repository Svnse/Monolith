"""CommandPicker — compact slash command autocomplete popup.

Follows the OmniBar/_OmniResult pattern: custom QFrame rows with accent
strip, keyboard navigation (Up/Down/Tab/Enter/Esc), and click selection.

Two modes:
  - list: filtered command matches (max 6 visible, scrollable)
  - hint: single command's arg hint (when full command typed)
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import core.style as _s

_MAX_VISIBLE = 8
_ROW_HEIGHT = 22


class CommandPickerRow(QFrame):
    clicked = Signal(int)
    hovered = Signal(int)

    def __init__(self, index: int, name: str, args: str, desc: str, parent=None):
        super().__init__(parent)
        self._index = index
        self._name = name
        self.setObjectName("cmd_picker_row")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(_ROW_HEIGHT)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 6, 0)
        row.setSpacing(0)

        self._accent = QFrame()
        self._accent.setFixedWidth(2)
        self._accent.setStyleSheet("background: transparent; border: none;")
        row.addWidget(self._accent)
        row.addSpacing(5)

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 9px; "
            f"font-family: Consolas, monospace; background: transparent;"
        )
        row.addWidget(name_lbl)

        if args:
            row.addSpacing(6)
            args_lbl = QLabel(args)
            args_lbl.setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 8px; "
                f"font-family: Consolas, monospace; background: transparent;"
            )
            row.addWidget(args_lbl)

        self._connector = QFrame()
        self._connector.setFixedHeight(1)
        self._connector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._connector.setStyleSheet("background: transparent; border: none;")
        row.addWidget(self._connector, 1)

        if desc:
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 8px; background: transparent;"
            )
            row.addWidget(desc_lbl)

    def set_selected_visual(self, selected: bool) -> None:
        accent = _s.ACCENT_PRIMARY if selected else "transparent"
        self._accent.setStyleSheet(f"background: {accent}; border: none;")
        connector = _s.FG_DIM if selected else "transparent"
        self._connector.setStyleSheet(f"background: {connector}; border: none;")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._index)
            event.accept()
            return
        super().mousePressEvent(event)

    def enterEvent(self, event) -> None:
        self.hovered.emit(self._index)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.set_selected_visual(False)
        super().leaveEvent(event)


class CommandPicker(QFrame):
    sig_command_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("command_picker")
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            f"""
            QFrame#command_picker {{
                background: {_s.BG_PANEL};
                border: 1px solid {_s.ACCENT_PRIMARY};
                border-radius: 3px;
            }}
            """
        )

        self._rows: list[CommandPickerRow] = []
        self._selected_index: int = -1
        self._mode: str = "list"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            f"QScrollBar:vertical {{ width: 4px; background: transparent; }}"
            f"QScrollBar::handle:vertical {{ background: {_s.FG_DIM}; border-radius: 2px; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        self._scroll.setMaximumHeight(_MAX_VISIBLE * _ROW_HEIGHT + 8)

        self._inner = QWidget()
        self._inner.setStyleSheet("background: transparent;")
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(1)
        self._inner_layout.addStretch()

        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)
        self.hide()

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._inner_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        self._selected_index = -1

    def update_matches(self, matches: list[tuple[str, str, str]]) -> None:
        self._clear_rows()
        self._mode = "list"
        if not matches:
            self.hide()
            return
        for i, (name, args, desc) in enumerate(matches):
            row = CommandPickerRow(i, name, args, desc, self._inner)
            row.clicked.connect(self._on_row_clicked)
            row.hovered.connect(self._set_selected_index)
            self._inner_layout.insertWidget(i, row)
            self._rows.append(row)
        self._set_selected_index(0)
        visible_count = min(len(self._rows), _MAX_VISIBLE)
        self._scroll.setMaximumHeight(visible_count * _ROW_HEIGHT + 8)

    def show_arg_hint(self, name: str, args: str, desc: str) -> None:
        self._clear_rows()
        self._mode = "hint"
        hint = QFrame(self._inner)
        hint.setObjectName("cmd_picker_row")
        hint.setFixedHeight(_ROW_HEIGHT)
        layout = QHBoxLayout(hint)
        layout.setContentsMargins(5, 0, 6, 0)
        layout.setSpacing(6)

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 9px; "
            f"font-family: Consolas, monospace; background: transparent;"
        )
        layout.addWidget(name_lbl)

        if args:
            args_lbl = QLabel(args)
            args_lbl.setStyleSheet(
                f"color: {_s.FG_TEXT}; font-size: 9px; "
                f"font-family: Consolas, monospace; background: transparent;"
            )
            args_lbl.setWordWrap(True)
            layout.addWidget(args_lbl)

        layout.addStretch()

        if desc:
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 8px; background: transparent;"
            )
            layout.addWidget(desc_lbl)

        self._inner_layout.insertWidget(0, hint)
        self._rows.append(hint)  # type: ignore[arg-type]
        self._scroll.setMaximumHeight(_ROW_HEIGHT + 6)

    def select_next(self) -> None:
        if self._mode != "list" or not self._rows:
            return
        self._set_selected_index(self._selected_index + 1)

    def select_prev(self) -> None:
        if self._mode != "list" or not self._rows:
            return
        self._set_selected_index(self._selected_index - 1)

    def accept_selected(self) -> str | None:
        if self._mode != "list" or not self._rows or self._selected_index < 0:
            return None
        row = self._rows[self._selected_index]
        if isinstance(row, CommandPickerRow):
            return row._name
        return None

    def _set_selected_index(self, index: int) -> None:
        if not self._rows:
            self._selected_index = -1
            return
        self._selected_index = max(0, min(index, len(self._rows) - 1))
        for i, row in enumerate(self._rows):
            selected = i == self._selected_index
            bg = _s.BG_BUTTON_HOVER if selected else "transparent"
            row.setStyleSheet(
                f"QFrame#cmd_picker_row {{ background: {bg}; border-radius: 3px; }}"
            )
            if hasattr(row, "set_selected_visual"):
                row.set_selected_visual(selected)
        # Ensure selected row is visible in scroll area
        if self._selected_index >= 0 and self._selected_index < len(self._rows):
            self._scroll.ensureWidgetVisible(self._rows[self._selected_index])

    def _on_row_clicked(self, index: int) -> None:
        if 0 <= index < len(self._rows):
            row = self._rows[index]
            if isinstance(row, CommandPickerRow):
                self.sig_command_selected.emit(row._name)

    def hide(self) -> None:
        self._selected_index = -1
        super().hide()
