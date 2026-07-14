from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import core.style as s
from core.style import refresh_styles
from core.theme_config import save_theme_config
from core.themes import (
    apply_theme,
    current_theme_key,
    delete_custom_theme,
    editable_theme_fields,
    get_theme,
    is_builtin_theme,
    list_theme_entries,
    save_custom_theme,
    theme_to_dict,
)
from ui.bridge import UIBridge
from ui.components.atoms import MonoButton, MonoGroupBox


class ThemeModule(QWidget):
    """Theme editor module for built-in + user-defined palettes."""

    def __init__(self, ui_bridge: UIBridge | None = None):
        super().__init__()
        self._ui_bridge = ui_bridge
        self._updating = False
        self._field_inputs: dict[str, QLineEdit] = {}
        self._field_swatches: dict[str, QFrame] = {}
        self._dim_labels: list[QLabel] = []   # labels that need FG_DIM on refresh

        # No outer MonoGroupBox("THEME") — the companion pane already titles
        # the tab. Body is the preset row + name row + scrollable color grid.
        box_layout = QVBoxLayout(self)
        box_layout.setContentsMargins(10, 10, 10, 10)
        box_layout.setSpacing(8)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        # Real QComboBox replacing the previous read-only QLineEdit that was
        # pretending to be a combo. PREV/NEXT remain as keyboard-friendly
        # steppers; the dropdown is the primary affordance now.
        self.theme_combo = QComboBox()
        self.theme_combo.setPlaceholderText("No theme selected")
        self.theme_combo.setFixedHeight(26)
        self.btn_prev = MonoButton("◀")
        self.btn_next = MonoButton("▶")
        self.btn_load = MonoButton("LOAD")
        self.btn_apply = MonoButton("APPLY", accent=True)
        self.btn_apply.setFixedHeight(26)
        self.btn_prev.setFixedSize(28, 26)
        self.btn_next.setFixedSize(28, 26)
        self.btn_load.setFixedHeight(26)
        preset_row.addWidget(self.theme_combo, 1)
        preset_row.addWidget(self.btn_prev)
        preset_row.addWidget(self.btn_next)
        preset_row.addWidget(self.btn_load)
        preset_row.addWidget(self.btn_apply)
        box_layout.addLayout(preset_row)

        name_row = QHBoxLayout()
        lbl_name = QLabel("Name")
        self._dim_labels.append(lbl_name)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Custom theme name")
        self.btn_apply_draft = MonoButton("APPLY DRAFT")
        self.btn_save_custom = MonoButton("SAVE CUSTOM")
        self.btn_delete_custom = MonoButton("DELETE CUSTOM")
        for btn in (self.btn_apply_draft, self.btn_save_custom, self.btn_delete_custom):
            btn.setFixedHeight(26)
        name_row.addWidget(lbl_name)
        name_row.addWidget(self.name_input, 1)
        name_row.addWidget(self.btn_apply_draft)
        name_row.addWidget(self.btn_save_custom)
        name_row.addWidget(self.btn_delete_custom)
        box_layout.addLayout(name_row)

        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        fields = editable_theme_fields()
        for row, field_name in enumerate(fields):
            label = QLabel(field_name.replace("_", " ").title())
            self._dim_labels.append(label)

            inp = QLineEdit()
            inp.setPlaceholderText("#RRGGBB or rgba(...)")
            inp.textChanged.connect(lambda _text, key=field_name: self._on_field_text_changed(key))

            swatch = QFrame()
            swatch.setFixedSize(28, 18)

            btn_pick = MonoButton("PICK")
            btn_pick.setFixedHeight(22)
            btn_pick.setFixedWidth(56)
            btn_pick.clicked.connect(lambda _checked=False, key=field_name: self._pick_color(key))

            grid.addWidget(label, row, 0)
            grid.addWidget(inp, row, 1)
            grid.addWidget(swatch, row, 2)
            grid.addWidget(btn_pick, row, 3)

            self._field_inputs[field_name] = inp
            self._field_swatches[field_name] = swatch

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_area.setWidget(grid_container)
        box_layout.addWidget(self._scroll_area, 1)

        self.status_label = QLabel("")
        box_layout.addWidget(self.status_label)

        self._theme_entries: list[tuple[str, str, bool]] = []
        self._active_index: int = 0

        self.theme_combo.currentIndexChanged.connect(self._on_combo_changed)
        self.btn_prev.clicked.connect(lambda: self._step_theme(-1))
        self.btn_next.clicked.connect(lambda: self._step_theme(1))
        self.btn_load.clicked.connect(self._load_selected_theme)
        self.btn_apply.clicked.connect(self._apply_selected_theme)
        self.btn_apply_draft.clicked.connect(self._apply_draft_theme)
        self.btn_save_custom.clicked.connect(self._save_custom_theme)
        self.btn_delete_custom.clicked.connect(self._delete_selected_theme)

        if self._ui_bridge is not None:
            self._ui_bridge.sig_theme_changed.connect(self._on_external_theme_changed)

        self._refresh_theme_entries(select_key=current_theme_key())
        self._refresh_widget_styles()

    def _refresh_widget_styles(self) -> None:
        """Re-apply all inline widget styles using current theme tokens."""
        linedit_ss = (
            f"QLineEdit {{ background: {s.BG_INPUT}; color: {s.FG_TEXT}; "
            f"border: 1px solid {s.BORDER_LIGHT}; padding: 4px 6px; "
            f"border-radius: 4px; font-size: 10px; }}"
        )
        combo_ss = (
            f"QComboBox {{ background: {s.BG_INPUT}; color: {s.FG_TEXT}; "
            f"border: 1px solid {s.BORDER_LIGHT}; padding: 4px 6px; "
            f"border-radius: 4px; font-size: 10px; }}"
            f"QComboBox QAbstractItemView {{ background: {s.BG_INPUT}; "
            f"color: {s.FG_TEXT}; border: 1px solid {s.BORDER_LIGHT}; }}"
        )
        self.theme_combo.setStyleSheet(combo_ss)
        self.name_input.setStyleSheet(linedit_ss)
        for inp in self._field_inputs.values():
            inp.setStyleSheet(linedit_ss)
        for lbl in self._dim_labels:
            lbl.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        self.status_label.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        self._scroll_area.setStyleSheet(
            f"QScrollArea {{ background: {s.BG_INPUT}; border: 1px solid {s.BORDER_SUBTLE}; "
            f"border-radius: 4px; }}"
            f"{s.SCROLLBAR_STYLE}"
        )
        # Refresh swatch borders (background is the actual color value, leave that)
        for field_name in self._field_swatches:
            self._update_swatch(field_name)

    def _set_status(self, text: str, warn: bool = False) -> None:
        color = s.FG_WARN if warn else s.FG_DIM
        self.status_label.setStyleSheet(f"color: {color}; font-size: 10px;")
        self.status_label.setText(text)

    def _on_external_theme_changed(self, key: str) -> None:
        self._refresh_widget_styles()
        self._refresh_theme_entries(select_key=key)

    def _refresh_theme_entries(self, select_key: str | None = None) -> None:
        self._theme_entries = list_theme_entries()
        if not self._theme_entries:
            self._theme_entries = [("midnight", "Midnight", True)]
        keys = [entry[0] for entry in self._theme_entries]
        target_key = select_key if select_key in keys else current_theme_key()
        self._active_index = keys.index(target_key) if target_key in keys else 0
        self._sync_theme_title()
        self._load_selected_theme()

    def _sync_theme_title(self) -> None:
        # Rebuild the combo's items from _theme_entries and select the
        # currently-active one. Block signals while rebuilding so the
        # internal index updates don't fire _on_combo_changed during init.
        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        for key, name, builtin in self._theme_entries:
            suffix = "Builtin" if builtin else "Custom"
            self.theme_combo.addItem(f"{name} [{suffix}]", key)
        if self._theme_entries:
            self._active_index = min(self._active_index, len(self._theme_entries) - 1)
            self.theme_combo.setCurrentIndex(self._active_index)
            key, _, builtin = self._theme_entries[self._active_index]
            self.theme_combo.setToolTip(key)
            self.btn_delete_custom.setEnabled(not builtin)
        else:
            self.btn_delete_custom.setEnabled(False)
        self.theme_combo.blockSignals(False)

    def _on_combo_changed(self, index: int) -> None:
        """User picked a theme from the dropdown — update active index and
        load the fields. Doesn't apply globally; user clicks APPLY for that.
        Matches the PREV/NEXT/_step_theme behavior."""
        if not (0 <= index < len(self._theme_entries)):
            return
        if index == self._active_index:
            return
        self._active_index = index
        key, _, builtin = self._theme_entries[self._active_index]
        self.theme_combo.setToolTip(key)
        self.btn_delete_custom.setEnabled(not builtin)
        self._load_selected_theme()

    def _selected_key(self) -> str:
        if not self._theme_entries:
            return "midnight"
        return self._theme_entries[self._active_index][0]

    def _step_theme(self, direction: int) -> None:
        if not self._theme_entries:
            return
        total = len(self._theme_entries)
        self._active_index = (self._active_index + direction) % total
        self._sync_theme_title()

    def _load_selected_theme(self) -> None:
        key = self._selected_key()
        theme = get_theme(key)
        payload = theme_to_dict(theme)
        self._updating = True
        for field_name, inp in self._field_inputs.items():
            inp.setText(str(payload.get(field_name, "")))
            self._update_swatch(field_name)
        self._updating = False
        self.name_input.setText(theme.name)
        self._set_status(f"Loaded '{theme.name}'")

    def _parse_color(self, raw: str) -> QColor | None:
        color = QColor(str(raw or "").strip())
        return color if color.isValid() else None

    def _update_swatch(self, field_name: str) -> None:
        swatch = self._field_swatches.get(field_name)
        inp = self._field_inputs.get(field_name)
        if swatch is None or inp is None:
            return
        color = self._parse_color(inp.text())
        if color is None:
            swatch.setStyleSheet(
                f"QFrame {{ background: transparent; border: 1px dashed {s.FG_ERROR}; border-radius: 3px; }}"
            )
            return
        alpha = color.alpha()
        if alpha < 255:
            val = f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"
        else:
            val = color.name()
        swatch.setStyleSheet(
            f"QFrame {{ background: {val}; border: 1px solid {s.BORDER_LIGHT}; border-radius: 3px; }}"
        )

    def _on_field_text_changed(self, field_name: str) -> None:
        self._update_swatch(field_name)
        if not self._updating:
            self._set_status("Draft modified")

    def _pick_color(self, field_name: str) -> None:
        inp = self._field_inputs.get(field_name)
        if inp is None:
            return
        seed = self._parse_color(inp.text()) or QColor("#ffffff")
        picked = QColorDialog.getColor(seed, self, f"Pick Color: {field_name}", QColorDialog.ShowAlphaChannel)
        if not picked.isValid():
            return
        if field_name.startswith("glass_"):
            value = f"rgba({picked.red()}, {picked.green()}, {picked.blue()}, {picked.alpha()})"
        else:
            value = picked.name(QColor.HexRgb)
        inp.setText(value)

    def _collect_values(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for field_name, inp in self._field_inputs.items():
            raw = inp.text().strip()
            values[field_name] = raw
        return values

    def _apply_theme_key(self, key: str, persist: bool = True) -> None:
        if persist:
            save_theme_config({"theme": key})
        if self._ui_bridge is not None:
            self._ui_bridge.sig_theme_changed.emit(key)
            return
        apply_theme(key)
        refresh_styles()

    def _apply_selected_theme(self) -> None:
        key = self._selected_key()
        self._apply_theme_key(key, persist=True)
        self._set_status(f"Applied '{get_theme(key).name}'")
        self._log("INFO", f"[theme] applied {key}")

    def _apply_draft_theme(self) -> None:
        name = self.name_input.text().strip() or "Draft Theme"
        values = self._collect_values()
        key = save_custom_theme(name=name, values=values, key="__draft__", persist=False)
        self._apply_theme_key(key, persist=False)
        self._set_status("Draft applied (not saved)")
        self._log("INFO", f"[theme] draft applied ({name})")

    def _save_custom_theme(self) -> None:
        name = self.name_input.text().strip() or "Custom Theme"
        values = self._collect_values()
        selected_key = self._selected_key()
        save_key = None if is_builtin_theme(selected_key) else selected_key
        key = save_custom_theme(name=name, values=values, key=save_key, persist=True)
        self._refresh_theme_entries(select_key=key)
        self._apply_theme_key(key, persist=True)
        self._set_status(f"Saved custom theme '{name}'")
        self._log("INFO", f"[theme] saved {name}")

    def _delete_selected_theme(self) -> None:
        key = self._selected_key()
        if is_builtin_theme(key):
            self._set_status("Built-in themes cannot be deleted", warn=True)
            self._log("WARNING", "[theme] delete blocked: builtin theme")
            return
        answer = QMessageBox.question(
            self,
            "Delete Theme",
            f"Delete custom theme '{get_theme(key).name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        if not delete_custom_theme(key):
            self._set_status("Delete failed", warn=True)
            self._log("ERROR", f"[theme] delete failed: {key}")
            return
        self._apply_theme_key(current_theme_key(), persist=True)
        self._refresh_theme_entries(select_key=current_theme_key())
        self._set_status("Custom theme deleted")
        self._log("INFO", f"[theme] deleted {key}")

    def _log(self, severity: str, message: str) -> None:
        if self._ui_bridge is None:
            return
        self._ui_bridge.sig_monitor_log.emit(severity, message)
