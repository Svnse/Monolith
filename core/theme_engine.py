from __future__ import annotations

from PySide6.QtWidgets import QApplication

from core.themes import current_theme


class ThemeEngine:
    """Builds and applies the single app-wide stylesheet from the active theme."""

    def build_stylesheet(self) -> str:
        t = current_theme()
        return f"""
QWidget {{
    background: {t.bg_main};
    color: {t.fg_text};
}}
QFrame {{
    border-color: {t.border_subtle};
}}
QLabel {{
    color: {t.fg_text};
    background: transparent;
}}
QPushButton {{
    background: {t.bg_button};
    color: {t.fg_dim};
    border: 1px solid {t.border_light};
    padding: 6px 12px;
    border-radius: 2px;
    font-size: 11px;
    font-weight: bold;
}}
QPushButton:hover {{
    background: {t.bg_button_hover};
    color: {t.accent_primary};
    border: 1px solid {t.accent_primary};
}}
QPushButton:pressed {{
    background: {t.bg_button_pressed};
    color: {t.accent_primary};
    border: 1px solid {t.accent_primary};
}}
QPushButton:disabled {{
    background: {t.bg_button_disabled};
    color: {t.border_light};
    border: 1px solid {t.border_subtle};
}}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {t.bg_input};
    color: {t.fg_text};
    border: 1px solid {t.border_dark};
    padding: 4px;
}}
QComboBox:hover {{
    border: 1px solid {t.accent_primary};
}}
QComboBox QAbstractItemView {{
    background: {t.bg_input};
    color: {t.fg_text};
    border: 1px solid {t.border_light};
    selection-background-color: {t.bg_button_hover};
    selection-color: {t.accent_primary};
}}
QScrollArea {{
    border: none;
}}
QScrollBar:vertical {{
    background: {t.bg_input};
    width: 10px;
    margin: 0px;
    border: 1px solid {t.border_dark};
}}
QScrollBar::handle:vertical {{
    background: {t.scrollbar_handle};
    min-height: 24px;
    border: 1px solid {t.accent_primary};
    border-radius: 2px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t.scrollbar_handle_hover};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
    width: 0px;
}}
QScrollBar:horizontal {{
    background: {t.bg_input};
    height: 10px;
    margin: 0px;
    border: 1px solid {t.border_dark};
}}
QScrollBar::handle:horizontal {{
    background: {t.scrollbar_handle};
    min-width: 24px;
    border: 1px solid {t.accent_primary};
    border-radius: 2px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {t.scrollbar_handle_hover};
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0px;
    height: 0px;
}}
"""

    def apply(self, app: QApplication) -> None:
        app.setStyleSheet(self.build_stylesheet())
