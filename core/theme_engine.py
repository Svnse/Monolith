from __future__ import annotations

from PySide6.QtWidgets import QApplication

from core.themes import current_theme


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.strip().lstrip("#")
    if len(color) != 6:
        return (0, 0, 0)
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


class ThemeEngine:
    """Builds and applies the single app-wide stylesheet from the active theme."""

    def build_stylesheet(self) -> str:
        t = current_theme()
        bg_main_rgb = ", ".join(str(v) for v in _hex_to_rgb(t.bg_main))
        return f"""
QWidget {{
    background: {t.bg_main};
    color: {t.fg_text};
}}
QFrame {{
    border-color: {t.border_subtle};
}}
QLabel {{ color: {t.fg_text}; background: transparent; }}
QPushButton {{
    background: {t.bg_button};
    color: {t.fg_dim};
    border: 1px solid {t.border_light};
    padding: 6px 12px;
    border-radius: 2px;
    font-size: 11px;
    font-weight: bold;
}}
QPushButton:hover {{ background: {t.bg_button_hover}; color: {t.accent_primary}; border-color: {t.accent_primary}; }}
QPushButton:pressed {{ background: {t.bg_button_pressed}; }}
QPushButton:disabled {{ background: {t.bg_button_disabled}; color: {t.border_light}; border-color: {t.border_subtle}; }}

QPushButton.MonoButton {{ font-size: 10px; letter-spacing: 1px; }}
QPushButton.MonoButton[accent="true"] {{ color: {t.fg_accent}; }}
QPushButton.MonoButton[accent="true"]:hover {{ color: {t.accent_primary}; }}
QPushButton.MonoTriangleButton {{ color: {t.fg_text}; }}
QPushButton.SidebarButton {{
    background: transparent;
    border: none;
    border-left: 2px solid transparent;
    border-right: 1px solid transparent;
    margin-right: -1px;
    padding: 6px 4px;
    color: {t.fg_dim};
}}
QPushButton.SidebarButton:hover {{
    background: rgba(255, 255, 255, 0.03);
}}
QPushButton.SidebarButton:checked {{
    border-left: 2px solid {t.accent_primary};
    background: {t.bg_main};
    border-right: 1px solid {t.bg_main};
}}
QPushButton#collapsible_toggle {{ background: transparent; border: none; text-align: left; font-weight: bold; font-size: 10px; color: {t.fg_dim}; padding: 4px; }}
QPushButton#collapsible_toggle:checked {{ color: {t.accent_primary}; }}
QWidget#collapsible_content {{ background: {t.bg_input}; }}
QFrame#mono_group_box {{ background: transparent; border: none; margin: 0px; padding: 0px; }}
QLabel#slider_label {{ color: {t.fg_dim}; font-size: 9px; }}
QLabel#slider_value {{ color: {t.fg_text}; font-size: 10px; font-weight: bold; }}
QSlider::groove:horizontal {{ border: none; height: 4px; background: {t.bg_button_hover}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -4px 0; background: {t.accent_primary}; border-radius: 6px; }}
QSlider::sub-page:horizontal {{ background: {t.accent_primary}; border-radius: 2px; }}

QDialog#vitals_window {{ background: rgba({bg_main_rgb}, 0.95); border: 1px solid {t.border_subtle}; border-radius: 6px; }}
QPushButton#mode_btn_active {{ background: {t.bg_input}; border: 1px solid {t.accent_primary}; color: {t.fg_text}; font-weight: bold; }}
QPushButton#mode_btn_inactive {{ background: transparent; border: 1px solid {t.border_subtle}; color: {t.fg_dim}; }}
QPushButton.SplitControl {{ background: transparent; border: 1px solid {t.border_subtle}; color: {t.fg_dim}; border-radius: 2px; }}
QPushButton.SplitControl:hover {{ color: {t.accent_primary}; border-color: {t.accent_primary}; }}
QPushButton#close_btn:hover {{ color: {t.fg_error}; }}
QFrame#tag_input_frame {{ background: {t.bg_input}; border: 1px solid {t.border_light}; border-radius: 3px; }}
QPushButton.tag_chip {{ background: transparent; border: 1px solid {t.border_subtle}; color: {t.accent_primary}; border-radius: 2px; padding: 2px 6px; font-size: 9px; }}

QWidget#palette_overlay {{ background: rgba({bg_main_rgb}, 0.85); }}
QFrame#palette_container {{ background: {t.bg_panel}; border: 1px solid {t.border_subtle}; border-radius: 6px; }}
QLineEdit#palette_input {{ background: {t.bg_input}; color: {t.fg_text}; border: 1px solid {t.border_subtle}; border-radius: 4px; padding: 8px 12px; font-size: 12px; }}
QFrame.PaletteResultRow {{ border: none; border-radius: 3px; padding: 6px 10px; }}
QFrame.PaletteResultRow:hover {{ background: {t.bg_group}; }}
QFrame.PaletteResultRow[selected="true"] {{ border-left: 2px solid {t.accent_primary}; }}

QWidget#drop_overlay {{ background: rgba({bg_main_rgb}, 0.88); }}
QFrame#drop_panel {{ background: transparent; border: 2px dashed {t.accent_primary}; border-radius: 8px; }}

QFrame.MessageWidget {{ border-left: 2px solid {t.border_subtle}; background: transparent; padding: 0px; }}
QFrame.MessageWidget[role="assistant"] {{ border-left-color: {t.accent_primary}; }}
QLabel#msg_role {{ font-size: 9px; font-weight: bold; color: {t.fg_dim}; background: transparent; }}
QLabel#msg_time {{ font-size: 8px; color: {t.fg_info}; background: transparent; }}
QLabel#msg_content {{ color: {t.fg_text}; font-size: 11px; background: transparent; padding: 0px; }}
QTextEdit#msg_content {{ color: {t.fg_text}; font-size: 11px; background: transparent; border: none; padding: 0px; margin: 0px; }}
QPushButton.msg_icon_action {{
    background: transparent;
    border: 1px solid {t.border_subtle};
    color: {t.fg_dim};
    font-size: 12px;
    border-radius: 3px;
    padding: 0px;
}}
QPushButton.msg_icon_action:hover {{
    color: {t.accent_primary};
    border-color: {t.accent_primary};
    background: {t.bg_button_hover};
}}

QWidget#MainFrame {{ border: 1px solid {t.border_light}; }}
QFrame#sidebar {{ background: {t.bg_sidebar}; border-right: 1px solid {t.border_subtle}; padding: 0px; }}
QFrame#top_bar {{ background: {t.bg_sidebar}; border-bottom: 1px solid {t.border_subtle}; }}
QLabel#lbl_monolith {{ color: {t.accent_primary_dark}; font-size: 14px; font-weight: bold; letter-spacing: 3px; background: transparent; }}
QLabel#lbl_chat_title {{ color: {t.fg_text}; font-size: 10px; font-weight: bold; letter-spacing: 1px; background: transparent; }}
QLabel#lbl_chat_time {{ color: {t.fg_dim}; font-size: 9px; font-weight: bold; letter-spacing: 1px; background: transparent; }}
QLabel#lbl_status {{ color: {t.fg_placeholder}; font-size: 8px; font-weight: bold; background: transparent; }}
QLabel#lbl_status[state="error"] {{ color: {t.fg_error}; }}
QLabel#lbl_status[state="loading"] {{ color: {t.fg_warn}; }}
QLabel#status_label {{ font-size: 10px; font-weight: bold; }}

QWidget#overseer_window {{ background: {t.overseer_bg}; color: {t.overseer_fg}; }}
QLabel.overseer_label {{ color: {t.overseer_fg}; }}
QLabel.overseer_dim {{ color: {t.overseer_dim}; }}
QPushButton.overseer_btn {{ background: transparent; border: 1px solid {t.overseer_border}; color: {t.overseer_fg}; padding: 4px 8px; border-radius: 2px; }}
QPushButton.overseer_btn:hover {{ border-color: {t.accent_primary}; color: {t.accent_primary}; }}
QTableWidget#active_tasks {{ background: {t.overseer_bg}; color: {t.overseer_fg}; gridline-color: {t.overseer_border}; }}

QPushButton._OperatorCard {{ background: {t.bg_input}; border: 1px solid {t.border_dark}; border-radius: 3px; }}
QPushButton._OperatorCard:hover {{ border-color: {t.accent_primary}; background: {t.bg_button_hover}; }}
QPushButton._OperatorCard[selected="true"] {{ border-color: {t.accent_primary}; background: {t.border_subtle}; }}
QFrame#chat_input_frame {{ background: {t.bg_input}; border: 1px solid {t.border_light}; border-radius: 4px; }}
QWidget#trace_display {{ background: {t.overseer_bg}; }}
QTreeView#file_tree {{ background: {t.bg_input}; color: {t.fg_secondary}; border: 1px solid {t.border_dark}; }}
QTreeView#file_tree::item:hover {{ background: {t.bg_button_hover}; }}
QTreeView#file_tree::item:selected {{ background: {t.accent_primary}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {t.bg_input}; color: {t.fg_text}; border: 1px solid {t.border_dark}; padding: 4px;
}}
QComboBox QAbstractItemView {{
    background: {t.bg_input}; color: {t.fg_text}; border: 1px solid {t.border_light};
    selection-background-color: {t.bg_button_hover}; selection-color: {t.accent_primary};
}}
QScrollArea {{ border: none; }}
"""

    def apply(self, app: QApplication) -> None:
        app.setStyleSheet(self.build_stylesheet())
