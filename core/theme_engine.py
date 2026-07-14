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
        scrollbar_ss = f"""
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 0px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {t.scrollbar_handle};
    min-height: 36px;
    border: none;
    border-radius: 5px;
    margin: 2px 3px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t.scrollbar_handle_hover};
}}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
    width: 0px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 0px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {t.scrollbar_handle};
    min-width: 36px;
    border: none;
    border-radius: 5px;
    margin: 3px 2px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {t.scrollbar_handle_hover};
}}
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: transparent;
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    height: 0px;
    width: 0px;
}}
"""
        return f"""
/* Regions own color; content is transparent (UI_CONTRACT §2). The old
   `QWidget {{ background: bg_main }}` default made EVERY plain container
   repaint the darkest color over its region's surface — the root cause of the
   panel-within-a-panel look. Content widgets now inherit their region's fill
   (#MainFrame, #companion_pane, #icon_rail, …); floating windows get explicit
   fills below so nothing renders transparent-to-desktop. */
QWidget {{
    background: transparent;
    color: {t.fg_text};
}}
QDialog {{ background: {t.bg_main}; }}
QMenu {{
    background: {t.bg_panel};
    color: {t.fg_text};
    border: 1px solid {t.border_subtle};
}}
QMenu::item:selected {{ background: {t.bg_button_hover}; color: {t.accent_primary}; }}
QToolTip {{
    background: {t.bg_panel};
    color: {t.fg_text};
    border: 1px solid {t.border_subtle};
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
QFrame.MessageWidget[role="system"] {{ border-left-color: {t.fg_info}; }}
QFrame.MessageWidget[editing="true"] {{
    background: rgba({bg_main_rgb}, 0.24);
    border-left-color: {t.accent_primary};
}}
QLabel#msg_role {{ font-size: 9px; font-weight: bold; color: {t.fg_dim}; background: transparent; }}
QLabel#msg_role[role="system"] {{ color: {t.fg_info}; }}
QLabel#msg_time {{ font-size: 8px; color: {t.fg_info}; background: transparent; }}
QLabel#msg_content {{ color: {t.fg_text}; font-size: 11px; background: transparent; padding: 0px; }}
QTextEdit#msg_content {{ color: {t.fg_text}; font-size: 11px; background: transparent; border: none; padding: 0px; margin: 0px; }}
QTextEdit#msg_content[role="system"] {{ color: {t.fg_secondary}; }}
QFrame#msg_think_badge {{ background: transparent; border: none; }}

QWidget#MainFrame {{ background: {t.bg_main}; border: 1px solid {t.border_light}; }}
QStackedWidget#conversation_stack,
QWidget#conversation_surface {{ background: {t.bg_main}; }}
QFrame#sidebar {{ background: {t.bg_sidebar}; border-right: 1px solid {t.border_subtle}; padding: 0px; }}
QFrame#top_bar {{ background: {t.bg_sidebar}; border-bottom: 1px solid {t.border_subtle}; }}
QWidget#icon_rail {{ background: {t.bg_sidebar}; border-left: 1px solid {t.border_subtle}; }}
QFrame#omni_bar_frame {{
    background: {getattr(t, "bg_surface_1", t.bg_panel)};
    border: 1px solid {t.border_subtle};
    border-radius: 16px;
}}
QFrame#omni_bar_frame[activeGlow="true"] {{
    border: 1px solid {t.accent_primary};
}}
QLineEdit#omni_bar_input {{
    background: transparent;
    color: {t.fg_text};
    border: none;
    padding: 8px 0px;
    font-size: 13px;
}}
QFrame#companion_pane {{
    background: {getattr(t, "bg_surface_1", t.bg_panel)};
    border-left: 1px solid {t.border_subtle};
}}
/* Companion-panel inset surfaces (UI_CONTRACT §2–§3): every feed, list, input
   and tab pane inside a panel is the SAME color as the pane itself
   (bg_surface_1 — one continuous surface, zero fill contrast), separated only
   by a border_subtle hairline + rounded corners. Widgets opt in with
   setProperty("panelInset", True); fonts stay per-widget. */
QListWidget[panelInset="true"], QTextEdit[panelInset="true"],
QPlainTextEdit[panelInset="true"], QScrollArea[panelInset="true"] {{
    background: {getattr(t, "bg_surface_1", t.bg_panel)};
    border: 1px solid {t.border_subtle};
    border-radius: 6px;
    color: {t.fg_text};
}}
QLineEdit[panelInset="true"] {{
    background: {getattr(t, "bg_surface_1", t.bg_panel)};
    border: 1px solid {t.border_subtle};
    border-radius: 4px;
    color: {t.fg_text};
    padding: 3px 6px;
}}
QLineEdit[panelInset="true"]:focus {{ border-color: {t.accent_primary}; }}
QTabWidget[panelInset="true"]::pane {{
    background: {getattr(t, "bg_surface_1", t.bg_panel)};
    border: 1px solid {t.border_subtle};
    border-radius: 6px;
}}
QTabWidget[panelInset="true"] QTabBar::tab {{
    background: transparent;
    color: {t.fg_dim};
    border: none;
    padding: 4px 10px;
    font-family: Consolas;
    font-size: 9px;
}}
QTabWidget[panelInset="true"] QTabBar::tab:selected {{
    color: {t.accent_primary};
    border-bottom: 2px solid {t.accent_primary};
}}
QFrame#vitals_footer {{
    background: {t.bg_sidebar};
    border-top: 1px solid {t.border_subtle};
}}
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
QTreeView#file_tree {{
    background: {getattr(t, "bg_surface_1", t.bg_panel)}; color: {t.fg_secondary};
    border: 1px solid {t.border_subtle}; border-radius: 6px;
}}
QTreeView#file_tree::item:hover {{ background: {t.bg_button_hover}; }}
QTreeView#file_tree::item:selected {{ background: {t.accent_primary}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {getattr(t, "bg_surface_1", t.bg_panel)}; color: {t.fg_text};
    border: 1px solid {t.border_subtle}; border-radius: 4px; padding: 4px;
}}

QLineEdit#archive_search {{
    background: {getattr(t, "bg_surface_1", t.bg_panel)}; color: {t.fg_text};
    border: 1px solid {t.border_subtle};
    padding: 5px 8px; font-size: 10px; border-radius: 4px;
}}
QLineEdit#archive_search:focus {{ border-color: {t.accent_primary}; }}
QLineEdit#chat_input,
QPlainTextEdit#chat_input {{
    background: {t.bg_input}; color: {t.fg_text}; border: 1px solid {t.border_light};
    padding: 8px; font-family: 'Verdana'; font-size: 11px;
}}
QPlainTextEdit#chat_input::viewport {{ background: {t.bg_input}; }}
QLineEdit#chat_input:focus,
QPlainTextEdit#chat_input:focus {{ border: 1px solid {t.accent_primary}; }}
QLineEdit#path_display {{
    background: {t.bg_input}; color: {t.fg_placeholder}; border: 1px solid {t.border_light}; padding: 5px;
}}
QTextEdit#trace_log {{
    background: {t.bg_input}; color: {t.fg_text}; border: 1px solid {t.border_subtle};
    font-family: 'Consolas', monospace; font-size: 10px;
}}
QListWidget#archive_list {{
    background: {getattr(t, "bg_surface_1", t.bg_panel)}; color: {t.fg_text};
    border: 1px solid {t.border_subtle}; border-radius: 6px;
    font-family: 'Consolas', monospace; font-size: 10px;
}}
QListWidget#archive_list::item {{ padding: 6px; }}
QListWidget#archive_list::item:selected {{ background: {t.bg_button_hover}; color: {t.accent_primary}; }}
QListWidget#message_list {{
    background: {t.bg_main}; color: {t.fg_text}; border: none; outline: 0;
    font-family: 'Consolas', monospace; font-size: 12px;
}}
QListWidget#message_list::viewport {{ background: {t.bg_main}; }}
QListWidget#message_list::item {{ border: none; background: transparent; padding: 0px; }}
QPushButton#panel_tab_btn {{
    background: {t.bg_button}; border: 1px solid {t.border_light}; color: {t.fg_dim};
    padding: 6px 12px; font-size: 10px; font-weight: bold; border-radius: 2px;
}}
QPushButton#panel_tab_btn:checked {{ background: {t.bg_button_hover}; color: {t.accent_primary}; border: 1px solid {t.accent_primary}; }}
QPushButton#panel_tab_btn:hover {{ color: {t.fg_text}; border: 1px solid {t.fg_text}; }}
QPushButton#options_toggle_btn {{
    background: transparent; border: none; color: {t.fg_dim};
    font-size: 9px; font-weight: bold; letter-spacing: 1px; text-align: left; padding: 4px 0;
}}
QPushButton#options_toggle_btn:hover {{ color: {t.accent_primary}; }}
QLabel#lbl_config_state {{ color: {t.fg_dim}; font-size: 10px; font-weight: bold; }}
QLabel#lbl_config_state[dirty="true"] {{ color: {t.fg_warn}; }}
QLabel#lbl_config_update {{ color: {t.accent_primary}; font-size: 10px; font-weight: bold; }}
QPushButton#btn_save_config {{
    background: {t.bg_button}; border: 1px solid {t.border_light}; color: {t.fg_dim};
    padding: 6px 12px; font-size: 11px; font-weight: bold; border-radius: 2px;
}}
QPushButton#btn_save_config:hover {{ background: {t.bg_button_hover}; color: {t.fg_dim}; }}
QPushButton#btn_save_config[dirty="true"] {{
    border-color: {t.accent_primary}; color: {t.accent_primary};
}}
QPushButton#btn_save_config[dirty="true"]:hover {{ background: {t.accent_primary}; color: black; }}
QPushButton#btn_save_config[dirty="true"]:pressed {{ background: {t.accent_primary_dark}; color: black; }}
QComboBox QAbstractItemView {{
    background: {t.bg_input}; color: {t.fg_text}; border: 1px solid {t.border_light};
    selection-background-color: {t.bg_button_hover}; selection-color: {t.accent_primary};
}}
/* Scroll areas are windows onto themed surfaces, never surfaces themselves.
   Without this, the viewport + its body widget paint the OS palette color —
   an untheme-able near-black that made every QScrollArea-based panel (Config,
   Audio, Stats) read as a panel-within-a-panel (UI_CONTRACT §2). */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget {{ background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
{scrollbar_ss}
"""

    def apply(self, app: QApplication) -> None:
        app.setStyleSheet(self.build_stylesheet())


# ─── Deep theme refresh ────────────────────────────────────────────────
# The application-level QSS rebuilt by ThemeEngine.apply() covers anything
# styled via global selectors. It does NOT cover per-widget stylesheets
# that f-string theme tokens at construction (e.g. `f"color: {_s.FG_DIM};"`)
# — those values are baked at the moment of the setStyleSheet call and
# stay frozen forever unless something re-runs the call.
#
# Most affected widgets opt in via `ui_bridge.sig_theme_changed`, but
# that signal misses widgets that were constructed after a theme change
# (no subscription yet) or whose subscription was lost (signal emitter
# replaced, parent destroyed, etc.). To make a theme switch deterministic,
# this walker descends from a root widget, finds every QWidget that
# exposes a known refresh entry point, and force-calls it.
#
# Recognized method names (first match per widget, in order):
#   - refresh_theme()         — canonical name; new widgets should prefer this
#   - _on_theme_changed(name) — pre-existing convention in chat/page surfaces
#   - _apply_chrome_theme()   — pre-existing convention in companion/omni
#   - _apply_input_theme()    — pre-existing convention in model_config
#
# Errors are swallowed per widget; one broken refresh shouldn't abort the
# rest of the tree.
_THEME_REFRESH_METHODS = (
    "refresh_theme",
    "_on_theme_changed",
    "_apply_chrome_theme",
    "_apply_input_theme",
)


def deep_refresh_theme(root, *, theme_name: str = "") -> int:
    """Walk every QWidget descendant of `root` and call its theme refresh
    method, if any. Returns the count of widgets refreshed (for debug).
    """
    from PySide6.QtWidgets import QWidget

    if root is None:
        return 0
    refreshed = 0
    candidates = [root]
    try:
        candidates.extend(root.findChildren(QWidget))
    except Exception:
        pass

    seen: set[int] = set()
    for widget in candidates:
        if widget is None:
            continue
        wid = id(widget)
        if wid in seen:
            continue
        seen.add(wid)
        for method_name in _THEME_REFRESH_METHODS:
            method = getattr(widget, method_name, None)
            if not callable(method):
                continue
            try:
                # Most refresh hooks accept an optional theme name; some
                # take no args. Try the named-arg form first, fall back.
                try:
                    method(theme_name)
                except TypeError:
                    method()
            except Exception:
                # One stale widget shouldn't abort the rest of the sweep.
                continue
            refreshed += 1
            break
    return refreshed
