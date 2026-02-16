# ======================
# DYNAMIC THEME BRIDGE
# ======================
# All constants are populated from the active theme.
# Other files import these names as before — no import changes needed.

from core.themes import current_theme


def _t():
    return current_theme()


# Backgrounds
BG_MAIN = _t().bg_main
BG_SIDEBAR = _t().bg_sidebar
BG_PANEL = _t().bg_panel
BG_GROUP = _t().bg_group
BG_INPUT = _t().bg_input

# Button states
BG_BUTTON = _t().bg_button
BG_BUTTON_HOVER = _t().bg_button_hover
BG_BUTTON_PRESSED = _t().bg_button_pressed
BG_BUTTON_DISABLED = _t().bg_button_disabled

# Borders
BORDER_DARK = _t().border_dark
BORDER_LIGHT = _t().border_light
BORDER_SUBTLE = _t().border_subtle

# Foreground / text
FG_TEXT = _t().fg_text
FG_DIM = _t().fg_dim
FG_ACCENT = _t().fg_accent
FG_ERROR = _t().fg_error
FG_WARN = _t().fg_warn
FG_PLACEHOLDER = _t().fg_placeholder
FG_INFO = _t().fg_info
FG_SECONDARY = _t().fg_secondary

# Primary accent
ACCENT_GOLD = _t().accent_primary
ACCENT_PRIMARY = _t().accent_primary
ACCENT_PRIMARY_DARK = _t().accent_primary_dark

# Glass surfaces
GLASS_BG = _t().glass_bg
GLASS_BORDER = _t().glass_border
GLASS_HOVER = _t().glass_hover

# Overseer palette
OVERSEER_BG = _t().overseer_bg
OVERSEER_FG = _t().overseer_fg
OVERSEER_DIM = _t().overseer_dim
OVERSEER_BORDER = _t().overseer_border

# Scrollbar
SCROLLBAR_HANDLE = _t().scrollbar_handle
SCROLLBAR_HANDLE_HOVER = _t().scrollbar_handle_hover

# Gradient
GRADIENT_COLOR = _t().gradient_color


def _build_scrollbar_style():
    t = _t()
    return f"""
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
    height: 0px;
    width: 0px;
}}
"""


SCROLLBAR_STYLE = _build_scrollbar_style()


def refresh_styles():
    """Rebuild all module-level constants from the currently active theme."""
    g = globals()
    t = _t()
    g["BG_MAIN"] = t.bg_main
    g["BG_SIDEBAR"] = t.bg_sidebar
    g["BG_PANEL"] = t.bg_panel
    g["BG_GROUP"] = t.bg_group
    g["BG_INPUT"] = t.bg_input
    g["BG_BUTTON"] = t.bg_button
    g["BG_BUTTON_HOVER"] = t.bg_button_hover
    g["BG_BUTTON_PRESSED"] = t.bg_button_pressed
    g["BG_BUTTON_DISABLED"] = t.bg_button_disabled
    g["BORDER_DARK"] = t.border_dark
    g["BORDER_LIGHT"] = t.border_light
    g["BORDER_SUBTLE"] = t.border_subtle
    g["FG_TEXT"] = t.fg_text
    g["FG_DIM"] = t.fg_dim
    g["FG_ACCENT"] = t.fg_accent
    g["FG_ERROR"] = t.fg_error
    g["FG_WARN"] = t.fg_warn
    g["FG_PLACEHOLDER"] = t.fg_placeholder
    g["FG_INFO"] = t.fg_info
    g["FG_SECONDARY"] = t.fg_secondary
    g["ACCENT_GOLD"] = t.accent_primary
    g["ACCENT_PRIMARY"] = t.accent_primary
    g["ACCENT_PRIMARY_DARK"] = t.accent_primary_dark
    g["GLASS_BG"] = t.glass_bg
    g["GLASS_BORDER"] = t.glass_border
    g["GLASS_HOVER"] = t.glass_hover
    g["OVERSEER_BG"] = t.overseer_bg
    g["OVERSEER_FG"] = t.overseer_fg
    g["OVERSEER_DIM"] = t.overseer_dim
    g["OVERSEER_BORDER"] = t.overseer_border
    g["SCROLLBAR_HANDLE"] = t.scrollbar_handle
    g["SCROLLBAR_HANDLE_HOVER"] = t.scrollbar_handle_hover
    g["GRADIENT_COLOR"] = t.gradient_color
    g["SCROLLBAR_STYLE"] = _build_scrollbar_style()
