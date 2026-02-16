from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class Theme:
    name: str

    # Backgrounds
    bg_main: str
    bg_sidebar: str
    bg_panel: str
    bg_group: str
    bg_input: str

    # Button states
    bg_button: str
    bg_button_hover: str
    bg_button_pressed: str
    bg_button_disabled: str

    # Borders
    border_dark: str
    border_light: str
    border_subtle: str

    # Foreground / text
    fg_text: str
    fg_dim: str
    fg_accent: str
    fg_error: str
    fg_warn: str
    fg_placeholder: str
    fg_info: str
    fg_secondary: str

    # Primary accent (identity color)
    accent_primary: str
    accent_primary_dark: str

    # Glass surfaces
    glass_bg: str
    glass_border: str
    glass_hover: str

    # Overseer palette
    overseer_bg: str
    overseer_fg: str
    overseer_dim: str
    overseer_border: str

    # Scrollbar
    scrollbar_handle: str
    scrollbar_handle_hover: str

    # Gradient / animated elements
    gradient_color: str


# ---------------------
# BUILT-IN PRESETS
# ---------------------

MIDNIGHT = Theme(
    name="Midnight",
    bg_main="#0e1117",
    bg_sidebar="#12151c",
    bg_panel="#161922",
    bg_group="#0e1117",
    bg_input="#0c0f14",
    bg_button="#171b24",
    bg_button_hover="#1e2330",
    bg_button_pressed="#0e1117",
    bg_button_disabled="#0e1117",
    border_dark="#252a36",
    border_light="#2e3442",
    border_subtle="#1a1f2b",
    fg_text="#d4d8e0",
    fg_dim="#6b7280",
    fg_accent="#4ade80",
    fg_error="#ef4444",
    fg_warn="#f59e0b",
    fg_placeholder="#4b5563",
    fg_info="#3b4252",
    fg_secondary="#9ca3af",
    accent_primary="#6d8cff",
    accent_primary_dark="#5a72cc",
    glass_bg="rgba(14, 17, 23, 220)",
    glass_border="rgba(109, 140, 255, 40)",
    glass_hover="rgba(109, 140, 255, 15)",
    overseer_bg="#080a0f",
    overseer_fg="#33ff33",
    overseer_dim="#1a7a1a",
    overseer_border="#151a24",
    scrollbar_handle="#1e2330",
    scrollbar_handle_hover="#2a3040",
    gradient_color="#6d8cff",
)

OBSIDIAN = Theme(
    name="Obsidian",
    bg_main="#0C0C0C",
    bg_sidebar="#111111",
    bg_panel="#141414",
    bg_group="#0C0C0C",
    bg_input="#0f0f0f",
    bg_button="#181818",
    bg_button_hover="#222222",
    bg_button_pressed="#111111",
    bg_button_disabled="#111111",
    border_dark="#2a2a2a",
    border_light="#333333",
    border_subtle="#1a1a1a",
    fg_text="#dcdcdc",
    fg_dim="#777777",
    fg_accent="#96c93d",
    fg_error="#d44e4e",
    fg_warn="#e0b020",
    fg_placeholder="#555555",
    fg_info="#444444",
    fg_secondary="#bbbbbb",
    accent_primary="#6d8cff",
    accent_primary_dark="#5a72cc",
    glass_bg="rgba(12, 12, 12, 220)",
    glass_border="rgba(109, 140, 255, 40)",
    glass_hover="rgba(109, 140, 255, 15)",
    overseer_bg="#080808",
    overseer_fg="#33ff33",
    overseer_dim="#1a7a1a",
    overseer_border="#1a1a1a",
    scrollbar_handle="#1c1c1c",
    scrollbar_handle_hover="#252525",
    gradient_color="#6d8cff",
)

MONOLITHIC = Theme(
    name="Monolithic",
    bg_main="#0C0C0C",
    bg_sidebar="#111111",
    bg_panel="#141414",
    bg_group="#0C0C0C",
    bg_input="#0f0f0f",
    bg_button="#181818",
    bg_button_hover="#222222",
    bg_button_pressed="#111111",
    bg_button_disabled="#111111",
    border_dark="#2a2a2a",
    border_light="#333333",
    border_subtle="#1a1a1a",
    fg_text="#dcdcdc",
    fg_dim="#777777",
    fg_accent="#96c93d",
    fg_error="#d44e4e",
    fg_warn="#e0b020",
    fg_placeholder="#555555",
    fg_info="#444444",
    fg_secondary="#bbbbbb",
    accent_primary="#D4AF37",
    accent_primary_dark="#8a7340",
    glass_bg="rgba(12, 12, 12, 220)",
    glass_border="rgba(212, 175, 55, 40)",
    glass_hover="rgba(212, 175, 55, 15)",
    overseer_bg="#080808",
    overseer_fg="#33ff33",
    overseer_dim="#1a7a1a",
    overseer_border="#1a1a1a",
    scrollbar_handle="#1c1c1c",
    scrollbar_handle_hover="#252525",
    gradient_color="#D4AF37",
)

SLATE = Theme(
    name="Slate",
    # Shell disappears → cognition foregrounded
    bg_main="#343541",           # Conversation field
    bg_sidebar="#202123",        # Base shell
    bg_panel="#444654",          # Assistant elevation
    bg_group="#343541",          # Groups match main
    bg_input="#2d2d3a",          # Inputs slightly recessed
    # Buttons: minimal, no gradient, no glow
    bg_button="#3e3f4b",
    bg_button_hover="#444654",
    bg_button_pressed="#343541",
    bg_button_disabled="#2a2b32",
    # Borders: nearly invisible, structural only
    border_dark="#3e3f4b",
    border_light="#4a4b57",
    border_subtle="#2f3040",
    # Text: restrained, high readability
    fg_text="#ececf1",           # Primary text
    fg_dim="#8e8ea0",            # Secondary / labels
    fg_accent="#10a37f",         # Single green accent
    fg_error="#ef4444",
    fg_warn="#f59e0b",
    fg_placeholder="#5a5a6e",
    fg_info="#4a4b57",
    fg_secondary="#c5c5d2",
    # Accent: controlled interactivity, not branding
    accent_primary="#10a37f",
    accent_primary_dark="#0e8c6b",
    # Glass: subtle, no flashy borders
    glass_bg="rgba(32, 33, 35, 235)",
    glass_border="rgba(16, 163, 127, 20)",
    glass_hover="rgba(16, 163, 127, 8)",
    # Overseer inherits shell darkness
    overseer_bg="#1a1b1e",
    overseer_fg="#10a37f",
    overseer_dim="#0d7a5f",
    overseer_border="#2f3040",
    # Scrollbar: disappears into surface
    scrollbar_handle="#3e3f4b",
    scrollbar_handle_hover="#4a4b57",
    # No gradient glow — neutrality over identity
    gradient_color="#10a37f",
)


# ---------------------
# THEME REGISTRY
# ---------------------

THEMES: Dict[str, Theme] = {
    "midnight": MIDNIGHT,
    "obsidian": OBSIDIAN,
    "monolithic": MONOLITHIC,
    "slate": SLATE,
}

_active_theme: Theme = MIDNIGHT


def current_theme() -> Theme:
    return _active_theme


def apply_theme(name: str) -> None:
    global _active_theme
    key = name.lower()
    # Migration: arctic was renamed to monolithic
    if key == "arctic":
        key = "monolithic"
    if key not in THEMES:
        key = "midnight"
    _active_theme = THEMES[key]


def list_themes() -> list[str]:
    return [t.name for t in THEMES.values()]
