from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Dict

from core.paths import CONFIG_DIR


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

ARCTIC = Theme(
    name="Arctic",
    # Apple HIG dark system backgrounds — layered depth, cool charcoal not blue-black
    bg_main="#1c1c1e",           # systemBackground (iOS dark)
    bg_sidebar="#1c1c1e",        # same level — no contrast split on sidebar
    bg_panel="#2c2c2e",          # secondarySystemBackground
    bg_group="#1c1c1e",          # groups match main
    bg_input="#2c2c2e",          # elevated input surface
    # Buttons: Apple material feel — barely lifted off the surface
    bg_button="#3a3a3c",         # tertiarySystemBackground
    bg_button_hover="#48484a",   # quaternary lift
    bg_button_pressed="#2c2c2e",
    bg_button_disabled="#1c1c1e",
    # Borders: structural only, almost invisible
    border_dark="#38383a",
    border_light="#48484a",
    border_subtle="#2c2c2e",
    # Foreground: Apple label hierarchy
    fg_text="#f2f2f7",           # label (primary)
    fg_dim="#8e8e93",            # secondaryLabel
    fg_accent="#30d158",         # Apple green (systemGreen dark)
    fg_error="#ff453a",          # Apple red (systemRed dark)
    fg_warn="#ff9f0a",           # Apple orange (systemOrange dark)
    fg_placeholder="#636366",    # quaternaryLabel
    fg_info="#3a3a3c",           # near-invisible structural info
    fg_secondary="#ebebf5",      # slightly dimmer than primary
    # Accent: iOS blue — clean, unmistakable Apple
    accent_primary="#0a84ff",    # systemBlue (dark mode)
    accent_primary_dark="#0071e3",
    # Glass: frosted-panel feel
    glass_bg="rgba(28, 28, 30, 230)",
    glass_border="rgba(10, 132, 255, 30)",
    glass_hover="rgba(10, 132, 255, 10)",
    # Overseer: deep surface, Apple green pulse
    overseer_bg="#0d0d0f",
    overseer_fg="#30d158",
    overseer_dim="#1a5c2a",
    overseer_border="#1c1c1e",
    # Scrollbar: flush with surface, subtle
    scrollbar_handle="#3a3a3c",
    scrollbar_handle_hover="#48484a",
    gradient_color="#0a84ff",
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

_BUILTIN_THEMES: Dict[str, Theme] = {
    "midnight": MIDNIGHT,
    "obsidian": OBSIDIAN,
    "monolithic": MONOLITHIC,
    "slate": SLATE,
    "arctic": ARCTIC,
}
_BUILTIN_ORDER: tuple[str, ...] = tuple(_BUILTIN_THEMES.keys())
_CUSTOM_THEME_PATH = CONFIG_DIR / "themes_custom.json"

THEMES: Dict[str, Theme] = dict(_BUILTIN_THEMES)
_custom_themes: Dict[str, Theme] = {}
_active_theme_key: str = "midnight"
_active_theme: Theme = _BUILTIN_THEMES[_active_theme_key]


def _normalize_theme_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return key or "custom_theme"


def _field_names() -> tuple[str, ...]:
    return tuple(Theme.__dataclass_fields__.keys())


def editable_theme_fields() -> tuple[str, ...]:
    return tuple(name for name in _field_names() if name != "name")


def theme_to_dict(theme: Theme) -> dict:
    return asdict(theme)


def _theme_from_payload(name: str, payload: dict) -> Theme:
    defaults = theme_to_dict(MIDNIGHT)
    values = {"name": str(name or defaults["name"])}
    for field_name in editable_theme_fields():
        raw = payload.get(field_name) if isinstance(payload, dict) else None
        if isinstance(raw, str) and raw.strip():
            values[field_name] = raw.strip()
        else:
            values[field_name] = defaults[field_name]
    return Theme(**values)


def _save_custom_themes() -> None:
    try:
        _CUSTOM_THEME_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: theme_to_dict(theme) for key, theme in _custom_themes.items()}
        with _CUSTOM_THEME_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:
        pass


def _rebuild_theme_registry() -> None:
    THEMES.clear()
    THEMES.update(_BUILTIN_THEMES)
    THEMES.update(_custom_themes)


def _resolve_existing_key(name_or_key: str | None) -> str | None:
    if not name_or_key:
        return None
    raw = str(name_or_key).strip()
    if not raw:
        return None
    if raw in THEMES:
        return raw

    key = _normalize_theme_key(raw)
    if key in THEMES:
        return key

    lower = raw.lower()
    for candidate_key, theme in THEMES.items():
        if theme.name.lower() == lower:
            return candidate_key
    return None


def _load_custom_themes() -> None:
    if not _CUSTOM_THEME_PATH.exists():
        return
    try:
        with _CUSTOM_THEME_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return

    if not isinstance(data, dict):
        return

    loaded: Dict[str, Theme] = {}
    for raw_key, raw_theme in data.items():
        key = _normalize_theme_key(raw_key)
        if key in _BUILTIN_THEMES:
            continue
        if not isinstance(raw_theme, dict):
            continue
        name = raw_theme.get("name")
        if not isinstance(name, str) or not name.strip():
            name = key.replace("_", " ").title()
        loaded[key] = _theme_from_payload(str(name), raw_theme)

    _custom_themes.clear()
    _custom_themes.update(loaded)
    _rebuild_theme_registry()


def current_theme_key() -> str:
    return _active_theme_key


def current_theme() -> Theme:
    return _active_theme


def apply_theme(name_or_key: str) -> None:
    global _active_theme_key, _active_theme
    key = _resolve_existing_key(name_or_key) or "midnight"
    _active_theme_key = key
    _active_theme = THEMES[key]


def list_themes() -> list[str]:
    return [name for _key, name, _builtin in list_theme_entries()]


def list_theme_entries() -> list[tuple[str, str, bool]]:
    entries: list[tuple[str, str, bool]] = []
    for key in _BUILTIN_ORDER:
        theme = THEMES.get(key)
        if theme is not None:
            entries.append((key, theme.name, True))

    custom_entries = sorted(
        ((key, theme.name) for key, theme in _custom_themes.items() if not key.startswith("__")),
        key=lambda item: item[1].lower(),
    )
    for key, name in custom_entries:
        entries.append((key, name, False))
    return entries


def get_theme(name_or_key: str | None = None) -> Theme:
    key = _resolve_existing_key(name_or_key) if name_or_key else current_theme_key()
    if key is None:
        key = "midnight"
    return THEMES[key]


def is_builtin_theme(name_or_key: str) -> bool:
    key = _resolve_existing_key(name_or_key)
    return bool(key in _BUILTIN_THEMES) if key else False


def save_custom_theme(name: str, values: dict, key: str | None = None, persist: bool = True) -> str:
    base_name = str(name or "").strip() or "Custom Theme"
    if key is not None and str(key).startswith("__"):
        requested_key = "__" + _normalize_theme_key(str(key).strip("_"))
    else:
        requested_key = _normalize_theme_key(key or base_name)
    if requested_key in _BUILTIN_THEMES and requested_key not in _custom_themes:
        requested_key = f"custom_{requested_key}"

    theme_key = requested_key
    if theme_key in _BUILTIN_THEMES:
        theme_key = f"custom_{theme_key}"

    _custom_themes[theme_key] = _theme_from_payload(base_name, values if isinstance(values, dict) else {})
    _rebuild_theme_registry()
    if persist:
        _save_custom_themes()
    return theme_key


def delete_custom_theme(name_or_key: str) -> bool:
    key = _resolve_existing_key(name_or_key)
    if key is None or key in _BUILTIN_THEMES or key not in _custom_themes:
        return False

    del _custom_themes[key]
    _rebuild_theme_registry()
    _save_custom_themes()

    global _active_theme_key, _active_theme
    if _active_theme_key not in THEMES:
        _active_theme_key = "midnight"
        _active_theme = THEMES[_active_theme_key]
    return True


_load_custom_themes()
