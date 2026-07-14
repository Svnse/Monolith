"""
Tests for core/themes.py — registry, custom themes, persistence, edge cases.
All tests use a temp directory so they never touch real config files.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers: reload themes module with a patched CONFIG_DIR
# ---------------------------------------------------------------------------

def _load_themes(tmp_path: Path):
    """Import (or re-import) core.themes with CONFIG_DIR pointing at tmp_path."""
    # Patch core.paths so CONFIG_DIR resolves to tmp_path
    paths_mod = types.ModuleType("core.paths")
    paths_mod.CONFIG_DIR = tmp_path
    sys.modules["core.paths"] = paths_mod

    # Force a fresh import of core.themes
    sys.modules.pop("core.themes", None)
    import core.themes as t
    return t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def th(tmp_path):
    """Fresh themes module isolated to tmp_path."""
    mod = _load_themes(tmp_path)
    yield mod
    # Cleanup: remove from sys.modules so next fixture gets a clean slate
    sys.modules.pop("core.themes", None)
    sys.modules.pop("core.paths", None)


# ---------------------------------------------------------------------------
# Built-in registry
# ---------------------------------------------------------------------------

class TestBuiltins:
    def test_five_builtin_themes(self, th):
        assert len(th._BUILTIN_THEMES) == 5

    def test_all_builtin_keys_present(self, th):
        for key in ("midnight", "obsidian", "monolithic", "slate", "arctic"):
            assert key in th.THEMES

    def test_builtin_order_matches_keys(self, th):
        assert set(th._BUILTIN_ORDER) == set(th._BUILTIN_THEMES.keys())

    def test_default_active_theme_is_midnight(self, th):
        assert th.current_theme_key() == "midnight"
        assert th.current_theme().name == "Midnight"

    def test_theme_dataclass_has_all_fields(self, th):
        t = th.MIDNIGHT
        assert t.bg_main and t.fg_text and t.accent_primary
        assert t.gradient_color

    def test_all_builtin_themes_have_names(self, th):
        for key, theme in th._BUILTIN_THEMES.items():
            assert isinstance(theme.name, str) and theme.name.strip()

    def test_editable_fields_excludes_name(self, th):
        fields = th.editable_theme_fields()
        assert "name" not in fields
        assert len(fields) == len(th.Theme.__dataclass_fields__) - 1

    def test_is_builtin_theme_true_for_builtins(self, th):
        for key in th._BUILTIN_THEMES:
            assert th.is_builtin_theme(key) is True

    def test_is_builtin_theme_false_for_unknown(self, th):
        assert th.is_builtin_theme("nonexistent") is False


# ---------------------------------------------------------------------------
# apply_theme / current_theme
# ---------------------------------------------------------------------------

class TestApplyTheme:
    def test_apply_by_key(self, th):
        th.apply_theme("obsidian")
        assert th.current_theme_key() == "obsidian"
        assert th.current_theme().name == "Obsidian"

    def test_apply_by_display_name(self, th):
        th.apply_theme("Arctic")
        assert th.current_theme_key() == "arctic"

    def test_apply_by_display_name_case_insensitive(self, th):
        th.apply_theme("SLATE")
        assert th.current_theme_key() == "slate"

    def test_apply_unknown_falls_back_to_midnight(self, th):
        th.apply_theme("doesnotexist")
        assert th.current_theme_key() == "midnight"

    def test_apply_empty_string_falls_back_to_midnight(self, th):
        th.apply_theme("")
        assert th.current_theme_key() == "midnight"

    def test_apply_then_get_theme_matches(self, th):
        th.apply_theme("monolithic")
        assert th.get_theme() == th.MONOLITHIC


# ---------------------------------------------------------------------------
# get_theme
# ---------------------------------------------------------------------------

class TestGetTheme:
    def test_get_by_key(self, th):
        assert th.get_theme("arctic") == th.ARCTIC

    def test_get_by_name(self, th):
        assert th.get_theme("Midnight") == th.MIDNIGHT

    def test_get_none_returns_active(self, th):
        th.apply_theme("slate")
        assert th.get_theme(None) == th.SLATE

    def test_get_unknown_falls_back_to_midnight(self, th):
        assert th.get_theme("bogus") == th.MIDNIGHT


# ---------------------------------------------------------------------------
# list_themes / list_theme_entries
# ---------------------------------------------------------------------------

class TestListThemes:
    def test_list_themes_returns_names(self, th):
        names = th.list_themes()
        assert "Midnight" in names
        assert "Arctic" in names

    def test_list_theme_entries_structure(self, th):
        entries = th.list_theme_entries()
        for key, name, is_builtin in entries:
            assert isinstance(key, str)
            assert isinstance(name, str)
            assert isinstance(is_builtin, bool)

    def test_builtins_come_before_custom(self, th):
        th.save_custom_theme("Zzzz", {}, persist=False)
        entries = th.list_theme_entries()
        builtin_indices = [i for i, (_, _, b) in enumerate(entries) if b]
        custom_indices = [i for i, (_, _, b) in enumerate(entries) if not b]
        if builtin_indices and custom_indices:
            assert max(builtin_indices) < min(custom_indices)


# ---------------------------------------------------------------------------
# theme_to_dict / _theme_from_payload round-trip
# ---------------------------------------------------------------------------

class TestThemeDict:
    def test_theme_to_dict_has_name(self, th):
        d = th.theme_to_dict(th.MIDNIGHT)
        assert d["name"] == "Midnight"

    def test_theme_to_dict_round_trip(self, th):
        d = th.theme_to_dict(th.OBSIDIAN)
        rebuilt = th._theme_from_payload(d["name"], d)
        assert rebuilt == th.OBSIDIAN

    def test_theme_from_payload_fills_missing_with_midnight_defaults(self, th):
        partial = {"bg_main": "#ff0000"}
        result = th._theme_from_payload("Test", partial)
        assert result.bg_main == "#ff0000"
        assert result.fg_text == th.MIDNIGHT.fg_text  # default filled in


# ---------------------------------------------------------------------------
# Custom theme CRUD
# ---------------------------------------------------------------------------

class TestCustomThemes:
    def test_save_custom_theme_appears_in_registry(self, th):
        key = th.save_custom_theme("My Theme", {}, persist=False)
        assert key in th.THEMES

    def test_save_custom_theme_not_builtin(self, th):
        key = th.save_custom_theme("My Theme", {}, persist=False)
        assert th.is_builtin_theme(key) is False

    def test_save_custom_theme_appears_in_list_entries(self, th):
        th.save_custom_theme("My Theme", {}, persist=False)
        custom = [e for e in th.list_theme_entries() if not e[2]]
        assert any("My Theme" in e[1] for e in custom)

    def test_delete_custom_theme_removes_from_registry(self, th):
        key = th.save_custom_theme("Temp", {}, persist=False)
        assert key in th.THEMES
        result = th.delete_custom_theme(key)
        assert result is True
        assert key not in th.THEMES

    def test_delete_builtin_returns_false(self, th):
        assert th.delete_custom_theme("midnight") is False

    def test_delete_nonexistent_returns_false(self, th):
        assert th.delete_custom_theme("totally_made_up") is False

    def test_delete_active_theme_resets_to_midnight(self, th):
        key = th.save_custom_theme("Active One", {}, persist=False)
        th.apply_theme(key)
        assert th.current_theme_key() == key
        th.delete_custom_theme(key)
        assert th.current_theme_key() == "midnight"

    def test_save_with_explicit_key(self, th):
        key = th.save_custom_theme("Named", {}, key="my_key", persist=False)
        assert key == "my_key"
        assert "my_key" in th.THEMES

    def test_save_custom_cannot_overwrite_builtin_key(self, th):
        key = th.save_custom_theme("Fake Midnight", {}, key="midnight", persist=False)
        assert key != "midnight"
        assert th.THEMES["midnight"] == th.MIDNIGHT  # builtin untouched

    def test_custom_theme_count(self, th):
        th.save_custom_theme("A", {}, persist=False)
        th.save_custom_theme("B", {}, persist=False)
        custom = [e for e in th.list_theme_entries() if not e[2]]
        assert len(custom) == 2


# ---------------------------------------------------------------------------
# Draft theme (__draft__)
# ---------------------------------------------------------------------------

class TestDraftTheme:
    def test_draft_not_in_list_entries(self, th):
        th.save_custom_theme("Draft", {}, key="__draft__", persist=False)
        keys = [e[0] for e in th.list_theme_entries()]
        assert "__draft__" not in keys

    def test_draft_is_in_themes_registry(self, th):
        key = th.save_custom_theme("Draft", {}, key="__draft__", persist=False)
        assert key in th.THEMES

    def test_draft_is_not_builtin(self, th):
        key = th.save_custom_theme("Draft", {}, key="__draft__", persist=False)
        assert th.is_builtin_theme(key) is False


# ---------------------------------------------------------------------------
# Key normalization
# ---------------------------------------------------------------------------

class TestKeyNormalization:
    def test_spaces_become_underscores(self, th):
        assert th._normalize_theme_key("My Theme") == "my_theme"

    def test_special_chars_stripped(self, th):
        assert th._normalize_theme_key("Hello!!World") == "hello_world"

    def test_empty_string_returns_fallback(self, th):
        assert th._normalize_theme_key("") == "custom_theme"

    def test_uppercase_lowercased(self, th):
        assert th._normalize_theme_key("MIDNIGHT") == "midnight"

    def test_leading_trailing_underscores_stripped(self, th):
        result = th._normalize_theme_key("__test__")
        assert not result.startswith("_")
        assert not result.endswith("_")


# ---------------------------------------------------------------------------
# Persistence (JSON round-trip)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_persists_to_json(self, th, tmp_path):
        th.save_custom_theme("Persist Me", {"bg_main": "#abcdef"}, persist=True)
        json_path = tmp_path / "themes_custom.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert any(v.get("name") == "Persist Me" for v in data.values())

    def test_reload_restores_custom_themes(self, tmp_path):
        # First load: save a custom theme
        th1 = _load_themes(tmp_path)
        key = th1.save_custom_theme("Reload Me", {"bg_main": "#123456"}, persist=True)
        sys.modules.pop("core.themes", None)

        # Second load: theme should be present
        th2 = _load_themes(tmp_path)
        assert key in th2.THEMES
        assert th2.THEMES[key].bg_main == "#123456"
        sys.modules.pop("core.themes", None)
        sys.modules.pop("core.paths", None)

    def test_delete_removes_from_json(self, th, tmp_path):
        key = th.save_custom_theme("Gone Soon", {}, persist=True)
        th.delete_custom_theme(key)
        json_path = tmp_path / "themes_custom.json"
        if json_path.exists():
            data = json.loads(json_path.read_text())
            assert key not in data

    def test_corrupt_json_does_not_crash(self, tmp_path):
        json_path = tmp_path / "themes_custom.json"
        json_path.write_text("{ not valid json !!!")
        # Should not raise
        th = _load_themes(tmp_path)
        assert "midnight" in th.THEMES
        sys.modules.pop("core.themes", None)
        sys.modules.pop("core.paths", None)

    def test_non_dict_json_does_not_crash(self, tmp_path):
        json_path = tmp_path / "themes_custom.json"
        json_path.write_text("[1, 2, 3]")
        th = _load_themes(tmp_path)
        assert "midnight" in th.THEMES
        sys.modules.pop("core.themes", None)
        sys.modules.pop("core.paths", None)

    def test_builtin_key_in_json_is_ignored(self, tmp_path):
        json_path = tmp_path / "themes_custom.json"
        payload = {"midnight": {"name": "Fake Midnight", "bg_main": "#ff0000"}}
        json_path.write_text(json.dumps(payload))
        th = _load_themes(tmp_path)
        assert th.THEMES["midnight"] == th.MIDNIGHT  # builtin wins
        sys.modules.pop("core.themes", None)
        sys.modules.pop("core.paths", None)


# ---------------------------------------------------------------------------
# llm_config — system prompt & config
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def _load(self):
        sys.modules.pop("core.llm_config", None)
        import core.llm_config as lc
        return lc

    def test_master_prompt_not_empty(self):
        lc = self._load()
        assert len(lc.MASTER_PROMPT) > 100

    def test_master_prompt_contains_identity(self):
        lc = self._load()
        assert "Monolith" in lc.MASTER_PROMPT

    def test_build_system_prompt_includes_tool_catalog(self):
        lc = self._load()
        prompt = lc.build_system_prompt()
        assert prompt.startswith("You are Monolith")
        assert prompt == lc.build_system_prompt({})
        assert "{skills_catalog}" not in prompt
        assert "[TOOL DISCOVERY KERNEL]" in prompt
        assert '<tool_call>{"name":"monosearch","arguments":' in prompt

    def test_build_system_prompt_injects_identity_block(self, monkeypatch):
        """{identity_block} placeholder is replaced with load_identity() output."""
        lc = self._load()

        def _fake_identity():
            return "## TEST IDENTITY\nUser: Alice\nRole: tester"

        monkeypatch.setattr(lc, "load_identity", _fake_identity)
        prompt = lc.build_system_prompt()
        # Block was substituted, not left raw
        assert "{identity_block}" not in prompt
        # Identity content is present
        assert "TEST IDENTITY" in prompt
        assert "User: Alice" in prompt
        # The [IDENTITY] section header from system.md is preserved
        assert "[IDENTITY]" in prompt

    def test_default_config_has_system_prompt(self):
        lc = self._load()
        assert "system_prompt" in lc.DEFAULT_CONFIG
        assert lc.DEFAULT_CONFIG["system_prompt"] == lc.MASTER_PROMPT

    def test_save_config_does_not_persist_system_prompt(self, tmp_path):
        lc = self._load()
        out = tmp_path / "cfg.json"

        def _fake_update_config_section(_section, payload, persist=True):
            assert persist is True
            out.write_text(json.dumps(payload), encoding="utf-8")

        lc.update_config_section = _fake_update_config_section
        lc.save_config({"system_prompt": "secret", "temp": 0.5})
        data = json.loads(out.read_text())
        assert "system_prompt" not in data
        assert data["temp"] == 0.5

