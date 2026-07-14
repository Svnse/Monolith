from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from ui.panels.cloud_profile_menu import build_profile_menu


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_menu_lists_profiles_and_marks_active(app):
    cfg = {
        "active_cloud_profile": "openai|https://api.deepseek.com",
        "cloud_profiles": [
            {"api_provider": "openai", "api_base": "https://api.deepseek.com", "last_model": "deepseek-v4-pro"},
            {"api_provider": "anthropic", "api_base": "https://api.anthropic.com", "last_model": "claude-opus-4-8"},
        ],
    }
    calls = {}
    menu = build_profile_menu(
        config=cfg,
        on_switch=lambda pid: calls.setdefault("switch", pid),
        on_save=lambda: calls.setdefault("save", True),
        on_delete=lambda pid: calls.setdefault("delete", pid),
    )
    texts = [a.text() for a in menu.actions() if a.text()]
    assert any("DeepSeek" in t for t in texts)
    assert any("Anthropic" in t for t in texts)
    assert any("Save current" in t for t in texts)
    # active profile action is checkable + checked
    active = [a for a in menu.actions() if "DeepSeek" in a.text()][0]
    assert active.isChecked()
    # triggering a profile action fires on_switch with its id
    other = [a for a in menu.actions() if "Anthropic" in a.text()][0]
    other.trigger()
    assert calls["switch"] == "anthropic|https://api.anthropic.com"


def test_menu_empty_state(app):
    menu = build_profile_menu(
        config={"cloud_profiles": []},
        on_switch=lambda pid: None,
        on_save=lambda: None,
        on_delete=lambda pid: None,
    )
    texts = [a.text() for a in menu.actions() if a.text()]
    assert any("No saved profiles" in t for t in texts)
    assert any("Save current" in t for t in texts)
