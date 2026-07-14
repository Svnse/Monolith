from __future__ import annotations

import pytest

import addons.system.bearing as bearing_pkg
from addons.system.bearing import compiler
from addons.system.bearing.provider import BearingProvider


def test_build_addon_returns_none_when_kill_switch_off(monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_V1", "0")
    addon = bearing_pkg.build_addon()
    assert addon is None


def test_build_addon_returns_object_when_kill_switch_on(monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_V1", "1")
    addon = bearing_pkg.build_addon()
    assert addon is not None
    assert isinstance(addon.provider, BearingProvider)
    assert addon.interceptor is compiler.bearing_interceptor


def test_build_addon_default_is_enabled(monkeypatch) -> None:
    monkeypatch.delenv("MONOLITH_BEARING_V1", raising=False)
    addon = bearing_pkg.build_addon()
    assert addon is not None


def test_build_addon_interceptor_is_callable(monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_V1", "1")
    addon = bearing_pkg.build_addon()
    assert addon is not None
    assert callable(addon.interceptor)
