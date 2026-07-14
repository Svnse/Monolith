from __future__ import annotations

import pytest

from addons.system.bearing import compiler
from addons.system.bearing import kill_switch
from addons.system.bearing import plane
from addons.system.bearing import schema as bs
from addons.system.bearing import store


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    yield tmp_path


# ── kill_switch.is_enabled ──────────────────────────────────────────


def test_default_is_enabled(monkeypatch) -> None:
    monkeypatch.delenv(kill_switch.FLAG_ENV, raising=False)
    assert kill_switch.is_enabled() is True


def test_explicit_zero_disables(monkeypatch) -> None:
    monkeypatch.setenv(kill_switch.FLAG_ENV, "0")
    assert kill_switch.is_enabled() is False


def test_explicit_off_disables(monkeypatch) -> None:
    monkeypatch.setenv(kill_switch.FLAG_ENV, "off")
    assert kill_switch.is_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "True", "ON"])
def test_truthy_values_enable(monkeypatch, value) -> None:
    monkeypatch.setenv(kill_switch.FLAG_ENV, value)
    assert kill_switch.is_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "False", "OFF", "garbage"])
def test_falsy_values_disable(monkeypatch, value) -> None:
    monkeypatch.setenv(kill_switch.FLAG_ENV, value)
    assert kill_switch.is_enabled() is False


# ── plane registration ──────────────────────────────────────────────


def test_plane_name_is_bearing() -> None:
    assert plane.PLANE_NAME == "bearing"


def test_plane_modes_are_on_off() -> None:
    assert plane.valid_modes() == frozenset({"on", "off"})


def test_plane_is_enabled_mirrors_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv(kill_switch.FLAG_ENV, "0")
    assert plane.is_enabled() is False
    monkeypatch.setenv(kill_switch.FLAG_ENV, "1")
    assert plane.is_enabled() is True


def test_plane_uses_correct_flag_env() -> None:
    assert plane.PLANE_CONFIG.flag_env == "MONOLITH_BEARING_V1"


# ── end-to-end: kill switch suppresses block, parser, ops ───────────


def test_kill_switch_off_suppresses_interceptor(monkeypatch, tmp_store) -> None:
    """Compiler interceptor returns None when MONOLITH_BEARING_V1=0,
    regardless of whether Bearing has content."""
    monkeypatch.setenv("MONOLITH_BEARING_V1", "0")
    store.set_bearing(bs.Bearing(active_goal="present but suppressed"))
    messages = [{"role": "user", "content": "u"}]
    assert compiler.bearing_interceptor(messages, {}) is None


def test_kill_switch_on_allows_interceptor(monkeypatch, tmp_store) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_V1", "1")
    store.set_bearing(bs.Bearing(active_goal="x"))
    messages = [{"role": "user", "content": "u"}]
    result = compiler.bearing_interceptor(messages, {})
    assert result is not None
    assert "active_goal: x" in result[0]["content"]
