from __future__ import annotations

import pytest

from addons.system.bearing import schema as bs
from addons.system.bearing import store
from addons.system.bearing.provider import BearingProvider


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    yield tmp_path


def test_get_active_goal_empty_when_no_bearing(tmp_store) -> None:
    provider = BearingProvider()
    assert provider.get_active_goal() == ""


def test_get_active_goal_returns_current(tmp_store) -> None:
    store.set_bearing(bs.Bearing(active_goal="ship Bearing V0"))
    provider = BearingProvider()
    assert provider.get_active_goal() == "ship Bearing V0"


def test_get_active_goal_picks_up_store_mutation(tmp_store) -> None:
    """Provider must NOT cache — Bearing changes between turns."""
    provider = BearingProvider()
    store.set_bearing(bs.Bearing(active_goal="goal A"))
    assert provider.get_active_goal() == "goal A"
    store.set_bearing(bs.Bearing(active_goal="goal B"))
    assert provider.get_active_goal() == "goal B"


def test_provider_swallows_store_errors(monkeypatch, tmp_store) -> None:
    """Provider is on a kernel hot path — must not raise even if store fails."""
    def boom():
        raise RuntimeError("simulated store failure")
    monkeypatch.setattr(store, "get_bearing", boom)
    provider = BearingProvider()
    assert provider.get_active_goal() == ""


def test_provider_has_no_mutation_surface() -> None:
    """V0 contract: provider exposes only get_active_goal(). No setters."""
    provider = BearingProvider()
    public_methods = [
        name for name in dir(provider)
        if not name.startswith("_") and callable(getattr(provider, name))
    ]
    # Any method that smells like mutation is a contract violation.
    forbidden = {"set", "update", "write", "delete", "clear", "mutate"}
    for method in public_methods:
        for fragment in forbidden:
            assert fragment not in method.lower(), (
                f"Provider exposes apparent mutation method: {method}"
            )
