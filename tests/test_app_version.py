from __future__ import annotations

import tomllib
from pathlib import Path

import bootstrap
import main as entrypoint
from core.version import APP_VERSION, __version__


def test_runtime_version_is_v1_release() -> None:
    assert APP_VERSION == "1.0.0"
    assert __version__ == APP_VERSION
    assert entrypoint.__version__ == APP_VERSION


def test_pyproject_version_matches_runtime_source() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == APP_VERSION
    assert "Development Status :: 5 - Production/Stable" in pyproject["project"]["classifiers"]


def test_event_ledger_receives_runtime_version(monkeypatch) -> None:
    database = object()
    captured: dict[str, object] = {}

    class FakeOverseer:
        def __init__(self, _guard, _ui_bridge) -> None:
            self.db = database

    def fake_ledger(db, *, app_version):
        captured["db"] = db
        captured["app_version"] = app_version
        return object()

    monkeypatch.setattr(bootstrap, "MonolithUI", lambda _state, _ui_bridge: object())
    monkeypatch.setattr(bootstrap, "OverseerWindow", FakeOverseer)
    monkeypatch.setattr(bootstrap, "EventLedger", fake_ledger)

    bootstrap.init_ui(object(), object(), object(), object())

    assert captured == {"db": database, "app_version": APP_VERSION}
