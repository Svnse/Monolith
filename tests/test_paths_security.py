import importlib
import sys
from pathlib import Path

import pytest


def _reload_paths(monkeypatch, **env):
    for key in ("MONOLITH_ROOT", "MONOLITH_ALLOW_UNANCHORED_ROOT", "APPDATA", "USERPROFILE", "HOME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    sys.modules.pop("core.paths", None)
    return importlib.import_module("core.paths")


def test_monolith_root_rejects_relative_env_path(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    appdata = tmp_path / "appdata"
    mod = _reload_paths(
        monkeypatch,
        MONOLITH_ROOT="relative\\unsafe",
        USERPROFILE=home,
        HOME=home,
        APPDATA=appdata,
    )
    expected = (appdata / "Monolith").resolve()
    assert mod.MONOLITH_ROOT == expected


def test_monolith_root_rejects_unanchored_absolute_path(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    appdata = tmp_path / "appdata"
    outsider = tmp_path / "outsider" / "monolith"
    mod = _reload_paths(
        monkeypatch,
        MONOLITH_ROOT=outsider,
        USERPROFILE=home,
        HOME=home,
        APPDATA=appdata,
    )
    expected = (appdata / "Monolith").resolve()
    assert mod.MONOLITH_ROOT == expected


def test_monolith_root_allows_unanchored_path_when_opted_in(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    appdata = tmp_path / "appdata"
    outsider = tmp_path / "outsider" / "monolith"
    mod = _reload_paths(
        monkeypatch,
        MONOLITH_ROOT=outsider,
        MONOLITH_ALLOW_UNANCHORED_ROOT="1",
        USERPROFILE=home,
        HOME=home,
        APPDATA=appdata,
    )
    assert mod.MONOLITH_ROOT == outsider.resolve()


def test_ensure_safe_local_path_accepts_regular_file_path(tmp_path: Path) -> None:
    from core.paths import ensure_safe_local_path

    path = tmp_path / "logs" / "app.log"
    resolved = ensure_safe_local_path(path)
    assert resolved == path
    assert path.parent.exists()


def test_ensure_safe_local_path_rejects_symlink_leaf(tmp_path: Path) -> None:
    from core.paths import ensure_safe_local_path

    target = tmp_path / "target.log"
    target.write_text("ok", encoding="utf-8")
    link = tmp_path / "link.log"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError, PermissionError):
        pytest.skip("Symlinks unavailable in this environment")

    with pytest.raises(OSError):
        ensure_safe_local_path(link)
