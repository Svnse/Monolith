from __future__ import annotations

import yaml

import pytest

from core.config import _write_yaml


def test_write_yaml_is_atomic_and_preserves_existing_on_replace_failure(monkeypatch, tmp_path) -> None:
    path = tmp_path / "config.yaml"
    original = {"llm": {"temp": 0.5}}
    path.write_text(yaml.safe_dump(original, sort_keys=False), encoding="utf-8")

    def _fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr("core.config.os.replace", _fail_replace)

    with pytest.raises(OSError):
        _write_yaml(path, {"llm": {"temp": 0.9}})

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded == original
    assert not (tmp_path / "config.yaml.tmp").exists()
