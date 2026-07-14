from __future__ import annotations

import json

import pytest

import core.identity as identity_mod
from core.operators import OperatorManager


def test_save_identity_preserves_existing_on_replace_failure(monkeypatch, tmp_path) -> None:
    identity_path = tmp_path / "identity.md"
    identity_path.write_text("original\n", encoding="utf-8")
    monkeypatch.setattr(identity_mod, "IDENTITY_PATH", identity_path)

    def _fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr("core.identity.os.replace", _fail_replace)

    with pytest.raises(OSError):
        identity_mod.save_identity("changed")

    assert identity_path.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob("identity.md.*.tmp")) == []


def test_save_operator_preserves_existing_on_replace_failure(monkeypatch, tmp_path) -> None:
    manager = OperatorManager()
    manager._operators_dir = tmp_path / "operators"
    path = manager._path_for_name("Atlas")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": "Atlas", "config": {"temperature": 0.5}}), encoding="utf-8")

    def _fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr("core.operators.os.replace", _fail_replace)

    with pytest.raises(OSError):
        manager.save_operator("Atlas", {"config": {"temperature": 0.9}})

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["config"]["temperature"] == 0.5
    assert list(path.parent.glob(f"{path.name}.*.tmp")) == []
