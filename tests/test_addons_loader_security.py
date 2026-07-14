from __future__ import annotations

from pathlib import Path

from ui.addons.loader import load_manifest_addons
from ui.addons.registry import AddonRegistry


def test_loader_blocks_spec_py_when_disabled(monkeypatch, tmp_path: Path) -> None:
    spec_file = tmp_path / "addon_spec.py"
    spec_file.write_text("def get_addons():\n    return []\n", encoding="utf-8")
    monkeypatch.setattr(
        "ui.addons.loader.load_addon_manifest",
        lambda: [{"id": "spec_mod", "name": "Spec Mod", "mode": "spec_py", "entry": str(spec_file)}],
    )
    monkeypatch.setattr("ui.addons.loader._ALLOW_SPEC_PY", False)
    errors: list[str] = []

    registry = AddonRegistry()
    load_manifest_addons(registry, on_error=errors.append, replace_dynamic=True)

    assert errors
    assert "spec_py addons are disabled for security" in errors[0]
    assert list(registry.all()) == []


def test_loader_blocks_external_command_entries(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "ext.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(
        "ui.addons.loader.load_addon_manifest",
        lambda: [
            {
                "id": "ext_mod",
                "name": "Ext Mod",
                "mode": "external_py",
                "entry": str(script),
                "command": "powershell -NoProfile -Command calc",
            }
        ],
    )
    errors: list[str] = []

    registry = AddonRegistry()
    load_manifest_addons(registry, on_error=errors.append, replace_dynamic=True)

    assert errors
    assert "external command addons are disabled" in errors[0]
    assert list(registry.all()) == []


def test_loader_accepts_external_python_entry(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "ext.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(
        "ui.addons.loader.load_addon_manifest",
        lambda: [{"id": "ext_mod", "name": "Ext Mod", "mode": "external_py", "entry": str(script)}],
    )
    monkeypatch.setattr("ui.addons.loader._ALLOW_UNTRUSTED_EXTERNAL_ENTRY", True)
    errors: list[str] = []

    registry = AddonRegistry()
    load_manifest_addons(registry, on_error=errors.append, replace_dynamic=True)

    specs = list(registry.all())
    assert errors == []
    assert len(specs) == 1
    assert specs[0].id == "ext_mod"


def test_loader_blocks_untrusted_external_entry_by_default(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "ext.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(
        "ui.addons.loader.load_addon_manifest",
        lambda: [{"id": "ext_mod", "name": "Ext Mod", "mode": "external_py", "entry": str(script)}],
    )
    monkeypatch.setattr("ui.addons.loader._ALLOW_UNTRUSTED_EXTERNAL_ENTRY", False)
    errors: list[str] = []

    registry = AddonRegistry()
    load_manifest_addons(registry, on_error=errors.append, replace_dynamic=True)

    assert errors
    assert "outside trusted roots" in errors[0]
    assert list(registry.all()) == []


def test_loader_forces_workdir_to_entry_parent(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "ext.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(
        "ui.addons.loader.load_addon_manifest",
        lambda: [
            {
                "id": "ext_mod",
                "name": "Ext Mod",
                "mode": "external_py",
                "entry": str(script),
                "workdir": str(tmp_path / "elsewhere"),
            }
        ],
    )
    monkeypatch.setattr("ui.addons.loader._ALLOW_UNTRUSTED_EXTERNAL_ENTRY", True)
    errors: list[str] = []

    registry = AddonRegistry()
    load_manifest_addons(registry, on_error=errors.append, replace_dynamic=True)

    specs = list(registry.all())
    assert errors == []
    assert len(specs) == 1
    closure_values = [cell.cell_contents for cell in (specs[0].factory.__closure__ or ())]
    assert str(script.parent) in closure_values
