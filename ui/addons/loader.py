from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Callable

from core.addon_manifest import load_addon_manifest
from core.paths import MONOLITH_ROOT
from ui.addons.registry import AddonRegistry
from ui.addons.spec import AddonSpec
from ui.modules.external_process import ExternalProcessModule


_manifest_addon_ids: set[str] = set()
_manifest_entry_map: dict[str, list[str]] = {}
_ALLOW_SPEC_PY = os.environ.get("MONOLITH_ALLOW_SPEC_PY", "").strip().lower() in {"1", "true", "yes"}
_ALLOW_UNTRUSTED_EXTERNAL_ENTRY = os.environ.get("MONOLITH_ALLOW_UNTRUSTED_EXTERNAL_ADDON", "").strip().lower() in {"1", "true", "yes"}
_ADDON_ID_PATTERN = re.compile(r"^[a-z0-9_.-]{1,64}$")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRUSTED_ADDON_ROOTS = (
    (Path(MONOLITH_ROOT) / "addons").resolve(),
    _REPO_ROOT.resolve(),
)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_trusted_addon_entry(path: Path) -> bool:
    return any(_is_relative_to(path, root) for root in _TRUSTED_ADDON_ROOTS)


def _load_module_from_path(path: Path):
    module_name = f"monolith_addon_{abs(hash(str(path)))}"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load addon module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _external_factory_builder(name: str, entry_path: str, command: str | None, workdir: str | None):
    def _factory(ctx):
        return ExternalProcessModule(
            name=name,
            entry_path=entry_path,
            command=command,
            workdir=workdir,
            ui_bridge=getattr(ctx, "ui_bridge", None),
            state=getattr(ctx, "state", None),
            guard=getattr(ctx, "guard", None),
        )
    return _factory


def load_manifest_addons(
    registry: AddonRegistry,
    on_error: Callable[[str], None] | None = None,
    replace_dynamic: bool = False,
) -> None:
    if replace_dynamic and _manifest_addon_ids:
        for addon_id in list(_manifest_addon_ids):
            registry.unregister(addon_id)
        _manifest_addon_ids.clear()
        _manifest_entry_map.clear()

    entries = load_addon_manifest()
    if not entries:
        return

    for entry in entries:
        try:
            entry_path_raw = str(entry.get("entry", "")).strip()
            entry_path = None
            if entry_path_raw:
                candidate = Path(entry_path_raw).expanduser()
                if candidate.is_symlink():
                    raise RuntimeError(f"symlink addon entry rejected: {candidate}")
                try:
                    entry_path = candidate.resolve(strict=True)
                except OSError:
                    raise FileNotFoundError(f"Addon entry not found: {candidate}") from None
                if not entry_path.is_file():
                    raise RuntimeError(f"addon entry must be a file: {entry_path}")
                if entry_path.suffix.lower() != ".py":
                    raise RuntimeError("addon entry must be a .py file")

            mode = entry.get("mode", "external_py")
            addon_id = str(entry.get("id", "")).strip()
            name = str(entry.get("name", addon_id)).strip() or addon_id
            icon = entry.get("icon") or "*"
            command = str(entry.get("command", "")).strip() or None
            workdir = str(entry.get("workdir", "")).strip() or None

            if addon_id and not _ADDON_ID_PATTERN.match(addon_id):
                raise RuntimeError(f"invalid addon id: {addon_id}")

            if mode == "spec_py":
                if not _ALLOW_SPEC_PY:
                    raise RuntimeError("spec_py addons are disabled for security")
                if entry_path is None:
                    raise RuntimeError("spec_py entry missing entry path")
                if not _is_trusted_addon_entry(entry_path):
                    raise RuntimeError(f"spec_py entry is outside trusted roots: {entry_path}")
                module = _load_module_from_path(entry_path)
                get_addons = getattr(module, "get_addons", None)
                if not callable(get_addons):
                    raise RuntimeError("spec_py entry missing get_addons()")
                specs = list(get_addons())
                registered_ids = []
                for spec in specs:
                    if not isinstance(spec, AddonSpec):
                        raise RuntimeError("get_addons() must return AddonSpec instances")
                    if spec.id in [s.id for s in registry.all()]:
                        raise RuntimeError(f"Addon id already exists: {spec.id}")
                    registry.register(spec)
                    _manifest_addon_ids.add(spec.id)
                    registered_ids.append(spec.id)
                if addon_id:
                    _manifest_entry_map[addon_id] = registered_ids
                continue

            if not addon_id:
                raise RuntimeError("external_py entry missing id")

            if addon_id in [s.id for s in registry.all()]:
                raise RuntimeError(f"Addon id already exists: {addon_id}")

            if command:
                raise RuntimeError("external command addons are disabled; provide a Python entry file")
            if entry_path is None:
                raise RuntimeError("external_py entry missing entry")
            if not _ALLOW_UNTRUSTED_EXTERNAL_ENTRY and not _is_trusted_addon_entry(entry_path):
                raise RuntimeError(f"external entry is outside trusted roots: {entry_path}")
            workdir = str(entry_path.parent)

            registry.register(
                AddonSpec(
                    id=addon_id,
                    kind="module",
                    title=name.upper(),
                    icon=icon,
                    factory=_external_factory_builder(name, str(entry_path), None, workdir),
                    descriptor=None,
                )
            )
            _manifest_addon_ids.add(addon_id)
            if addon_id:
                _manifest_entry_map[addon_id] = [addon_id]
        except Exception as e:
            if on_error:
                on_error(f"[ADDON] {e}")
