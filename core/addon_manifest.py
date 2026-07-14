import json
from pathlib import Path
from typing import Any

from core.paths import ADDON_MANIFEST
from core.slug import normalize_id


def _slugify(name: str) -> str:
    """Addon IDs use underscores and preserve dots."""
    return normalize_id(name, "addon")


def _ensure_unique_id(entries: list[dict[str, Any]], base: str) -> str:
    existing = {e.get("id", "") for e in entries}
    if base not in existing:
        return base
    i = 2
    while True:
        candidate = f"{base}_{i}"
        if candidate not in existing:
            return candidate
        i += 1


def load_addon_manifest() -> list[dict[str, Any]]:
    path = Path(ADDON_MANIFEST)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return data
    return []


def save_addon_manifest(entries: list[dict[str, Any]]) -> None:
    path = Path(ADDON_MANIFEST)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")


def upsert_addon(entry: dict[str, Any]) -> dict[str, Any]:
    entries = load_addon_manifest()
    addon_id = str(entry.get("id", "")).strip()
    if not addon_id:
        addon_id = _ensure_unique_id(entries, _slugify(str(entry.get("name", ""))))
        entry["id"] = addon_id
    updated = False
    for i, existing in enumerate(entries):
        if existing.get("id") == addon_id:
            entries[i] = entry
            updated = True
            break
    if not updated:
        entries.append(entry)
    save_addon_manifest(entries)
    return entry


def delete_addon(addon_id: str) -> bool:
    entries = load_addon_manifest()
    new_entries = [e for e in entries if e.get("id") != addon_id]
    if len(new_entries) == len(entries):
        return False
    save_addon_manifest(new_entries)
    return True


def add_external_addon(
    name: str,
    entry_path: str | None = None,
    command: str | None = None,
    workdir: str | None = None,
    icon: str | None = None,
) -> dict[str, Any]:
    entries = load_addon_manifest()
    addon_id = _ensure_unique_id(entries, _slugify(name))
    entry = {
        "id": addon_id,
        "name": name.strip() or addon_id,
        "entry": str(Path(entry_path)) if entry_path else "",
        "mode": "external_py",
    }
    if command:
        entry["command"] = command
    if workdir:
        entry["workdir"] = workdir
    if icon:
        entry["icon"] = icon
    entries.append(entry)
    save_addon_manifest(entries)
    return entry
