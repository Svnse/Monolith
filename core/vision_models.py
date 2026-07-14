"""Shared vision model scanning and matching utilities.

Used by both the SD module UI and the generate_image skill.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.paths import MONOLITH_ROOT

DEFAULT_MODEL_ROOT = MONOLITH_ROOT / "models" / "vision"


@dataclass
class ModelEntry:
    path: str
    label: str
    backend: str


def detect_backend(path: Path) -> str:
    """Heuristic backend detection without importing diffusers."""
    try:
        if path.is_dir():
            idx = path / "model_index.json"
            if idx.exists():
                data = json.loads(idx.read_text(encoding="utf-8"))
                cls = str(data.get("_class_name", "")).lower()
                if "flux" in cls:
                    return "flux"
                if "xl" in cls:
                    return "sdxl"
            name = path.name.lower()
            if "flux" in name:
                return "flux"
            if "xl" in name:
                return "sdxl"
            return "sd15"
        name = path.name.lower()
        if "flux" in name:
            return "flux"
        if "xl" in name:
            return "sdxl"
    except Exception:
        pass
    return "sd15"


def scan_model_root(root: Path | None = None) -> list[ModelEntry]:
    """Scan a directory for vision models (diffusers dirs, .safetensors, .ckpt)."""
    if root is None:
        root = DEFAULT_MODEL_ROOT
    if not root.exists() or not root.is_dir():
        return []

    entries: list[ModelEntry] = []
    seen: set[str] = set()

    # Diffusers directories
    try:
        for idx_file in root.rglob("model_index.json"):
            model_dir = idx_file.parent
            key = str(model_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            backend = detect_backend(model_dir)
            rel = model_dir.relative_to(root)
            label = f"[{backend}] {rel.as_posix()}/"
            entries.append(ModelEntry(path=str(model_dir), label=label, backend=backend))
    except Exception:
        pass

    # Single-file checkpoints
    for pattern in ("*.safetensors", "*.ckpt"):
        try:
            for file_path in root.rglob(pattern):
                key = str(file_path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                backend = detect_backend(file_path)
                rel = file_path.relative_to(root)
                label = f"[{backend}] {rel.as_posix()}"
                entries.append(ModelEntry(path=str(file_path), label=label, backend=backend))
        except Exception:
            pass

    entries.sort(key=lambda e: (e.backend, e.label.lower()))
    return entries


def fuzzy_match_model(query: str, entries: list[ModelEntry]) -> ModelEntry | None:
    """Match a partial model name against scanned entries. Returns best match or None."""
    if not query or not entries:
        return None
    q = query.strip().lower()

    # Exact path match
    for entry in entries:
        if entry.path.lower() == q:
            return entry

    # Exact label/name match
    for entry in entries:
        name = Path(entry.path).stem.lower()
        if name == q:
            return entry

    # Substring match
    matches = [e for e in entries if q in e.label.lower() or q in Path(e.path).stem.lower()]
    if len(matches) == 1:
        return matches[0]

    return None
