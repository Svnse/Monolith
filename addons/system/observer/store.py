"""Observer V0 store.

The Observer owns only its advisory next-turn snapshot. It does not mutate
ACUs, Bearing, canonical_log, identity, or user-visible chat text.
"""
from __future__ import annotations

import json
import os
from typing import Any

from core.paths import CONFIG_DIR


_STORE_PATH = CONFIG_DIR / "observer.json"
_SCHEMA_VERSION = 1


def _empty() -> dict[str, Any]:
    return {"schema_version": _SCHEMA_VERSION, "latest": None}


def read_store() -> dict[str, Any]:
    if not _STORE_PATH.exists():
        return _empty()
    try:
        data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    data.setdefault("schema_version", _SCHEMA_VERSION)
    data.setdefault("latest", None)
    return data


def write_latest(snapshot: dict[str, Any]) -> None:
    data = read_store()
    data["latest"] = dict(snapshot)
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STORE_PATH.with_name(_STORE_PATH.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, _STORE_PATH)


def read_latest() -> dict[str, Any] | None:
    raw = read_store().get("latest")
    return raw if isinstance(raw, dict) else None


def clear() -> None:
    if _STORE_PATH.exists():
        _STORE_PATH.unlink()
