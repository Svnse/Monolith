"""Adapter registry — one place adapters declare themselves. Keyed by name so
re-registration (e.g. at a re-init) replaces rather than duplicates.
"""
from __future__ import annotations

from core.monosearch.adapter import SourceAdapter

_ADAPTERS: dict[str, SourceAdapter] = {}


def register(adapter: SourceAdapter) -> None:
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str) -> SourceAdapter | None:
    return _ADAPTERS.get(name)


def all_adapters() -> list[SourceAdapter]:
    return list(_ADAPTERS.values())


def clear() -> None:
    """Test/re-init helper."""
    _ADAPTERS.clear()
