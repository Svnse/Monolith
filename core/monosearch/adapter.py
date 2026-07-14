"""The SourceAdapter contract (spec §4). Each adapter wraps one store's existing
in-process read primitives and maps rows to Records. Concrete adapters also
implement the internal _to_record / _recurrence_key / _provenance helpers; only
the three public methods are part of the ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.monosearch.record import EvidenceTier, Record


class SourceAdapter(ABC):
    name: str
    evidence_tier: EvidenceTier

    @abstractmethod
    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        """Keyword/metadata search, live against this source's own store."""

    @abstractmethod
    def get(self, namespaced_id: str) -> Record | None:
        """Direct lookup by this source's namespaced id (e.g. 'fault:991')."""

    @abstractmethod
    def list(self, filters: dict, limit: int) -> list[Record]:
        """Filtered listing (also the iteration path used by salience.rebuild)."""
