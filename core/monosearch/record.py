"""The unified MonoSearch record envelope (spec §4).

Every adapter maps its native rows into a Record. The per-source recurrence_key
and provenance are NOT uniform — see each adapter. evidence_tier is an IntEnum so
that ordering IS the merge guard (invariant #3): a lower tier can never outrank a
higher one.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any


class EvidenceTier(IntEnum):
    LITERAL = 0      # a faithful record of what actually happened
    DERIVED = 1      # computed/interpreted from evidence
    TELEMETRY = 2    # operational metrics about the machinery
    SPECULATIVE = 3  # not-yet-accepted / aspirational


class Provenance(Enum):
    SELF = "self"
    USER = "user"
    WORLD = "world"


@dataclass(frozen=True)
class Record:
    namespaced_id: str            # e.g. "fault:991", "clog:1840"
    source: str                   # adapter name
    provenance: Provenance        # derived per source
    recurrence_key: str | None    # deterministic, non-LLM; None => not salience-eligible
    text: str                     # human/model-readable content
    metadata: dict[str, Any]      # source-specific filterable fields
    ts: float | None              # epoch seconds (for temporal filters + decay)
    evidence_tier: EvidenceTier
