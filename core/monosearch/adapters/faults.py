"""fault_traces adapter (spec §5). Wraps core.fault_response reads. The single
highest-value dark store — the faults the model has never been able to see.
provenance is a constant SELF (all rows are the runtime's own self-detections).
"""
from __future__ import annotations

from datetime import datetime

from core import fault_response as _fr
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "fault:"


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


class FaultAdapter(SourceAdapter):
    name = "fault_traces"
    evidence_tier = EvidenceTier.LITERAL

    def _recurrence_key(self, rec: "_fr.FaultRecord") -> str:
        # KIND-LEVEL recurrence (advisor catch, 2026-06-03): "what I keep failing"
        # recurs by fault_kind, NOT by exact evidence. In prod the evidence is the
        # actual leaked snippet — different every turn — so hashing it would collapse
        # the signal to a pile of count-1 entries and `failing` would never surface
        # "I keep tripping think_leak". The recurring UNIT for a fault is its kind.
        return rec.fault_kind

    def _to_record(self, rec: "_fr.FaultRecord") -> Record:
        return Record(
            namespaced_id=f"{_ID_PREFIX}{rec.id}",
            source=self.name,
            provenance=Provenance.SELF,
            recurrence_key=self._recurrence_key(rec),
            text=f"[{rec.fault_kind}] {rec.evidence or ''}".strip(),
            metadata={
                "fault_kind": rec.fault_kind,
                "severity": rec.severity,
                "detector_name": rec.detector_name,
                "turn_id": rec.turn_id,
                "emitted_at": rec.detected_at,
            },
            ts=_iso_to_epoch(rec.detected_at),
            evidence_tier=EvidenceTier.LITERAL,
        )

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        since = (filters or {}).get("since")
        kind = (filters or {}).get("fault_kind")
        # The keyword is post-filtered (the reads have no text index), so when
        # searching we must fetch a BROAD window and narrow it — otherwise a small
        # `limit` fetches only the most-recent N rows BEFORE filtering and misses
        # matches beyond them ("search faults for 'tool'" -> 0 despite tool_no_fire
        # ×36). Read up to the cap, filter, then take `limit`.
        fetch = 200 if query else limit
        if kind:
            rows = _fr.read_by_kind(kind, limit=fetch, since=since, keyword=query or None)
        else:
            rows = _fr.read_recent(limit=fetch, since=since, keyword=query or None)
        return [self._to_record(r) for r in rows[:limit]]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        try:
            fault_id = int(namespaced_id[len(_ID_PREFIX):])
        except ValueError:
            return None
        rec = _fr.read_one(fault_id)
        return self._to_record(rec) if rec is not None else None

    def list(self, filters: dict, limit: int) -> list[Record]:
        # Used by salience.rebuild — iterate via the paged ascending read.
        out: list[Record] = []
        after = 0
        while len(out) < limit:
            page = _fr.read_since_id(after, limit=min(500, limit - len(out)))
            if not page:
                break
            out.extend(self._to_record(r) for r in page)
            after = page[-1].id
        return out
