"""canonical_log adapter (spec §5). Wraps core.acatalepsy.canonical_log reads.
Owns the raw conversation + the learning-event timeline. NOTE: the table has NO
turn_id column — the dimension is session_id + acu_id.
"""
from __future__ import annotations

from core.acatalepsy import canonical_log as _cl
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "clog:"


class CanonicalLogAdapter(SourceAdapter):
    name = "canonical_log"
    evidence_tier = EvidenceTier.LITERAL

    def _provenance(self, ev: "_cl.Event") -> Provenance:
        return Provenance.USER if ev.kind == "user_message" else Provenance.SELF

    def _recurrence_key(self, ev: "_cl.Event") -> str | None:
        # canonical_log is a SEARCH/LOOKUP source, NOT a recurrence source. Empirically
        # (2026-06-03) text-hash keys let a boilerplate message recur ×68 and dominate
        # `recurring`, burying the real "attend to this" signal (faults, bearing
        # rejections). A repeated conversation message is not a problem-that-recurs.
        # So messages no longer feed salience — same call as continuity/turn_trace.
        return None

    def _text(self, ev: "_cl.Event") -> str:
        if isinstance(ev.payload, dict) and "text" in ev.payload:
            return str(ev.payload["text"])
        return f"[{ev.kind}]"

    def _to_record(self, ev: "_cl.Event") -> Record:
        return Record(
            namespaced_id=f"{_ID_PREFIX}{ev.event_id}",
            source=self.name,
            provenance=self._provenance(ev),
            recurrence_key=self._recurrence_key(ev),
            text=self._text(ev),
            metadata={"kind": ev.kind, "session_id": ev.session_id, "acu_id": ev.acu_id},
            ts=ev.ts,
            evidence_tier=EvidenceTier.LITERAL,
        )

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        evs = _cl.search(query, limit=limit) if query else _cl.read_recent(limit=limit)
        since = (filters or {}).get("since")
        recs = [self._to_record(e) for e in evs]
        if since:
            cutoff = _iso_since_to_epoch(since)
            if cutoff is not None:
                recs = [r for r in recs if r.ts is None or r.ts >= cutoff]
        return recs

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        try:
            event_id = int(namespaced_id[len(_ID_PREFIX):])
        except ValueError:
            return None
        ev = _cl.read_one(event_id)
        return self._to_record(ev) if ev is not None else None

    def list(self, filters: dict, limit: int) -> list[Record]:
        return [self._to_record(e) for e in _cl.read_recent(limit=limit)]


def _iso_since_to_epoch(since: str) -> float | None:
    from datetime import datetime
    try:
        return datetime.fromisoformat(since).timestamp()
    except (ValueError, TypeError):
        return None
