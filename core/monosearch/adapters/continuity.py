"""continuity adapter (spec §5). Wraps core.continuity.read — the first-person
pin store (CONFIG_DIR/continuity.json). Pins are the model's self-curated
session-survival notes, so provenance is a constant SELF and the tier is DERIVED
(interpreted self-state, not a literal event record).

recurrence_key is None for every pin. A pin is a UNIQUE self-curated commitment,
not a recurring event: hashing the pin text would yield all-count-1 keys (a
byte-identical re-pin is rare and not the signal we track), which only pollutes
the `recurring` selector. continuity is a lookup/search source like turn_trace,
not a salience-feed source.

NOTE on the store shape (read the source, don't guess): continuity.read returns
a dict {"active": [...pins...], "counts": {...}, "retired": [...]} — NOT a flat
list. There is NO `retired` field on a pin; active vs retired is derived from
WHICH LIST the pin came from. Timestamps are ISO strings (`created_at` on every
pin; `retired_at` added when retired); both are parsed to epoch for Record.ts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from core import continuity as _cont
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "continuity:"


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


class ContinuityAdapter(SourceAdapter):
    name = "continuity"
    evidence_tier = EvidenceTier.DERIVED

    def _provenance(self, pin: dict[str, Any]) -> Provenance:
        # Constant: pins are the model's own self-curation. SELF regardless of
        # the pin's `source` field (user_said/i_inferred/evidence describes WHERE
        # the lesson came from, not who authored the record — the model wrote it).
        return Provenance.SELF

    def _recurrence_key(self, pin: dict[str, Any]) -> None:
        # None for every pin. A pin is a UNIQUE self-curated commitment, not a
        # recurring event. Hashing the text (the previous design) yields all
        # count-1 keys — byte-identical re-pins are rare and not the signal we
        # track — and only pollutes the `recurring` selector. continuity is a
        # lookup/search source like turn_trace, never salience-eligible.
        return None

    def _to_record(self, pin: dict[str, Any], *, active: bool) -> Record:
        # ts: active pins date from created_at; retired pins date from retired_at
        # (fall back to created_at if somehow absent). Both ISO strings → epoch.
        if active:
            ts = _iso_to_epoch(pin.get("created_at"))
        else:
            ts = _iso_to_epoch(pin.get("retired_at")) or _iso_to_epoch(pin.get("created_at"))
        return Record(
            namespaced_id=f"{_ID_PREFIX}{pin.get('id')}",
            source=self.name,
            provenance=self._provenance(pin),
            recurrence_key=self._recurrence_key(pin),
            text=str(pin.get("text", "")),
            metadata={
                "active": active,  # derived from list membership, not a pin field
                "category": pin.get("category"),
                "pin_source": pin.get("source"),
                "evidence": pin.get("evidence"),
                "supersedes": pin.get("supersedes"),
                "created_at": pin.get("created_at"),
                "retired_at": pin.get("retired_at"),
                "retire_reason": pin.get("retire_reason"),
            },
            ts=ts,
            evidence_tier=EvidenceTier.DERIVED,
        )

    def _all_records(self) -> list[Record]:
        # retired_limit=0 → read() returns ALL retired pins (store-capped at 16),
        # not the default tail of 5. Faithful map: expose every pin the store has.
        data = _cont.read(include_retired=True, retired_limit=0)
        out: list[Record] = []
        for pin in data.get("active", []):
            out.append(self._to_record(pin, active=True))
        for pin in data.get("retired", []):
            out.append(self._to_record(pin, active=False))
        return out

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        recs = self._all_records()
        if query:
            needle = query.lower()
            recs = [r for r in recs if needle in r.text.lower()]
        since = (filters or {}).get("since")
        if since:
            cutoff = _iso_to_epoch(since)
            if cutoff is not None:
                recs = [r for r in recs if r.ts is None or r.ts >= cutoff]
        return recs[:limit]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        try:
            pin_id = int(namespaced_id[len(_ID_PREFIX):])
        except ValueError:
            return None
        # Search BOTH active and retired lists — a retired pin still has an id.
        target = f"{_ID_PREFIX}{pin_id}"
        for r in self._all_records():
            if r.namespaced_id == target:
                return r
        return None

    def list(self, filters: dict, limit: int) -> list[Record]:
        # The iteration path salience.rebuild uses. Returns all pins (active +
        # retired) so recurrence is counted across re-pins.
        recs = self._all_records()
        since = (filters or {}).get("since")
        if since:
            cutoff = _iso_to_epoch(since)
            if cutoff is not None:
                recs = [r for r in recs if r.ts is None or r.ts >= cutoff]
        return recs[:limit]
