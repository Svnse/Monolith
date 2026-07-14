"""outcome_traces adapter — wraps core.turn_trace's outcome reads.

The ratings ledger (outcome_traces; kind='rating' carries failure_tags) was the
dark store behind "monosearch premise -> 0": premise_unchecked lives here, and no
adapter read it. Registering this lets the existing substring matcher reach the
ledger (search) and lets rated failures flow into the salience ledger (recurring,
and thus monopulse).

Two per-source choices (see spec §4 — recurrence_key/provenance are NOT uniform):
  * provenance = USER  — a rating is an EXTERNAL evaluation of the runtime (E or
    the examiner), not the runtime's own self-detection (which is fault_traces=SELF).
  * evidence_tier = DERIVED — a graded judgment, ranked just under the LITERAL
    self-detected faults in an un-scoped merge (irrelevant when scoped to ratings).
"""
from __future__ import annotations

from datetime import datetime

from core import turn_trace as _tt
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "outcome:"


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def _primary_tag(row: "_tt.OutcomeReadRow") -> str | None:
    tags = row.metadata.get("failure_tags") if isinstance(row.metadata, dict) else None
    if isinstance(tags, list):
        for t in tags:
            t = str(t).strip()
            if t:
                return t
    return None


class OutcomeTraceAdapter(SourceAdapter):
    name = "outcome_traces"
    evidence_tier = EvidenceTier.DERIVED

    def _recurrence_key(self, row: "_tt.OutcomeReadRow") -> str | None:
        # TAG-LEVEL recurrence (mirrors FaultAdapter's kind-level key): "what I
        # keep getting rated down for" recurs by failure_tag, so repeated
        # premise_unchecked aggregates instead of scattering into count-1 rows.
        # A rating with no failure_tag (a clean/positive score) is not a recurring
        # failure -> None (not salience-eligible, but still searchable).
        # Multi-tag ratings key on the primary tag (the trainer concentrate-fires
        # single tags); the full list rides in metadata/text so search matches any.
        return _primary_tag(row)

    def _to_record(self, row: "_tt.OutcomeReadRow") -> Record:
        meta = row.metadata if isinstance(row.metadata, dict) else {}
        tags = meta.get("failure_tags")
        note = meta.get("surface_note")
        head = f"[{row.kind}]" if row.rating_value is None else f"[{row.kind} {row.rating_value}]"
        text = " ".join(
            p for p in (head, row.reason or "", str(note) if note else "") if p
        ).strip()
        return Record(
            namespaced_id=f"{_ID_PREFIX}{row.id}",
            source=self.name,
            provenance=Provenance.USER,
            recurrence_key=self._recurrence_key(row),
            text=text,
            metadata={
                "kind": row.kind,
                "rating_value": row.rating_value,
                "failure_tags": tags if isinstance(tags, list) else [],
                "turn_id": row.turn_id,
                "recorded_at": row.recorded_at,
            },
            ts=_iso_to_epoch(row.recorded_at),
            evidence_tier=EvidenceTier.DERIVED,
        )

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        since = (filters or {}).get("since")
        # Mirror FaultAdapter: the keyword is post-filtered in the read (no text
        # index), so fetch a broad window when querying, then take `limit` —
        # otherwise a small limit fetches only the most-recent N before filtering
        # and misses older matches.
        fetch = 200 if query else limit
        rows = _tt.read_recent_outcomes(fetch, since=since, keyword=query or None)
        return [self._to_record(r) for r in rows[:limit]]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        try:
            outcome_id = int(namespaced_id[len(_ID_PREFIX):])
        except ValueError:
            return None
        row = _tt.read_outcome(outcome_id)
        return self._to_record(row) if row is not None else None

    def list(self, filters: dict, limit: int) -> list[Record]:
        # Used by salience.rebuild. The ratings ledger is small, so a single
        # recent read covers it (no ascending paging as for the large fault table).
        rows = _tt.read_recent_outcomes(limit)
        return [self._to_record(r) for r in rows]
