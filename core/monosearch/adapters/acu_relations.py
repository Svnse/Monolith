"""acatalepsy-relations adapter — surfaces the acu_relations contradiction/overlap
graph (the ``contradicts`` edges ``verifier.py`` writes, the ``overlaps`` edges
``intake.py`` writes) that nothing read on a live turn until now. Wraps the read
primitives in ``core.acatalepsy.relations`` (the same way ``FaultAdapter`` wraps
``fault_response``'s reads).

An edge's text is its two RESOLVED endpoints, so:
  * recurrence_key is always None — an edge is a one-off relationship, not a
    recurring unit (not salience-eligible);
  * keyword search post-filters the rendered text (the read layer has no keyword
    param — acu_relations has no text column of its own);
  * a dangling edge (endpoint ACU deleted -> NULL canonical from the LEFT JOIN) is
    SKIPPED — an unrenderable relationship is noise;
  * a LOCKED Origin-0 endpoint is RENDERED (the namespaced_id is ``relation:``,
    not ``acu:``, so we name it inside a relationship, we do not double-serve the
    locked ACU as its own record) with the locked/state flags surfaced in metadata.

provenance is a constant SELF (every edge is the runtime's own derived judgment),
evidence_tier = DERIVED (an interpreted relationship, same tier as an ACU).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from core.acatalepsy import relations
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "relation:"


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


class RelationsAdapter(SourceAdapter):
    name = "acatalepsy-relations"
    evidence_tier = EvidenceTier.DERIVED

    def _to_record(self, row: dict[str, Any]) -> Record | None:
        # Dangling edge: an endpoint ACU is gone (LEFT JOIN -> NULL canonical). An
        # unrenderable relationship is noise -> skip (spec D2).
        src_text = row.get("source_canonical")
        tgt_text = row.get("target_canonical")
        if src_text is None or tgt_text is None:
            return None
        relation = row.get("relation")
        return Record(
            namespaced_id=f"{_ID_PREFIX}{row['id']}",
            source=self.name,
            provenance=Provenance.SELF,
            recurrence_key=None,
            text=f"#{row['source_id']} {src_text} --[{relation}]--> #{row['target_id']} {tgt_text}",
            metadata={
                "relation": relation,
                "source_id": row.get("source_id"),
                "target_id": row.get("target_id"),
                "score": row.get("score"),
                "source_state": row.get("source_state"),
                "target_state": row.get("target_state"),
                "source_locked": bool(row.get("source_locked")),
                "target_locked": bool(row.get("target_locked")),
            },
            ts=_iso_to_epoch(row.get("created_at")),
            evidence_tier=EvidenceTier.DERIVED,
        )

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        f = filters or {}
        # No text index on edges; when searching, fetch a broad window and
        # post-filter the rendered text (mirrors FaultAdapter), else dangling
        # skips + endpoint text wouldn't be matchable.
        fetch = 200 if query else int(limit)
        rows = relations.read_recent(limit=fetch, since=f.get("since"), relation=f.get("relation"))
        recs = [r for r in (self._to_record(row) for row in rows) if r is not None]
        if query:
            q = query.lower()
            recs = [r for r in recs if q in r.text.lower()]
        return recs[: int(limit)]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        try:
            edge_id = int(namespaced_id[len(_ID_PREFIX):])
        except ValueError:
            return None
        row = relations.read_one(edge_id)
        return self._to_record(row) if row is not None else None

    def list(self, filters: dict, limit: int) -> list[Record]:
        # The salience.rebuild iteration path — page ascending by id, drop dangling,
        # optional relation filter, take limit.
        f = filters or {}
        rel = f.get("relation")
        out: list[Record] = []
        after = 0
        while len(out) < int(limit):
            page = relations.read_since_id(after, limit=min(500, int(limit) - len(out)))
            if not page:
                break
            for row in page:
                rec = self._to_record(row)
                if rec is not None and (rel is None or rec.metadata.get("relation") == rel):
                    out.append(rec)
            after = page[-1]["id"]
        return out[: int(limit)]
