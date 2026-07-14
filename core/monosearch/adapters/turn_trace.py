"""turn_trace adapter (spec §5, Layers A/B/D). One Record per TURN.

Wraps the EXISTING joined reads in core.turn_trace — `get_turn_trace`,
`list_recent_turns`, `search_turns` — the Cognitive Darkness Map's "dead
readers": these reads exist but had no production caller; MonoSearch becomes it.

Scope narrowing (conscious, deviates from the full §5 row): the full spec row
describes per-record evidence tiers (frame LITERAL / outcome DERIVED / stage
TELEMETRY) and outcome-level recurrence. This adapter collapses that to ONE
LITERAL / self / recurrence_key=None record per turn — a turn is a unique
search/lookup unit, not a recurring one (the recurring fault/outcome signals are
served by the fault_traces + canonical_log adapters). recurrence_key is always
None: this is a lookup source, not a salience source.

provenance is a constant SELF — a turn is the runtime's own act.
"""
from __future__ import annotations

from datetime import datetime

from core import turn_trace as _tt
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "turn:"


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


class TurnTraceAdapter(SourceAdapter):
    name = "turn_trace"
    evidence_tier = EvidenceTier.LITERAL

    def _provenance(self, joined: "_tt.TurnTraceJoined") -> Provenance:
        # A turn is the runtime's own act — always self-sourced.
        return Provenance.SELF

    def _recurrence_key(self, joined: "_tt.TurnTraceJoined") -> str | None:
        # KIND-LEVEL note (advisor catch): faults recur by kind, but a TURN is
        # unique by construction (turn_id is a per-turn UUID). This is a
        # search/lookup source, NOT a recurrence source — every turn is its own
        # row, so a recurrence_key would only ever produce count-1 entries.
        # None => not salience-eligible.
        return None

    def _ts(self, joined: "_tt.TurnTraceJoined") -> float | None:
        # Prefer the frame's captured_at; fall back for substrate-only turns
        # (frame=None) to an outcome's recorded_at, then to the first stage's
        # entered_at. All three are ISO strings in the real store.
        if joined.frame is not None:
            ts = _iso_to_epoch(joined.frame.captured_at)
            if ts is not None:
                return ts
        for oc in joined.outcomes:
            ts = _iso_to_epoch(oc.recorded_at)
            if ts is not None:
                return ts
        if joined.stages:
            return _iso_to_epoch(joined.stages[0].entered_at)
        return None

    def _summary_text(self, joined: "_tt.TurnTraceJoined") -> str:
        # Short, deterministic turn summary — no LLM, real fields only.
        frame = joined.frame
        parts: list[str] = [f"turn {joined.turn_id}"]
        if frame is not None:
            if frame.effort_tier:
                parts.append(f"effort={frame.effort_tier}")
            if frame.backend:
                parts.append(f"backend={frame.backend}")
            if isinstance(frame.classification, dict):
                mode = frame.classification.get("conversation_mode")
                if mode:
                    parts.append(f"mode={mode}")
        rating = joined.summary.get("last_rating") if isinstance(joined.summary, dict) else None
        if rating is not None:
            parts.append(f"rating={rating}")
        errored = joined.summary.get("errored_stage_count") if isinstance(joined.summary, dict) else None
        if errored:
            parts.append(f"errored_stages={errored}")
        return " ".join(parts)

    def _to_record(self, joined: "_tt.TurnTraceJoined") -> Record:
        frame = joined.frame
        # Only real, present fields — be a faithful map. effort_tier /
        # classification / backend / total_chars live ONLY on the frame, so they
        # are absent for substrate-only turns (frame=None). rating comes from the
        # joined summary's last_rating (turn_trace already folds outcomes there).
        metadata: dict = {}
        if frame is not None:
            if frame.effort_tier:
                metadata["effort_tier"] = frame.effort_tier
            if isinstance(frame.classification, dict):
                metadata["classification"] = frame.classification
            if frame.backend:
                metadata["backend"] = frame.backend
            metadata["total_chars"] = frame.total_chars
            # Gate C (Source-Tier): surface the persisted tier on resurfaced
            # TURNS so the monosearch contract can require re-grounding
            # generation hits. Flag-gated → byte-identical when off. Clean for
            # turns (the tier lives on this same frame; no linkage gap). ACU
            # resurfacing is deferred to the linkage spec.
            if isinstance(frame.metadata, dict) and frame.metadata.get("source_tier"):
                from core.source_tier import source_tier_enabled
                if source_tier_enabled():
                    metadata["source_tier"] = frame.metadata["source_tier"]
                    _rt = frame.metadata.get("region_tiers")
                    if isinstance(_rt, dict):
                        metadata["region_tiers"] = _rt
        rating = joined.summary.get("last_rating") if isinstance(joined.summary, dict) else None
        if rating is not None:
            metadata["rating"] = rating
        if joined.parent_turn_id:
            metadata["parent_turn_id"] = joined.parent_turn_id
        ts = self._ts(joined)
        if ts is not None:
            metadata["ts"] = ts
        return Record(
            namespaced_id=f"{_ID_PREFIX}{joined.turn_id}",
            source=self.name,
            provenance=self._provenance(joined),
            recurrence_key=self._recurrence_key(joined),
            text=self._summary_text(joined),
            metadata=metadata,
            ts=ts,
            evidence_tier=EvidenceTier.LITERAL,
        )

    def _from_turn_id(self, turn_id: str) -> Record | None:
        joined = _tt.get_turn_trace(turn_id)
        return self._to_record(joined) if joined is not None else None

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        # turn_trace has NO free-text index over turns (search_turns filters by
        # since/until/backend/has_errored_stage only). So a non-empty `query`
        # with NO filters cannot be honoured: returning recent turns for a
        # keyword would be keyword-irrelevant noise. Return [] in that case —
        # turn_trace is a lookup/filter source, not a free-text search source.
        # (Empty query, or any filter present, still works: we map the filters
        # MonoSearch passes onto search_turns, else fall back to recent turns.)
        f = filters or {}
        since = f.get("since")
        until = f.get("until")
        backend = f.get("backend")
        has_errored = f.get("has_errored_stage")
        has_filters = bool(since or until or backend or has_errored is not None)
        if (query or "").strip() and not has_filters:
            return []
        if has_filters:
            summaries = _tt.search_turns(
                since=since, until=until, backend=backend,
                has_errored_stage=has_errored, limit=limit,
            )
        else:
            summaries = _tt.list_recent_turns(limit=limit)
        # Enrich each lean summary via the rich joined read — effort_tier /
        # classification / rating live only there, not on TurnTraceSummary.
        out: list[Record] = []
        for s in summaries:
            rec = self._from_turn_id(s.turn_id)
            if rec is not None:
                out.append(rec)
        return out

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        turn_id = namespaced_id[len(_ID_PREFIX):]
        if not turn_id:
            return None
        return self._from_turn_id(turn_id)

    def list(self, filters: dict, limit: int) -> list[Record]:
        # salience.rebuild iteration path. recurrence_key is None on every turn
        # record, so the ledger reads NOTHING from this source — "make it work"
        # means yield valid Records without crashing, NOT a paged full scan.
        out: list[Record] = []
        for s in _tt.list_recent_turns(limit=limit):
            rec = self._from_turn_id(s.turn_id)
            if rec is not None:
                out.append(rec)
        return out
