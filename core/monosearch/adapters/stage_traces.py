"""Stage trace adapter.

Layer A was the original "dark" turn_trace layer: every interceptor/prompt stage
records what happened, but only pull tools could inspect it. This adapter keeps
the mapping read-only and selective: all stages are searchable, but only errored
stages contribute recurrence keys.
"""
from __future__ import annotations

from datetime import datetime

from core import turn_trace as _tt
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "stage:"


def _iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


class StageTraceAdapter(SourceAdapter):
    name = "stage_traces"
    evidence_tier = EvidenceTier.TELEMETRY

    def _to_record(self, stage: "_tt.StageTraceRecord") -> Record:
        recurrence_key = None
        if stage.outcome == "errored":
            recurrence_key = f"stage_error:{stage.stage_name}"
        text = f"[{stage.outcome}] {stage.stage_name}"
        if stage.outcome_reason:
            text += f" - {stage.outcome_reason}"
        return Record(
            namespaced_id=f"{_ID_PREFIX}{stage.turn_id}:{stage.seq}",
            source=self.name,
            provenance=Provenance.SELF,
            recurrence_key=recurrence_key,
            text=text,
            metadata={
                "turn_id": stage.turn_id,
                "parent_turn_id": stage.parent_turn_id,
                "seq": stage.seq,
                "stage_name": stage.stage_name,
                "stage_kind": stage.stage_kind,
                "outcome": stage.outcome,
                "outcome_reason": stage.outcome_reason,
                "messages_in": stage.messages_in,
                "messages_out": stage.messages_out,
            },
            ts=_iso_to_epoch(stage.entered_at),
            evidence_tier=self.evidence_tier,
        )

    def _recent_records(self, limit: int) -> list[Record]:
        out: list[Record] = []
        for summary in _tt.list_recent_turns(limit=max(20, int(limit or 20))):
            joined = _tt.get_turn_trace(summary.turn_id)
            if joined is None:
                continue
            out.extend(self._to_record(stage) for stage in joined.stages)
            if len(out) >= limit:
                break
        return out[: max(1, int(limit))]

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        filters = filters or {}
        recs = self._recent_records(max(50, int(limit or 20)))
        q = (query or "").strip().lower()
        if q:
            recs = [r for r in recs if q in r.text.lower()]
        outcome = filters.get("outcome")
        if outcome:
            recs = [r for r in recs if r.metadata.get("outcome") == outcome]
        stage_name = filters.get("stage_name")
        if stage_name:
            recs = [r for r in recs if r.metadata.get("stage_name") == stage_name]
        since = filters.get("since")
        if since:
            cutoff = _iso_to_epoch(str(since))
            if cutoff is not None:
                recs = [r for r in recs if r.ts is None or r.ts >= cutoff]
        return recs[: max(1, int(limit))]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        raw = namespaced_id[len(_ID_PREFIX):]
        if ":" not in raw:
            return None
        turn_id, seq_s = raw.rsplit(":", 1)
        try:
            seq = int(seq_s)
        except ValueError:
            return None
        joined = _tt.get_turn_trace(turn_id)
        if joined is None:
            return None
        for stage in joined.stages:
            if int(stage.seq) == seq:
                return self._to_record(stage)
        return None

    def list(self, filters: dict, limit: int) -> list[Record]:
        return self.search("", filters or {}, limit)
