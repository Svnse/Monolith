from __future__ import annotations

from datetime import datetime

from core import investigation_runs as _runs
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "investigation:"


def _iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


class InvestigationAdapter(SourceAdapter):
    name = "investigations"
    evidence_tier = EvidenceTier.DERIVED

    def _to_record(self, run: "_runs.InvestigationRun") -> Record:
        source_count = len(run.source_refs)
        text = f"[{run.status}] {run.goal} ({source_count} source{'s' if source_count != 1 else ''})"
        if run.synthesis_markdown:
            text += f" - {run.synthesis_markdown[:180].strip()}"
        recurrence_key = f"investigation:{run.status}" if run.status != "done" else None
        return Record(
            namespaced_id=f"{_ID_PREFIX}{run.run_id}",
            source=self.name,
            provenance=Provenance.SELF,
            recurrence_key=recurrence_key,
            text=text,
            metadata={
                "run_id": run.run_id,
                "goal": run.goal,
                "status": run.status,
                "created_at": run.created_at,
                "updated_at": run.updated_at,
                "source_count": source_count,
                "linked_plan_uids": list(run.linked_plan_uids),
                "linked_acu_ids": list(run.linked_acu_ids),
            },
            ts=_iso_to_epoch(run.updated_at),
            evidence_tier=self.evidence_tier,
        )

    def _all_records(self, limit: int) -> list[Record]:
        return [self._to_record(run) for run in _runs.list_investigations(limit=max(1, int(limit or 20)))]

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        filters = filters or {}
        recs = self._all_records(max(50, int(limit or 20)))
        q = (query or "").strip().lower()
        if q:
            recs = [r for r in recs if q in r.text.lower()]
        status = filters.get("status")
        if status:
            recs = [r for r in recs if r.metadata.get("status") == status]
        return recs[: max(1, int(limit))]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        run_id = namespaced_id[len(_ID_PREFIX):]
        run = _runs.load_investigation(run_id)
        return self._to_record(run) if run is not None else None

    def list(self, filters: dict, limit: int) -> list[Record]:
        return self.search("", filters or {}, limit)
