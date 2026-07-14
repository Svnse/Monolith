from __future__ import annotations

from datetime import datetime, timezone

from core.health import get_runtime_health
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "health:"


class HealthAdapter(SourceAdapter):
    name = "runtime_health"
    evidence_tier = EvidenceTier.TELEMETRY

    def _all_records(self) -> list[Record]:
        now = datetime.now(timezone.utc).timestamp()
        health = get_runtime_health(probe_endpoint_now=False)
        records: list[Record] = []
        for check in health.checks:
            recurrence_key = f"health:{check.name}:{check.status}" if check.status != "ok" else None
            records.append(
                Record(
                    namespaced_id=f"{_ID_PREFIX}{check.name}",
                    source=self.name,
                    provenance=Provenance.SELF,
                    recurrence_key=recurrence_key,
                    text=f"[{check.status}] {check.name}: {check.message}",
                    metadata=check.to_dict(),
                    ts=now,
                    evidence_tier=self.evidence_tier,
                )
            )
        return records

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        filters = filters or {}
        recs = self._all_records()
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
        for rec in self._all_records():
            if rec.namespaced_id == namespaced_id:
                return rec
        return None

    def list(self, filters: dict, limit: int) -> list[Record]:
        return self.search("", filters or {}, limit)
