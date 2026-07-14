from __future__ import annotations

from datetime import datetime, timezone

from core import plan_reminders as _reminders
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "reminder:"


def _iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _is_due(row: dict) -> bool:
    due = _iso_to_epoch(row.get("due_at"))
    if due is None:
        return False
    return row.get("status") == "pending" and due <= datetime.now(timezone.utc).timestamp()


class PlanReminderAdapter(SourceAdapter):
    name = "plan_reminders"
    evidence_tier = EvidenceTier.DERIVED

    def _to_record(self, row: dict) -> Record:
        due = _is_due(row)
        status = str(row.get("status") or "pending")
        recurrence_key = "due_reminder" if due else (f"reminder:{status}" if status != "seen" else None)
        text = f"[{status}] {row.get('message', '')}".strip()
        if row.get("due_at"):
            text += f" due={row['due_at']}"
        return Record(
            namespaced_id=f"{_ID_PREFIX}{row.get('reminder_uid', '')}",
            source=self.name,
            provenance=Provenance.SELF,
            recurrence_key=recurrence_key,
            text=text,
            metadata=dict(row) | {"due": due},
            ts=_iso_to_epoch(row.get("due_at")),
            evidence_tier=self.evidence_tier,
        )

    def _all_records(self, limit: int) -> list[Record]:
        return [self._to_record(row) for row in _reminders.list_reminders(limit=max(1, int(limit or 20)))]

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        filters = filters or {}
        recs = self._all_records(max(50, int(limit or 20)))
        q = (query or "").strip().lower()
        if q:
            recs = [r for r in recs if q in r.text.lower()]
        status = filters.get("status")
        if status:
            recs = [r for r in recs if r.metadata.get("status") == status]
        if filters.get("due") is True:
            recs = [r for r in recs if r.metadata.get("due") is True]
        return recs[: max(1, int(limit))]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        uid = namespaced_id[len(_ID_PREFIX):]
        for rec in self._all_records(500):
            if rec.namespaced_id == f"{_ID_PREFIX}{uid}":
                return rec
        return None

    def list(self, filters: dict, limit: int) -> list[Record]:
        return self.search("", filters or {}, limit)
