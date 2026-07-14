from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record
from core.paths import LOG_DIR

_ID_PREFIX = "lag:"
_LOG_PATH = LOG_DIR / "lag_watch.jsonl"


def _iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def _read_rows(path: Path | None = None) -> list[dict[str, Any]]:
    log_path = path or _LOG_PATH
    if not log_path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    for idx, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            row["_seq"] = idx
            out.append(row)
    return out


class LagWatchAdapter(SourceAdapter):
    name = "lag_watch"
    evidence_tier = EvidenceTier.TELEMETRY

    def _to_record(self, row: dict[str, Any]) -> Record:
        shape = row.get("system_class") if isinstance(row.get("system_class"), dict) else {}
        task_type = shape.get("task_type")
        effort = shape.get("effort_tier")
        recurrence_key = f"turn_shape:{task_type}:{effort}" if task_type and effort else None
        text = f"shape task={task_type or 'unknown'} effort={effort or 'unknown'} user={row.get('user_preview', '')}"
        return Record(
            namespaced_id=f"{_ID_PREFIX}{row.get('_seq', 0)}",
            source=self.name,
            provenance=Provenance.SELF,
            recurrence_key=recurrence_key,
            text=text.strip(),
            metadata={
                "seq": row.get("_seq"),
                "ts": row.get("ts"),
                "user_preview": row.get("user_preview"),
                "system_class": shape,
                "llm_class": row.get("llm_class"),
            },
            ts=_iso_to_epoch(row.get("ts")),
            evidence_tier=self.evidence_tier,
        )

    def _all_records(self, limit: int) -> list[Record]:
        rows = _read_rows()
        return [self._to_record(row) for row in rows[-max(1, int(limit or 20)):]]

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        filters = filters or {}
        recs = self._all_records(max(50, int(limit or 20)))
        q = (query or "").strip().lower()
        if q:
            recs = [r for r in recs if q in r.text.lower()]
        task_type = filters.get("task_type")
        if task_type:
            recs = [r for r in recs if (r.metadata.get("system_class") or {}).get("task_type") == task_type]
        return recs[-max(1, int(limit)):]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        try:
            seq = int(namespaced_id[len(_ID_PREFIX):])
        except ValueError:
            return None
        for row in _read_rows():
            if int(row.get("_seq", 0) or 0) == seq:
                return self._to_record(row)
        return None

    def list(self, filters: dict, limit: int) -> list[Record]:
        return self.search("", filters or {}, limit)
