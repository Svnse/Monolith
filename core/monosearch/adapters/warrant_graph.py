"""Acatalepsy warrant graph adapter.

This is a read-only MonoSearch surface over ``core.acatalepsy.warrant_graph``.
It does not create claims, evidence, warrants, or relation edges; it renders the
existing ACU provenance spine into a single model-readable record per claim root.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from core.acatalepsy import warrant_graph
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "warrant:"

_PROVENANCE = {
    "self": Provenance.SELF,
    "model": Provenance.SELF,
    "agent": Provenance.SELF,
    "auditor": Provenance.SELF,
    "auditor_monolith": Provenance.SELF,
    "user": Provenance.USER,
    "user_e": Provenance.USER,
    "world": Provenance.WORLD,
}


def _iso_to_epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _first_epoch(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        ts = _iso_to_epoch(row.get(key))
        if ts is not None:
            return ts
    return None


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _provenance(row: dict[str, Any]) -> Provenance:
    for key in ("acu_provenance", "candidate_source", "acu_source", "decided_by"):
        raw = _clean(row.get(key)).lower()
        if raw in _PROVENANCE:
            return _PROVENANCE[raw]
        if raw.startswith("user"):
            return Provenance.USER
        if raw.startswith("world"):
            return Provenance.WORLD
    return Provenance.SELF


def _node_id(row: dict[str, Any]) -> str:
    if row.get("node_kind") == "acu":
        return f"{_ID_PREFIX}acu:{row['acu_id']}"
    return f"{_ID_PREFIX}candidate:{row['candidate_id']}"


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "warrant_graph",
        "node_kind": row.get("node_kind"),
        "acu_id": row.get("acu_id"),
        "candidate_id": row.get("candidate_id"),
        "decision_id": row.get("decision_id"),
        "evidence_log_id": row.get("evidence_log_id"),
        "evidence_event_kind": row.get("evidence_event_kind"),
        "evidence_session_id": row.get("evidence_session_id"),
        "candidate_state": row.get("candidate_state"),
        "acu_state": row.get("acu_state"),
        "decision": row.get("decision"),
        "decided_by": row.get("decided_by"),
        "reject_reason": row.get("reject_reason"),
        "l_level": row.get("l_level"),
        "truth": row.get("truth"),
        "truth_confidence": row.get("truth_confidence"),
        "truth_method": row.get("truth_method"),
        "relation_count": row.get("relation_count"),
        "has_defeater": _has_defeater(row),
        "contradicts_acu_id": row.get("contradicts_acu_id"),
        "cid": row.get("cid"),
        "eqid": row.get("eqid"),
        "locked": bool(row.get("locked")),
    }


def _has_defeater(row: dict[str, Any]) -> bool:
    if row.get("contradicts_acu_id") is not None:
        return True
    if _clean(row.get("acu_state")) == "-inf":
        return True
    if _clean(row.get("truth")).lower() == "contradicted":
        return True
    return "contradicts" in _clean(row.get("relation_summary")).lower()


def _relation_lines(row: dict[str, Any]) -> list[str]:
    raw = _clean(row.get("relation_summary"))
    if not raw:
        return []
    return [f"- {line}" for line in raw.splitlines() if line.strip()]


def _text(row: dict[str, Any]) -> str:
    claim = _clean(row.get("claim_text")) or _clean(row.get("candidate_claim"))
    evidence_id = row.get("evidence_log_id") or row.get("evidence_event_id")
    evidence_span = _clean(row.get("evidence_span"))
    warrant = _clean(row.get("warrant_text"))
    relation_lines = _relation_lines(row)

    lines = [
        "[WARRANT GRAPH]",
        f"node: {_node_id(row)}",
        f"claim: {claim}",
    ]
    if row.get("node_kind") == "acu":
        lines.append(
            "status: "
            f"acu_state={_clean(row.get('acu_state')) or 'unknown'} "
            f"l_level={_clean(row.get('l_level')) or 'unknown'} "
            f"truth={_clean(row.get('truth')) or 'unverified'}"
        )
    else:
        lines.append(f"status: candidate_state={_clean(row.get('candidate_state')) or 'unknown'}")

    if evidence_id:
        start = row.get("evidence_char_start")
        end = row.get("evidence_char_end")
        lines.append(f"evidence: canonical_log:{evidence_id} chars={start}-{end} span={evidence_span}")
    elif evidence_span:
        lines.append(f"evidence: {evidence_span}")
    else:
        lines.append("evidence: none linked")

    lines.append(f"warrant: {warrant or 'none recorded'}")

    decision = _clean(row.get("decision"))
    if decision:
        by = _clean(row.get("decided_by")) or "unknown"
        at = _clean(row.get("decided_at")) or "unknown"
        lines.append(f"decision: {decision} by={by} at={at}")
        if row.get("reject_reason"):
            lines.append(f"reject_reason: {_clean(row.get('reject_reason'))}")
        if row.get("decision_note"):
            lines.append(f"decision_note: {_clean(row.get('decision_note'))}")
    else:
        lines.append("decision: unresolved")

    if row.get("truth_method") or row.get("truth_confidence") is not None or row.get("evidence_url"):
        lines.append(
            "truth_evidence: "
            f"method={_clean(row.get('truth_method')) or 'none'} "
            f"confidence={row.get('truth_confidence')} "
            f"url={_clean(row.get('evidence_url')) or 'none'}"
        )

    if _has_defeater(row):
        pieces = []
        if row.get("contradicts_acu_id") is not None:
            pieces.append(f"candidate_contradicts_acu:{row.get('contradicts_acu_id')}")
        if _clean(row.get("truth")).lower() == "contradicted":
            pieces.append("truth=contradicted")
        if _clean(row.get("acu_state")) == "-inf":
            pieces.append("state=-inf")
        if "contradicts" in _clean(row.get("relation_summary")).lower():
            pieces.append("relation=contradicts")
        lines.append(f"defeaters: {', '.join(pieces)}")
    else:
        lines.append("defeaters: none linked")

    if relation_lines:
        lines.append("relations:")
        lines.extend(relation_lines)
    else:
        lines.append("relations: none linked")

    return "\n".join(lines)


def _bool_filter(raw: Any) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


class WarrantGraphAdapter(SourceAdapter):
    name = "acatalepsy-warrants"
    evidence_tier = EvidenceTier.DERIVED

    def _to_record(self, row: dict[str, Any]) -> Record:
        return Record(
            namespaced_id=_node_id(row),
            source=self.name,
            provenance=_provenance(row),
            recurrence_key=None,
            text=_text(row),
            metadata=_metadata(row),
            ts=_first_epoch(
                row,
                ("decided_at", "candidate_created_at", "acu_last_seen", "acu_created_at", "evidence_event_ts"),
            ),
            evidence_tier=self.evidence_tier,
        )

    def _read(self, filters: dict, limit: int, *, fetch: int | None = None) -> list[Record]:
        f = filters or {}
        rows = warrant_graph.read_recent(
            limit=fetch or limit,
            since=f.get("since"),
            node_kind=f.get("node_kind"),
            state=f.get("state"),
            truth=f.get("truth"),
            decision=f.get("decision"),
            relation=f.get("relation"),
            has_defeater=_bool_filter(f.get("has_defeater")),
        )
        return [self._to_record(row) for row in rows]

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        fetch = 200 if query else int(limit)
        recs = self._read(filters or {}, int(limit), fetch=fetch)
        if query:
            q = str(query).lower()
            recs = [r for r in recs if q in r.text.lower()]
        return recs[: int(limit)]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        rest = namespaced_id[len(_ID_PREFIX):]
        try:
            kind, raw_id = rest.split(":", 1)
            node_id = int(raw_id)
        except ValueError:
            return None
        if kind not in {"acu", "candidate"}:
            return None
        row = warrant_graph.read_one(kind, node_id)
        return self._to_record(row) if row is not None else None

    def list(self, filters: dict, limit: int) -> list[Record]:
        return self._read(filters or {}, int(limit))
