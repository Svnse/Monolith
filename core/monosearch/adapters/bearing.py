"""bearing adapter (spec §5). Wraps addons.system.bearing.audit — the append-only
JSONL history of Bearing transitions + verifier outcomes.

A FAITHFUL MAP of the real store: every audit row carries only {ts, turn_id, kind}
(written by audit.append) plus per-kind extra fields merged verbatim
(failed_rules: list on `rejected`/`grounding_failed`; slots_changed on `applied`;
streak/fault_emitted on `escalated`). There is NO rejection_kind/outcome/seq column
— rows carry no sequence at all, so we synthesize one from the JSONL line index
(audit.read_all tags each row with `_seq`).

provenance is a constant SELF (all rows are the runtime's own bearing self-checks).
evidence_tier is DERIVED (these are verifier verdicts computed from envelopes, not
literal records of an external event).

recurrence_key is the KIND for every row that has one (rejected / grounding_failed /
applied / cleared / escalated). The meaningful bearing-cycle signal is "rejected ×N"
— how often each kind of bearing outcome recurs — so we aggregate on kind rather than
fragmenting on the exact failed-rule set (the {kind}|{rules} form split "rejected
[D1,D3]" from "rejected [D1]" into separate count-1 buckets and buried the signal).
A row with no kind has no recurrence_key → None. `ts` is an ISO string in the store,
parsed to epoch here.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from addons.system.bearing import audit as _audit
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "bearing_audit:"


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


# Bearing audit kinds that represent a PROBLEM (these recur into the signal);
# successes (applied/cleared) are deliberately NOT keyed so `recurring` means
# "problems that recur" — consistent with the faults adapter, which only ever
# keys failure kinds. (advisor catch, 2026-06-03)
_PROBLEM_KINDS = frozenset({"rejected", "grounding_failed", "escalated", "parse_error"})


class BearingAdapter(SourceAdapter):
    name = "bearing"
    evidence_tier = EvidenceTier.DERIVED

    def _provenance(self, row: dict[str, Any]) -> Provenance:
        # All bearing audit rows are the runtime's own self-checks.
        return Provenance.SELF

    def _recurrence_key(self, row: dict[str, Any]) -> str | None:
        # KIND-LEVEL recurrence, but ONLY for problem kinds: "rejected ×N" is the
        # bearing-cycle signal worth surfacing; `applied`/`cleared` are successes
        # and must not dilute `recurring`. (The previous form keyed EVERY kind,
        # injecting operational successes into the "what keeps coming up" signal.)
        kind = row.get("kind")
        if kind in _PROBLEM_KINDS:
            return str(kind)
        return None

    def _seq(self, row: dict[str, Any]) -> int:
        # read_all tags each row with its physical JSONL line index. Fall back to 0
        # for hand-built rows / unexpected shapes so an id is always synthesizable.
        seq = row.get("_seq")
        try:
            return int(seq)
        except (TypeError, ValueError):
            return 0

    def _text(self, row: dict[str, Any]) -> str:
        kind = row.get("kind", "")
        rules = row.get("failed_rules")
        if rules:
            return f"[{kind}] failed_rules: {', '.join(str(r) for r in rules)}".strip()
        detail = row.get("detail")
        if detail:
            return f"[{kind}] {detail}".strip()
        return f"[{kind}]".strip()

    def _to_record(self, row: dict[str, Any]) -> Record:
        turn_id = row.get("turn_id")
        seq = self._seq(row)
        meta: dict[str, Any] = {
            "kind": row.get("kind"),
            "turn_id": turn_id,
            "seq": seq,
        }
        if "failed_rules" in row:
            meta["failed_rules"] = row.get("failed_rules")
        return Record(
            namespaced_id=f"{_ID_PREFIX}{turn_id}:{seq}",
            source=self.name,
            provenance=self._provenance(row),
            recurrence_key=self._recurrence_key(row),
            text=self._text(row),
            metadata=meta,
            ts=_iso_to_epoch(row.get("ts")),
            evidence_tier=EvidenceTier.DERIVED,
        )

    def _haystack(self, row: dict[str, Any]) -> str:
        parts = [str(row.get("kind", "")), str(row.get("turn_id", ""))]
        rules = row.get("failed_rules")
        if rules:
            parts.extend(str(r) for r in rules)
        detail = row.get("detail")
        if detail:
            parts.append(str(detail))
        return " ".join(parts).lower()

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        filters = filters or {}
        rows = _audit.read_all()
        q = (query or "").strip().lower()
        if q:
            rows = [r for r in rows if q in self._haystack(r)]
        kind = filters.get("kind")
        if kind:
            rows = [r for r in rows if r.get("kind") == kind]
        recs = [self._to_record(r) for r in rows]
        since = filters.get("since")
        if since:
            cutoff = _iso_to_epoch(since)
            if cutoff is not None:
                recs = [r for r in recs if r.ts is None or r.ts >= cutoff]
        return recs[: max(0, int(limit))]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        for row in _audit.read_all():
            rec = self._to_record(row)
            if rec.namespaced_id == namespaced_id:
                return rec
        return None

    def list(self, filters: dict, limit: int) -> list[Record]:
        # The iteration path salience.rebuild uses — full-file read, oldest first.
        rows = _audit.read_all()
        return [self._to_record(r) for r in rows[: max(0, int(limit))]]
