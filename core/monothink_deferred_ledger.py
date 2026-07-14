"""Deferred concern ledger for MonoThink.

The ledger makes "pattern, not instance" mechanical. A single rating records a
reservation. A scaffold edit is eligible only when the same
(tag, section, failure_signature) recurs on distinct turns with distinct trace
spans, or when a mechanically validated critical invariant break occurs.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.paths import LOG_DIR


LEDGER_PATH = LOG_DIR / "monothink_deferred_ledger.jsonl"
MONITOR_PATH = LOG_DIR / "monothink_edit_monitor.jsonl"
EXPIRY_RATED_TURNS = int(os.environ.get("MONOLITH_MONOTHINK_LEDGER_EXPIRY", "30") or 30)


_SIG_RE = re.compile(r"[^a-z0-9]+")


def normalize_signature(text: str) -> str:
    sig = _SIG_RE.sub("-", str(text or "").strip().lower()).strip("-")
    sig = re.sub(r"-{2,}", "-", sig)
    return sig[:96] or "unspecified-mechanism"


def trace_span_id(turn_id: str, span: str | None) -> str:
    text = " ".join(str(span or "").split())
    if len(text) > 160:
        text = text[:160]
    return f"{turn_id}:{text}"


@dataclass(frozen=True)
class LedgerRow:
    tag: str
    section: str
    failure_signature: str
    signature_status: str
    first_seen_turn: str
    last_seen_turn: str
    count: int
    trace_spans: list[str] = field(default_factory=list)
    rater_notes: list[str] = field(default_factory=list)
    status: str = "open"
    last_seen_rated_index: int = 0
    updated_ts: float = 0.0

    def key(self) -> tuple[str, str, str]:
        return self.tag, self.section, self.failure_signature


class DeferredLedger:
    def __init__(self, path: Path | None = None, expiry_rated_turns: int = EXPIRY_RATED_TURNS):
        self.path = path or LEDGER_PATH
        self.expiry_rated_turns = max(1, int(expiry_rated_turns))

    def _append(self, row: LedgerRow) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    def all(self) -> list[LedgerRow]:
        if not self.path.exists():
            return []
        out: list[LedgerRow] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(LedgerRow(**obj))
                except Exception:
                    continue
        return out

    def current_rows(self) -> list[LedgerRow]:
        by_key: dict[tuple[str, str, str], LedgerRow] = {}
        for row in self.all():
            by_key[row.key()] = row
        return list(by_key.values())

    def find(self, tag: str, section: str, signature: str) -> LedgerRow | None:
        nsig = normalize_signature(signature)
        for row in self.current_rows():
            if row.tag == tag and row.section == section and row.failure_signature == nsig:
                return row
        return None

    def signatures_for(self, tag: str, section: str) -> list[str]:
        return sorted({
            row.failure_signature for row in self.current_rows()
            if row.tag == tag and row.section == section and row.status == "open"
        })

    def lookup_block(self, tag: str, section: str | None = None) -> str:
        rows = [
            r for r in self.current_rows()
            if r.tag == tag and r.status == "open" and (section is None or r.section == section)
        ]
        if not rows:
            return "LEDGER: no open reservations for this tag."
        lines = [
            "LEDGER: open reservations. Reuse an existing failure_signature when the mechanism matches."
        ]
        for row in sorted(rows, key=lambda r: (r.section, r.failure_signature)):
            lines.append(
                f"- ({row.tag}, {row.section}, {row.failure_signature}) -> "
                f"count={row.count}, status={row.signature_status}, "
                f"last_seen={row.last_seen_turn}"
            )
        return "\n".join(lines)

    def record_reservation(
        self,
        *,
        tag: str,
        section: str,
        failure_signature: str,
        signature_status: str,
        turn_id: str,
        trace_span: str | None,
        rater_note: str | None,
        rated_index: int,
        status: str = "open",
    ) -> LedgerRow:
        signature = normalize_signature(failure_signature)
        existing = self.find(tag, section, signature)
        span_id = trace_span_id(turn_id, trace_span)
        spans = list(existing.trace_spans) if existing else []
        notes = list(existing.rater_notes) if existing else []
        turns = {s.split(":", 1)[0] for s in spans}
        if turn_id not in turns and span_id not in spans:
            spans.append(span_id)
        if rater_note and str(rater_note).strip():
            notes.append(str(rater_note).strip()[:500])
            notes = notes[-8:]

        normalized_status = "canonical" if signature_status == "canonical" else "provisional"
        if existing and existing.failure_signature == signature and len(spans) >= 2:
            normalized_status = "canonical"

        distinct_turns = {s.split(":", 1)[0] for s in spans}
        row = LedgerRow(
            tag=tag,
            section=section or "NONE",
            failure_signature=signature,
            signature_status=normalized_status,
            first_seen_turn=existing.first_seen_turn if existing else str(turn_id),
            last_seen_turn=str(turn_id),
            count=len(distinct_turns) if normalized_status == "canonical" else min(len(distinct_turns), 1),
            trace_spans=spans,
            rater_notes=notes,
            status=status,
            last_seen_rated_index=max(int(rated_index or 0), existing.last_seen_rated_index if existing else 0),
            updated_ts=time.time(),
        )
        self._append(row)
        return row

    def mark_promoted(self, tag: str, section: str, failure_signature: str, *, turn_id: str, rated_index: int) -> None:
        existing = self.find(tag, section, failure_signature)
        if not existing:
            return
        self._append(LedgerRow(
            tag=existing.tag,
            section=existing.section,
            failure_signature=existing.failure_signature,
            signature_status=existing.signature_status,
            first_seen_turn=existing.first_seen_turn,
            last_seen_turn=str(turn_id),
            count=existing.count,
            trace_spans=list(existing.trace_spans),
            rater_notes=list(existing.rater_notes),
            status="promoted",
            last_seen_rated_index=max(int(rated_index or 0), existing.last_seen_rated_index),
            updated_ts=time.time(),
        ))

    def expire_stale(self, current_rated_index: int) -> list[LedgerRow]:
        expired: list[LedgerRow] = []
        now_idx = int(current_rated_index or 0)
        for row in self.current_rows():
            if row.status != "open":
                continue
            if now_idx - int(row.last_seen_rated_index or 0) < self.expiry_rated_turns:
                continue
            ex = LedgerRow(
                tag=row.tag,
                section=row.section,
                failure_signature=row.failure_signature,
                signature_status=row.signature_status,
                first_seen_turn=row.first_seen_turn,
                last_seen_turn=row.last_seen_turn,
                count=row.count,
                trace_spans=list(row.trace_spans),
                rater_notes=list(row.rater_notes),
                status="expired",
                last_seen_rated_index=row.last_seen_rated_index,
                updated_ts=time.time(),
            )
            self._append(ex)
            expired.append(ex)
        return expired


def next_rated_index(path: Path | None = None) -> int:
    ledger = DeferredLedger(path)
    indices = [row.last_seen_rated_index for row in ledger.current_rows()]
    return (max(indices) if indices else 0) + 1


def record_monitor_event(event: dict[str, Any], path: Path | None = None) -> None:
    path = path or MONITOR_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(event)
        row.setdefault("ts", time.time())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
