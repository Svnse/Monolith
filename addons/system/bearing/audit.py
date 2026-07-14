"""Bearing audit — owns CONFIG_DIR/bearing.audit.jsonl.

Append-only history of Bearing transitions and verifier outcomes.

Replayable: the JSONL alone is sufficient to reconstruct what Bearing
saw across turns. No in-memory-only state.

Event kinds:
  "applied"           — structural verifier passed; Bearing state updated
  "rejected"          — structural verifier failed; Bearing unchanged
  "grounding_failed"  — grounding verifier failed (post-commit); referent downgraded
  "escalated"         — N=3 streak; emit_fault was called
  "cleared"           — explicit bearing_clear
  "staleness_nudged"  — channel-staleness nudge injected this turn (streak tracked)
  "staleness_cleared" — a tracked staleness episode resolved (mismatch gone)

Audit failures must never break Bearing operation — append() catches
all exceptions and silently drops. The cost of a missed audit row is
lower than the cost of a generation-path failure.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from core.paths import CONFIG_DIR

_AUDIT_PATH = CONFIG_DIR / "bearing.audit.jsonl"

VALID_KINDS = frozenset({
    "applied", "rejected", "grounding_failed", "escalated", "cleared",
    "staleness_nudged", "staleness_cleared",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(kind: str, turn_id: str, **fields: Any) -> None:
    """Append a JSONL row. Safe under any error condition.

    Always writes ts + turn_id + kind. Caller passes additional fields as
    keyword args; they're merged into the row verbatim.
    """
    try:
        if kind not in VALID_KINDS:
            return
        row: dict[str, Any] = {
            "ts": _now_iso(),
            "turn_id": str(turn_id),
            "kind": kind,
        }
        for k, v in fields.items():
            row[k] = v
        _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        # Audit must not break generation path.
        pass


def read_recent(limit: int = 20) -> list[dict[str, Any]]:
    """Return up to *limit* most recent audit rows, newest last.

    Reading is best-effort: malformed lines are skipped.
    """
    if not _AUDIT_PATH.exists():
        return []
    try:
        with _AUDIT_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[-max(1, int(limit)):]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def read_all() -> list[dict[str, Any]]:
    """Return EVERY audit row, oldest first, each tagged with its 0-based JSONL
    line index in ``_seq``.

    ``read_recent`` is tail-only (it slices the last *limit* lines), which loses
    older history and gives no stable line position. MonoSearch needs both the
    full history (to count recurrence across all turns) and a per-row sequence to
    synthesize ids — JSONL rows themselves carry no seq/rowid. ``_seq`` is the raw
    physical line index (blank/malformed lines included in the count) so an id
    stays stable for a given append-only file. Reading is best-effort: blank and
    malformed lines are skipped from the output but still advance ``_seq``.
    """
    if not _AUDIT_PATH.exists():
        return []
    try:
        with _AUDIT_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            obj = dict(obj)
            obj["_seq"] = idx
            rows.append(obj)
    return rows
