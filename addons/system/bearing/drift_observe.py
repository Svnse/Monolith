"""Frame-drift OBSERVE layer (v0) — log only, never nudge.

Every turn (flag-gated), logs the cheap drift signal — lexical content overlap
between bearing.current_frame and the recent user asks — to
CONFIG_DIR/frame_drift.ledger.jsonl. Acting on drift (the nudge) is a SEPARATE,
later flag: offline calibration (tools/frame_drift) showed the cheap signal is a
weak proxy (F1 ~0.6) on truncated trace previews, so it earns actuation only
after the live ledger — clean full-text frames + asks, true frame age derivable
from consecutive same-frame rows — shows it separates DRIFT from MATCH.

Mirrors frame_observe.py: pure detection + safe append, NEVER mutates bearing,
NEVER raises into the chat path. Flag MONOLITH_FRAME_DRIFT_V1 (default OFF -> no
writes -> byte-identical to flag-off).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from core.paths import CONFIG_DIR

from . import drift

_FLAG_ENV = "MONOLITH_FRAME_DRIFT_V1"
_TRUTHY = {"1", "true", "yes", "on"}

# Module-level so tests can monkeypatch it (mirrors frame_observe._LEDGER).
_LEDGER = CONFIG_DIR / "frame_drift.ledger.jsonl"
_FRAME_CAP = 300
_ASK_CAP = 200


def enabled() -> bool:
    """True only when MONOLITH_FRAME_DRIFT_V1 is set to a truthy value (default OFF)."""
    return os.environ.get(_FLAG_ENV, "0").strip().lower() in _TRUTHY


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(turn_id: str, current_frame: str, messages: list[dict]) -> None:
    """Append one drift-signal row to the ledger. No-op if disabled; never raises.

    Logs the raw signal (overlap + frame + asks), NOT a binary nudge decision —
    age and thresholds are applied offline from the ledger so the operating point
    can be tuned without a code change (observe -> compare -> commit).
    """
    if not enabled():
        return
    try:
        frame = (current_frame or "").strip()
        if not frame:
            return
        asks = drift.recent_asks(messages, k=3)
        overlap, frame_tokens, ask_tokens = drift.overlap_of(frame, asks)
        row: dict[str, Any] = {
            "ts": _now_iso(),
            "turn_id": str(turn_id),
            "overlap": round(overlap, 3),
            "frame_tokens": frame_tokens,
            "ask_tokens": ask_tokens,
            "n_asks": len(asks),
            "current_frame": frame[:_FRAME_CAP],
            "recent_asks": [a[:_ASK_CAP] for a in asks],
        }
        _LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with _LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # Never raise into the chat path.
        pass


def read_recent(limit: int = 50) -> list[dict[str, Any]]:
    """Best-effort tail read of the drift ledger. Mirrors frame_observe.read_recent."""
    if not _LEDGER.exists():
        return []
    try:
        with _LEDGER.open("r", encoding="utf-8") as f:
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
