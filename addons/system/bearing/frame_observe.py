"""Observable frame fastpath (v0) — OBSERVE only, no mutation.

Records, every finalize turn (flag-gated), what the model's raw output did re:
<bearing_update> / <frame>, so silent-model vs emitted-and-dropped vs
no-op-warranted is inspectable. Writes CONFIG_DIR/frame.ledger.jsonl. Pure
detection + safe append; NEVER mutates bearing, NEVER raises into the chat path.
Flag MONOLITH_OBSERVABLE_FRAME_V0 (default OFF -> no writes -> byte-identical).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from core.paths import CONFIG_DIR

_FLAG_ENV = "MONOLITH_OBSERVABLE_FRAME_V0"
_TRUTHY = {"1", "true", "yes", "on"}

_FRAME_RE = re.compile(r"<frame>\s*(.*?)\s*</frame>", re.DOTALL | re.IGNORECASE)
_FRAME_CAP = 600

# Module-level ledger path so tests can monkeypatch it.
_LEDGER = CONFIG_DIR / "frame.ledger.jsonl"


def enabled() -> bool:
    """Return True only when MONOLITH_OBSERVABLE_FRAME_V0 is set to a truthy value."""
    return os.environ.get(_FLAG_ENV, "0").strip().lower() in _TRUTHY


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_FENCE_RE = re.compile(r"^(?:```+|`+)([\s\S]*?)(?:```+|`+)$")


def _strip_fences(text: str) -> str:
    """Remove surrounding code-fence or backtick wrappers, then strip whitespace."""
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    if m is not None:
        return m.group(1).strip()
    return stripped


def observe(raw: str) -> dict[str, Any]:
    """Pure detection — no I/O.

    Returns a dict with:
      has_bearing_update: bool
      has_frame: bool
      observed_frame: str  (last captured frame text, fence-stripped+capped to 600, else "")
      raw_len: int
      frame_count: int     — number of <frame>...</frame> pairs found
      frame_multiline: bool — does observed_frame contain a newline?
    """
    safe_raw = raw or ""
    has_bearing_update = "<bearing_update" in safe_raw.lower()

    matches = _FRAME_RE.findall(safe_raw)
    frame_count = len(matches)

    # Take the last non-empty match (heartbeat is emitted at the end).
    observed_frame = ""
    for candidate in reversed(matches):
        cleaned = _strip_fences(candidate)
        if cleaned:
            observed_frame = cleaned[:_FRAME_CAP]
            break

    has_frame = frame_count > 0
    frame_multiline = "\n" in observed_frame

    return {
        "has_bearing_update": has_bearing_update,
        "has_frame": has_frame,
        "observed_frame": observed_frame,
        "raw_len": len(safe_raw),
        "frame_count": frame_count,
        "frame_multiline": frame_multiline,
    }


def disparity(observed_frame: str, current_frame: str, *, has_frame: bool) -> str:
    """Pure comparison of what the model stated vs what the bearing stores.

    Returns one of: "no_frame", "empty_bearing", "match", "differ".
    """
    if not has_frame:
        return "no_frame"
    if not (current_frame or "").strip():
        return "empty_bearing"
    def _norm(s: str) -> str:
        return " ".join(s.split()).casefold()
    if _norm(observed_frame) == _norm(current_frame):
        return "match"
    return "differ"


def record(
    turn_id: str,
    raw: str,
    *,
    bu_outcome: str,
    current_frame: str,
) -> None:
    """Append one JSON row to frame.ledger.jsonl.

    No-op if not enabled(). Wraps everything in try/except — never raises.
    """
    if not enabled():
        return
    try:
        obs = observe(raw)
        has_bu = obs["has_bearing_update"]
        has_fr = obs["has_frame"]

        # Derive decision from outcome + observation, in priority order.
        if has_bu and bu_outcome == "applied":
            decision = "bearing_update_applied"
        elif has_bu and bu_outcome in ("rejected", "parse_failed"):
            decision = "bearing_update_dropped"
        elif has_bu and bu_outcome == "noop":
            decision = "bearing_update_noop"
        elif has_bu:
            # Tag present but outcome not one of the handled cases (e.g. bearing
            # kill-switch off → not processed). NEVER label a tag-present row as
            # silent — that would re-conflate present-but-unhandled with absent.
            decision = "bearing_update_unprocessed"
        elif has_fr:
            decision = "frame_only"
        else:
            decision = "no_emission"

        row: dict[str, Any] = {
            "ts": _now_iso(),
            "turn_id": str(turn_id),
            "decision": decision,
            "bu_outcome": bu_outcome,
            **obs,
            "current_frame_len": len(current_frame or ""),
            "frame_vs_bearing": disparity(
                obs["observed_frame"], current_frame, has_frame=obs["has_frame"]
            ),
        }

        _LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with _LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        # Never raise into the chat path.
        pass


def read_recent(limit: int = 20) -> list[dict[str, Any]]:
    """Best-effort tail read of the frame ledger. Mirrors audit.py:read_recent."""
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
