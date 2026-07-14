"""MonoFrame v2 — the standing frame-selection recorder (a TRACE CONTRACT).

It records the frame-selection Monolith COMMITS TO before answering — candidates
considered, the selected frame, the rejected runner-up, and the reason — into a
durable JSONL trace (CONFIG_DIR/frame_selection.jsonl). This is NOT a probe of
hidden cognition; it is a commitment the model's subsequent output can be judged
against. Automatic every turn (source="auto") when the model emits a
<frame_selection> block, or on a /frame demand (source="requested").

The model emits, before its answer:
    <frame_selection>
    CANDIDATES: frame a | frame b | frame c
    SELECTED: frame a
    REJECTED: frame b
    REASON: ...
    </frame_selection>

Schema per the cold-proof spec. Pure parse + safe append; never raises into the
chat path. Flag MONOLITH_MONOFRAME_V1 (default OFF -> no writes -> byte-identical).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from core.paths import CONFIG_DIR

_FLAG_ENV = "MONOLITH_MONOFRAME_V1"
_TRUTHY = {"1", "true", "yes", "on"}

MONOFRAME_VERSION = "v2.0"
STORAGE_SURFACE = "frame_selection.jsonl"
_STORE = CONFIG_DIR / STORAGE_SURFACE

_BLOCK_RE = re.compile(r"<frame_selection>\s*(.*?)\s*</frame_selection>", re.DOTALL | re.IGNORECASE)


def enabled() -> bool:
    return os.environ.get(_FLAG_ENV, "0").strip().lower() in _TRUTHY


def digest(text: str | None) -> str:
    """Stable SHA256 hex of *text* (None -> hash of empty string)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def has_selection(raw_output: str) -> bool:
    return bool(_BLOCK_RE.search(raw_output or ""))


def _field(text: str, label: str) -> str:
    m = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", text or "", re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_frame_selection(raw_output: str) -> dict[str, Any]:
    """Extract the committed <frame_selection> block. Tolerant; missing -> empty."""
    m = _BLOCK_RE.search(raw_output or "")
    if not m:
        return {"candidate_frames": [], "selected_frame": "", "rejected_runner_up": "", "rejection_reason": ""}
    body = m.group(1)
    cands_raw = _field(body, "CANDIDATES")
    candidates = [c.strip() for c in cands_raw.split("|") if c.strip()]
    return {
        "candidate_frames": candidates,
        "selected_frame": _field(body, "SELECTED"),
        "rejected_runner_up": _field(body, "REJECTED"),
        "rejection_reason": _field(body, "REASON"),
    }


@dataclass(frozen=True)
class FrameSelectionRecord:
    turn_id: str
    session_id: str
    timestamp_utc: str
    input_digest: str
    candidate_frames: list[str]
    selected_frame: str
    rejected_runner_up: str
    rejection_reason: str
    source: str                 # "auto" | "requested"
    storage_surface: str
    monoframe_version: str
    bearing_before_hash: str
    artifact_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "timestamp_utc": self.timestamp_utc,
            "input_digest": self.input_digest,
            "candidate_frames": list(self.candidate_frames),
            "selected_frame": self.selected_frame,
            "rejected_runner_up": self.rejected_runner_up,
            "rejection_reason": self.rejection_reason,
            "source": self.source,
            "storage_surface": self.storage_surface,
            "monoframe_version": self.monoframe_version,
            "bearing_before_hash": self.bearing_before_hash,
            "artifact_hash": self.artifact_hash,
        }


def build_record(
    *,
    raw_output: str,
    turn_id: str,
    session_id: str,
    timestamp_utc: str,
    user_input: str,
    bearing_before: str,
    source: str,
) -> FrameSelectionRecord:
    """Assemble a record from the committed block + turn metadata. ``artifact_hash``
    is computed over the content (every field except itself), so the trace is
    self-verifying — a row whose content doesn't hash to its artifact_hash was edited."""
    parsed = parse_frame_selection(raw_output)
    base = {
        "turn_id": turn_id,
        "session_id": session_id,
        "timestamp_utc": timestamp_utc,
        "input_digest": digest(user_input),
        "candidate_frames": parsed["candidate_frames"],
        "selected_frame": parsed["selected_frame"],
        "rejected_runner_up": parsed["rejected_runner_up"],
        "rejection_reason": parsed["rejection_reason"],
        "source": source,
        "storage_surface": STORAGE_SURFACE,
        "monoframe_version": MONOFRAME_VERSION,
        "bearing_before_hash": digest(bearing_before),
    }
    artifact_hash = digest(json.dumps(base, sort_keys=True, ensure_ascii=False))
    return FrameSelectionRecord(artifact_hash=artifact_hash, **base)


def record_selection(record: FrameSelectionRecord) -> None:
    """Append one record to frame_selection.jsonl. No-op if disabled; never raises."""
    if not enabled():
        return
    try:
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        with _STORE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    except Exception:
        pass


def record_from_output(
    *,
    raw_output: str,
    turn_id: str,
    session_id: str,
    timestamp_utc: str,
    user_input: str,
    bearing_before: str,
    source: str,
) -> bool:
    """Turn-finalizer seam: if the output carries a committed <frame_selection>
    block, build + record it. Returns whether a record was written. No block ->
    no record (the model didn't commit a selection this turn)."""
    if not has_selection(raw_output):
        return False
    rec = build_record(
        raw_output=raw_output, turn_id=turn_id, session_id=session_id,
        timestamp_utc=timestamp_utc, user_input=user_input,
        bearing_before=bearing_before, source=source,
    )
    record_selection(rec)
    return enabled()


def read_recent(limit: int = 20) -> list[dict[str, Any]]:
    if not _STORE.exists():
        return []
    try:
        with _STORE.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-max(1, int(limit)):]:
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out
