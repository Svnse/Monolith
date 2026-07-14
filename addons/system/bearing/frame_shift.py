"""frame_shift — consecutive-frame CHANGE detector (Phase 1: observe-only).

A SEPARATE rail from ``frame_drift`` — do not conflate them:

  * ``frame_drift``  compares current_frame vs the recent user ASKS
                     ("staleness": should this frame have moved but didn't?).
                     Mushy lexical signal (offline F1 ~0.6) -> observe-only.
  * ``frame_shift``  compares current_frame[t-1] vs current_frame[t]
                     ("change event": did the frame move, and is that a new
                     trajectory leg?). Sharply BIMODAL — an offline replay over
                     244 transitions split ~149 HOLD / ~87 SHIFT / ~8 AMBIG, a
                     near-empty middle — strong enough to promote into a detector.

Architecture this serves (E's split-rail design):
    current_frame   -> model-authored "now" state
    frame_shift     -> runtime-detected change event  (THIS module)
    trajectory      -> runtime-derived, low-authority arc            (Phase 2)
    bearing_update  -> model-authored rich posture, nudged on shift  (Phase 2)

Phase 1 (this module): OBSERVE ONLY. Classify each frame transition
(HOLD | SHIFT | AMBIG) and append it to CONFIG_DIR/frame_shift.ledger.jsonl.
NO mutation, NO trajectory write, NO nudge — those are Phase 2 behind a second
flag, gated on this ledger first confirming the SHIFT verdicts line up with real
topic/work transitions on live turns.

Pure detection + safe append; NEVER raises into the chat path. Flag
``MONOLITH_FRAME_SHIFT_V1`` (default OFF -> no writes -> byte-identical to off).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.paths import CONFIG_DIR
from .drift import _content_tokens  # reuse frame_drift's content tokenizer (same signal basis)

_FLAG_ENV = "MONOLITH_FRAME_SHIFT_V1"
_TRUTHY = {"1", "true", "yes", "on"}

# Module-level so tests can monkeypatch it (mirrors drift_observe._LEDGER).
_LEDGER = CONFIG_DIR / "frame_shift.ledger.jsonl"
_FRAME_CAP = 300

# Conservative thresholds from the offline bimodal split. Editable without
# touching logic; observe-first exists to confirm them on live turns.
SHIFT_BELOW = 0.2        # sim <  0.2        -> SHIFT
HOLD_AT_OR_ABOVE = 0.6   # sim >= 0.6        -> HOLD
                         # 0.2 <= sim < 0.6  -> AMBIG


def enabled() -> bool:
    """True only when MONOLITH_FRAME_SHIFT_V1 is set truthy (default OFF)."""
    return os.environ.get(_FLAG_ENV, "0").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class ShiftObservation:
    verdict: str        # "HOLD" | "SHIFT" | "AMBIG"
    sim: float          # symmetric content-token Jaccard, 0..1
    confidence: float   # how decisively classified (distance into the verdict band), 0..1
    prev_tokens: int
    new_tokens: int
    detail: str


def _jaccard(a: set[str], b: set[str]) -> float:
    """Symmetric content-token Jaccard. 0.0 when the union is empty; an empty
    prev vs a non-empty new yields 0.0 — establishing a frame reads as a SHIFT."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def classify(
    prev_frame: str,
    new_frame: str,
    *,
    shift_below: float = SHIFT_BELOW,
    hold_at_or_above: float = HOLD_AT_OR_ABOVE,
) -> ShiftObservation:
    """Pure: classify the prev->new frame transition. Never raises.

    SHIFT when content-token Jaccard < shift_below; HOLD when >= hold_at_or_above;
    AMBIG in between. ``confidence`` is how far the similarity sits inside its
    verdict band (1.0 = unambiguous, 0.0 = on the boundary / AMBIG).
    """
    pt = _content_tokens(prev_frame or "")
    nt = _content_tokens(new_frame or "")
    sim = _jaccard(pt, nt)
    if sim < shift_below:
        verdict = "SHIFT"
        confidence = (shift_below - sim) / shift_below if shift_below else 1.0
    elif sim >= hold_at_or_above:
        verdict = "HOLD"
        denom = (1.0 - hold_at_or_above) or 1.0
        confidence = (sim - hold_at_or_above) / denom
    else:
        verdict = "AMBIG"
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    detail = f"jaccard {sim:.2f} -> {verdict} ({len(pt)} prev / {len(nt)} new content words)"
    return ShiftObservation(verdict, round(sim, 3), round(confidence, 3), len(pt), len(nt), detail)


def record(
    turn_id: str,
    prev_frame: str,
    new_frame: str,
    *,
    turn_n: int = 0,
    source: str = "",
    session_id: str = "",
) -> ShiftObservation | None:
    """Observe-only: classify + append one transition row. Returns the
    ShiftObservation, or None when disabled or when there is no new frame.

    No-op (and writes nothing) when the flag is off. Never raises into the chat
    path: a write failure still returns the in-memory observation.
    """
    if not enabled():
        return None
    new = (new_frame or "").strip()
    if not new:
        return None
    obs = classify(prev_frame, new)
    try:
        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "turn_id": str(turn_id),
            "turn_n": int(turn_n or 0),
            "previous_frame": (prev_frame or "")[:_FRAME_CAP],
            "new_frame": new[:_FRAME_CAP],
            "lexical_sim": obs.sim,
            "verdict": obs.verdict,
            "confidence": obs.confidence,
            "source": str(source or ""),
            "session_id": str(session_id or ""),
        }
        _LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with _LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return obs


def read_recent(limit: int = 50) -> list[dict[str, Any]]:
    """Best-effort tail read of the shift ledger. Mirrors drift_observe.read_recent."""
    if not _LEDGER.exists():
        return []
    try:
        lines = _LEDGER.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[-max(1, int(limit)):]:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if isinstance(o, dict):
            rows.append(o)
    return rows


def replay_frames(frames: list[str]) -> list[ShiftObservation]:
    """Phase-1 verification helper: classify a chronological list of
    current_frame strings (e.g. extracted from the existing frame_drift ledger)
    into the consecutive transition verdicts. Pure; for offline replay."""
    return [classify(frames[i - 1], frames[i]) for i in range(1, len(frames))]
