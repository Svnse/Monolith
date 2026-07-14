"""friction_telemetry — the [FRICTION] coalescer contributor (reader).

Surfaces the friction TREND back into the next turn so the model can see
whether its read of the user is hardening — the live frame-collapse signal.
Shape-first: raw channel counts + a verbatim repair span, with a minimal
hedged gloss. It is a MIRROR, not an objective (the model weighs it; nothing
optimizes to minimize it). Relocated interpretation about the MODEL's own
pattern, not a stored fact about the user (spec §5/§15).

Built reader-first; mirrors core/fault_telemetry.contribute_section. Two flags:
  * MONOLITH_FRICTION_V1        — store/predict/settle (artifact)
  * MONOLITH_FRICTION_INJECT_V1 — surface this block (gated on calibration)
Silent (returns None) on flag-off, CONNECT peer turns, or a calm/uptake stretch.
"""
from __future__ import annotations

import os
from typing import Any

from core import friction_store as _fs

_INJECT_FLAG = "MONOLITH_FRICTION_INJECT_V1"
_TRUTHY = {"1", "true", "yes", "on"}
_TAG = "[FRICTION]"

_WINDOW = 8                 # recent settled rows to scan
_HIGH = 0.6                 # friction_score >= this = a real re-mesh event
_RECENT_HIGH_LOOKBACK = 3   # surface if any of the last N is high
_MIN_HIGH_IN_WINDOW = 2     # ...or >= this many high events in the window
_MAX_VERBATIM = 90
# low-signal types we don't tally as friction in the headline
_CALM_TYPES = frozenset({"uptake", "unresolved"})


def _inject_enabled() -> bool:
    return os.environ.get(_INJECT_FLAG, "0").strip().lower() in _TRUTHY


def _is_peer_turn(messages: list[dict]) -> bool:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            return "[CHANNEL: connect/" in str(msg.get("content") or "")
    return False


def _trim(text: Any) -> str:
    s = str(text or "").strip().replace("\n", " ")
    if len(s) <= _MAX_VERBATIM:
        return s
    return s[: _MAX_VERBATIM - 1].rstrip() + "…"


def _should_surface(rows: list[dict]) -> bool:
    if not rows:
        return False
    recent_high = any((r.get("friction_score") or 0) >= _HIGH for r in rows[:_RECENT_HIGH_LOOKBACK])
    total_high = sum(1 for r in rows if (r.get("friction_score") or 0) >= _HIGH)
    return recent_high or total_high >= _MIN_HIGH_IN_WINDOW


def render_friction_block(rows: list[dict]) -> str | None:
    """Build the shape-first [FRICTION] trend block, or None if nothing worth showing.

    `rows` are recent settled predictions, newest-first."""
    if not _should_surface(rows):
        return None

    # tally friction types (skip calm) over the window
    counts: dict[str, int] = {}
    for r in rows:
        t = str(r.get("friction_type") or "")
        if t and t not in _CALM_TYPES:
            counts[t] = counts.get(t, 0) + 1
    tally = ", ".join(f"{k} ×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))

    # overlap trend (chronological: oldest -> newest)
    overlaps = []
    for r in reversed(rows):
        cj = r.get("channel_json") or {}
        ov = cj.get("answer_overlap") if isinstance(cj, dict) else None
        if isinstance(ov, (int, float)) and ov >= 0:
            overlaps.append(ov)
    overlap_line = ""
    if len(overlaps) >= 2:
        overlap_line = f"; answer-overlap {overlaps[0]:.2f}→{overlaps[-1]:.2f}"

    # newest high-friction verbatim span
    verbatim = ""
    for r in rows:
        if (r.get("friction_score") or 0) >= _HIGH and r.get("observation"):
            verbatim = _trim(r.get("observation"))
            break

    n = len(rows)
    lines = [_TAG]
    head = f"- recent re-mesh: {tally or 'low'} over last {n} settled turns{overlap_line}"
    lines.append(head)
    if verbatim:
        lines.append(f'- last divergence: "{verbatim}"')
    lines.append("- a possible hardening of your read; weigh it, don't optimize it away")
    return "\n".join(lines)


def contribute_section(messages: list[dict], config: dict):
    from core.ephemeral_coalescer import SectionResult
    if not _inject_enabled() or _is_peer_turn(messages):
        return None
    rows = _fs.recent_settled(limit=_WINDOW)
    block = render_friction_block(rows)
    if block is None:
        return None

    shown_ids = [int(r["id"]) for r in rows if "id" in r]

    def _commit() -> None:
        for pid in shown_ids:
            _fs.mark_surfaced(pid)

    return SectionResult(name="friction", text=block, on_commit=_commit)
