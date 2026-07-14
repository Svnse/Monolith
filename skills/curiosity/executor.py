"""curiosity — inspect what Monolith is drawn toward (M3 V0).

Dispatch surface only; scoring lives in core.curiosity. One op, ``detect``,
deterministic (no LLM). Propose-only: surfaces and ranks pulls, never pursues.
"""
from __future__ import annotations

from typing import Any

from core import curiosity as _curiosity

_TOP_N = 8


def _threshold(cmd: dict) -> float:
    try:
        return float(cmd.get("threshold", _curiosity._DEFAULT_ALIGN))
    except (TypeError, ValueError):
        return _curiosity._DEFAULT_ALIGN


def _op_detect(cmd: dict) -> str:
    rep = _curiosity.detect_pulls(align_threshold=_threshold(cmd), force=True)
    lines = [f"[curiosity: {rep.message}]"]
    for p in rep.pulls[:_TOP_N]:
        lines.append(
            f"  - {p['canonical']}  (pull {p['pull_strength']}, "
            f"confidentity {p['confidentity']}, stability {p['stability']}, {p['provenance']})"
        )
    if not rep.pulls:
        lines.append("  (nothing drawing curiosity right now)")
    return "\n".join(lines)


def _op_kill(cmd: dict) -> str:
    canonical = str(cmd.get("canonical") or "").strip()
    if not canonical:
        return "[curiosity: kill requires 'canonical' — the pull text to retire]"
    reason = str(cmd.get("reason") or "")
    if _curiosity.kill_pull(canonical, reason):
        return (f"[curiosity: retired pull (reversible) — {canonical} "
                f"| reason: {reason or 'unspecified'}]")
    return "[curiosity: kill failed — empty canonical]"


def run(cmd: dict, ctx: Any) -> str:
    op = str((cmd or {}).get("op") or "detect").strip().lower()
    if op == "detect":
        return _op_detect(cmd)
    if op in ("kill", "resolve", "retire"):
        return _op_kill(cmd)
    return f"[curiosity: unknown op {op!r} — use 'detect' or 'kill']"
