"""Observer V0 turn-boundary runtime.

Fires once at assistant turn finalization, reads existing substrate state, and
persists an advisory [OBSERVER] block for the next generation turn.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from core import irp
from . import store


_FLAG_ENV = "MONOLITH_OBSERVER_V0"
_MAX_LINE_CHARS = 220


def is_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on", ""}


def _short(text: Any, limit: int = _MAX_LINE_CHARS) -> str:
    body = " ".join(str(text or "").split())
    if len(body) <= limit:
        return body
    return body[: limit - 1].rstrip() + "..."


def _locked_identity_claims(limit: int = 3) -> list[dict[str, Any]]:
    try:
        from core.acu_store import ACUStore
        s = ACUStore()
        try:
            rows = s.retrieve(limit=200)
        finally:
            s.close()
    except Exception:
        return []
    out = [row for row in rows if irp.is_locked_claim(row)]
    return out[:limit]


def _recent_log_events(limit: int = 8):
    try:
        from core.acatalepsy import canonical_log
        latest = canonical_log.latest_event_id()
        if latest <= 0:
            return []
        start = max(0, latest - limit)
        return canonical_log.read_since(start, limit=limit)
    except Exception:
        return []


def _bearing_input() -> dict[str, Any]:
    try:
        from addons.system.bearing import store as bearing_store
        return bearing_store.get_bearing().to_observer_input()
    except Exception:
        return {}


def _latest_emergence_signal() -> dict[str, Any] | None:
    """Read-only: the M2 emergence detector writes this to the milestone
    ledger; the Observer only reads it (no mutation authority preserved)."""
    try:
        from core import identity_milestones
        return identity_milestones.get_latest_emergence_signal()
    except Exception:
        return None


def _latest_curiosity_signal() -> dict[str, Any] | None:
    """Read-only: the M3 curiosity detector writes this; Observer only reads.

    Gated on liveness — a fully-retired signal is a ghost (live tool says 0
    while the cache still says N), so we never surface it. Pure read."""
    try:
        from core import curiosity
        return curiosity.latest_surfaceable_signal()
    except Exception:
        return None


def _planner_enabled() -> bool:
    return str(os.environ.get("MONOLITH_PLANNER_V1", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _active_plan_line() -> str:
    """Read-only: surface the M1 active plan + dependency-ready next step.
    Ships dark (MONOLITH_PLANNER_V1 default OFF)."""
    if not _planner_enabled():
        return ""
    try:
        from core import plans
        p = plans.get_active_plan()
        if not p or not p.get("steps"):
            return ""
        done = sum(1 for s in p["steps"] if s["status"] == "done")
        total = len(p["steps"])
        ready = plans.next_ready_steps(p["plan_uid"])
        nxt = f"{ready[0]['verb']} {ready[0]['target']}" if ready else "—"
        return (f"Active plan: {_short(p['goal'])} — next ready step: {_short(nxt, 80)} "
                f"({done}/{total} done). Use the plan skill to show/mark (propose-only).")
    except Exception:
        return ""


def _line(label: irp.IRPLabel, text: str) -> str:
    return irp.label_text(text, scope="observer", label=label)


def build_observer_snapshot(turn_id: str = "") -> dict[str, Any]:
    """Build an advisory Observer snapshot from allowed substrate inputs."""

    lines: list[str] = []

    for row in _locked_identity_claims(limit=2):
        canonical = _short(row.get("canonical"))
        if canonical:
            lines.append(_line("LOCKED", f"Origin 0 identity claim remains active: {canonical}"))

    # M2: surface the identity-emergence signal (read-only) high in the block so
    # it isn't truncated by the 8-line cap. Points the model at the bidden skill.
    _sig = _latest_emergence_signal()
    if _sig:
        _msg = _short(_sig.get("message") or "self-derived identity claims have emerged for review")
        lines.append(_line(
            "PROVISIONAL",
            f"Identity emergence: {_msg} Run the identity_review skill to inspect/draft "
            f"(propose-only; Origin 0 stays frozen).",
        ))

    # M1: surface the active plan + next ready step (read-only; ships dark).
    _plan_line = _active_plan_line()
    if _plan_line:
        lines.append(_line("PROVISIONAL", _plan_line))

    bearing = _bearing_input()
    active_goal = _short(bearing.get("active_goal"))
    if active_goal:
        lines.append(_line("PROVISIONAL", f"Bearing active goal: {active_goal}"))
    next_move = _short(bearing.get("next_move"))
    if next_move:
        lines.append(_line("PROVISIONAL", f"Bearing next move: {next_move}"))
    trajectory = _short(bearing.get("trajectory"))
    if trajectory:
        lines.append(_line("PROVISIONAL", f"Bearing trajectory: {trajectory}"))

    events = list(_recent_log_events(limit=8))
    for ev in reversed(events[-3:]):
        payload = ev.payload or {}
        text = payload.get("text") or payload.get("message") or payload.get("decision_id") or ""
        detail = f"{ev.kind}"
        if text:
            detail += f" - {_short(text, 120)}"
        lines.append(_line("PROVISIONAL", f"Recent canonical signal: {detail}"))

    # Keep the block compact and useful. Header/footer are contract framing;
    # every content line above is IRP-labeled.
    lines = lines[:8]
    block_lines = [
        "[OBSERVER] - advisory turn-boundary read. No mutation authority.",
        *[f"- {line}" for line in lines],
        "[/OBSERVER]",
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_after_turn_id": str(turn_id or ""),
        "authority": "advisory",
        "mutation_power": "none",
        "lines": lines,
        "block": "\n".join(block_lines) if lines else "",
    }


def _emit_canonical_log_event(snapshot: dict[str, Any]) -> None:
    """Write an observer_fired event to canonical_log for Auditor visibility."""
    try:
        from core.acatalepsy import canonical_log
        canonical_log.append(
            "observer_fired",
            payload={
                "turn_id": snapshot.get("generated_after_turn_id", ""),
                "lines_count": len(snapshot.get("lines") or []),
                "block_chars": len(snapshot.get("block") or ""),
                "labels_used": sorted(set(
                    line.split("]")[0].lstrip("[")
                    for line in (snapshot.get("lines") or [])
                    if line.startswith("[")
                )),
            },
        )
    except Exception:
        pass


def fire_turn_boundary(turn_id: str = "") -> dict[str, Any] | None:
    """Compute and persist the next-turn Observer block."""

    if not is_enabled():
        return None
    snapshot = build_observer_snapshot(turn_id=turn_id)
    if not snapshot.get("block"):
        return None
    store.write_latest(snapshot)
    _emit_canonical_log_event(snapshot)
    return snapshot
