"""monoexplore — drive a curiosity expedition and read its coherence (V0).

Dispatch surface only; logic lives in core.monoexplore. Ops: start | status |
coherence. Deterministic except `start` (one decompose LLM call). force=True
bypasses the dark flag, mirroring skills/curiosity.
"""
from __future__ import annotations

from typing import Any

from core import monoexplore as _mx


def _op_start(cmd: dict) -> str:
    goal = str(cmd.get("goal") or "").strip()
    exp = _mx.start_expedition(goal, force=True)
    if exp is None:
        return "[monoexplore: could not start — no decomposable goal]"
    return (
        f"[monoexplore: expedition started — goal {exp['goal']!r} "
        f"(source {exp['source']}, plan {exp['plan_uid'][:8]})]"
    )


def _op_status(cmd: dict) -> str:
    from core import plans
    p = plans.get_active_plan()
    rep = _mx.coherence_report()
    if p is None:
        return f"[monoexplore: no active expedition | coherence {rep['verdict']} — {rep['reason']}]"
    done = sum(1 for s in p["steps"] if s["status"] == "done")
    return (
        f"[monoexplore: expedition {p['goal']!r} — {done}/{len(p['steps'])} steps, "
        f"coherence {rep['verdict']}: {rep['reason']}]"
    )


def _op_coherence(cmd: dict) -> str:
    rep = _mx.coherence_report()
    d = rep["dims"]
    return (
        f"[monoexplore coherence: {rep['verdict']} — {rep['reason']} | "
        f"grounded {d['grounded']}/{d['referents']}, progress {d['progress']}, "
        f"drift_overlap {d['drift_overlap']}]"
    )


def run(cmd: dict, ctx: Any) -> str:
    op = str((cmd or {}).get("op") or "status").strip().lower()
    if op == "start":
        return _op_start(cmd)
    if op in ("status", "show"):
        return _op_status(cmd)
    if op == "coherence":
        return _op_coherence(cmd)
    return f"[monoexplore: unknown op {op!r} — use start | status | coherence]"
