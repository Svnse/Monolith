"""plan — propose-only task/goal planner skill (M1 V0).

Dispatch surface only; decomposition lives in core.planner, persistence in
core.plans. The planner never executes steps — `mark` only records progress as
the human / existing gated tool loop completes them.
"""
from __future__ import annotations

from typing import Any

from core import planner as _planner
from core import plans as _plans

_TOP_N = 8


def _fmt_plan(p: dict) -> list[str]:
    done = sum(1 for s in p["steps"] if s["status"] == "done")
    total = len(p["steps"])
    lines = [f"  goal: {p['goal']}  (source: {p['source']}, {done}/{total} done, {p['status']})"]
    for s in p["steps"]:
        dep = f" ← {s['depends_on']}" if s["depends_on"] else ""
        lines.append(f"    {s['seq']}. [{s['status']}] {s['verb']} {s['target']}{dep}")
    ready = _plans.next_ready_steps(p["plan_uid"])
    if ready:
        nxt = ready[0]
        lines.append(f"  next ready: step {nxt['seq']} — {nxt['verb']} {nxt['target']}")
    if _plans.done_gate_enabled():
        crits = _plans.get_criteria(p["plan_uid"])
        if crits:
            lines.append("  criteria:")
            for c in crits:
                cite = f"  [cite: {c['cite_handle']}]" if c.get("cite_handle") else ""
                lines.append(f"    {c['seq']}. [{c['status']}] {c['criterion']}{cite}")
        finds = _plans.list_findings(p["plan_uid"])
        if finds:
            lines.append("  grounds (cite one with [cite: obs:N] when you attest):")
            for f in finds:
                lines.append(f"    obs:{f['id']} — {f['content']}")
    return lines


def _op_candidates(cmd: dict) -> str:
    cands = _planner.propose_candidates()
    lines = [f"[plan: {len(cands)} candidate goal(s) — surface only, pick one to decompose]"]
    for c in cands[:_TOP_N]:
        lines.append(f"  - ({c['source']}) {c['goal']}  [{c.get('basis','')}]")
    if not cands:
        lines.append("  (no candidate goals — bearing empty and no curiosity pulls)")
    return "\n".join(lines)


def _op_decompose(cmd: dict) -> str:
    goal = str(cmd.get("goal") or "").strip()
    if not goal:
        return "[plan: decompose requires a non-empty 'goal']"
    source = str(cmd.get("source") or "explicit")
    plan = _planner.decompose(goal, source=source)
    if plan is None:
        return f"[plan: could not decompose {goal!r} into steps — nothing queued]"
    return "[plan: proposed]\n" + "\n".join(_fmt_plan(plan))


def _op_show(cmd: dict) -> str:
    p = _plans.get_active_plan()
    if p is None:
        return "[plan: no active plan]"
    return "[plan: active]\n" + "\n".join(_fmt_plan(p))


def _op_mark(cmd: dict) -> str:
    active = _plans.get_active_plan()
    uid = str(cmd.get("plan_uid") or (active["plan_uid"] if active else "")).strip()
    if not uid:
        return "[plan: no active plan to mark]"
    try:
        seq = int(cmd.get("step"))
    except (TypeError, ValueError):
        return "[plan: mark requires an integer 'step']"
    status = str(cmd.get("status") or "done")
    try:
        _plans.mark_step(uid, seq, status)
    except ValueError as exc:
        return f"[plan: {exc}]"
    p = _plans.get_plan(uid)
    done = sum(1 for s in p["steps"] if s["status"] == "done")
    total = len(p["steps"])
    _emit("plan_step_marked", {"plan_uid": uid, "step": seq, "status": status})
    if not _plans.done_gate_enabled() and done == total and p["status"] != "done":
        _plans.set_plan_status(uid, "done")
        _emit("plan_status_changed", {"plan_uid": uid, "status": "done"})
    return f"[plan: step {seq} → {status} ({done}/{total} done)]"


def _op_criteria(cmd: dict) -> str:
    active = _plans.get_active_plan()
    uid = str(cmd.get("plan_uid") or (active["plan_uid"] if active else "")).strip()
    if not uid:
        return "[plan: no active plan for criteria]"
    raw = cmd.get("criteria")
    crits = ([c.strip() for c in raw.split(";")] if isinstance(raw, str)
             else [str(c).strip() for c in (raw or [])])
    crits = [c for c in crits if c]
    if not crits:
        return "[plan: criteria requires at least one non-empty criterion]"
    n = _plans.set_criteria(uid, crits)
    _emit("plan_criteria_set", {"plan_uid": uid, "count": n})
    return f"[plan: {n} success criteria set]"


def _op_ground(cmd: dict) -> str:
    active = _plans.get_active_plan()
    uid = str(cmd.get("plan_uid") or (active["plan_uid"] if active else "")).strip()
    if not uid:
        return "[plan: no active plan to ground]"
    text = str(cmd.get("ground") or cmd.get("finding") or "").strip()
    if not text:
        return "[plan: ground requires a non-empty finding]"
    obs_id = _plans.record_finding(uid, text)
    if obs_id is None:
        return "[plan: could not record finding]"
    return f"[plan: ground recorded — cite it as [cite: obs:{obs_id}]]"


def _op_attest(cmd: dict) -> str:
    from core.plan_grounding import make_plan_resolver
    active = _plans.get_active_plan()
    uid = str(cmd.get("plan_uid") or (active["plan_uid"] if active else "")).strip()
    if not uid:
        return "[plan: no active plan to attest]"
    seq_raw = cmd.get("seq") if cmd.get("seq") is not None else cmd.get("step")
    try:
        seq = int(seq_raw)
    except (TypeError, ValueError):
        return "[plan: attest requires an integer 'seq']"
    evidence = str(cmd.get("evidence") or "").strip()
    if not evidence:
        return "[plan: attest requires 'evidence' citing a ground, e.g. [cite: obs:N]]"
    crit = _plans.attest_criterion(uid, seq, evidence, make_plan_resolver(uid))
    if not crit:
        return f"[plan: no criterion at seq {seq}]"
    return f"[plan: criterion {seq} → {crit['status']}]"


def _op_complete(cmd: dict) -> str:
    active = _plans.get_active_plan()
    uid = str(cmd.get("plan_uid") or (active["plan_uid"] if active else "")).strip()
    if not uid:
        return "[plan: no active plan to complete]"
    audit = _plans.complete_plan(uid)
    if audit.get("ok"):
        _emit("plan_status_changed", {"plan_uid": uid, "status": "done"})
        return "[plan: COMPLETE — all steps done and all criteria grounded]"
    parts = [f"[plan: NOT complete — {audit.get('reason', 'incomplete')}]"]
    if audit.get("steps_open"):
        parts.append(f"  steps not done: {audit['steps_open']}")
    if not audit.get("criteria_total"):
        parts.append("  no success_criteria set (use `plan criteria ...`)")
    elif audit.get("criteria_unmet"):
        parts.append(f"  criteria not met: {audit['criteria_unmet']}")
    return "\n".join(parts)


def _emit(kind: str, payload: dict) -> None:
    try:
        from core.acatalepsy import canonical_log
        canonical_log.append(kind, payload=payload)
    except Exception:
        pass


def run(cmd: dict, ctx: Any) -> str:
    op = str((cmd or {}).get("op") or "show").strip().lower()
    if op == "candidates":
        return _op_candidates(cmd)
    if op == "decompose":
        return _op_decompose(cmd)
    if op in ("show", "status"):
        return _op_show(cmd)
    if op == "mark":
        return _op_mark(cmd)
    if op == "criteria":
        return _op_criteria(cmd)
    if op == "ground":
        return _op_ground(cmd)
    if op == "attest":
        return _op_attest(cmd)
    if op == "complete":
        return _op_complete(cmd)
    return (f"[plan: unknown op {op!r} — use candidates | decompose | show | mark "
            f"| criteria | ground | attest | complete]")
