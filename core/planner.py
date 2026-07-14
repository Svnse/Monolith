"""Planner (M1 V0) — propose-only goal decomposition.

Two surfaces:
  * propose_candidates() — deterministic (no LLM): candidate goals the system
    could pursue, drawn from the Bearing active_goal + curiosity pulls. Surfaced
    for selection; the planner does NOT choose unbidden.
  * decompose(goal, source) — ONE grounded LLM call turning a goal into an
    ordered, acyclic step DAG, persisted to the plan store (turn_trace.sqlite3).
    Propose-only: steps execute via the EXISTING gated tool loop / human; the
    planner never drives execution.

Constructive counterpart to commitment_detector: a stated "I'll do X" becomes a
tracked plan with explicit steps + status, not a forgotten promise.
"""
from __future__ import annotations

import re

from core import plans

_STEP_RE = re.compile(r"^\s*STEP\s*:\s*(.+)$", re.IGNORECASE)
_DEPENDS_RE = re.compile(r"depends\s*:?\s*(.*)$", re.IGNORECASE)
_MAX_STEPS = 12
_VERB_CAP = 60
_TARGET_CAP = 160


# ── candidate goals (deterministic) ───────────────────────────────────

def _bearing_goal() -> str:
    try:
        from addons.system.bearing import store as bearing_store
        return str(bearing_store.get_bearing().active_goal or "").strip()
    except Exception:
        return ""


def _curiosity_pulls() -> list[dict]:
    try:
        from core import curiosity
        return list(curiosity.detect_pulls(force=True).pulls)
    except Exception:
        return []


def propose_candidates() -> list[dict]:
    """Candidate goals to pursue: the Bearing active_goal + curiosity pulls.
    Surfaced for selection (propose-only); de-duplicated by goal text."""
    candidates: list[dict] = []
    seen: set[str] = set()
    bg = _bearing_goal()
    if bg:
        candidates.append({"goal": bg, "source": "bearing", "basis": "active goal"})
        seen.add(bg)
    for p in sorted(_curiosity_pulls(), key=lambda x: x.get("pull_strength", 0), reverse=True):
        goal = str(p.get("canonical", "")).strip()
        if goal and goal not in seen:
            seen.add(goal)
            candidates.append({
                "goal": goal, "source": "curiosity",
                "basis": f"pull_strength {p.get('pull_strength')}",
            })
    return candidates


# ── decomposition (one LLM call, propose-only) ────────────────────────

def _call_llm(prompt: str) -> str:
    """Direct LLM call (monkeypatched in tests). Broad-except — decompose must
    never break the caller."""
    try:
        from core.llm_config import load_config
        from engine.sync_bridge import generate_sync_from_config
        cfg = load_config()
        text = generate_sync_from_config(
            cfg, [{"role": "user", "content": prompt}],
            llm_config={"max_tokens": 1024, "temp": 0.3}, thinking_enabled=False,
        )
        return str(text or "")
    except Exception:
        return ""


def _build_prompt(goal: str) -> str:
    return (
        "Decompose the goal below into a SHORT ordered plan of concrete steps. "
        "Each step is one action. Reference earlier steps in `depends` only "
        "(no forward references). Output EXACTLY this format, nothing else:\n\n"
        "PLAN: <one-line restatement of the goal>\n"
        "STEP: <verb> | <target> | depends: <comma-separated earlier step numbers, or none>\n"
        "STEP: ...\n\n"
        f"GOAL: {goal}"
    )


def _parse_steps(text: str) -> list[dict]:
    steps: list[dict] = []
    for raw in str(text or "").splitlines():
        m = _STEP_RE.match(raw)
        if not m:
            continue
        body = m.group(1)
        parts = [p.strip() for p in body.split("|")]
        verb = (parts[0] if parts else "").strip()[:_VERB_CAP]
        target = (parts[1] if len(parts) > 1 else "").strip()[:_TARGET_CAP]
        deps: list[int] = []
        if len(parts) > 2:
            dm = _DEPENDS_RE.search(parts[2])
            depstr = (dm.group(1) if dm else parts[2]) or ""
            for tok in re.split(r"[,\s]+", depstr.strip()):
                if tok.isdigit():
                    deps.append(int(tok))
        if not verb:
            continue
        seq = len(steps) + 1
        # forward/self references dropped — depend only on EARLIER steps.
        deps = [d for d in deps if 0 < d < seq]
        steps.append({"verb": verb, "target": target, "depends_on": deps})
        if len(steps) >= _MAX_STEPS:
            break
    return steps


def decompose(goal: str, source: str = "explicit", *, turn_id: str | None = None,
              backend: str | None = None) -> dict | None:
    """Decompose a goal into a persisted, acyclic plan (propose-only). Returns
    the plan dict, or None if the LLM produced no usable steps."""
    goal_s = str(goal or "").strip()
    if not goal_s:
        return None
    steps = _parse_steps(_call_llm(_build_prompt(goal_s)))
    if not steps:
        return None
    uid = plans.create_plan(goal=goal_s, source=source, steps=steps, turn_id=turn_id)
    try:
        from core.acatalepsy import canonical_log
        canonical_log.append(
            "plan_proposed",
            payload={"plan_uid": uid, "goal": goal_s, "source": source, "step_count": len(steps)},
        )
    except Exception:
        pass
    return plans.get_plan(uid)
