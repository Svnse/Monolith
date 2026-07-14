"""MonoExplore (V0) — deterministic curiosity-expedition core. Ships dark.

See docs/superpowers/plans/2026-06-03-monoexplore-emergence.md. V0 is the
LLM-FREE core: goal-seeding, expedition lifecycle over core.plans, and the
coherence signal (the external-evidence anti-drift metric). The autonomous
model-driving tick loop is V1 (a separately-gated increment) — V0 deliberately
ships the deterministic guard *before* the autonomy it will guard.

MonoExplore is the "body" that drives organs that already exist propose-only:
curiosity/bearing supply the will (planner.propose_candidates), the planner the
task-list (planner.decompose -> core.plans), and the BEARING is the coherence
anchor + state (active_goal/trajectory/referents, cross-session).

INVARIANTS:
  * external-evidence: growth credit only for observed file/tool referents
    (self-narration is already capped by identity_alignment._PROV_WEIGHT and the
    bearing's grounding_verifier; MonoExplore measures against that, never
    launders self-claims into growth);
  * coherence anchored to bearing.trajectory + active_goal, scored WITHOUT an LLM
    (the model never grades its own coherence);
  * no new store: state = plan (turn_trace.sqlite3) + bearing.json;
  * ships dark behind MONOLITH_MONOEXPLORE_V1 (the skill bypasses via force=True).

Flag: MONOLITH_MONOEXPLORE_V1 (default OFF).
"""
from __future__ import annotations

import json
import os
import re

from core.identity_alignment import _content_tokens
from core.paths import CONFIG_DIR

_FLAG_ENV = "MONOLITH_MONOEXPLORE_V1"

# Grounding gate (INV-1): a referent counts as real exploration evidence only
# when the model claimed it observed AND it is a concrete file / tool result.
_GROUNDED_KINDS = frozenset({"file", "tool_result"})
_GROUNDED_STATUS = frozenset({"observed"})

# Thresholds — placeholders to calibrate; deterministic so they stay testable.
_MIN_REFS = 3            # below this, too little signal to judge grounding/drift
_GROUNDING_RED = 0.34    # < this fraction grounded -> performing, not observing
_DRIFT_YELLOW = 0.20     # < this referent/trajectory token overlap -> drifting

_DEFAULT_GOAL = "explore the workspace filesystem and surface what is there"

# Referent names are usually PATHS (engine/turn_pipeline.py). identity_alignment.
# _content_tokens preserves '/' and '.' INSIDE tokens, so a path becomes ONE atom
# that can never lexically overlap a prose trajectory — drift would read YELLOW for
# every healthy exploration. _path_tokens splits on path/extension separators so
# components (engine, turn_pipeline) can match the trajectory prose. Deliberately
# NOT _content_tokens here.
_PATH_SPLIT_RE = re.compile(r"[^a-z0-9_]+")


def _path_tokens(text: str) -> set[str]:
    return {t for t in _PATH_SPLIT_RE.split(str(text or "").lower()) if t}


def flag_enabled() -> bool:
    return str(os.environ.get(_FLAG_ENV, "0")).strip().lower() in {"1", "true", "yes", "on"}


# ── goal-seeding + expedition lifecycle ───────────────────────────────


def seed_goal(explicit: str = "") -> dict:
    """Resolve the expedition goal (INV-6 propose-only). Explicit wins; else the
    top curiosity/bearing candidate; else a concrete default (never None)."""
    goal = str(explicit or "").strip()
    if goal:
        return {"goal": goal, "source": "explicit"}
    from core import planner
    cands = planner.propose_candidates()
    if cands:
        top = cands[0]
        return {"goal": str(top["goal"]).strip(), "source": str(top.get("source", "candidate"))}
    return {"goal": _DEFAULT_GOAL, "source": "default"}


def start_expedition(goal: str = "", *, force: bool = False) -> dict | None:
    """Seed a goal and decompose it into an active plan (INV-5: state lives in
    the plan store + bearing). Returns {plan_uid, goal, source}, or None if dark
    or the goal could not be decomposed."""
    if not force and not flag_enabled():
        return None
    from core import planner, plans
    seed = seed_goal(goal)
    plan = planner.decompose(seed["goal"], source="monoexplore")
    if plan is None:
        return None
    if plans.done_gate_enabled():
        plans.set_criteria(
            plan["plan_uid"],
            ["expedition goal explored with >=1 grounded finding"],
        )
    plans.set_plan_status(plan["plan_uid"], "active")
    return {"plan_uid": plan["plan_uid"], "goal": seed["goal"], "source": seed["source"]}


# ── coherence / anti-drift signal (the keystone, INV-2) ───────────────


def coherence_report() -> dict:
    """Deterministic, LLM-free expedition health. Dominance ordering, first-RED-
    wins (grounding RED > drift YELLOW); marked-step progress is a reported
    dimension, not a gate — never a single weighted scalar (mirrors the corpus
    PerfVector design so failure stays legible)."""
    from addons.system.bearing import store as bstore
    from core import plans

    bearing = bstore.get_bearing()
    refs = bearing.referents
    total = len(refs)
    grounded = sum(
        1 for r in refs if r.status in _GROUNDED_STATUS and r.kind in _GROUNDED_KINDS
    )
    grounding_ratio = (grounded / total) if total else 0.0

    plan = plans.get_active_plan()
    steps = plan["steps"] if plan else []
    done = sum(1 for s in steps if s["status"] == "done")
    progress = (done / len(steps)) if steps else 0.0

    traj_tokens = _content_tokens(f"{bearing.trajectory} {bearing.active_goal}")
    ref_tokens: set[str] = set()
    for r in refs:
        ref_tokens |= _path_tokens(r.name)
    drift_overlap = (len(ref_tokens & traj_tokens) / len(ref_tokens)) if ref_tokens else 1.0

    dims = {
        "referents": total,
        "grounded": grounded,
        "grounding_ratio": round(grounding_ratio, 4),
        "progress": round(progress, 4),
        "drift_overlap": round(drift_overlap, 4),
    }

    if total >= _MIN_REFS and grounding_ratio < _GROUNDING_RED:
        return {
            "verdict": "RED",
            "reason": "ungrounded: mostly self/inferred referents — performing growth, not observing",
            "dims": dims,
        }
    # marked-step `progress` is a REPORTED dimension only, not a RED gate (advisor
    # stage-4): gathering grounded, on-trajectory referents is itself real
    # exploration progress that marked plan steps don't capture. Grounding is the
    # anti-spin gate; drift is the off-course gate.
    if ref_tokens and traj_tokens and drift_overlap < _DRIFT_YELLOW:
        return {
            "verdict": "YELLOW",
            "reason": "referents drifting off the bearing trajectory",
            "dims": dims,
        }
    return {"verdict": "GREEN", "reason": "grounded and on-trajectory", "dims": dims}


# ── V1: read-biased tools + leash config (INV-3, INV-6) ───────────────

# Atomic intent: read-only tool names (canonical, underscored). 'recall' is NOT a
# registered SKILL.md tool (verified absent from list_tools()); search_history is
# the memory-read substitute if needed. NEVER include write/run tools.
READ_ONLY_SET = frozenset({"open_file", "read_file", "grep", "find_files", "list_files", "calculate"})

_LEASH_PATH = CONFIG_DIR / "monoexplore.json"
_DEFAULT_LEASH = {"tool_policy": "read_only", "max_ticks_per_wake": 6, "tick_interval_s": 20}


def load_leash() -> dict:
    """The expedition leash (tool_policy / max_ticks_per_wake / tick_interval_s).
    Defaults are conservative — read-only, short — so first unattended runs are
    reversible (INV-6: small new config file, not a new claims store)."""
    try:
        with _LEASH_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {**_DEFAULT_LEASH, **(data if isinstance(data, dict) else {})}
    except Exception:
        return dict(_DEFAULT_LEASH)


def save_leash(leash: dict) -> None:
    merged = {**_DEFAULT_LEASH, **{k: leash[k] for k in _DEFAULT_LEASH if k in leash}}
    _LEASH_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _LEASH_PATH.with_name(_LEASH_PATH.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, _LEASH_PATH)


# ── V1: grounded-finding ingest — the external-evidence gate (INV-4) ──


def _acu_store_factory():
    from core.acu_store import ACUStore
    return ACUStore()


def _evidence_tokens(text: str) -> set[str]:
    # content tokens + path tokens, so both "GeneratorWorker" and "engine/llm.py"
    # in tool output can be matched against a finding triple.
    return _content_tokens(text) | _path_tokens(text)


def ingest_grounded_findings(claims: list[str], *, evidence_text: str) -> int:
    """Ingest model-emitted finding triples as world-provenance ACUs — but ONLY the
    findings whose content overlaps the ACTUAL tool-output text gathered this tick
    (INV-4: deterministic per-finding external-evidence gate, NOT "a tool ran").

    A finding about something absent from the evidence is dropped: 'world' must be
    earned by evidence, never asserted — this is what keeps an unattended loop from
    self-radicalizing (the V0 echo-chamber failure, one level up). source='tool'
    maps to provenance 'world' (acu_store.py:46); non-atomic claims are also
    dropped by intake (ingest returns -1). Returns the count actually written."""
    ev = _evidence_tokens(evidence_text)
    if not ev:
        return 0
    store = _acu_store_factory()
    n = 0
    try:
        for c in claims:
            c = str(c or "").strip()
            if not c or not (_evidence_tokens(c) & ev):  # finding must overlap real evidence
                continue
            if store.ingest(c, source="tool") != -1:
                n += 1
    finally:
        store.close()
    return n
