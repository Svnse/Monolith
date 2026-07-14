# M1 — Task/Goal Planner (V0) — Design & Spec

- **Date:** 2026-06-02
- **Author:** Monolith agent (Claude), planned WITH E (3 forks answered), autonomous-build mandate.
- **Status:** **BUILT (V0, 2026-06-02).** Propose-only. Ships dark (`MONOLITH_PLANNER_V1` default OFF gates the Observer line; the `plan` skill works regardless). Full suite green (1522 passed / 1525). Live check: `propose_candidates()` surfaced 11 real candidate goals (1 bearing + 10 curiosity pulls) — arc closed. Impl: `core/plans.py`, `core/planner.py`, `skills/plan/`, `tests/test_plan*.py` + `test_planner*.py`.
- **Closes the arc:** identity (M2) → what to pursue (M3 curiosity) → **how to pursue it (M1 plan)**.

## 0. Decisions (E, 2026-06-02)

| Fork | Decision |
|------|----------|
| Scope/autonomy | **Propose-only planner** — decompose + persist + surface + track progress. Steps execute through the EXISTING gated tool loop / human; the planner NEVER auto-drives execution. |
| Plan storage | **`turn_trace.sqlite3` tables** (`plans` + `plan_steps`), via a dedicated `core/plans.py` mirroring the turn_trace store pattern. |
| Goal source | **Both** — (a) an explicitly handed goal, and (b) candidate goals surfaced from the Bearing `active_goal` + curiosity pulls. The planner *surfaces* candidates; it does not pick unbidden (propose-only). |

## 1. Concept

M1 is the **constructive counterpart to `commitment_detector`** (which only *diagnoses* "I'll do X" with no matching tool call — the identity-seed refusal "do not announce action as a substitute for taking it"). M1 turns "I'll do X" into a **tracked, decomposed plan** with explicit steps, dependencies, and status — so a stated intention becomes a first-class, auditable artifact rather than a forgotten promise.

```
goal (handed | bearing active_goal | curiosity pull)
        │
   plan.decompose  ── ONE grounded LLM call ── ordered steps {verb, target, depends_on}
        │            validate: DAG acyclic, deps reference earlier steps
        ▼
   plans / plan_steps tables (turn_trace.sqlite3)  +  canonical_log: plan_proposed
        │
   surfaced: `plan` skill (show / candidates) + read-only Observer line (active plan + next step)
        │
   steps execute via the EXISTING gated tool loop / human → plan.mark(step, done)
        │
   canonical_log: plan_step_marked → plan_status_changed (done when all steps done)
```

Nothing auto-executes; nothing is pursued unbidden. The planner's "decision" is the **decomposition** + the **next-ready-step proposal**; pursuit stays behind the existing `PolicyDecision`/tool-loop gates.

## 2. What gets built

1. **`core/plans.py`** — plan store in `turn_trace.sqlite3` (own connection, `set_db_path` test override, `_db_lock`, `busy_timeout`, best-effort writes, mirrors turn_trace):
   - `plans(id, plan_uid, goal, source, status, created_at, turn_id)` — status: `proposed|active|done|abandoned`; source: `explicit|bearing|curiosity`.
   - `plan_steps(id, plan_id, seq, verb, target, depends_on, status, note)` — status: `pending|done|failed|skipped`; `depends_on` = JSON array of earlier seqs.
   - API: `create_plan`, `get_plan`, `get_active_plan`, `list_plans`, `mark_step`, `set_plan_status`, `next_ready_steps` (steps whose deps are all `done`).
2. **`core/planner.py`** — logic:
   - `propose_candidates()` — deterministic (no LLM): candidate goals = Bearing `active_goal` + top curiosity pulls (`curiosity.detect_pulls(force=True)`), ranked + de-duplicated. The "what could I pursue" surface.
   - `decompose(goal, source, *, turn_id=None, backend=None)` — ONE grounded LLM call (`_call_llm`, monkeypatchable) → parse `STEP` lines → validate DAG (acyclic; `depends_on` reference earlier seqs only) → `create_plan` → emit `plan_proposed`. Propose-only.
3. **`skills/plan/`** — ops (all force-capable, propose-only):
   - `candidates` — surface candidate goals (deterministic).
   - `decompose` — decompose a handed goal (or a referenced candidate/pull) into a plan.
   - `show` / `status` — active plan + next ready step(s) + progress (X/Y done).
   - `mark` — mark a step `done`/`failed`/`skipped` (progress tracking; the human/tool-loop drives actual execution).
4. **`canonical_log_kinds.py`** — `plan_proposed`, `plan_step_marked`, `plan_status_changed`; `KIND_VERSION` 7→8.
5. **Observer line** — read-only: when an active plan exists, surface "Active plan: <goal> — next: <verb target> (X/Y done)". Gated by the planner flag (ships dark).
6. **Flag** — `MONOLITH_PLANNER_V1` (default OFF / ships dark) gates the Observer auto-surfacing; the `plan` skill works regardless (explicit invocation / force).

No auto-run heartbeat: plans are created on invocation (decompose is an LLM call — must not fire unbidden, consistent with propose-only). The only background surface is the read-only Observer plan line.

## 3. Decompose LLM contract (parse like identity_review)

```
PLAN: <one-line restatement of the goal>
STEP: <verb> | <target> | depends: <comma-separated earlier step numbers, or none>
STEP: ...
```
Parser: seq = line order (1-based); `depends_on` must reference only earlier seqs (else reject that dep). Reject empty plans. Cap steps (e.g. ≤12) + per-field length. Acyclic by construction (deps point backward).

## 4. Done-gate (acceptance)

`tests/test_planner_firerate.py` (+ live check): a realistic goal decomposes into a valid, acyclic, persisted plan with ≥2 ordered steps and correct `depends_on`; `next_ready_steps` returns the dependency-respecting frontier; `propose_candidates()` returns goals drawn from a non-empty Bearing goal and/or live curiosity pulls. Verify read-only against the live DB before declaring done.

## 5. Propose-only / frozen / Mad Cow

The planner proposes + tracks; it never executes a step or mutates ACUs/identity. Execution rides the existing gated tool loop. Frozen (untouched): `monokernel/*`, `engine/bridge.py`, `core/world_state.py`, `core/task.py` (M1 builds its OWN plan model above the flat frozen `Task`, never modifying it).

## 6. Deferred to V1

- **Auto-execution / DAG-driven scheduling** (the planner driving the tool loop) — explicitly OUT (cuts against propose-only; needs new rails).
- Autonomous goal *selection* (vs surfacing candidates).
- `commitment_detector` integration (auto-open a plan from a detected "I'll do X").
- Re-planning / step edits / plan revision; richer step typing (tool_required, rollback_safe) from the SESSION_STATE Plan-stage spec.
