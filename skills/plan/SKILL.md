---
name: plan
description: Propose-only task/goal planner (M1). Decompose a goal into an ordered, tracked plan of steps and watch progress — the planner NEVER executes steps itself; execution rides the existing gated tool loop / you. Four ops. (1) candidates — deterministic, no LLM — surface candidate goals to pursue, drawn from the Bearing active_goal + curiosity pulls (the 'pursue a pull' verb). (2) decompose — ONE grounded LLM call turning a goal into an ordered acyclic step DAG (verb, target, depends_on), persisted. (3) show/status — the active plan, its steps, the dependency-ready next step(s), and progress. (4) mark — record a step done/failed/skipped as it completes. It's the constructive counterpart to the commitment detector: a stated 'I'll do X' becomes a tracked plan, not a forgotten promise.
---

Surface candidate goals to pursue (no LLM — from bearing goal + curiosity pulls):
{"name":"plan","arguments":{"op":"candidates"}}

Decompose a goal into an ordered plan (handed goal, or a candidate/pull text):
{"name":"plan","arguments":{"op":"decompose","goal":"add a retry policy to the HTTP client"}}

Show the active plan + the next ready step + progress:
{"name":"plan","arguments":{"op":"show"}}

Mark a step done (or failed / skipped) as it completes — propose-only progress tracking:
{"name":"plan","arguments":{"op":"mark","step":1,"status":"done"}}
