---
name: monoexplore
description: Drive a self-directed curiosity EXPEDITION and read its coherence (V0, deterministic, dark by default behind MONOLITH_MONOEXPLORE_V1; this skill bypasses the flag for manual inspection). An expedition = a goal (explicit, or the top curiosity/bearing candidate) decomposed into a plan and pursued with the BEARING as the coherence anchor; growth counts only externally-grounded evidence (observed file/tool referents), never self-narration. Three ops. start — seed a goal (bare seeds from curiosity/bearing) and decompose it into an active plan; optional 'goal'. status — the active expedition plus its coherence verdict. coherence — the deterministic anti-drift signal only: GREEN when grounded and on-trajectory, RED when referents are mostly ungrounded (performing rather than observing), YELLOW on trajectory drift. The coherence signal is computed without an LLM — the model never grades its own coherence. Propose-only / read-biased: V0 never drives execution (the autonomous model-driving tick loop is V1).
---

Begin an expedition from your own curiosity/bearing (no goal given):
{"name":"monoexplore","arguments":{"op":"start"}}

Begin an expedition toward an explicit goal:
{"name":"monoexplore","arguments":{"op":"start","goal":"map the engine turn pipeline"}}

Check the active expedition and its coherence verdict:
{"name":"monoexplore","arguments":{"op":"status"}}

Read just the deterministic anti-drift signal:
{"name":"monoexplore","arguments":{"op":"coherence"}}
