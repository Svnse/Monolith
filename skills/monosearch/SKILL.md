---
name: monosearch
description: Read-only federated search across your own stores and capability catalog. Verbs: failing (fault kinds ranked by recurrence/recency), recurring, pulling, unresolved, search/find (keyword across stores, evidence-tier ranked), get (one record by id like tool:edit_file / skill:monosearch / fault:991 / acu:42 / clog:1840 / warrant:acu:42). Use meta=tools, skills, capabilities, debug, memory, or workflows for high-level discovery. Use source=warrants or source=claim_graph for Acatalepsy claim/evidence/warrant chains. Does not write.
---

What am I failing at? (the fault kinds you keep tripping, most-recurrent first):
{"name":"monosearch","arguments":{"verb":"failing","limit":10}}

What keeps recurring across my stores?:
{"name":"monosearch","arguments":{"verb":"recurring","limit":10}}

What's drawing my curiosity right now? (the real pulls, not a nudge):
{"name":"monosearch","arguments":{"verb":"pulling","limit":10}}

What self-claims have emerged that I haven't resolved?:
{"name":"monosearch","arguments":{"verb":"unresolved","limit":10}}

Search all stores by keyword (returns a cross-section):
{"name":"monosearch","arguments":{"verb":"search","query":"think tag","limit":10}}

Search Acatalepsy warrant / claim-evidence graph:
{"name":"monosearch","arguments":{"verb":"search","source":"warrants","query":"friction","limit":10}}
{"name":"monosearch","arguments":{"source":"warrants","query":"friction","limit":10}}

Search routing rule:
- Use `meta` for broad intent buckets: tools, skills, capabilities, debug, memory, workflows.
- Use `source` when you know the store. If a meta search misses, retry the likely exact source before concluding no record exists.
- `meta=debug` searches faults, turns, stages, ratings, and health. It does not search canonical_log.

Canonical timeline / conversation log:
{"name":"monosearch","arguments":{"verb":"search","source":"history","query":"closing tag","limit":10}}
{"name":"monosearch","arguments":{"verb":"search","source":"canonical_log","query":"","limit":10}}
{"name":"monosearch","arguments":{"verb":"get","id":"clog:<event_id>"}}

Delivery or tool-loop debugging:
- Search `source=history` for canonical events.
- Search `source=turns` for turn/frame telemetry.
- Search both before saying something did not happen or did not reach the user.

Find executable tools by intent:
{"name":"monosearch","arguments":{"verb":"find","meta":"tools","query":"edit file","limit":5}}
{"name":"monosearch","arguments":{"verb":"find","meta":"tools","query":"latest online web search","limit":3}}

Find procedural skills or broad capabilities:
{"name":"monosearch","arguments":{"verb":"find","meta":"skills","query":"workflow card","limit":5}}
{"name":"monosearch","arguments":{"verb":"find","meta":"capabilities","query":"run tests after editing","limit":8}}

Fetch an exact tool schema before using an unfamiliar tool:
{"name":"monosearch","arguments":{"verb":"get","id":"tool:edit_file"}}

Search one store. Use source when you know the target store:
tools | skills | faults | knowledge/claims | warrants or claim_graph | history or canonical_log | turns | stages | memory/pins | bearing | identity | curiosity | reminders | investigations | lag | health

If you are looking for a skill/tool by name or task, use `source="skills"` or `meta="capabilities"` and put the skill/task in `query`.

{"name":"monosearch","arguments":{"verb":"search","query":"operator preference","source":"knowledge"}}
{"name":"monosearch","arguments":{"verb":"search","query":"friction","source":"warrants"}}
{"name":"monosearch","arguments":{"verb":"search","query":"tool","source":"faults"}}

Fetch one record by its namespaced id:
{"name":"monosearch","arguments":{"verb":"get","id":"fault:991"}}

## Unverified results

Some results are marked **[unverified]**. The runtime has not externally confirmed
them; they came from earlier output, not a tool result or checked claim. An
**[unverified]** result is **not a citable premise.** If you need it, re-ground it
first by running the relevant tool or re-deriving it before relying on it.
