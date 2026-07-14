---
name: scratchpad
description: Continuity workspace + working-memory slot + substrate-amendment queue + confidence log + review loop + subsystem introspection. Six surfaces, thirteen ops. (1) Pins — pin / retire / read for cross-session continuity (anchor, pending, lesson categories; 8 active cap; anchor pins with rule-language + third-person Monolith framing are refused at write time as pin-10 self-violation pattern unless bypass_self_violation_check=true). (2) WORKING MEMORY — working_memory_set / working_memory_get / working_memory_clear for per-session in-flight reasoning state (single paragraph, ≤1000 chars, cleared on session boundary or model swap). (3) Proposed amendments — propose_amendment / list_proposals for Monolith-authored substrate edits to identity.md or system.md (E reviews and applies manually). (4) Confidence log — record_confidence to persist ANALYSIS-loop confidence numbers (value 0-100, claim, premise; 200-record cap; surfaced as [CONFIDENCE TRAJECTORY] each turn). (5) Review loop — review_read / review_mark / observe for unresolved substrate signals. (6) Subsystem introspection — introspect to enumerate registered pipeline policies, planes, skills, and interceptors from the bootstrap-time map (closes the "can't enumerate own subsystems without grep" gap).
---

Pin a calibration lesson with concrete evidence:
{"name":"scratchpad","arguments":{"op":"pin","category":"lesson","source":"i_inferred","text":"When user says 'no need to draft doc', paste in chat — don't write reports.","evidence":"asked to re-draft scratchpad mid-task"}}

Pin an open promise:
{"name":"scratchpad","arguments":{"op":"pin","category":"pending","source":"user_said","text":"Formalize the per-spec template once tool-call work lands."}}

Pin a load-bearing anchor (rare; never auto-decays):
{"name":"scratchpad","arguments":{"op":"pin","category":"anchor","source":"user_said","text":"We're in the Monolith checkout; another repository is reference-only."}}

Supersede an old lesson with a refined one (predecessor auto-retires):
{"name":"scratchpad","arguments":{"op":"pin","category":"lesson","source":"i_inferred","text":"Refined: terse only when user is in flow — long form is fine for spec work.","supersedes":7}}

Retire a pin you've resolved or that's no longer correct:
{"name":"scratchpad","arguments":{"op":"retire","id":7,"reason":"resolved"}}

Read current store (include retired tail to see the arc):
{"name":"scratchpad","arguments":{"op":"read","include_retired":true}}

Write WORKING MEMORY at turn-end if there's something load-bearing to carry into the next turn (single paragraph, ≤1000 chars, no internal-whitespace normalization, cleared on session boundary or model swap):
{"name":"scratchpad","arguments":{"op":"working_memory_set","text":"Mid-derivation: traced bug to skill_runtime.py:271 dynamic loader; next turn finish reading the executor.py file and write the fix."}}

Read the current WORKING MEMORY slot:
{"name":"scratchpad","arguments":{"op":"working_memory_get"}}

Clear the WORKING MEMORY slot explicitly (use when its content is obsolete; otherwise no-op = carry-forward by design):
{"name":"scratchpad","arguments":{"op":"working_memory_clear"}}

Propose an amendment to prompts/system.md, or to the EMERGENT region of identity.md (E reviews and applies manually — your role is to author the proposal, not to apply it). Required fields: target (must be "identity.md" or "system.md"), section (the section header being amended), current_text (verbatim from current substrate, ≤2000 chars), proposed_text (replacement, ≤2000 chars), rationale (one paragraph, ≤800 chars). Reject conditions: missing fields, invalid target, oversize text, no-op (current == proposed), and — for identity.md — any amendment that targets an ORIGIN-0 section or line (Origin-0 is frozen; M2). For identity evolution prefer the dedicated identity_review skill, which scores emergence and drafts an Emergent-region amendment for you:
{"name":"scratchpad","arguments":{"op":"propose_amendment","target":"system.md","section":"RESPONSE DISCIPLINE","current_text":"<verbatim slice of the section being replaced>","proposed_text":"<verbatim slice with your change>","rationale":"<why this amendment captures an earned observation>"}}

List queued proposed amendments (most recent first):
{"name":"scratchpad","arguments":{"op":"list_proposals"}}

Record an ANALYSIS-loop confidence value (value int 0-100, claim ≤200 chars, premise ≤200 chars; appended to confidence_log.jsonl; surfaced in [CONFIDENCE TRAJECTORY] each turn):
{"name":"scratchpad","arguments":{"op":"record_confidence","value":85,"claim":"that the spec correctly captures the requirements","premise":"my reading of the user's intent in turn 3"}}

Record a low-confidence value when the load-bearing premise is uncertain:
{"name":"scratchpad","arguments":{"op":"record_confidence","value":55,"claim":"that the regex handles all edge cases","premise":"I read the test cases but did not run them yet"}}

Read unresolved substrate review items:
{"name":"scratchpad","arguments":{"op":"review_read","kind":"audit_claim","limit":5}}

Mark a review item without mutating its source store:
{"name":"scratchpad","arguments":{"op":"review_mark","item_id":"acu:25","action":"escalate","note":"Needs E attention after runtime check."}}

Record an experiment/probe observation before it becomes a pin or ACU:
{"name":"scratchpad","arguments":{"op":"observe","summary":"Die-roll tautology probe improved after tool-result receipt ablation.","reason":"Probe result needs review before becoming a pin or ACU.","severity":2,"subkind":"probe"}}

Enumerate registered subsystems (full snapshot — all kinds):
{"name":"scratchpad","arguments":{"op":"introspect"}}

Filter to a specific kind (policies | planes | skills | interceptors | all):
{"name":"scratchpad","arguments":{"op":"introspect","kind":"planes"}}

Filter by name substring across the chosen kind:
{"name":"scratchpad","arguments":{"op":"introspect","kind":"skills","name":"scratchpad"}}
