---
name: curiosity
description: Inspect what Monolith is currently curious about — the fresh, identity-aligned claims it is drawn toward but has not yet integrated into its identity (the not-yet-stable disposition of the identity signal; the stable disposition is the identity_review skill). Two ops. detect — deterministic (no LLM): returns ranked "pulls" with pull-strength, confidentity, stability. kill — retire a pull judged as noise (the SAFE half of closing the curiosity loop: reversible, audited to canonical_log, excluded from future surfacing; promotion-into-identity stays human-gated). Propose-only otherwise — it forms and ranks what it is curious about, but never promotes or spends compute pursuing (pursuit is the planner's job). A pull also auto-retires after resurfacing a few times or once the underlying claim reinforces into the identity.
---

See what Monolith is curious about right now (deterministic, no LLM):
{"name":"curiosity","arguments":{"op":"detect"}}

Tune the identity-alignment bar:
{"name":"curiosity","arguments":{"op":"detect","threshold":0.2}}

Retire a pull you judge as noise (reversible, audited; kill not promote — the safe half of closing the loop). Rule: never kill a pull that's externally checkable before you've checked it — kill the ungroundable rumination, keep the falsifiable:
{"name":"curiosity","arguments":{"op":"kill","canonical":"monolith | values | precision","reason":"ungroundable rumination; did not survive a kill attempt"}}
