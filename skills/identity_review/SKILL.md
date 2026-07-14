---
name: identity_review
description: Review whether self-derived ACUs have accumulated that the operative identity does not yet reflect, and (when asked) draft a propose-only amendment to the Emergent region of identity.md. Two ops. (1) detect — deterministic, no LLM — reports the high-confidentity self-derived claims (NOVEL × identity-aligned) that have emerged, with confidentity scores. (2) draft — ONE grounded LLM call that turns the top emergent candidate(s) into a first-person Emergent-region claim and queues it via the proposals queue for E to review/apply. Origin-0 is frozen and never touched (code-enforced); nothing is ever auto-applied (propose-only, explicit human gate). This is M2 identity evolution.
---

See what self-derived identity material has emerged (no LLM, deterministic):
{"name":"identity_review","arguments":{"op":"detect"}}

Tune the surfacing threshold and accrual gate:
{"name":"identity_review","arguments":{"op":"detect","threshold":0.3,"min_new":1}}

Draft a propose-only amendment to the Emergent region from the top emergent candidate (queues a proposal; E reviews and applies manually — Origin-0 is never touched):
{"name":"identity_review","arguments":{"op":"draft"}}
