---
name: acu_mirror
description: Read-only ACU Mirror V1. Inspect emerging ACU state without promoting, editing, retiring, scoring-persisting, or bypassing E's manual gates. Shows identity_review-surfaceable claims, near-threshold claims, blocked-above-threshold claims, contradictions, decay preview, pending candidates, scope buckets, confidentity, stability, provenance, and surfacing reasons. Deterministic; no LLM; no writes.
---

Inspect the mirror as text:
{"name":"acu_mirror","arguments":{"op":"snapshot"}}

Tune the identity_review threshold band:
{"name":"acu_mirror","arguments":{"op":"snapshot","threshold":0.2,"near_band":0.05,"limit":8}}

Return machine-readable JSON:
{"name":"acu_mirror","arguments":{"op":"snapshot","format":"json"}}
