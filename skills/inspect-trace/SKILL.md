---
name: inspect_trace
description: Read-only inspection of stage_traces (interceptor + prompt-build stage events) stored in turn_trace.sqlite3. Verbs — recent (last N stage records across turns), for_turn (all stage records for a given turn_id). Diagnostic surface for E and Monolith — does not write.
---

Inspect the most recent stage traces across all turns:
{"name":"inspect_trace","arguments":{"verb":"recent","limit":10}}

Inspect all stage traces for a specific turn:
{"name":"inspect_trace","arguments":{"verb":"for_turn","turn_id":"<turn_id>"}}
