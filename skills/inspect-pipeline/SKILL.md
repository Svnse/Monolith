---
name: inspect_pipeline
description: Read-only inspection of turn-pipeline traces stored in turn_trace.sqlite3. Verbs — events (frame_traces for one turn), last (frame_traces for the most-recent turn with events), one (a single event by turn_id), faults (recent fault_traces records, optionally filtered). Diagnostic surface for E and Monolith — does not write.
---

Inspect events for a specific turn:
{"name":"inspect_pipeline","arguments":{"verb":"events","turn_id":"<turn_id>"}}

Inspect the most-recent turn that recorded pipeline events:
{"name":"inspect_pipeline","arguments":{"verb":"last"}}

Inspect a single event row by turn_id:
{"name":"inspect_pipeline","arguments":{"verb":"one","turn_id":"<turn_id>"}}

Recent fault traces (newest first; optional filters):
{"name":"inspect_pipeline","arguments":{"verb":"faults","limit":10}}

Fault traces filtered by kind (markdown_corruption / tool_no_fire / think_leak / regen_mismatch):
{"name":"inspect_pipeline","arguments":{"verb":"faults","fault_kind":"tool_no_fire"}}

Fault traces since an ISO timestamp:
{"name":"inspect_pipeline","arguments":{"verb":"faults","since":"2026-05-15T00:00:00"}}
