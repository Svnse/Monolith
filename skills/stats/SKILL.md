---
name: stats
description: Query the user's Monolith usage statistics (lifetime, rollups, distributions, records, achievements, substrate health) for the stats addon, the Wrapped narrative generator, or any other feature that needs aggregate telemetry. Verbs: `lifetime` (total turns/tokens/streak), `rollups range=week|month|year|all` (per-day aggregates), `distribution plane=effort|reasoning range=...` (counts per mode), `records` (six personal-record cards), `achievements limit=N` (newest unlock events), `substrate` (backends + monothink + continuity), `time_rhythm range=...` (per weekday-hour density), `pipeline_cost range=...` (mean stage latency), `wrapped_brief range=week|month|year` (combined brief sized for the Wrapped prompt). All numbers are read-only; this tool does not mutate state.
---

Lifetime summary (turns, tokens, first-turn date, day count):
{"name":"stats","arguments":{"verb":"lifetime"}}

Per-day rollups within a range:
{"name":"stats","arguments":{"verb":"rollups","range":"month"}}

Effort tier distribution this week:
{"name":"stats","arguments":{"verb":"distribution","plane":"effort","range":"week"}}

Reasoning mode distribution all-time:
{"name":"stats","arguments":{"verb":"distribution","plane":"reasoning","range":"all"}}

Six personal records:
{"name":"stats","arguments":{"verb":"records"}}

Newest 5 achievement unlocks:
{"name":"stats","arguments":{"verb":"achievements","limit":5}}

Substrate health (backends + monothink + continuity):
{"name":"stats","arguments":{"verb":"substrate"}}

Per (weekday, hour) density for the last month:
{"name":"stats","arguments":{"verb":"time_rhythm","range":"month"}}

Mean stage latency per pipeline stage for the last week:
{"name":"stats","arguments":{"verb":"pipeline_cost","range":"week"}}

Wrapped brief — all aggregates packaged for the Wrapped prompt:
{"name":"stats","arguments":{"verb":"wrapped_brief","range":"month"}}
