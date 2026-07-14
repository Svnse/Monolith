---
name: self_maint
description: Start/stop and pace the autonomous self-maintenance daemon (review-queue triage). Use to begin observe-first self-maintenance, stop it, or change how often it wakes (interval) and the daily cap — without a launcher flag or restart.
---

# self_maint — runtime control of the self-maintenance daemon

Controls the autonomous daemon that periodically wakes to triage Monolith's own
`[REVIEW QUEUE]` (snooze/escalate only). This tool starts/stops it and tunes its
cadence at runtime — no launcher flag, no restart.

**Safety:** starting here begins the OBSERVE loop only. Whether a snooze/escalate
actually APPLIES is still gated by `MONOLITH_SELF_MAINT_V1` (E's hard gate) — starting
the daemon cannot grant the power to mutate the review queue. The daemon's own wakes
cannot call this tool, so it cannot re-arm or re-pace itself in a loop.

## Ops

- **start** — begin observe-first self-maintenance. Optional `seconds` (wake interval)
  and `cap` (max wakes/day) applied first.
- **stop** — halt the daemon (asymmetric: an in-flight wake finishes).
- **set_interval** — change the wake cadence (`seconds`; floored at 15s).
- **set_cap** — change the daily wake cap (`cap`). Bounded by an absolute ceiling
  (~2880/day) so the total can't run away. Raise it only as much as you actually need —
  each wake is a real (paid) model call; the cap × interval sets the daily cost.
- **status** — report the daemon's current status / interval / apply-gate / wake count.

## Examples

```json
{"name":"self_maint","arguments":{"op":"start"}}
{"name":"self_maint","arguments":{"op":"start","seconds":60}}
{"name":"self_maint","arguments":{"op":"set_interval","seconds":60}}
{"name":"self_maint","arguments":{"op":"set_cap","cap":240}}
{"name":"self_maint","arguments":{"op":"stop"}}
{"name":"self_maint","arguments":{"op":"status"}}
```
