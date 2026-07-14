---
name: monopulse
description: Pull-only runtime attention view over MonoSearch, plans, reminders, investigations, health, drift, and recent traces. Verbs: pulse (combined attention), hotspots (recurring failures/signals), stalled (plans/reminders/investigations needing next action), drift (coherence/bearing/unresolved identity), changed (recent health/trace/investigation movement). Read-only and no automatic prompt injection.
---

Combined attention pulse:
{"name":"monopulse","arguments":{"verb":"pulse","limit":12}}

Recurring problems and signals:
{"name":"monopulse","arguments":{"verb":"hotspots","limit":10}}

Open work that needs a next action:
{"name":"monopulse","arguments":{"verb":"stalled","limit":10}}

Coherence, bearing rejection, and unresolved self-claims:
{"name":"monopulse","arguments":{"verb":"drift","limit":10}}

Recent runtime movement:
{"name":"monopulse","arguments":{"verb":"changed","limit":10}}
