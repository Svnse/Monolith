---
name: review_act
description: Act on a [REVIEW QUEUE] item — snooze (defer when it re-surfaces) or escalate (flag for E). SAFE actions only; resolve and dismiss are E's judgment call (they silently remove a real claim from her queue), so this tool refuses them — escalate instead. Single-purpose by design: routed through the flag-gated self-maintenance actuator, it exposes nothing but these two reversible actions (so an autonomous wake can be confined to it).
---

Snooze a stale review item (defers when it next surfaces):
{"name":"review_act","arguments":{"item_id":"acu:87","action":"snooze"}}

Escalate an item that needs E's attention:
{"name":"review_act","arguments":{"item_id":"pin:6","action":"escalate","note":"blocking — needs your call"}}
