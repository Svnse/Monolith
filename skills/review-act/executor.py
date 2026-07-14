"""review_act — the dedicated, single-purpose review-queue actuator tool.

Keystone of the 2026-06-19 trigger-plan audit fix. The 13-op `scratchpad` tool
could not be safely allow-listed for an autonomous wake (allow-listing it leaks
op=pin / op=propose_amendment / etc.). This tool exposes ONLY the two safe,
reversible review actions — snooze, escalate — and delegates to the flag-gated
`core.self_maint.safe_review_act`, so observe-first (the MONOLITH_SELF_MAINT_V1
gate) is real and the resolve/dismiss refusal is enforced three deep (here, in
self_maint, and at review_mark's authz). resolve/dismiss are E's judgment call.
"""
from __future__ import annotations

from typing import Any

from core import self_maint

_SAFE_ACTIONS = ("snooze", "escalate")


def run(cmd: dict, ctx: Any = None) -> str:
    item_id = str(cmd.get("item_id") or cmd.get("id") or "").strip()
    action = str(cmd.get("action") or "").strip().lower()
    if not item_id:
        return "[review_act: item_id is required]"
    if action not in _SAFE_ACTIONS:
        return ("[review_act: action must be 'snooze' or 'escalate'. "
                "resolve/dismiss are E's call — use escalate to flag those.]")
    note = cmd.get("note")
    res = self_maint.safe_review_act(item_id, action, note=str(note) if note is not None else None)
    if res.get("ok"):
        return f"[review_act: {action} {item_id} ok]"
    return f"[review_act: not applied ({res.get('refused', 'error')}): {res.get('detail', '')}]"
