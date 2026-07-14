"""Self-maintenance actuator (v0) — the model's SAFE hand on its own review queue.

Lets Monolith act on review_loop items, restricted to the two actions that carry
NO un-checked judgment: snooze (reversible defer) and escalate (flag for E). It
REFUSES resolve/dismiss — resolving or dismissing a real claim is a consequential
judgment with no fake-proof check, so it stays E's call. Every call is logged to
CONFIG_DIR/self_maint.ledger.jsonl. Flag MONOLITH_SELF_MAINT_V1 (default OFF ->
refuses all -> byte-identical to having no actuator).

Plan 1 of docs/superpowers/specs/2026-06-19-autonomous-self-maintenance-design.md.
Deliberately excludes the autonomous trigger, bug-claim re-verification, and the
self-modify gate (later plans).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from core.paths import CONFIG_DIR
from core import review_loop

_FLAG_ENV = "MONOLITH_SELF_MAINT_V1"
_TRUTHY = {"1", "true", "yes", "on"}
_LEDGER = CONFIG_DIR / "self_maint.ledger.jsonl"
# The only actions safe to take without a human decision.
_SAFE_ACTIONS = frozenset({"snooze", "escalate"})


def enabled() -> bool:
    """True only when MONOLITH_SELF_MAINT_V1 is set truthy (default OFF)."""
    return os.environ.get(_FLAG_ENV, "0").strip().lower() in _TRUTHY


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(row: dict) -> None:
    """Append one JSON row to the actuator ledger. Best-effort; never raises."""
    try:
        _LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with _LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def safe_review_act(item_id: str, action: str, *, note: str | None = None) -> dict[str, Any]:
    """Apply a SAFE review action (snooze|escalate) on behalf of Monolith.

    Refuses (does not apply) when the flag is off, or when the action is not in
    the safe set — resolve/dismiss are judgment calls that route to E. Logs every
    call (applied or refused). Never raises into the caller.
    """
    act = str(action or "").strip().lower()
    if not enabled():
        out = {"ok": False, "refused": "flag_off", "item_id": item_id, "action": act}
        _log({"ts": _now_iso(), **out})
        return out
    if act not in _SAFE_ACTIONS:
        out = {"ok": False, "refused": "unsafe_action", "item_id": item_id, "action": act,
               "detail": "resolve/dismiss are judgment calls — escalate to E instead"}
        _log({"ts": _now_iso(), **out})
        return out
    try:
        res = review_loop.review_mark(item_id, act, actor="monolith", note=note)
        out = {"ok": True, "item_id": item_id, "action": act, "state": res.get("state")}
    except review_loop.ReviewAuthorizationError as exc:
        out = {"ok": False, "refused": "not_authorized", "item_id": item_id, "action": act, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001 — never raise into the model's turn
        out = {"ok": False, "refused": "error", "item_id": item_id, "action": act, "detail": str(exc)}
    _log({"ts": _now_iso(), "note": note, **out})
    return out
