from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


_ALLOWED_COMMANDS = {
    "set_path",
    "set_history",
    "set_ctx_limit",
    "load",
    "unload",
    "generate",
}


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

class PolicyDecision(str, Enum):
    AUTO_APPROVE     = "auto_approve"
    REQUIRE_APPROVAL = "require_approval"
    BLOCKED          = "blocked"


# Commands that are purely config mutations — no side-effects, safe to auto-run.
_AUTO_APPROVE_COMMANDS: frozenset[str] = frozenset({
    "set_path",
    "set_history",
    "set_ctx_limit",
})

# Commands that are always rejected (reserved for future enforcement).
_BLOCKED_COMMANDS: frozenset[str] = frozenset()


def check_policy(action: dict[str, Any]) -> PolicyDecision:
    """Return the policy decision for a proposed world action."""
    if not isinstance(action, dict):
        return PolicyDecision.BLOCKED
    if action.get("type") == "engine_stop":
        return PolicyDecision.REQUIRE_APPROVAL
    cmd = str(action.get("command", ""))
    if cmd in _BLOCKED_COMMANDS:
        return PolicyDecision.BLOCKED
    if cmd in _AUTO_APPROVE_COMMANDS:
        return PolicyDecision.AUTO_APPROVE
    return PolicyDecision.REQUIRE_APPROVAL


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class WorldActionResult:
    ok: bool
    error: str | None = None
    action: dict[str, Any] | None = None


def validate_action(raw: dict[str, Any]) -> WorldActionResult:
    if not isinstance(raw, dict):
        return WorldActionResult(False, "action must be an object")
    kind = str(raw.get("type") or "")
    if kind not in {"engine_command", "engine_stop"}:
        return WorldActionResult(False, f"unknown action type: {kind}")

    if kind == "engine_stop":
        engine = str(raw.get("engine") or "")
        if not engine:
            return WorldActionResult(False, "engine_stop requires engine")
        return WorldActionResult(True, action={"type": "engine_stop", "engine": engine})

    engine = str(raw.get("engine") or "")
    command = str(raw.get("command") or "")
    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    priority = int(raw.get("priority", 2))

    if not engine:
        return WorldActionResult(False, "engine_command requires engine")
    if command not in _ALLOWED_COMMANDS:
        return WorldActionResult(False, f"unsupported command: {command}")
    if priority not in (1, 2, 3):
        priority = 2

    return WorldActionResult(
        True,
        action={
            "type": "engine_command",
            "engine": engine,
            "command": command,
            "payload": payload,
            "priority": priority,
        },
    )
