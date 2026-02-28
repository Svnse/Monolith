from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.loop.events import (
    RUN_REASON_CODES,
    RUN_STATE_CREATED,
    RUN_STATES,
    RUN_STATE_TRANSITIONS,
)


@dataclass(frozen=True)
class LifecycleTransitionResult:
    record: dict[str, Any]
    warnings: list[str]


def apply_lifecycle_transition(
    *,
    current: dict[str, Any] | None,
    next_state: str,
    reason: str = "",
    extra: dict[str, Any] | None = None,
) -> LifecycleTransitionResult:
    rec = dict(current or {})
    warnings: list[str] = []

    ns = str(next_state or "").strip().lower()
    if ns not in RUN_STATES:
        warnings.append(f"invalid state ignored: {next_state}")
        return LifecycleTransitionResult(record=rec, warnings=warnings)

    rsn = str(reason or "").strip()
    if rsn and rsn not in RUN_REASON_CODES:
        warnings.append(f"non-standard reason: {rsn}")

    prev = str(rec.get("state") or RUN_STATE_CREATED)
    allowed = RUN_STATE_TRANSITIONS.get(prev, frozenset())
    if prev != ns and prev in RUN_STATE_TRANSITIONS and ns not in allowed:
        warnings.append(f"illegal transition: {prev} -> {ns} (reason={rsn or '?'})")

    rec["state"] = ns
    if rsn:
        rec["reason"] = rsn
    if extra:
        rec.update(dict(extra))
    return LifecycleTransitionResult(record=rec, warnings=warnings)

