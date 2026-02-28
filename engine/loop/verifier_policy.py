from __future__ import annotations

from engine.loop.contracts import Pad
from engine.loop.tool_intelligence import FAILURE_MISSING_DEPENDENCY


def recommend_verification_mode(pad: Pad) -> dict[str, str]:
    failures = list(getattr(pad, "tool_failures", []) or [])
    if not failures:
        return {"mode": "auto", "reason": "no_recent_failures"}
    last = failures[-1]
    failure_class = str(last.get("failure_class") or "").strip().lower()
    if failure_class == FAILURE_MISSING_DEPENDENCY:
        return {
            "mode": "static_preferred",
            "reason": "missing_dependency_detected",
        }
    if failure_class in {"permission", "path_not_found"}:
        return {
            "mode": "path_correction",
            "reason": failure_class,
        }
    return {"mode": "auto", "reason": failure_class or "recent_failure"}

