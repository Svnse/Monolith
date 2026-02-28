from __future__ import annotations

from typing import Any


def compute_gap_state(*, pad, action_failure_class: dict[str, str] | None = None) -> dict[str, Any]:
    mission = pad.mission_contract if isinstance(getattr(pad, "mission_contract", {}), dict) else {}
    criteria = mission.get("success_criteria") if isinstance(mission.get("success_criteria"), list) else []
    done = set(str(r).strip() for r in (getattr(pad, "mission_refs", []) or []) if str(r).strip())

    required_ids: list[str] = []
    all_ids: list[str] = []
    for item in criteria:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "").strip()
        if not cid:
            continue
        all_ids.append(cid)
        if bool(item.get("required", True)):
            required_ids.append(cid)

    completed_requirements = [cid for cid in all_ids if cid in done]
    open_requirements = [cid for cid in required_ids if cid not in done]
    blocked_requirements: list[str] = []

    last_failure_class = "none"
    failures = list(getattr(pad, "tool_failures", []) or [])
    if failures:
        last_failure_class = str(failures[-1].get("failure_class") or "unknown")
        if last_failure_class in {"missing_dependency", "permission", "path_not_found"}:
            if "verification_evidence" in open_requirements:
                blocked_requirements.append("verification_evidence")

    available_strategies: list[str] = []
    if last_failure_class == "missing_dependency":
        available_strategies.extend([
            "attempt_install_dependency_if_allowed",
            "switch_to_static_verification",
        ])
    elif last_failure_class in {"permission", "path_not_found"}:
        available_strategies.extend([
            "correct_path_or_boundary",
            "re-discover_workspace_state",
        ])
    elif last_failure_class in {"syntax_error", "runtime_error", "timeout"}:
        available_strategies.extend([
            "read_and_fix_source",
            "use_narrower_verification_command",
        ])
    else:
        available_strategies.extend([
            "implement_open_requirements",
            "verify_with_evidence",
        ])

    # If no explicit mission criteria, fallback to a single generic gap.
    if not criteria:
        open_requirements = ["fulfill_request"]
        completed_requirements = [] if "fulfill_request" not in done else ["fulfill_request"]

    return {
        "completed_requirements": completed_requirements,
        "open_requirements": open_requirements,
        "blocked_requirements": blocked_requirements,
        "last_failure_class": last_failure_class,
        "available_strategies": available_strategies,
    }

