from __future__ import annotations

import re
from engine.loop.mission_decomposer import validate_mission_contract


def compile_turn_intent(user_prompt: str) -> tuple[dict, dict]:
    text = str(user_prompt or "").strip()
    lowered = text.lower()

    intent_type = _classify_intent(lowered)
    response_mode = "reply"
    completion_policy = "single_turn"
    artifacts: list[dict[str, str]] = []
    criteria: list[dict[str, object]] = []
    constraints: list[str] = []
    confidence_target = 0.7

    if intent_type == "coding":
        response_mode = "reply_then_act"
        completion_policy = "iterative"
        confidence_target = 0.85
        artifact_name = _extract_named_artifact(text)
        if artifact_name:
            artifacts.append({"name": artifact_name, "type": "file"})
            criteria.append({
                "id": "artifact_created",
                "text": f"Create/update requested artifact: {artifact_name}",
                "required": True,
            })
        if "game" in lowered:
            criteria.append({
                "id": "functional_behavior",
                "text": "Implement functional behavior (not only placeholder scaffolding).",
                "required": True,
            })
        if any(k in lowered for k in ("fix", "bug", "issue", "error")):
            criteria.append({
                "id": "root_cause_fix",
                "text": "Address the root cause and verify the fix with evidence.",
                "required": True,
            })
        criteria.append({
            "id": "verification_evidence",
            "text": "Provide verification evidence for core requirements.",
            "required": True,
        })
    elif intent_type == "social":
        criteria.append({
            "id": "social_reply",
            "text": "Respond naturally and relevantly to the user's message.",
            "required": True,
        })
        confidence_target = 0.6
    elif intent_type == "qa":
        criteria.append({
            "id": "answer_question",
            "text": "Provide a direct, relevant answer to the user's question.",
            "required": True,
        })
        criteria.append({
            "id": "state_uncertainty",
            "text": "If uncertain, state uncertainty or assumptions clearly.",
            "required": True,
        })
    else:
        criteria.append({
            "id": "fulfill_request",
            "text": "Fulfill the request directly with a useful response.",
            "required": True,
        })

    turn_intent = {
        "intent_type": intent_type,
        "response_mode": response_mode,
        "objective": text,
        "completion_policy": completion_policy,
        "confidence_target": confidence_target,
    }
    mission_contract = {
        "objective": text,
        "constraints": constraints,
        "success_criteria": criteria,
        "artifacts": artifacts,
    }
    return turn_intent, validate_mission_contract(mission_contract)


def render_mission_contract_block(turn_intent: dict, mission_contract: dict) -> str:
    intent_type = str(turn_intent.get("intent_type") or "unknown")
    response_mode = str(turn_intent.get("response_mode") or "reply")
    completion_policy = str(turn_intent.get("completion_policy") or "single_turn")
    confidence_target = turn_intent.get("confidence_target")
    objective = str(mission_contract.get("objective") or turn_intent.get("objective") or "").strip()
    constraints = mission_contract.get("constraints") if isinstance(mission_contract.get("constraints"), list) else []
    criteria = mission_contract.get("success_criteria") if isinstance(mission_contract.get("success_criteria"), list) else []
    artifacts = mission_contract.get("artifacts") if isinstance(mission_contract.get("artifacts"), list) else []

    lines = [
        "MISSION CONTRACT",
        f"- intent_type: {intent_type}",
        f"- response_mode: {response_mode}",
        f"- completion_policy: {completion_policy}",
        f"- confidence_target: {confidence_target}",
        f"- objective: {objective or '<none>'}",
        "- success_criteria:",
    ]
    if criteria:
        for item in criteria:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or "").strip() or "criterion"
            text = str(item.get("text") or "").strip() or "<empty>"
            required = bool(item.get("required", True))
            req = "required" if required else "optional"
            lines.append(f"  - [{cid}] ({req}) {text}")
    else:
        lines.append("  - <none>")

    lines.append("- constraints:")
    if constraints:
        for c in constraints:
            lines.append(f"  - {str(c)}")
    else:
        lines.append("  - <none>")

    lines.append("- artifacts:")
    if artifacts:
        for a in artifacts:
            if isinstance(a, dict):
                lines.append(f"  - {a.get('name', 'artifact')} ({a.get('type', 'unknown')})")
    else:
        lines.append("  - <none>")
    return "\n".join(lines).strip()


def _classify_intent(lowered: str) -> str:
    if not lowered:
        return "meta"
    if re.fullmatch(r"(hi|hello|hey|yo|sup|good morning|good afternoon|good evening)[!. ]*", lowered):
        return "social"
    if "?" in lowered or lowered.startswith(("what ", "why ", "how ", "when ", "where ", "who ")):
        if any(k in lowered for k in ("file", "code", "bug", "python", ".py", "build", "implement", "fix")):
            return "coding"
        return "qa"
    if any(k in lowered for k in (
        "code", "file", "bug", "python", ".py", "function", "class",
        "build", "create", "implement", "fix", "refactor", "script", "game",
    )):
        return "coding"
    return "meta"


def _extract_named_artifact(text: str) -> str:
    patterns = [
        r"\bname it\s+([A-Za-z0-9_\-./\\]+\.[A-Za-z0-9_]+)\b",
        r"\bnamed\s+([A-Za-z0-9_\-./\\]+\.[A-Za-z0-9_]+)\b",
        r"\bcreate\s+([A-Za-z0-9_\-./\\]+\.[A-Za-z0-9_]+)\b",
        r"\bmake\s+([A-Za-z0-9_\-./\\]+\.[A-Za-z0-9_]+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""

