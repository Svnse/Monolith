from __future__ import annotations

from typing import Any


def validate_mission_contract(mission_contract: dict[str, Any]) -> dict[str, Any]:
    contract = dict(mission_contract or {})
    objective = str(contract.get("objective") or "").strip()
    constraints = contract.get("constraints") if isinstance(contract.get("constraints"), list) else []
    artifacts = contract.get("artifacts") if isinstance(contract.get("artifacts"), list) else []
    raw_criteria = contract.get("success_criteria") if isinstance(contract.get("success_criteria"), list) else []
    raw_pipeline = contract.get("intent_pipeline") if isinstance(contract.get("intent_pipeline"), list) else []

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(raw_criteria):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "").strip() or f"criterion_{idx+1}"
        if cid in seen_ids:
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        normalized.append({
            "id": cid,
            "text": text,
            "required": bool(item.get("required", True)),
        })
        seen_ids.add(cid)

    if not normalized:
        normalized = [{
            "id": "fulfill_request",
            "text": "Fulfill the request directly with useful, evidence-backed output.",
            "required": True,
        }]

    # Guarantee at least one required criterion.
    if not any(bool(c.get("required", True)) for c in normalized):
        normalized[0]["required"] = True

    pipeline: list[str] = []
    seen_pipeline: set[str] = set()
    for item in raw_pipeline:
        text = str(item or "").strip()
        if not text or text in seen_pipeline:
            continue
        pipeline.append(text)
        seen_pipeline.add(text)
    if len(pipeline) > 12:
        pipeline = pipeline[:12]

    return {
        "objective": objective,
        "constraints": [str(c) for c in constraints],
        "success_criteria": normalized,
        "artifacts": [a for a in artifacts if isinstance(a, dict)],
        "intent_pipeline": pipeline,
    }
