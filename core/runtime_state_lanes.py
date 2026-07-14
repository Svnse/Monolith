"""Single source of truth for runtime-state lane order and lead phrases."""
from __future__ import annotations

from dataclasses import dataclass


CONTRACT_PLACEHOLDER = "{runtime_state_lane_contract}"


@dataclass(frozen=True)
class RuntimeStateLane:
    name: str
    lead_phrase: str
    contract_note: str


LANES: tuple[RuntimeStateLane, ...] = (
    RuntimeStateLane(
        "identity_material",
        "Identity material, as operating law:",
        "operating-law register only; not evidence for backend, locality, model, or live state",
    ),
    RuntimeStateLane(
        "continuity",
        "What I carry forward from continuity:",
        "durable pins from the continuity store; ambient state, not this turn's request",
    ),
    RuntimeStateLane(
        "recall",
        "From recalled memory:",
        "retrieved ACU claims relevant to the latest user turn; verify stale claims before acting",
    ),
    RuntimeStateLane(
        "current_model_execution",
        "Current execution facts:",
        "live model/backend/context facts from describe_self current_model_execution",
    ),
    RuntimeStateLane(
        "temporal_context",
        "Current local time:",
        "local wall-clock time/date from the temporal context provider",
    ),
    RuntimeStateLane(
        "temporal_relative",
        "Time elapsed:",
        "elapsed since the last turn / since the previous session; relative grounding, not an absolute clock",
    ),
)

LANE_ORDER: tuple[str, ...] = tuple(lane.name for lane in LANES)
_LANE_BY_NAME: dict[str, RuntimeStateLane] = {lane.name: lane for lane in LANES}


def lane_for(name: str) -> RuntimeStateLane:
    return _LANE_BY_NAME[name]


def lead_phrase(name: str) -> str:
    return lane_for(name).lead_phrase


def render_lane_contract() -> str:
    """Render the cached prompt contract from the same registry the runtime uses."""
    lines = [
        "[RUNTIME STATE] is ambient state, not this turn's request.",
        "Its lanes emit in this exact order with these exact lead phrases:",
    ]
    for lane in LANES:
        lines.append(f"- `{lane.name}` -> `{lane.lead_phrase}` {lane.contract_note}.")
    lines.extend(
        [
            "The identity_material lane stays in third-person operating-law register.",
            "Do not treat identity_material as proof of backend locality, current model, context window, or live runtime state.",
            "Use current_model_execution for backend/model/context claims and temporal_context for wall-clock claims.",
            "Use temporal_relative only for elapsed-time or session-gap orientation, not as an absolute clock.",
        ]
    )
    return "\n".join(lines)
