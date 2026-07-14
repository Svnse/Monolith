"""Bearing schema — typed slots for the model's situational posture.

V0 is flat-and-fixed. No nested recursion; lossiness gets surfaced as faults,
structural upgrades earned by evidence not anticipated. See
`docs/superpowers/plans/<bearing-plan>.md` §6 for the layer-accounting
justification.

Total Bearing block target ≤ ~3KB to preserve KV cache.

This module defines pure data — no IO, no validation logic beyond default
construction. Validation lives in `structural_verifier.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── per-slot character limits ────────────────────────────────────────

MAX_CURRENT_FRAME = 400
MAX_ACTIVE_GOAL = 200
MAX_TRAJECTORY = 600
MAX_TENSION = 200
MAX_TENSIONS = 5
MAX_REFERENT_NAME = 120
MAX_REFERENTS = 8
MAX_BRANCH_TEXT = 200
MAX_BRANCH_REASON = 200
MAX_BRANCHES = 6
MAX_USER_INTENT_READ = 300
MAX_NEXT_MOVE = 300
MAX_STAKES_COST_IF_WRONG = 300

# ── enum value sets ──────────────────────────────────────────────────

VALID_BRANCH_STATUS = frozenset({"active", "dormant", "closed", "rejected", "superseded"})
VALID_REFERENT_STATUS = frozenset({"observed", "inferred", "predicted", "unverified"})
VALID_REFERENT_KIND = frozenset({"file", "peer", "entity", "claim", "tool_result"})
VALID_REVERSIBILITY = frozenset({"easy", "moderate", "hard", "unknown"})
VALID_URGENCY = frozenset({"low", "medium", "high"})
VALID_REGISTER = frozenset({"literal", "performative", "ironic", "exploratory"})

SCHEMA_VERSION = 1

# Observer V0 may read only these Bearing slots. It is advisory and must not
# infer private state beyond this typed posture surface.
OBSERVER_INPUT_FIELDS = (
    "current_frame",
    "active_goal",
    "trajectory",
    "open_tensions",
    "referents",
    "modal_branches",
    "stakes",
    "user_model",
    "next_move",
    "updated_at_turn",
)


# ── nested dataclasses ───────────────────────────────────────────────


@dataclass(frozen=True)
class Tension:
    text: str = ""
    opened_at_turn: str = ""


@dataclass(frozen=True)
class Referent:
    name: str = ""
    kind: str = "entity"
    status: str = "observed"
    grounded_at_turn: str = ""


@dataclass(frozen=True)
class ModalBranch:
    text: str = ""
    status: str = "active"
    reason: str = ""
    last_touched_turn: str = ""


@dataclass(frozen=True)
class Stakes:
    """V0: descriptive only. V2 (Cost Surface) makes these load-bearing."""
    reversibility: str = "unknown"
    urgency: str = "low"
    cost_if_wrong: str = ""


@dataclass(frozen=True)
class UserModel:
    intent_read: str = ""
    register: str = "literal"
    confidence: float = 0.0


# ── root Bearing dataclass ───────────────────────────────────────────


@dataclass(frozen=True)
class Bearing:
    """The model's current situational posture across turns.

    Empty Bearing (all defaults) represents "no situational state established
    yet" — used on first turn of first session or after explicit bearing_clear.
    `is_empty()` reports this.
    """

    schema_version: int = SCHEMA_VERSION
    current_frame: str = ""
    active_goal: str = ""
    trajectory: str = ""
    open_tensions: tuple[Tension, ...] = ()
    referents: tuple[Referent, ...] = ()
    modal_branches: tuple[ModalBranch, ...] = ()
    stakes: Stakes | None = None
    user_model: UserModel | None = None
    next_move: str = ""
    last_writer_model_id: str = ""
    updated_at_turn: str = ""
    # Readable monotonic turn-count, stamped alongside updated_at_turn (which
    # stays the UUID for trace-join). 0 = never stamped / feature off. Lets the
    # compiler render "N turns ago" so the model self-judges frame staleness.
    updated_at_turn_n: int = 0

    def is_empty(self) -> bool:
        return (
            self.current_frame == ""
            and self.active_goal == ""
            and self.trajectory == ""
            and self.next_move == ""
            and not self.open_tensions
            and not self.referents
            and not self.modal_branches
            and self.stakes is None
            and self.user_model is None
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "current_frame": self.current_frame,
            "active_goal": self.active_goal,
            "trajectory": self.trajectory,
            "open_tensions": [
                {"text": t.text, "opened_at_turn": t.opened_at_turn}
                for t in self.open_tensions
            ],
            "referents": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "status": r.status,
                    "grounded_at_turn": r.grounded_at_turn,
                }
                for r in self.referents
            ],
            "modal_branches": [
                {
                    "text": b.text,
                    "status": b.status,
                    "reason": b.reason,
                    "last_touched_turn": b.last_touched_turn,
                }
                for b in self.modal_branches
            ],
            "stakes": None if self.stakes is None else {
                "reversibility": self.stakes.reversibility,
                "urgency": self.stakes.urgency,
                "cost_if_wrong": self.stakes.cost_if_wrong,
            },
            "user_model": None if self.user_model is None else {
                "intent_read": self.user_model.intent_read,
                "register": self.user_model.register,
                "confidence": self.user_model.confidence,
            },
            "next_move": self.next_move,
            "last_writer_model_id": self.last_writer_model_id,
            "updated_at_turn": self.updated_at_turn,
            "updated_at_turn_n": self.updated_at_turn_n,
        }
        return out

    def to_observer_input(self) -> dict[str, Any]:
        """Return the read-only Bearing view available to Observer V0."""

        data = self.to_dict()
        return {key: data.get(key) for key in OBSERVER_INPUT_FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Bearing":
        if not isinstance(data, dict):
            return cls()
        tensions = tuple(
            Tension(
                text=str(t.get("text", "")),
                opened_at_turn=str(t.get("opened_at_turn", "")),
            )
            for t in (data.get("open_tensions") or [])
            if isinstance(t, dict)
        )
        referents = tuple(
            Referent(
                name=str(r.get("name", "")),
                kind=str(r.get("kind", "entity")),
                status=str(r.get("status", "observed")),
                grounded_at_turn=str(r.get("grounded_at_turn", "")),
            )
            for r in (data.get("referents") or [])
            if isinstance(r, dict)
        )
        branches = tuple(
            ModalBranch(
                text=str(b.get("text", "")),
                status=str(b.get("status", "active")),
                reason=str(b.get("reason", "")),
                last_touched_turn=str(b.get("last_touched_turn", "")),
            )
            for b in (data.get("modal_branches") or [])
            if isinstance(b, dict)
        )
        stakes_raw = data.get("stakes")
        stakes = None
        if isinstance(stakes_raw, dict):
            stakes = Stakes(
                reversibility=str(stakes_raw.get("reversibility", "unknown")),
                urgency=str(stakes_raw.get("urgency", "low")),
                cost_if_wrong=str(stakes_raw.get("cost_if_wrong", "")),
            )
        try:
            turn_n = int(data.get("updated_at_turn_n", 0) or 0)
        except (TypeError, ValueError):
            turn_n = 0
        if turn_n < 0:
            turn_n = 0
        um_raw = data.get("user_model")
        user_model = None
        if isinstance(um_raw, dict):
            try:
                conf = float(um_raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            user_model = UserModel(
                intent_read=str(um_raw.get("intent_read", "")),
                register=str(um_raw.get("register", "literal")),
                confidence=conf,
            )
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            current_frame=str(data.get("current_frame", "")),
            active_goal=str(data.get("active_goal", "")),
            trajectory=str(data.get("trajectory", "")),
            open_tensions=tensions,
            referents=referents,
            modal_branches=branches,
            stakes=stakes,
            user_model=user_model,
            next_move=str(data.get("next_move", "")),
            last_writer_model_id=str(data.get("last_writer_model_id", "")),
            updated_at_turn=str(data.get("updated_at_turn", "")),
            updated_at_turn_n=turn_n,
        )
