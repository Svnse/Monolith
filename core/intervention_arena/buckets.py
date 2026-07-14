"""Bucket signatures over TurnShape.

A bucket is a deterministic predicate over a TurnShape tuple. Signatures
are stored as JSON on intervention rows. Matching is pure equality / set
membership; no embedding similarity (per spec §14).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from core.turn_shape import TurnShape


@dataclass(frozen=True)
class BucketSignature:
    """A serializable bucket predicate.

    Conditions are AND-ed across the dataclass fields: every non-empty
    condition must match the TurnShape for evaluate() to return True.
    """
    name: str
    task_type_in: tuple[str, ...] = ()
    effort_tier_in: tuple[str, ...] = ()
    intent_tags_any_of: tuple[str, ...] = ()
    min_confidence: float | None = None
    complexity_min: int | None = None
    complexity_max: int | None = None

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "task_type_in": list(self.task_type_in),
            "effort_tier_in": list(self.effort_tier_in),
            "intent_tags_any_of": list(self.intent_tags_any_of),
            "min_confidence": self.min_confidence,
            "complexity_min": self.complexity_min,
            "complexity_max": self.complexity_max,
        })

    @classmethod
    def from_json(cls, raw: str) -> "BucketSignature":
        data = json.loads(raw)
        return cls(
            name=data["name"],
            task_type_in=tuple(data.get("task_type_in") or ()),
            effort_tier_in=tuple(data.get("effort_tier_in") or ()),
            intent_tags_any_of=tuple(data.get("intent_tags_any_of") or ()),
            min_confidence=data.get("min_confidence"),
            complexity_min=data.get("complexity_min"),
            complexity_max=data.get("complexity_max"),
        )


def evaluate(shape: TurnShape, sig: BucketSignature) -> bool:
    """Return True iff the shape matches every non-empty condition in sig."""
    if sig.task_type_in and shape.task_type not in sig.task_type_in:
        return False
    if sig.effort_tier_in and shape.effort_tier not in sig.effort_tier_in:
        return False
    if sig.intent_tags_any_of:
        if not (set(shape.intent_tags) & set(sig.intent_tags_any_of)):
            return False
    if sig.min_confidence is not None and shape.confidence < sig.min_confidence:
        return False
    if sig.complexity_min is not None and shape.complexity_score < sig.complexity_min:
        return False
    if sig.complexity_max is not None and shape.complexity_score > sig.complexity_max:
        return False
    return True


BUCKET_REGISTRY: dict[str, BucketSignature] = {
    "analysis_high_plus": BucketSignature(
        name="analysis_high_plus",
        task_type_in=("analysis",),
        intent_tags_any_of=("plan", "analysis", "review"),
        effort_tier_in=("high", "xhigh", "ultimate"),
    ),
    "action_with_tools": BucketSignature(
        name="action_with_tools",
        task_type_in=("action",),
        intent_tags_any_of=("debug", "code", "repair", "retrieval"),
    ),
    "attribution_calibration": BucketSignature(
        name="attribution_calibration",
        min_confidence=0.5,
    ),
}
