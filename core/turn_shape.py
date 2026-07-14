"""TurnShape — system-side classification record for one turn.

A frozen dataclass produced by `core.turn_classifier.classify()` BEFORE the
LLM dispatch. Captures the deterministic shape of the current turn:
effort tier, complexity score, intent tags, task type, confidence.

Consumers:
  - core/effort.py Layer 4 — reads `effort_tier` to resolve the tier
    when no user override is active.
  - core/turn_trace.py FrameTraceRecord.classification — inspector
    visibility via /trace and inspect_trace.
  - core/lag_watch.py — per-turn JSONL drift telemetry.

Never read by the LLM, never round-tripped through it. The whole point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TurnShape:
    """Classification record for one turn — pure function of the input."""
    effort_tier: str  # classifier output: low, med, high, xhigh, ultimate; user override may set off
    complexity_score: int  # 0-100 from adaptive_budget.compute_complexity_score
    intent_tags: tuple[str, ...]  # multi-valued: chat, debug, code, analysis, plan, learn, review, retrieval, creative, vent
    task_type: str  # primary category: conversation | action | analysis
    confidence: float  # 0.0-1.0; lower means heuristics weren't certain

    def to_dict(self) -> dict[str, Any]:
        return {
            "effort_tier": self.effort_tier,
            "complexity_score": self.complexity_score,
            "intent_tags": list(self.intent_tags),
            "task_type": self.task_type,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> "TurnShape | None":
        """Reverse of to_dict. Returns None for invalid input (best-effort)."""
        if not isinstance(raw, dict):
            return None
        try:
            return cls(
                effort_tier=str(raw.get("effort_tier") or "med"),
                complexity_score=int(raw.get("complexity_score") or 0),
                intent_tags=tuple(str(t) for t in (raw.get("intent_tags") or [])),
                task_type=str(raw.get("task_type") or "conversation"),
                confidence=float(raw.get("confidence") or 0.0),
            )
        except (TypeError, ValueError):
            return None
