"""Deterministic Kind classifier (B1) — the gate that decides which validation
rules apply to a claim.

Kinds:
  world-fact  externally-checkable claim about the world  -> Tavily-groundable
  self        claim about the system / Monolith itself    -> kept, E-confirmed
  meta        design principle / normative maxim           -> kept, E-confirmed
  causal      "X causes/influences Y"                      -> contested (hypothesis)
  emotional   affect/valence about a person                -> the Affect lane

This is a K1/K2 floor — deterministic, from the canonical triple. The auditor /
verifier may refine it later (K3/K4). Resolution order is fixed:
emotional > causal > meta > self > world-fact.
"""
from __future__ import annotations

from core.acatalepsy.normalize import CanonicalTriple

KINDS = ("world-fact", "self", "meta", "causal", "emotional")

_AFFECT = {
    "likes", "dislikes", "dislike", "hates", "hate", "loves", "love", "fears",
    "fear", "prefers", "prefer", "wants", "want", "feels", "feel", "enjoys",
    "resents", "angry", "sad", "happy", "frustrated", "afraid",
}
_CAUSAL = {
    "causes", "cause", "caused", "influences", "influence", "leads", "results",
    "drives", "drive", "enables", "prevents", "because", "triggers", "shifts",
}
_NORMATIVE = {
    "should", "must", "requires", "require", "selects", "select", "ought",
    "needs", "need", "shall",
}
# Monolith-specific subject tokens. Deliberately NOT generic words like "core"
# or "engine" (which appear in world-facts, e.g. "earth core | composed_of | iron").
_SELF = {
    "monolith", "acu", "acus", "substrate", "auditor", "observer", "bearing",
    "kernel", "monobase", "acatalepsy", "monokernel", "ofac", "scratchpad",
    "continuity", "intake",
}


def _tokens(s: object) -> set[str]:
    return set(str(s or "").lower().replace("_", " ").replace("/", " ").replace("|", " ").split())


def classify_kind(triple: CanonicalTriple | None) -> str:
    """Classify a claim's Kind from its canonical triple. Non-atomic prose
    (``triple is None``) defaults to ``self`` (identity material)."""
    if triple is None:
        return "self"
    rel = _tokens(triple.relation)
    if rel & _AFFECT:
        return "emotional"
    if rel & _CAUSAL:
        return "causal"
    if rel & _NORMATIVE:
        return "meta"
    entity_a = str(triple.entity_a or "")
    if (_tokens(entity_a) & _SELF) or "/" in entity_a or entity_a.lower().endswith(".py"):
        return "self"
    return "world-fact"
