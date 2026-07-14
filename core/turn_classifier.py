"""turn_classifier — system-side pure classification of a turn.

Runs once per turn before LLM dispatch. Input: (messages, config).
Output: TurnShape (effort_tier, complexity_score, intent_tags, task_type, confidence).

Replaces the LLM-emit → world_state → next-turn-inject feedback loop
that caused the lag bug. The model is no longer asked to self-classify;
the system classifies upstream from the user message content.

Heuristics:
  - complexity_score wraps core.adaptive_budget.compute_complexity_score()
    (already present in Monolith, parked behind its own interceptor).
  - effort_tier maps complexity_score to {low, med, high, xhigh, ultimate}.
  - intent_tags from regex patterns over the latest user message.
  - task_type derived from which intent class dominates.
  - confidence floors high for clear signals (greetings, exact-keyword hits)
    and drops on ambiguity.
"""
from __future__ import annotations

import re
from typing import Any

from core.adaptive_budget import AdaptiveBudgetConfig, compute_complexity_score
from core.turn_shape import TurnShape

# Bearing addon injects a read-only provider here at bootstrap time. None
# when MONOLITH_BEARING_V1=0 or the addon failed to construct — classifier
# then behaves exactly as before. The classifier MUST NOT import the
# Bearing store directly; only this DI handle is reachable.
_bearing_provider: Any = None


def set_bearing_provider(provider: Any) -> None:
    """Wire the Bearing read-only provider. Called from bootstrap.py.

    Idempotent. Passing None disconnects the dependency (kill-switch behavior).
    """
    global _bearing_provider
    _bearing_provider = provider

# Score → tier boundaries. Tiers correspond to prompts/effort/<tier>.md
# scaffolds (depth-only after the plane-separation refactor — experimental,
# monolith, and monothink moved to /conversation/, /linguency/, and /reasoning/
# respectively). The classifier only infers depth tiers; cross-plane modes
# (conversation, reasoning, linguency) are explicit-only and never inferred here.
_TIER_BOUNDARIES: tuple[tuple[int, str], ...] = (
    (15, "low"),
    (40, "med"),
    (65, "high"),
    (85, "xhigh"),
    (100, "ultimate"),
)

_INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("debug", re.compile(r"\b(bug|error|exception|traceback|crash|stack ?trace|broken|not working|fix this)\b", re.I)),
    ("code", re.compile(r"```|\b(function|class|def |import |refactor|implement|method)\b|\.(?:py|ts|tsx|js|jsx|go|rs|cpp|java|rb|kt|swift)\b", re.I)),
    ("plan", re.compile(r"\b(plan|design|architect(?:ure)?|approach|strategy|how would you|steps to|outline)\b", re.I)),
    ("analysis", re.compile(r"\b(analyz|compare|trade-?off|critique|reason|because|why does|why is)\b", re.I)),
    ("learn", re.compile(r"\b(teach|tutorial|walk me through|explain|how does .+ work|what is)\b", re.I)),
    ("review", re.compile(r"\b(review|audit|check|inspect|verify|validate)\b", re.I)),
    ("retrieval", re.compile(r"\b(find|search|grep|where is|locate|list (?:all|the))\b", re.I)),
    ("creative", re.compile(r"\b(write me a|story|poem|brainstorm|imagine|draft)\b", re.I)),
)

_GREETING_RE = re.compile(
    r"^\s*(hey|hi|hello|yo|sup|howdy|good (?:morning|afternoon|evening))\b[!.\s]*$",
    re.I,
)

_VENT_RE = re.compile(
    r"\b(frustrated|exhausted|tired of|hate|fed up|burned out|overwhelmed)\b",
    re.I,
)

_ACTION_INTENTS = frozenset({"debug", "code", "refactor", "repair", "retrieval"})
_ANALYSIS_INTENTS = frozenset({"analysis", "plan", "review", "learn"})


def _last_user_text(messages: list[dict]) -> str:
    """Return the most recent non-ephemeral user message content."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        if msg.get("ephemeral"):
            continue
        content = msg.get("content", "")
        # Skip injection-style sentinels that shouldn't shape classification
        if any(tag in str(content) for tag in ("[BUDGET GUIDANCE]", "[SYSTEM REMINDER]", "[OBSERVED STATE", "[RUNTIME STATE]")):
            continue
        return str(content or "")
    return ""


def _score_to_tier(score: int) -> str:
    for boundary, name in _TIER_BOUNDARIES:
        if score <= boundary:
            return name
    return "ultimate"


def _classify_intent(text: str) -> tuple[tuple[str, ...], str, float]:
    """Return (intent_tags, task_type, intent_confidence) from the user text."""
    stripped = text.strip()
    if not stripped:
        return (("chat",), "conversation", 0.3)
    if _GREETING_RE.match(stripped):
        return (("chat",), "conversation", 1.0)
    if _VENT_RE.search(stripped):
        return (("vent", "chat"), "conversation", 0.85)

    matched: list[str] = []
    for tag, pat in _INTENT_PATTERNS:
        if pat.search(stripped):
            matched.append(tag)
    if not matched:
        matched.append("chat")

    if any(t in _ACTION_INTENTS for t in matched):
        task_type = "action"
    elif any(t in _ANALYSIS_INTENTS for t in matched):
        task_type = "analysis"
    else:
        task_type = "conversation"

    # Confidence: at least 0.5 (we always have *some* signal); +0.2 per match.
    confidence = min(1.0, 0.5 + 0.2 * len(matched))
    return (tuple(matched), task_type, confidence)


def _maybe_carry_forward(
    task_type: str,
    intent_tags: tuple[str, ...],
    text: str,
    config: dict,
) -> str:
    """Bearing carry-forward (V0 heuristic shape — see deviation in plan §13.3).

    The Bearing V0 plan §4.A spec says "carry `task_type` forward from prior
    turn's classification trace." V0 implementation is narrower than the
    spec phrasing implies: classify() is still a pure function, and the
    turn engine doesn't yet populate `config["_prior_task_type"]`. So in
    practice V0 ALWAYS overrides to "analysis" on weak-signal + active_goal,
    unless a caller explicitly threads a prior task_type through config.

    True prior-turn-trace plumbing is V1 work; the substrate is wired for it
    via `config["_prior_task_type"]` (read here; written by no one in V0).

    Returns the (possibly overridden) task_type.

    Preconditions for override: classifier resolved to "conversation" via the
    default-branch fall-through (intent_tags == ("chat",) AND not a greeting
    AND not a vent). Anything stronger (greeting, vent, code/debug/plan/etc.)
    bypasses carry-forward — those ARE the strong intent signals the plan §4.A
    excludes.

    Override target: prefer `config["_prior_task_type"]` (V1 plumbing point),
    fallback to "analysis" (V0 default — continuation of mental work).
    """
    if _bearing_provider is None:
        return task_type
    if task_type != "conversation":
        return task_type
    if intent_tags != ("chat",):
        return task_type
    stripped = (text or "").strip()
    if not stripped:
        return task_type
    if _GREETING_RE.match(stripped):
        return task_type
    if _VENT_RE.search(stripped):
        return task_type
    try:
        active_goal = _bearing_provider.get_active_goal()
    except Exception:
        return task_type
    if not (isinstance(active_goal, str) and active_goal.strip()):
        return task_type
    prior = config.get("_prior_task_type") if isinstance(config, dict) else None
    if isinstance(prior, str) and prior in ("action", "analysis", "conversation"):
        return prior
    return "analysis"


def classify(messages: list[dict], config: dict | None = None) -> TurnShape:
    """Deterministic system-side classification of the current turn.

    Pure function: same `(messages, config)` always produces the same TurnShape,
    given a fixed `_bearing_provider` state.

    Bearing carry-forward (plan §4.A; V0 deviation in §13.3): when a
    BearingProvider is wired AND active_goal is non-empty AND user signal is
    weak, `task_type` is overridden from "conversation" to "analysis" (V0
    heuristic default) — unless `config["_prior_task_type"]` is set by the
    caller, in which case that wins. The V0 spec said "carry from prior turn's
    trace"; the V0 implementation is a heuristic-shaped narrowing of that. True
    prior-turn-trace plumbing is V1 work. See `_maybe_carry_forward` docstring.

    With the provider None (kill switch off or addon disabled), classifier is
    the same pure function it was before.
    """
    cfg = AdaptiveBudgetConfig.load()
    score, _features = compute_complexity_score(messages, config or {}, cfg)
    text = _last_user_text(messages)
    intent_tags, task_type, intent_conf = _classify_intent(text)
    # Bearing stakes boost: high-urgency or hard-reversibility situations
    # get a complexity floor so the effort tier can't drop below orient-level.
    if _bearing_provider is not None:
        try:
            stakes = _bearing_provider.get_stakes()
            if isinstance(stakes, dict):
                if stakes.get("urgency") == "high" or stakes.get("reversibility") == "hard":
                    score = max(score, 50)
        except Exception:
            pass
    tier = _score_to_tier(score)
    # Score-side confidence: extreme scores (clearly low or clearly high) are
    # more reliable than middling scores.
    score_conf = 1.0 if (score < 20 or score > 70) else 0.7
    confidence = round((intent_conf + score_conf) / 2.0, 2)
    task_type = _maybe_carry_forward(task_type, intent_tags, text, config or {})
    return TurnShape(
        effort_tier=tier,
        complexity_score=score,
        intent_tags=intent_tags,
        task_type=task_type,
        confidence=confidence,
    )
