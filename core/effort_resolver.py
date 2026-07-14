"""Effort resolver — M2 of the execution-plane sync.

The classifier (core/turn_classifier) emits an ``effort_tier`` every turn but
nothing consumes it. This module resolves the effective tier and maps it to the
behavioral lever the 2026-05-09 smart-spec reserved for ``/effort``: the backend
reasoning mode (``enable_thinking``), which is pure decode-side (KV-stable) and
produces a real behavioral delta — unlike a ``max_tokens`` cap, which can only
truncate or no-op.

Non-performative: this shapes *how hard the model thinks* via a backend mode,
silently. Nothing is injected into the prompt; the model never narrates effort.

Monotonic-upgrade discipline (see ``resolve_thinking``): effort only ever ADDS
thinking for hard turns. An explicit manual "thinking on" always wins and is
never turned off by effort.

Tiers (from core/turn_classifier): low < med < high < xhigh < ultimate.
"""
from __future__ import annotations

import os


_FLAG_ENV = "MONOLITH_EFFORT_V1"


def effort_governance_enabled() -> bool:
    """Whether effort governs generation this session. Default ON.

    Kill switch for all of M2. When off, ``engine/llm.py`` skips the resolve +
    thinking-upgrade entirely, restoring prior behavior (thinking = manual
    toggle only; ``_resolved_effort_tier`` unwritten).
    """
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


# Tiers at/above which a turn is "hard enough" to warrant backend reasoning
# mode. low/med stay fast; high+ think. med is the default tier, so the default
# turn does NOT pay for thinking unless the user asked for it.
_THINKING_TIERS: frozenset[str] = frozenset({"high", "xhigh", "ultimate"})

# Canonical tier set (mirrors core/turn_classifier._TIER_BOUNDARIES).
_VALID_TIERS: frozenset[str] = frozenset({"low", "med", "high", "xhigh", "ultimate"})
_DEFAULT_TIER = "med"


def resolve_effort_tier(config: object) -> str:
    """Resolve the effective effort tier for this turn and stamp it on *config*.

    V0 precedence: classifier suggestion (``config["_turn_shape"].effort_tier``,
    object or dict) > ``"med"`` default. A future manual ``/effort`` surface
    (``world_state`` baseline/once) layers in ABOVE the classifier when that
    command exists — its plumbing (``world_state.consume_effort_tier``) is built
    but has no command yet, so it is intentionally not consulted in V0.

    Writes ``config["_resolved_effort_tier"]`` — the key engine/llm.py already
    reads (previously only for telemetry; now load-bearing). Unknown/garbage
    tiers fall back to the default. Never raises.
    """
    tier = _DEFAULT_TIER
    if isinstance(config, dict):
        shape = config.get("_turn_shape")
        raw = None
        if isinstance(shape, dict):
            raw = shape.get("effort_tier")
        elif shape is not None:
            raw = getattr(shape, "effort_tier", None)
        if isinstance(raw, str) and raw in _VALID_TIERS:
            tier = raw
        config["_resolved_effort_tier"] = tier
    return tier


def effort_enables_thinking(tier: str) -> bool:
    """Whether *tier* is deliberate enough to engage backend reasoning mode."""
    return tier in _THINKING_TIERS


def resolve_thinking(*, manual: bool | None, tier: str) -> bool | None:
    """Resolve backend reasoning mode from the manual toggle + effort tier.

    Monotonic upgrade:
      - an explicit manual ``True`` always wins (user intent is never overridden);
      - else effort engages thinking for hard tiers (high/xhigh/ultimate);
      - else the manual value passes through unchanged — effort NEVER forces
        thinking off. ``None`` ("not set") passes through as ``None`` so the
        engine keeps its backend default.
    """
    if manual is True:
        return True
    if effort_enables_thinking(tier):
        return True
    return manual
