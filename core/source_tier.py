"""Source-Tier Gate — verifiability classification of a turn's output.

Three tiers (ordering IS the trust guard, lower == more trusted), mirroring
monosearch.record.EvidenceTier's IntEnum convention:

    TOOL           externally checkable — derived from a tool call this turn
    FAITHFUL_TRACE a reasoning trace that PASSED the Stage-2 withholding test
    GENERATION     unverified — the floor

Stage 1a mints ONLY TOOL and GENERATION. FAITHFUL_TRACE is reserved for the
Stage 2 withholding promoter: no cheap signal proves a trace faithful (a
tool-grounded trace already rolls up to TOOL), so an untested <think> trace
stays at the GENERATION floor. See the spec §0 decisions 2 and 7
(docs/superpowers/specs/2026-06-07-source-tier-gate-design.md).

This module is PURE — classify_source_tiers has no I/O and no flag gate. The
MONOLITH_SOURCE_TIER_V1 flag gates the SIDE EFFECTS (persistence + surfacing)
at the wiring sites, not the classification itself.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import IntEnum

from core.internal_tags import INTERNAL_LEAK_TAGS

_FLAG_ENV = "MONOLITH_SOURCE_TIER_V1"


def source_tier_enabled() -> bool:
    """Feature flag. Default OFF (dark) — flag-off path is byte-identical."""
    return str(os.environ.get(_FLAG_ENV, "0")).strip().lower() in {"1", "true", "yes", "on"}


class SourceTier(IntEnum):
    TOOL = 0
    FAITHFUL_TRACE = 1
    GENERATION = 2


_LABELS = {
    SourceTier.TOOL: "tool",
    SourceTier.FAITHFUL_TRACE: "faithful-trace",
    SourceTier.GENERATION: "generation",
}


def tier_label(tier: SourceTier) -> str:
    return _LABELS[tier]


# Partition the canonical internal-tag set (single source of truth in
# core.internal_tags, shared with output_sanitizer + response_verifier) into
# tool vs reasoning-trace tags. A new internal tag added there propagates here.
_TOOL_TAGS = tuple(t for t in INTERNAL_LEAK_TAGS if "tool" in t.lower())
_TRACE_TAGS = tuple(t for t in INTERNAL_LEAK_TAGS if "tool" not in t.lower())


def _tag_present(text: str, tags: tuple[str, ...]) -> bool:
    if not text or not tags:
        return False
    pat = re.compile(r"<(?:" + "|".join(re.escape(t) for t in tags) + r")\b", re.IGNORECASE)
    return pat.search(text) is not None


@dataclass(frozen=True)
class SourceTierResult:
    answer_tier: SourceTier
    region_tiers: dict[str, str]
    had_tool: bool
    had_trace: bool


def classify_source_tiers(
    raw: str, public: str = "", tools_used: tuple[str, ...] = ()
) -> SourceTierResult:
    """Classify a turn's output regions. Pure; deterministic; no LLM.

    raw:        the full assistant turn text WITH internal tags intact.
    public:     the post-normalization public answer (accepted for forward
                compat / Stage 2; unused in the V0 rollup).
    tools_used: tool names the turn invoked (may be empty at the chat finalize
                call site — had_tool then falls back to the <tool_call> tag).
    """
    raw = raw or ""
    had_tool = bool(tools_used) or _tag_present(raw, _TOOL_TAGS)
    had_trace = _tag_present(raw, _TRACE_TAGS)

    # V0 (Stage 1a): TOOL if externally grounded, else the GENERATION floor.
    answer_tier = SourceTier.TOOL if had_tool else SourceTier.GENERATION

    region_tiers: dict[str, str] = {"answer": _LABELS[answer_tier]}
    if had_tool:
        region_tiers["tool"] = _LABELS[SourceTier.TOOL]
    if had_trace:
        # Untested in V0 → floored. Stage 2 promotes to faithful-trace.
        region_tiers["trace"] = _LABELS[SourceTier.GENERATION]

    return SourceTierResult(
        answer_tier=answer_tier,
        region_tiers=region_tiers,
        had_tool=had_tool,
        had_trace=had_trace,
    )
