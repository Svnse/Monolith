"""Single source of truth for Monolith's internal-tag taxonomy.

Three callsites used to inline their own near-identical regex over the
same conceptual tag set:

  * core/pipeline_policies/output_sanitizer.py:_INTERNAL_TAG_RE — flags a
    leak when an internal tag reaches the public answer lane.
  * core/response_verifier.py:_RAW_INTERNAL_TAG_RE — terminal verifier
    re-check on READY-time public answer (defense-in-depth).
  * engine/agent_server.py — six per-tag full-block strip regexes
    applied before exposing model output to external peers (MCP/HTTP).

Drift between those lists is how internal-only markup leaks to external
consumers. This module owns the names and the regex *shape* for each
purpose; callsites import and build, they don't re-inline. Adding a tag
once here flows everywhere downstream.

Two name sets — they overlap but are not identical, on purpose:

  INTERNAL_LEAK_TAGS: tags the in-band detectors flag as "leaked into
    the public answer." Tighter set — only includes tags that, by
    themselves, indicate a normalizer regression. tool_evidence/axes/
    intent are excluded because the verifier has tighter dedicated
    checks for those (e.g. _UNCLOSED_TOOL_EVIDENCE_RE in
    response_verifier).

  EXTERNAL_STRIP_TAGS: tags scrubbed before exposing model output to
    external peers. Broader set — strips anything the runtime produces
    that an outside caller shouldn't see, including internal-directive
    tags (monolith_cmd) that should never reach a peer even if they
    somehow escape internal handling.

Adding a tag: update the appropriate constant here, run the test
suite. Both detectors and the external stripper pick up the change
without per-site edits.
"""
from __future__ import annotations

import re


# ── canonical tag-name sets ─────────────────────────────────────────


# Tags the in-band leak detectors flag. Tight set; only tags that mean
# "the normalizer let internal markup into the public answer." If you
# add one here, both output_sanitizer (live + terminal) and
# response_verifier (terminal) pick it up.
INTERNAL_LEAK_TAGS: tuple[str, ...] = (
    "think",
    "analysis",
    "reasoning",
    "monolith_cmd",
    "tool_call",
    "curiosity",
    "acatalepsy",
)


# Tags the agent_server strips before exposing model output to MCP /
# HTTP peers. Broader: includes all detector tags + tags whose presence
# is benign internally but inappropriate externally (tool_evidence,
# axes, intent, monolith_cmd directive markup).
EXTERNAL_STRIP_TAGS: tuple[str, ...] = (
    "think",
    "analysis",
    "reasoning",
    "monolith_cmd",
    "tool_call",
    "curiosity",
    "tool_evidence",
    "acatalepsy",
    "axes",
    "intent",
    "bearing_update",
)


# ── regex builders ─────────────────────────────────────────────────


def make_leak_detection_pattern(tags: tuple[str, ...]) -> re.Pattern[str]:
    """Compile the open-or-close tag detector used by the leak detectors.

    Matches ``<tag`` or ``</tag`` at a word boundary, case-insensitive.
    The trailing ``\\b`` keeps ``<thinking>`` from spuriously matching
    ``think`` while still catching ``<think>`` and ``</think>``.

    Shape contract: both output_sanitizer and response_verifier rely on
    this exact pattern. Do not deviate per callsite — that's the drift
    this module exists to prevent.
    """
    if not tags:
        # Empty alternation would compile to invalid regex; return a
        # never-matches pattern instead.
        return re.compile(r"(?!x)x")
    alternation = "|".join(re.escape(t) for t in tags)
    return re.compile(rf"</?(?:{alternation})\b", flags=re.IGNORECASE)


def strip_tag_blocks(text: str, tags: tuple[str, ...]) -> str:
    """Remove ``<tag>...</tag>`` blocks (DOTALL, IGNORECASE) for each tag.

    Used by agent_server's _clean_agent_response. One pass per tag keeps
    the behavior identical to the prior per-tag .sub() chain so the
    extraction is observably the same.
    """
    if not text:
        return text
    out = text
    for tag in tags:
        pattern = re.compile(
            rf"<{re.escape(tag)}>.*?</{re.escape(tag)}>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        out = pattern.sub("", out)
    # Second pass: remove ORPHAN open/close tags left by malformed model output
    # (e.g. a bare "</think>" with no matching opener, or content emitted after a
    # stray closing tag). The balanced pass above misses these, which is how
    # raw "</think>" fragments leaked into peer replies. Strip the tag token
    # itself, preserving surrounding answer text.
    if tags:
        alternation = "|".join(re.escape(t) for t in tags)
        out = re.sub(rf"</?(?:{alternation})\b[^>]*>", "", out, flags=re.IGNORECASE)
    return out
