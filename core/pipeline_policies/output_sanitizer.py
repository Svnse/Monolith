"""Output sanitizer — in-stream answer-lane defense against render-breaking output.

Authority tier: MUTATION. Kill switch: MONOLITH_PIPELINE_SANITIZER_V1.

Two handlers:

  1. Live chunk guard — on TagRoutedEvent(lane="answer"). Detects internal-
     tag leaks (a <think>/<analysis>/<reasoning>/<acatalepsy>/<tool_call> tag
     that reached the answer lane, meaning the normalizer regressed) and
     emits FaultDetectedEvent(fault_kind="internal_tag_leak", severity="hard").
     Does NOT mutate the live chunk — defense-in-depth observation only at
     this stage; the normalizer is the primary line.

  2. Terminal balancer — on TurnReadyEvent. Mirrors the rules in
     ui/components/markdown_renderer.py:_balance_markdown (fence balance,
     single-backtick balance, ** balance) and emits OutputSanitizedEvent
     when a rule fires. The render-time _balance_markdown call is KEPT as
     a fallback (defense-in-depth — a regression here doesn't immediately
     ship visible corruption).

Independence: this policy does not import from engine/* or any ACU surface.
"""
from __future__ import annotations

import re
from typing import Any

from core.internal_tags import INTERNAL_LEAK_TAGS, make_leak_detection_pattern, strip_tag_blocks
from core.pipeline_registry import PolicyRegistration
from core.turn_pipeline_events import (
    AuthorityTier,
    FaultDetectedEvent,
    OutputSanitizedEvent,
    PipelineEvent,
    TagRoutedEvent,
    TurnReadyEvent,
)


NAME = "output_sanitizer"
KILL_SWITCH = "MONOLITH_PIPELINE_SANITIZER_V1"


# Single source of truth for the leak detector's tag set: core/internal_tags.
# response_verifier.py builds its detector from the same constant, so a new
# internal tag added there propagates here automatically.
_INTERNAL_TAG_RE = make_leak_detection_pattern(INTERNAL_LEAK_TAGS)

# Match a closed fenced code block — used to scrub fenced regions before
# counting inline markers so backticks inside a closed fence don't pollute.
_FENCE_RE = re.compile(r"```[^\n`]*\n.*?```\s*?", re.MULTILINE | re.DOTALL)

# Open-only internal-tag matcher, built from the same canonical set as the
# leak detector. Used by compute_terminal_correction to find an unterminated
# internal-open tag (leaked reasoning that ran to end-of-stream).
_INTERNAL_OPEN_RE = re.compile(
    r"<(?P<tag>" + "|".join(re.escape(t) for t in INTERNAL_LEAK_TAGS) + r")\b[^>]*>",
    re.IGNORECASE,
)

# Orphan closing-tag matcher (a stray ``</tag>`` with no surviving open).
_INTERNAL_CLOSE_RE = re.compile(
    r"</(?:" + "|".join(re.escape(t) for t in INTERNAL_LEAK_TAGS) + r")\s*>",
    re.IGNORECASE,
)


REGISTRATION = PolicyRegistration(
    name=NAME,
    module_path="core.pipeline_policies.output_sanitizer",
    subscribes_to=("TagRoutedEvent", "TurnReadyEvent"),
    depends_on=(),  # runs early; verifier_bridge depends on us via TurnReadyEvent
    authority_tier=AuthorityTier.MUTATION,
    kill_switch_env_flag=KILL_SWITCH,
)


def compute_terminal_correction(text: str) -> str | None:
    """Strip internal-tag leaks from the public answer.

    Returns the corrected text when a leak was removed, else None (no change).
    Conservative: removes paired ``<tag>...</tag>`` internal blocks for every
    tag in INTERNAL_LEAK_TAGS. Markdown imbalance is intentionally NOT corrected
    here — the render-time ``_balance_markdown`` already handles it; this organ
    fixes the live gap (leaked internal markup, which has no other remedy).

    Pure function — the testable core of the consumption seam (M1). The bus
    handler (``_check_terminal``) writes the result onto ``ctx.sanitized_text``;
    the finalize site applies it.
    """
    if not text:
        return None
    source = text
    # An unterminated internal-open tag = leaked reasoning that ran to the end
    # of the stream; drop from the tag onward.
    for open_m in _INTERNAL_OPEN_RE.finditer(text):
        tag = re.escape(open_m.group("tag"))
        close_re = re.compile(rf"</{tag}\s*>", re.IGNORECASE)
        if close_re.search(text, open_m.end()) is None:
            source = text[: open_m.start()].rstrip()
            break
    corrected = strip_tag_blocks(source, INTERNAL_LEAK_TAGS)
    # Any remaining closing tag is an orphan (its open was already stripped or
    # never existed) — drop the bare marker, keep the surrounding answer text.
    corrected = _INTERNAL_CLOSE_RE.sub("", corrected)
    if corrected == text:
        return None
    return corrected


def register_with(pipeline) -> None:
    pipeline.register(REGISTRATION, _handle)


def _handle(event: PipelineEvent, ctx) -> None:
    if isinstance(event, TagRoutedEvent):
        _check_live_chunk(event, ctx)
    elif isinstance(event, TurnReadyEvent):
        _check_terminal(event, ctx)


def _check_live_chunk(event: TagRoutedEvent, ctx) -> None:
    """Live chunk: detect internal-tag leaks into the answer lane."""
    if event.lane != "answer" or not event.delta_text:
        return
    if not _INTERNAL_TAG_RE.search(event.delta_text):
        return
    # The normalizer should never route an internal tag into the answer lane.
    # If we see one here, that's a regression — emit a hard fault.
    fault = FaultDetectedEvent(
        fault_kind="internal_tag_leak",
        severity="hard",
        source_event_seq=event.seq,
        detail={
            "lane": event.lane,
            "delta_preview": event.delta_text[:280],
            "tag_state": event.tag_state,
        },
    )
    _publish_via_kernel(fault, ctx)


def _check_terminal(event: TurnReadyEvent, ctx) -> None:
    """Terminal: run balance checks against the public answer."""
    public = event.public_answer or ""
    if not public:
        return

    # MUTATION: strip internal-tag leaks and write the corrected answer onto
    # ctx for the finalize site to re-commit. Markdown imbalance is left to the
    # render-time _balance_markdown; this organ fixes the leak-strip gap only.
    corrected = compute_terminal_correction(public)
    if corrected is not None:
        ctx.sanitized_text = corrected

    fires: list[str] = []

    # Triple-backtick fence balance.
    if public.count("```") % 2 == 1:
        fires.append("fence_imbalance")

    # Strip closed fences before counting inline markers.
    scrub = _FENCE_RE.sub("", public)
    if scrub.count("`") % 2 == 1:
        fires.append("single_backtick_imbalance")
    if scrub.count("**") % 2 == 1:
        fires.append("double_star_imbalance")

    # Internal-tag leak check on the public answer surface — defense in depth
    # against any path that built public_answer without going through the
    # normalizer.
    if _INTERNAL_TAG_RE.search(public):
        fires.append("internal_tag_leak_terminal")

    for rule in fires:
        out = OutputSanitizedEvent(
            lane="answer",
            before=public[:280],
            after=public[:280],  # Phase 2 reports only; mutation hooks into the
                                 # render path land in Phase 4 migration.
            rule_fired=rule,
        )
        _publish_via_kernel(out, ctx)
        ctx.mutation_count += 1
        if rule == "internal_tag_leak_terminal":
            fault = FaultDetectedEvent(
                fault_kind="internal_tag_leak",
                severity="hard",
                source_event_seq=event.seq,
                detail={"surface": "terminal", "preview": public[:280]},
            )
            _publish_via_kernel(fault, ctx)


def _publish_via_kernel(event: PipelineEvent, ctx) -> None:
    """Publish an event back onto the bus as this policy's source."""
    from monokernel.turn_pipeline import get_pipeline
    get_pipeline().publish(
        event,
        ctx,
        source_kind="policy",
        source_name=NAME,
    )
