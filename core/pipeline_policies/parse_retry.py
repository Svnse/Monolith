"""Parse retry — bounded retry policy for malformed tool_call blocks.

Authority tier: DISPATCH. Kill switch: MONOLITH_PIPELINE_PARSE_RETRY_V1.
Retry budget: 2 per turn.

Subscribes to ToolParseFailedEvent. On the first or second attempt, emits
LoopContinuationRequestedEvent(reason="parse_retry") so the kernel re-
invokes the producer with a parse-retry hint in context. After two
attempts, emits LoopContinuationSuppressedEvent(reason="parse_budget_exhausted")
and FaultDetectedEvent(fault_kind="tool_parse_unrecoverable").

The actual hint text comes from the existing helper at
ui/pages/chat.py:_build_parse_retry_prompt (kept in place; this policy
observes the failure and decides retry, but the prompt construction stays
where it is for Phase 2 — wiring the hint into the producer's next call
is part of Phase 4 migration).

Independence: no engine/* or ACU coupling.
"""
from __future__ import annotations

from core.pipeline_registry import PolicyRegistration
from core.turn_pipeline_events import (
    AuthorityTier,
    FaultDetectedEvent,
    LoopContinuationRequestedEvent,
    LoopContinuationSuppressedEvent,
    PipelineEvent,
    ToolParseFailedEvent,
)


NAME = "parse_retry"
KILL_SWITCH = "MONOLITH_PIPELINE_PARSE_RETRY_V1"
RETRY_BUDGET = 2


REGISTRATION = PolicyRegistration(
    name=NAME,
    module_path="core.pipeline_policies.parse_retry",
    subscribes_to=("ToolParseFailedEvent",),
    depends_on=(),
    authority_tier=AuthorityTier.DISPATCH,
    kill_switch_env_flag=KILL_SWITCH,
    retry_budget=RETRY_BUDGET,
)


def register_with(pipeline) -> None:
    pipeline.register(REGISTRATION, _handle)


def _handle(event: PipelineEvent, ctx) -> None:
    if not isinstance(event, ToolParseFailedEvent):
        return
    from monokernel.turn_pipeline import get_pipeline
    pipeline = get_pipeline()

    used = ctx.retry_budget_used.get(NAME, 0)
    if used >= RETRY_BUDGET:
        pipeline.publish(
            LoopContinuationSuppressedEvent(reason="parse_budget_exhausted"),
            ctx,
            source_kind="policy",
            source_name=NAME,
        )
        pipeline.publish(
            FaultDetectedEvent(
                fault_kind="tool_parse_unrecoverable",
                severity="hard",
                source_event_seq=event.seq,
                detail={"error": event.error, "attempts": used},
            ),
            ctx,
            source_kind="policy",
            source_name=NAME,
        )
        ctx.suppressed_continuation = True
        return

    ctx.retry_budget_used[NAME] = used + 1
    ctx.requeue_count += 1
    pipeline.publish(
        LoopContinuationRequestedEvent(
            reason="parse_retry",
            retry_count=used + 1,
        ),
        ctx,
        source_kind="policy",
        source_name=NAME,
    )
