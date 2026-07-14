"""Tool-loop continuation — detects silent loop drops and requests requeue.

Authority tier: DISPATCH. Kill switch: MONOLITH_PIPELINE_LOOP_CONT_V1.
Retry budget: 2 (scales with effort tier — 4 at ultimate, 1 at low — read
from ctx metadata at fire time; default 2 applies when no tier is present).

Subscribes to TurnStreamEndedEvent and ToolFailureClassifiedEvent.

Decision rules (TurnStreamEndedEvent):
  - had_tool_call=True AND had_continuation=False → emit
    FaultDetectedEvent(fault_kind="tool_no_fire", severity="warn") and,
    if retry budget remains, LoopContinuationRequestedEvent.
  - had_tool_call=False → nothing to do; loop closed cleanly.

ToolFailureClassifiedEvent:
  - classification="hard_failure" → emit LoopContinuationSuppressedEvent
    so the no-fire detector doesn't trigger a useless retry.

Independence: no engine/* or ACU imports. The actual requeue dispatch is
the kernel's job; this policy only emits the request event.
"""
from __future__ import annotations

from core.pipeline_registry import PolicyRegistration
from core.turn_pipeline_events import (
    AuthorityTier,
    FaultDetectedEvent,
    LoopContinuationRequestedEvent,
    LoopContinuationSuppressedEvent,
    PipelineEvent,
    ToolFailureClassifiedEvent,
    ToolFailureKind,
    TurnStreamEndedEvent,
)


NAME = "tool_loop_continuation"
KILL_SWITCH = "MONOLITH_PIPELINE_LOOP_CONT_V1"
DEFAULT_RETRY_BUDGET = 2


# Effort tier → retry budget. Read at fire time so the policy stays
# tier-aware without importing core/effort.py directly.
_TIER_BUDGET = {
    "low": 1,
    "default": 2,
    "high": 3,
    "ultimate": 4,
}


REGISTRATION = PolicyRegistration(
    name=NAME,
    module_path="core.pipeline_policies.tool_loop_continuation",
    subscribes_to=("TurnStreamEndedEvent", "ToolFailureClassifiedEvent"),
    depends_on=("tool_failure_classifier",),  # classifier must run first on same chain
    authority_tier=AuthorityTier.DISPATCH,
    kill_switch_env_flag=KILL_SWITCH,
    retry_budget=DEFAULT_RETRY_BUDGET,
)


def register_with(pipeline) -> None:
    pipeline.register(REGISTRATION, _handle)


def _handle(event: PipelineEvent, ctx) -> None:
    if isinstance(event, TurnStreamEndedEvent):
        _on_stream_ended(event, ctx)
    elif isinstance(event, ToolFailureClassifiedEvent):
        _on_failure_classified(event, ctx)


def _on_stream_ended(event: TurnStreamEndedEvent, ctx) -> None:
    if not event.had_tool_call:
        return
    if event.had_continuation:
        return
    if ctx.suppressed_continuation:
        # Hard failure already short-circuited this turn's loop.
        return

    # Fault first — record the symptom before deciding the response.
    from monokernel.turn_pipeline import get_pipeline
    pipeline = get_pipeline()
    pipeline.publish(
        FaultDetectedEvent(
            fault_kind="tool_no_fire",
            severity="warn",
            source_event_seq=event.seq,
            detail={"closed_lanes": list(event.closed_lanes)},
        ),
        ctx,
        source_kind="policy",
        source_name=NAME,
    )

    used = ctx.retry_budget_used.get(NAME, 0)
    budget = _effective_budget(ctx)
    if used >= budget:
        pipeline.publish(
            LoopContinuationSuppressedEvent(reason="budget_exhausted"),
            ctx,
            source_kind="policy",
            source_name=NAME,
        )
        return

    ctx.retry_budget_used[NAME] = used + 1
    ctx.requeue_count += 1
    pipeline.publish(
        LoopContinuationRequestedEvent(reason="silent_loop_drop", retry_count=used + 1),
        ctx,
        source_kind="policy",
        source_name=NAME,
    )


def _on_failure_classified(event: ToolFailureClassifiedEvent, ctx) -> None:
    if event.classification != ToolFailureKind.HARD_FAILURE.value:
        return
    ctx.suppressed_continuation = True
    from monokernel.turn_pipeline import get_pipeline
    get_pipeline().publish(
        LoopContinuationSuppressedEvent(reason="hard_failure"),
        ctx,
        source_kind="policy",
        source_name=NAME,
    )


def _effective_budget(ctx) -> int:
    """Read effort tier from ctx and pick the budget. Default if missing."""
    tier = getattr(ctx, "effort_tier", None)
    if isinstance(tier, str) and tier in _TIER_BUDGET:
        return _TIER_BUDGET[tier]
    return DEFAULT_RETRY_BUDGET
