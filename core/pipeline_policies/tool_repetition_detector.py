"""Tool-repetition detector — breaks loops when the model re-emits the
same tool_call envelope consecutively within a turn.

Authority tier: DISPATCH. Kill switch: MONOLITH_PIPELINE_TOOL_REPEAT_V1.

Subscribes to ToolCallParsedEvent and TurnCompleteEvent.

Decision rule (ToolCallParsedEvent):
  - Hash (tool_name, args). When the new hash matches the previous parsed
    tool_call's hash in the same turn, increment the consecutive-repeat
    counter. On reaching REPEAT_THRESHOLD, emit FaultDetectedEvent
    (fault_kind="tool_call_repeated") + LoopContinuationSuppressedEvent
    (reason="tool_repetition") and flip ctx.suppressed_continuation.

  - A different hash resets the counter to 1 and clears the per-turn
    suppression flag so a single repeat-loop can be broken without
    silencing later legitimate calls in the same turn.

TurnCompleteEvent:
  - Drop the per-turn state so memory does not accumulate across turns.

Independence: no engine/* or ACU imports. State lives in a module-level
dict keyed by turn_id and is reaped on TurnCompleteEvent.
"""
from __future__ import annotations

import hashlib
import json

from core.pipeline_registry import PolicyRegistration
from core.turn_pipeline_events import (
    AuthorityTier,
    FaultDetectedEvent,
    LoopContinuationSuppressedEvent,
    PipelineEvent,
    ToolCallParsedEvent,
    TurnCompleteEvent,
)


NAME = "tool_repetition_detector"
KILL_SWITCH = "MONOLITH_PIPELINE_TOOL_REPEAT_V1"
REPEAT_THRESHOLD = 2  # 2 consecutive identical calls → suppress


REGISTRATION = PolicyRegistration(
    name=NAME,
    module_path="core.pipeline_policies.tool_repetition_detector",
    subscribes_to=("ToolCallParsedEvent", "TurnCompleteEvent"),
    depends_on=(),
    authority_tier=AuthorityTier.DISPATCH,
    kill_switch_env_flag=KILL_SWITCH,
    retry_budget=None,
)


def register_with(pipeline) -> None:
    pipeline.register(REGISTRATION, _handle)


# Per-turn state: turn_id → (last_hash, consecutive_count, suppressed_already).
# Reaped by the TurnCompleteEvent handler so memory does not leak.
_TURN_STATE: dict[str, tuple[str, int, bool]] = {}


def _hash_call(tool_name: str, args: dict) -> str:
    try:
        payload = json.dumps(args or {}, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        payload = repr(args)
    return hashlib.sha1(f"{tool_name}|{payload}".encode("utf-8")).hexdigest()[:16]


def _handle(event: PipelineEvent, ctx) -> None:
    if isinstance(event, ToolCallParsedEvent):
        _on_parsed(event, ctx)
    elif isinstance(event, TurnCompleteEvent):
        _TURN_STATE.pop(ctx.turn_id, None)


def _on_parsed(event: ToolCallParsedEvent, ctx) -> None:
    new_hash = _hash_call(event.tool_name, event.payload or {})
    last_hash, count, suppressed = _TURN_STATE.get(ctx.turn_id, ("", 0, False))

    if new_hash == last_hash:
        count += 1
    else:
        count = 1
        suppressed = False

    if suppressed:
        _TURN_STATE[ctx.turn_id] = (new_hash, count, True)
        return

    if count < REPEAT_THRESHOLD:
        _TURN_STATE[ctx.turn_id] = (new_hash, count, False)
        return

    from monokernel.turn_pipeline import get_pipeline
    pipeline = get_pipeline()
    pipeline.publish(
        FaultDetectedEvent(
            fault_kind="tool_call_repeated",
            severity="warn",
            source_event_seq=event.seq,
            detail={
                "tool_name": event.tool_name,
                "consecutive_count": count,
                "call_hash": new_hash,
            },
        ),
        ctx,
        source_kind="policy",
        source_name=NAME,
    )
    ctx.suppressed_continuation = True
    pipeline.publish(
        LoopContinuationSuppressedEvent(reason="tool_repetition"),
        ctx,
        source_kind="policy",
        source_name=NAME,
    )
    _TURN_STATE[ctx.turn_id] = (new_hash, count, True)
