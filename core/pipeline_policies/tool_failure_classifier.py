"""Tool-failure classifier — decides recoverable / hard / informational.

Authority tier: DISPATCH (the classifier's verdict drives suppression
behavior in tool_loop_continuation). Kill switch:
MONOLITH_PIPELINE_TOOL_CLASS_V1.

Subscribes to ToolFailedEvent. Emits ToolFailureClassifiedEvent and,
when classification=RECOVERABLE_WITH_HINT, HintInjectionRequestedEvent
so the loop_continuation policy can inject the hint on requeue.

Classification rules are declared as code in this module — not config,
not scattered constants — so they're auditable in one screen.

Independence: no engine/* or ACU coupling.
"""
from __future__ import annotations

from typing import Any

from core.pipeline_registry import PolicyRegistration
from core.turn_pipeline_events import (
    AuthorityTier,
    HintInjectionRequestedEvent,
    PipelineEvent,
    ToolFailedEvent,
    ToolFailureClassifiedEvent,
    ToolFailureKind,
)


NAME = "tool_failure_classifier"
KILL_SWITCH = "MONOLITH_PIPELINE_TOOL_CLASS_V1"


REGISTRATION = PolicyRegistration(
    name=NAME,
    module_path="core.pipeline_policies.tool_failure_classifier",
    subscribes_to=("ToolFailedEvent",),
    depends_on=(),
    authority_tier=AuthorityTier.DISPATCH,
    kill_switch_env_flag=KILL_SWITCH,
    retry_budget=None,  # classifier doesn't dispatch retries itself
)


def register_with(pipeline) -> None:
    pipeline.register(REGISTRATION, _handle)


# ── classification rules (declared inline; readable in one screen) ─


# Substrings on the error message → RECOVERABLE_WITH_HINT with a structured
# hint the model can act on next turn.
_RECOVERABLE_RULES: tuple[tuple[str, str], ...] = (
    ("missing required field",
     "The tool call is missing a required field. Re-emit the call with the "
     "field present and a value."),
    ("file not found",
     "The path you used does not exist. Use list_files or grep first to "
     "discover a valid path, then retry."),
    ("expected string",
     "An argument was the wrong type. Check the tool schema and re-emit "
     "with the correct types."),
    ("strict_mode_missing_close_tag",
     "Your previous emission looked like a missing closing </tool_call> "
     "tag. Re-emit a single complete <tool_call>...</tool_call> block."),
    ("invalid json",
     "The tool arguments were not valid JSON. Re-emit with strictly valid "
     "JSON (double-quoted keys, no trailing commas, escaped backslashes)."),
)


# Substrings → HARD_FAILURE: retry will not help, suppress the loop.
_HARD_RULES: tuple[str, ...] = (
    "sandbox violation",
    "permission denied",
    "operation not permitted",
    "out of memory",
    "killed by signal",
    "network is unreachable",
    "unknown tool",
)


# Substrings → INFORMATIONAL: tool succeeded technically, returned an empty
# or no-match result that the model should treat as a fact, not a retry
# trigger.
_INFORMATIONAL_RULES: tuple[str, ...] = (
    "(empty)",
    "no matches",
    "0 lines",
    "no results",
)


def _classify(error_text: str) -> tuple[str, str | None]:
    """Return (classification, hint). hint is non-None only for recoverable."""
    low = (error_text or "").lower()
    for marker, hint in _RECOVERABLE_RULES:
        if marker in low:
            return ToolFailureKind.RECOVERABLE_WITH_HINT.value, hint
    for marker in _HARD_RULES:
        if marker in low:
            return ToolFailureKind.HARD_FAILURE.value, None
    for marker in _INFORMATIONAL_RULES:
        if marker in low:
            return ToolFailureKind.INFORMATIONAL.value, None
    # Default: hard failure — conservative. Better to suppress a retry than
    # to burn a turn re-running a tool with no diagnostic added.
    return ToolFailureKind.HARD_FAILURE.value, None


def _handle(event: PipelineEvent, ctx) -> None:
    if not isinstance(event, ToolFailedEvent):
        return
    error_text = ""
    summary = event.envelope_summary or {}
    if isinstance(summary, dict):
        error_text = str(summary.get("error", "") or summary.get("message", "") or "")
    classification, hint = _classify(error_text)

    from monokernel.turn_pipeline import get_pipeline
    pipeline = get_pipeline()
    pipeline.publish(
        ToolFailureClassifiedEvent(
            call_id=event.call_id,
            classification=classification,
            hint=hint,
        ),
        ctx,
        source_kind="policy",
        source_name=NAME,
    )
    if classification == ToolFailureKind.RECOVERABLE_WITH_HINT.value and hint:
        pipeline.publish(
            HintInjectionRequestedEvent(call_id=event.call_id, hint_text=hint),
            ctx,
            source_kind="policy",
            source_name=NAME,
        )
