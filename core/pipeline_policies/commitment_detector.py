"""Commitment detector — catches stated intent without corresponding action.

Authority tier: OBSERVATION. Kill switch: MONOLITH_PIPELINE_COMMITMENT_V1.

Subscribes to TurnReadyEvent. Scans the model's output for commitment
patterns (stated intent to perform an action) and cross-references against
actual tool calls present in the response. Emits FaultDetectedEvent for
each unfulfilled commitment.

Generalizes the existing detect_tool_no_fire detector (which only catches
tool-intent language) to cover:
  - File write commitments: "Save this as X", "I'll write to X"
  - Verification commitments: "I'll check/verify/confirm X"
  - Creation commitments: "Let me create X", "I'll build X"

Unlike detect_tool_no_fire (post-hoc detector in core/fault_detectors.py),
this runs as a pipeline policy with access to the full turn context and
structured event recording.
"""
from __future__ import annotations

import re

from core.pipeline_registry import PolicyRegistration
from core.turn_pipeline_events import (
    AuthorityTier,
    FaultDetectedEvent,
    PipelineEvent,
    TurnReadyEvent,
)


NAME = "commitment_detector"
KILL_SWITCH = "MONOLITH_PIPELINE_COMMITMENT_V1"

# Commitment patterns: (compiled_regex, commitment_kind, description)
# Each pattern fires independently. A response can have multiple commitments.
_COMMITMENTS: list[tuple[re.Pattern, str, str]] = [
    # File write/save commitments
    (
        re.compile(
            r"[Ss]ave (?:this|it|the \w+) (?:as|to|in) [`'\"]?(?:prompts/|skills/|docs/|\w+\.(?:md|py|json|txt))",
            re.IGNORECASE,
        ),
        "file_save",
        "Stated intent to save a file",
    ),
    (
        re.compile(
            r"(?:I'll|I will|Let me) (?:write|save|create|add) (?:it |this |the \w+ )?(?:to |as |at )?[`'\"]?\w+[/.]",
            re.IGNORECASE,
        ),
        "file_write",
        "Stated intent to write/create a file",
    ),
    # Verification commitments
    (
        re.compile(
            r"(?:I'll|I will|Let me|I need to) (?:check|verify|confirm|validate|test|run)\b",
            re.IGNORECASE,
        ),
        "verification",
        "Stated intent to verify/check something",
    ),
    # Read/inspect commitments
    (
        re.compile(
            r"(?:I'll|I will|Let me|I need to) (?:read|inspect|look at|examine|scan)\b",
            re.IGNORECASE,
        ),
        "inspection",
        "Stated intent to read/inspect something",
    ),
]

# Tool call presence — if ANY tool_call block exists, tool-based commitments
# are considered potentially fulfilled (the specific match is hard to verify
# without parsing the tool call args).
_TOOL_CALL_RE = re.compile(r"<tool_call\b")

# Write-specific tool names that fulfill file_save/file_write commitments
_WRITE_TOOL_RE = re.compile(r'"(?:tool|name)"\s*:\s*"(?:write_file|edit_file|save_note)"')


REGISTRATION = PolicyRegistration(
    name=NAME,
    module_path="core.pipeline_policies.commitment_detector",
    subscribes_to=("TurnReadyEvent",),
    depends_on=(),
    authority_tier=AuthorityTier.OBSERVATION,
    kill_switch_env_flag=KILL_SWITCH,
)


def register_with(pipeline) -> None:
    pipeline.register(REGISTRATION, _handle)


def _handle(event: PipelineEvent, ctx) -> None:
    if isinstance(event, TurnReadyEvent):
        _on_turn_ready(event, ctx)


def _on_turn_ready(event: TurnReadyEvent, ctx) -> None:
    text = event.raw_answer or ""
    if not text:
        return

    has_any_tool_call = bool(_TOOL_CALL_RE.search(text))
    has_write_tool = bool(_WRITE_TOOL_RE.search(text))

    for pattern, kind, description in _COMMITMENTS:
        match = pattern.search(text)
        if match is None:
            continue

        # Check if the commitment was fulfilled
        fulfilled = False
        if kind in ("file_save", "file_write"):
            fulfilled = has_write_tool
        elif kind in ("verification", "inspection"):
            fulfilled = has_any_tool_call

        if fulfilled:
            continue

        # Unfulfilled commitment — emit fault
        evidence = text[max(0, match.start() - 30):match.end() + 50].strip()
        from monokernel.turn_pipeline import get_pipeline
        get_pipeline().publish(
            FaultDetectedEvent(
                fault_kind=f"commitment_unfulfilled:{kind}",
                severity="warn",
                source_event_seq=event.seq,
                detail={
                    "commitment_kind": kind,
                    "description": description,
                    "evidence": evidence,
                    "had_any_tool_call": has_any_tool_call,
                },
            ),
            ctx,
            source_kind="policy",
            source_name=NAME,
        )
