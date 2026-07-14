"""Turn Pipeline event taxonomy — the closed set of events published on the bus.

Every event has identity (turn_id, parent_turn_id, seq, emitted_at) and
provenance (source_kind, source_name). seq is assigned by the kernel at
publish() time; producers and policies do not set it.

Persistence: every published event is recorded to the fault_traces table
in turn_trace.sqlite3 by the kernel — that store is the single source of
truth for replay. Events serialize via to_payload() into payload_json.

Scope: live-turn observability and recovery only. No identity work, no
slow-lifecycle features, no ACU/acatalepsy coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AuthorityTier(str, Enum):
    """Declared authority a policy holds at registration."""
    OBSERVATION = "observation"  # may emit events; may not mutate stream or dispatch
    MUTATION = "mutation"        # may mutate stream content (e.g., sanitizer)
    DISPATCH = "dispatch"        # may force requeue, suppress retry, escalate


class ToolFailureKind(str, Enum):
    """Classification verdict from the tool_failure_classifier policy."""
    RECOVERABLE_WITH_HINT = "recoverable_with_hint"
    HARD_FAILURE = "hard_failure"
    INFORMATIONAL = "informational"


class ProducerKind(str, Enum):
    """Which producer adapter fed the stream into the pipeline."""
    LOCAL_LLM = "local_llm"
    REMOTE_CONNECT = "remote_connect"
    OTHER = "other"


@dataclass(frozen=True)
class PipelineEvent:
    """Base event. Concrete subclasses add payload fields.

    turn_id / parent_turn_id / seq / emitted_at / source_kind / source_name
    are stamped by the kernel at publish() time. Subclasses should not
    override them — they're declared here so type-checkers see them on every
    subtype.
    """
    turn_id: str = ""
    parent_turn_id: str | None = None
    seq: int = 0
    emitted_at: str = ""
    source_kind: str = ""  # "producer" | "policy" | "kernel"
    source_name: str = ""

    @property
    def kind(self) -> str:
        return type(self).__name__

    def payload_fields(self) -> dict[str, Any]:
        """Subclass-specific payload. Base returns empty; subclasses override."""
        return {}

    def to_payload(self) -> dict[str, Any]:
        """Serialize the event for fault_traces.payload_json."""
        base = {
            "turn_id": self.turn_id,
            "parent_turn_id": self.parent_turn_id,
            "seq": self.seq,
            "emitted_at": self.emitted_at,
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "kind": self.kind,
        }
        base.update(self.payload_fields())
        return base


# ── kernel-emitted lifecycle events ────────────────────────────────


@dataclass(frozen=True)
class TurnStreamStartedEvent(PipelineEvent):
    producer_kind: str = ProducerKind.OTHER.value

    def payload_fields(self) -> dict[str, Any]:
        return {"producer_kind": self.producer_kind}


@dataclass(frozen=True)
class TurnStreamEndedEvent(PipelineEvent):
    closed_lanes: tuple[str, ...] = ()
    had_tool_call: bool = False
    had_continuation: bool = False

    def payload_fields(self) -> dict[str, Any]:
        return {
            "closed_lanes": list(self.closed_lanes),
            "had_tool_call": self.had_tool_call,
            "had_continuation": self.had_continuation,
        }


@dataclass(frozen=True)
class TurnReadyEvent(PipelineEvent):
    raw_answer: str = ""
    public_answer: str = ""
    tools_used: tuple[str, ...] = ()
    # Distinguishes the first model response of an exchange ("initial")
    # from a continuation triggered by a tool follow-up ("tool_followup").
    # parent_turn_id alone doesn't distinguish — chat.py constructs fresh
    # TurnContext per emit without populating parent_turn_id, so every
    # event currently looks like a root-turn event. Consumers that want
    # to query "did the post-tool answer regress?" need this lane.
    # Default "initial" preserves current behavior for callers that don't
    # set it. Valid values: "initial" | "tool_followup".
    turn_phase: str = "initial"

    def payload_fields(self) -> dict[str, Any]:
        return {
            "raw_answer_chars": len(self.raw_answer),
            "public_answer_chars": len(self.public_answer),
            "tools_used": list(self.tools_used),
            "turn_phase": self.turn_phase,
        }


@dataclass(frozen=True)
class TurnCompleteEvent(PipelineEvent):
    outcome: str = "ok"  # "ok" | "faulted" | "cancelled"
    fault_count: int = 0
    mutation_count: int = 0
    requeue_count: int = 0
    duration_ms: float = 0.0

    def payload_fields(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "fault_count": self.fault_count,
            "mutation_count": self.mutation_count,
            "requeue_count": self.requeue_count,
            "duration_ms": round(float(self.duration_ms), 3),
        }


# ── stream events (producer + stream-parse policy) ────────────────


@dataclass(frozen=True)
class StreamChunkReceivedEvent(PipelineEvent):
    text: str = ""
    lane_hint: str | None = None

    def payload_fields(self) -> dict[str, Any]:
        return {"chars": len(self.text), "lane_hint": self.lane_hint}


@dataclass(frozen=True)
class TagRoutedEvent(PipelineEvent):
    lane: str = ""  # "answer" | "thinking" | "acu" | "tool_call" | "axes" | "intent" | "tool_evidence"
    delta_text: str = ""
    tag_state: str = ""

    def payload_fields(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "delta_chars": len(self.delta_text),
            "tag_state": self.tag_state,
        }


@dataclass(frozen=True)
class OutputSanitizedEvent(PipelineEvent):
    lane: str = ""
    before: str = ""
    after: str = ""
    rule_fired: str = ""

    def payload_fields(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "before_chars": len(self.before),
            "after_chars": len(self.after),
            "rule_fired": self.rule_fired,
        }


# ── tool events ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolCallParsedEvent(PipelineEvent):
    call_id: str = ""
    tool_name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def payload_fields(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "args": dict(self.payload),
        }


@dataclass(frozen=True)
class ToolParseFailedEvent(PipelineEvent):
    raw: str = ""
    error: str = ""
    attempt: int = 0

    def payload_fields(self) -> dict[str, Any]:
        return {
            "raw_preview": self.raw[:280],
            "error": self.error,
            "attempt": self.attempt,
        }


@dataclass(frozen=True)
class ToolExecutedEvent(PipelineEvent):
    call_id: str = ""
    tool_name: str = ""
    envelope_summary: dict[str, Any] = field(default_factory=dict)

    def payload_fields(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "envelope_summary": dict(self.envelope_summary),
        }


@dataclass(frozen=True)
class ToolFailedEvent(PipelineEvent):
    call_id: str = ""
    tool_name: str = ""
    envelope_summary: dict[str, Any] = field(default_factory=dict)
    classification: str | None = None  # filled later by classifier policy

    def payload_fields(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "envelope_summary": dict(self.envelope_summary),
            "classification": self.classification,
        }


@dataclass(frozen=True)
class ToolFailureClassifiedEvent(PipelineEvent):
    call_id: str = ""
    classification: str = ToolFailureKind.HARD_FAILURE.value
    hint: str | None = None

    def payload_fields(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "classification": self.classification,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class HintInjectionRequestedEvent(PipelineEvent):
    call_id: str = ""
    hint_text: str = ""

    def payload_fields(self) -> dict[str, Any]:
        return {"call_id": self.call_id, "hint_chars": len(self.hint_text)}


# ── loop continuation events (dispatch tier) ──────────────────────


@dataclass(frozen=True)
class LoopContinuationRequestedEvent(PipelineEvent):
    reason: str = ""
    retry_count: int = 0

    def payload_fields(self) -> dict[str, Any]:
        return {"reason": self.reason, "retry_count": self.retry_count}


@dataclass(frozen=True)
class LoopContinuationSuppressedEvent(PipelineEvent):
    reason: str = ""

    def payload_fields(self) -> dict[str, Any]:
        return {"reason": self.reason}


# ── verifier event ──────────────────────────────────────────────────


@dataclass(frozen=True)
class VerifierVerdictEvent(PipelineEvent):
    verdict: str = "pass"
    findings: tuple[dict[str, Any], ...] = ()

    def payload_fields(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "findings": list(self.findings)}


# ── identity-retry events (subordinate-clause detector) ──────────


@dataclass(frozen=True)
class IdentityRetryRequestedEvent(PipelineEvent):
    """Detector requests a retry with describe_self facts injected.

    Published by `subordinate_clause_detector` on a positive embedded-premise
    hit. The kernel consumes this event (separate ticket) to dispatch a retry
    with the original messages + describe_self() facts injected, the original
    output discarded. The retry context must contain facts the retry can act
    on — never verifier verdicts or grades on the prior turn (preserves the
    no-self-judgment protection from core/response_verifier.py:5-13).
    """
    fixture_hint: str = ""  # e.g. "embedded_premise"
    premise_trigger: str = ""  # detector pattern label, e.g. "adverbial_local_system"
    source_fact_keys: tuple[str, ...] = ()  # describe_self keys to inject on retry
    public_answer_chars: int = 0  # for trace correlation

    def payload_fields(self) -> dict[str, Any]:
        return {
            "fixture_hint": self.fixture_hint,
            "premise_trigger": self.premise_trigger,
            "source_fact_keys": list(self.source_fact_keys),
            "public_answer_chars": self.public_answer_chars,
        }


# ── fault event (single surface for Layer E recording) ────────────


@dataclass(frozen=True)
class FaultDetectedEvent(PipelineEvent):
    fault_kind: str = ""
    severity: str = "warn"  # "warn" | "hard"
    source_event_seq: int = -1
    detail: dict[str, Any] = field(default_factory=dict)

    def payload_fields(self) -> dict[str, Any]:
        return {
            "fault_kind": self.fault_kind,
            "severity": self.severity,
            "source_event_seq": self.source_event_seq,
            "detail": dict(self.detail),
        }


# ── helper: stamp identity onto a freshly-built event ─────────────


def stamp_event(
    event: PipelineEvent,
    *,
    turn_id: str,
    parent_turn_id: str | None,
    seq: int,
    emitted_at: str,
    source_kind: str,
    source_name: str,
) -> PipelineEvent:
    """Return a copy of *event* with kernel-assigned fields filled in.

    Frozen dataclasses can't mutate in place; the kernel calls this at
    publish() to stamp identity. Subclasses inherit fields from the base
    dataclass so dataclasses.replace works uniformly.
    """
    from dataclasses import replace
    return replace(
        event,
        turn_id=turn_id,
        parent_turn_id=parent_turn_id,
        seq=seq,
        emitted_at=emitted_at,
        source_kind=source_kind,
        source_name=source_name,
    )
