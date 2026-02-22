"""
Execution Contract + Agent Outcome — Phase 3 of Monolith Agent Contract V2.

ExecutionContract is immutable after first inference (Invariant B).
AgentOutcome replaces boolean completion with typed terminal states.
RuntimeState defines the authoritative FSM states.

ContractFactory produces contracts from page context + prompt analysis.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# AgentOutcome — typed terminal states (replaces bool completed)
# ---------------------------------------------------------------------------

class AgentOutcome(str, Enum):
    """All possible terminal states for an agent run. Disjoint and exhaustive."""
    COMPLETED_WITH_TOOLS = "COMPLETED_WITH_TOOLS"
    COMPLETED_CHAT_ONLY = "COMPLETED_CHAT_ONLY"
    FAILED_PREFLIGHT = "FAILED_PREFLIGHT"
    FAILED_PROTOCOL_NO_TOOLS = "FAILED_PROTOCOL_NO_TOOLS"
    FAILED_PROTOCOL_MALFORMED = "FAILED_PROTOCOL_MALFORMED"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    FAILED_BUDGET_EXHAUSTED = "FAILED_BUDGET_EXHAUSTED"
    FAILED_TIMEOUT = "FAILED_TIMEOUT"
    FAILED_CONTRACT_VIOLATION = "FAILED_CONTRACT_VIOLATION"
    INTERRUPTED = "INTERRUPTED"

    @property
    def is_success(self) -> bool:
        return self in (AgentOutcome.COMPLETED_WITH_TOOLS, AgentOutcome.COMPLETED_CHAT_ONLY)

    @property
    def is_failure(self) -> bool:
        return not self.is_success and self != AgentOutcome.INTERRUPTED


# ---------------------------------------------------------------------------
# Tool policy
# ---------------------------------------------------------------------------

class ToolPolicy(str, Enum):
    REQUIRED = "required"       # success requires >= 1 validated tool call
    OPTIONAL = "optional"       # tools may or may not be used; both are valid
    FORBIDDEN = "forbidden"     # tool calls are rejected


# ---------------------------------------------------------------------------
# RuntimeState — authoritative FSM states (Phase 3)
# ---------------------------------------------------------------------------

class RuntimeState(str, Enum):
    """
    Authoritative FSM states for the agent execution loop.

    OFAC v0.2 Final Form — 8 states:
      PRECHECK → INFER → VALIDATE_CALLS → WAIT_ACK → EXECUTE → OBSERVE → COMMIT → TERMINATE
    """
    PRECHECK = "PRECHECK"
    INFER = "INFER"
    VALIDATE_CALLS = "VALIDATE_CALLS"
    WAIT_ACK = "WAIT_ACK"
    EXECUTE = "EXECUTE"
    OBSERVE = "OBSERVE"
    COMMIT = "COMMIT"
    TERMINATE = "TERMINATE"


FSM_TRANSITIONS: dict[RuntimeState, frozenset[RuntimeState]] = {
    RuntimeState.PRECHECK: frozenset({RuntimeState.INFER, RuntimeState.TERMINATE}),
    RuntimeState.INFER: frozenset({RuntimeState.VALIDATE_CALLS, RuntimeState.COMMIT, RuntimeState.TERMINATE}),
    RuntimeState.VALIDATE_CALLS: frozenset({RuntimeState.WAIT_ACK, RuntimeState.EXECUTE, RuntimeState.TERMINATE}),
    RuntimeState.WAIT_ACK: frozenset({RuntimeState.EXECUTE, RuntimeState.TERMINATE}),
    RuntimeState.EXECUTE: frozenset({RuntimeState.OBSERVE, RuntimeState.TERMINATE}),
    RuntimeState.OBSERVE: frozenset({RuntimeState.COMMIT}),
    RuntimeState.COMMIT: frozenset({RuntimeState.INFER, RuntimeState.TERMINATE}),
    RuntimeState.TERMINATE: frozenset(),  # terminal, no outgoing transitions
}


# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextBudget:
    context_window: int = 8192
    reserved_system: int = 512
    reserved_synthesis: int = 256
    force_synthesis_at_ratio: float = 0.85


# ---------------------------------------------------------------------------
# Tool output budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolOutputBudget:
    max_bytes_per_call: int = 32768
    truncation_marker: str = "[TRUNCATED]"


# ---------------------------------------------------------------------------
# ExecutionContract — immutable after first inference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionContract:
    """
    Immutable execution contract for a single agent run.

    Created by ContractFactory before runtime loop begins.
    Cannot be mutated after first inference (Invariant B).
    """
    # identity
    contract_id: str = ""
    contract_hash: str = ""
    parent_contract_hash: str | None = None

    # policy
    tool_policy: ToolPolicy = ToolPolicy.OPTIONAL
    allowed_tools: tuple[str, ...] | None = None
    strict_mode: bool = False

    # budgets
    max_inferences: int = 25
    max_tokens_consumed: int = 0          # 0 = unlimited
    max_format_retries: int = 1
    step_timeout_ms: int = 30000
    total_timeout_ms: int = 300000

    # context budget
    context_budget: ContextBudget = field(default_factory=ContextBudget)

    # tool output budget
    tool_output_budget: ToolOutputBudget = field(default_factory=ToolOutputBudget)

    # boundary/runtime config
    adapter_version: str = "2a.0"
    model_profile_id: str = "local_xml"
    token_gate: bool = False

    # anti-cycle controls
    cycle_forbid: tuple[tuple[str, str], ...] = ()

    # Phase 5 — Open Foundation
    contract_format_version: str = "3.0"
    model_fingerprint: str = ""
    grammar_profile: str | None = None

    # provenance
    source_page: str = ""                 # "chat" | "code"
    creation_timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "parent_contract_hash": self.parent_contract_hash,
            "tool_policy": self.tool_policy.value,
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools else None,
            "strict_mode": self.strict_mode,
            "max_inferences": self.max_inferences,
            "max_tokens_consumed": self.max_tokens_consumed,
            "max_format_retries": self.max_format_retries,
            "step_timeout_ms": self.step_timeout_ms,
            "total_timeout_ms": self.total_timeout_ms,
            "context_budget": {
                "context_window": self.context_budget.context_window,
                "reserved_system": self.context_budget.reserved_system,
                "reserved_synthesis": self.context_budget.reserved_synthesis,
                "force_synthesis_at_ratio": self.context_budget.force_synthesis_at_ratio,
            },
            "tool_output_budget": {
                "max_bytes_per_call": self.tool_output_budget.max_bytes_per_call,
                "truncation_marker": self.tool_output_budget.truncation_marker,
            },
            "adapter_version": self.adapter_version,
            "model_profile_id": self.model_profile_id,
            "token_gate": self.token_gate,
            "cycle_forbid": [list(pair) for pair in self.cycle_forbid],
            "contract_format_version": self.contract_format_version,
            "model_fingerprint": self.model_fingerprint,
            "grammar_profile": self.grammar_profile,
            "source_page": self.source_page,
            "creation_timestamp": self.creation_timestamp,
        }


# ---------------------------------------------------------------------------
# AgentRunResult — full result payload from runtime
# ---------------------------------------------------------------------------

@dataclass
class AgentRunResult:
    """Complete result from an agent run, replacing the (bool, str, list) tuple."""
    outcome: AgentOutcome
    output: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    contract: ExecutionContract | None = None
    steps_used: int = 0
    inferences_used: int = 0
    tools_executed: int = 0
    format_retries_used: int = 0
    tokens_consumed: int = 0
    termination_reason: str = ""

    @property
    def success(self) -> bool:
        return self.outcome.is_success


# ---------------------------------------------------------------------------
# RunSummary — frozen telemetry receipt emitted once at termination
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSummary:
    """
    Sealed execution receipt. Produced by runtime, consumed by governance.

    Contains only factual execution metrics. No scoring, no interpretation,
    no LLM calls. Immutable after creation.
    """
    # identity
    contract_id: str = ""
    run_id: str = ""

    # outcome
    termination_reason: str = ""
    outcome: str = ""                    # AgentOutcome.value

    # counters
    llm_calls: int = 0
    tool_calls: int = 0
    format_retries: int = 0
    steps_used: int = 0                  # total STEP_START events

    # budget
    max_inferences: int = 0
    budget_remaining: int = 0
    tokens_consumed: int = 0

    # timing
    elapsed_ms: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    # error flags
    had_protocol_error: bool = False     # any adapter rejected/recovered
    had_validation_error: bool = False   # any tool validation failure
    had_cycle_violation: bool = False    # cycle_forbid triggered
    had_budget_exhaustion: bool = False  # max_inferences hit

    # model profile
    model_profile_id: str = ""

    # duplicate detection (Phase 4)
    unique_tool_signatures: int = 0
    total_tool_invocations: int = 0

    # transcript chain (Phase 5)
    transcript_chain_head: str = ""
    transcript_chain_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "run_id": self.run_id,
            "termination_reason": self.termination_reason,
            "outcome": self.outcome,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "format_retries": self.format_retries,
            "steps_used": self.steps_used,
            "max_inferences": self.max_inferences,
            "budget_remaining": self.budget_remaining,
            "tokens_consumed": self.tokens_consumed,
            "elapsed_ms": self.elapsed_ms,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "had_protocol_error": self.had_protocol_error,
            "had_validation_error": self.had_validation_error,
            "had_cycle_violation": self.had_cycle_violation,
            "had_budget_exhaustion": self.had_budget_exhaustion,
            "model_profile_id": self.model_profile_id,
            "unique_tool_signatures": self.unique_tool_signatures,
            "total_tool_invocations": self.total_tool_invocations,
            "transcript_chain_head": self.transcript_chain_head,
            "transcript_chain_length": self.transcript_chain_length,
        }


# ---------------------------------------------------------------------------
# PerfVector — multi-dimensional run evaluation (Phase 4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerfVector:
    """
    Multi-dimensional evaluation of a single agent run.

    Computed deterministically from RunSummary fields. No LLM calls,
    no randomness. Frozen after creation.
    """
    run_id: str = ""
    protocol_compliant: bool = True
    retry_count: int = 0
    budget_efficiency: float = 1.0       # inferences_used / max_inferences (lower = more efficient)
    duplicate_call_ratio: float = 0.0    # 0.0 = no duplicates, 1.0 = all duplicates
    anomaly_ignored_count: int = 0
    hard_failure: bool = False
    dominance: str = "GREEN"             # GREEN | YELLOW | RED
    composite_score: float = 1.0         # weighted scalar 0.0–1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "protocol_compliant": self.protocol_compliant,
            "retry_count": self.retry_count,
            "budget_efficiency": self.budget_efficiency,
            "duplicate_call_ratio": self.duplicate_call_ratio,
            "anomaly_ignored_count": self.anomaly_ignored_count,
            "hard_failure": self.hard_failure,
            "dominance": self.dominance,
            "composite_score": self.composite_score,
        }


# ---------------------------------------------------------------------------
# StateDigest — runtime snapshot for UI consumption (Phase 3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateDigest:
    """Snapshot of runtime state emitted on every FSM transition."""
    fsm_state: str = ""
    inferences_used: int = 0
    inferences_remaining: int = 0
    tokens_consumed: int = 0
    tokens_remaining: int = 0
    tools_executed: int = 0
    format_retries_used: int = 0
    elapsed_ms: float = 0.0
    timeout_remaining_ms: float = 0.0
    context_ratio: float = 0.0
    force_synthesis_pending: bool = False
    last_outcome: str | None = None
    contract_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fsm_state": self.fsm_state,
            "inferences_used": self.inferences_used,
            "inferences_remaining": self.inferences_remaining,
            "tokens_consumed": self.tokens_consumed,
            "tokens_remaining": self.tokens_remaining,
            "tools_executed": self.tools_executed,
            "format_retries_used": self.format_retries_used,
            "elapsed_ms": self.elapsed_ms,
            "timeout_remaining_ms": self.timeout_remaining_ms,
            "context_ratio": self.context_ratio,
            "force_synthesis_pending": self.force_synthesis_pending,
            "last_outcome": self.last_outcome,
            "contract_id": self.contract_id,
        }


# ---------------------------------------------------------------------------
# Intent classification heuristics (for ContractFactory)
# ---------------------------------------------------------------------------

# Lazy-compiled regex patterns for intent classification
# Compiled on first use to avoid cost for chat-only sessions
_TOOL_REQUIRED_PATTERNS: list[re.Pattern] | None = None
_CHAT_ONLY_PATTERNS: list[re.Pattern] | None = None


def _get_tool_required_patterns() -> list[re.Pattern]:
    """Return compiled patterns that suggest tool use is required."""
    global _TOOL_REQUIRED_PATTERNS
    if _TOOL_REQUIRED_PATTERNS is None:
        _TOOL_REQUIRED_PATTERNS = [
            re.compile(r"\b(?:create|write|make|build|generate|implement)\b.*\b(?:file|code|script|program|app|game|project|website|server|api|function|class|module|component)\b", re.IGNORECASE),
            re.compile(r"\b(?:edit|modify|change|update|fix|patch|refactor)\b.*\b(?:file|code|line|function|class|bug|error)\b", re.IGNORECASE),
            re.compile(r"\b(?:run|execute|test|compile|build|install)\b", re.IGNORECASE),
            re.compile(r"\b(?:read|show|cat|open|view|display)\b.*\b(?:file|contents?|source)\b", re.IGNORECASE),
            re.compile(r"\b(?:find|search|grep|locate|list)\b.*\b(?:file|dir|folder|pattern|function|class)\b", re.IGNORECASE),
            re.compile(r"\b(?:delete|remove|rename|move|copy)\b.*\b(?:file|dir|folder)\b", re.IGNORECASE),
        ]
    return _TOOL_REQUIRED_PATTERNS


def _get_chat_only_patterns() -> list[re.Pattern]:
    """Return compiled patterns that suggest chat-only / explanatory response."""
    global _CHAT_ONLY_PATTERNS
    if _CHAT_ONLY_PATTERNS is None:
        _CHAT_ONLY_PATTERNS = [
            re.compile(r"\b(?:explain|what is|what are|how does|why does|describe|tell me about|difference between)\b", re.IGNORECASE),
            re.compile(r"\b(?:help me understand|can you explain|what do you think)\b", re.IGNORECASE),
            re.compile(r"^\s*(?:hi|hello|hey|thanks|thank you)\b", re.IGNORECASE),
        ]
    return _CHAT_ONLY_PATTERNS


def classify_intent(prompt: str, source_page: str) -> ToolPolicy:
    """
    Deterministic intent classification.

    Rules:
      - chat page always returns FORBIDDEN (no tools available)
      - code page with tool-required pattern returns REQUIRED
      - code page with chat-only pattern returns OPTIONAL
      - code page default returns OPTIONAL
    """
    if source_page == "chat":
        return ToolPolicy.FORBIDDEN

    # Code page: analyze prompt
    prompt_lower = prompt.strip()

    # Check for strong chat-only signals (lazy-compiled on first use)
    for pattern in _get_chat_only_patterns():
        if pattern.search(prompt_lower):
            return ToolPolicy.OPTIONAL

    # Check for tool-required signals (lazy-compiled on first use)
    for pattern in _get_tool_required_patterns():
        if pattern.search(prompt_lower):
            return ToolPolicy.REQUIRED

    # Default for code page: optional (model decides)
    return ToolPolicy.OPTIONAL


# ---------------------------------------------------------------------------
# ContractFactory — produces ExecutionContract from context
# ---------------------------------------------------------------------------

def _compute_contract_hash(contract_dict: dict[str, Any]) -> str:
    """Deterministic hash of contract for transcript chain."""
    serialized = json.dumps(contract_dict, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ContractFactory:
    """
    Produces ExecutionContract from page context + prompt.

    This is the "intent-to-execution" bridge. The contract it produces
    is immutable and threaded through the entire runtime.
    """

    def __init__(
        self,
        default_profile_id: str = "local_xml",
        default_max_inferences: int = 25,
        default_ctx_limit: int = 8192,
    ) -> None:
        self._default_profile_id = default_profile_id
        self._default_max_inferences = default_max_inferences
        self._default_ctx_limit = default_ctx_limit

    def create(
        self,
        prompt: str,
        *,
        source_page: str = "code",
        allowed_tools: list[str] | None = None,
        model_profile_id: str | None = None,
        ctx_limit: int | None = None,
        max_inferences: int | None = None,
        strict_mode: bool = False,
        parent_contract_hash: str | None = None,
        model_fingerprint: str = "",
    ) -> ExecutionContract:
        """
        Create an immutable ExecutionContract for a single agent run.

        The tool_policy is determined by classify_intent unless source_page
        forces a specific policy (chat = forbidden).
        """
        profile_id = model_profile_id or self._default_profile_id
        tool_policy = classify_intent(prompt, source_page)

        context_window = ctx_limit or self._default_ctx_limit
        budget = max_inferences or self._default_max_inferences

        # For forbidden policy, zero out tool-related budgets
        if tool_policy == ToolPolicy.FORBIDDEN:
            budget = 1  # single inference, no looping

        contract_id = str(uuid.uuid4())
        creation_ts = time.time()

        # Build the contract without hash first, then compute hash
        proto = {
            "contract_id": contract_id,
            "contract_format_version": "3.0",
            "tool_policy": tool_policy.value,
            "allowed_tools": sorted(allowed_tools) if allowed_tools else None,
            "strict_mode": strict_mode,
            "max_inferences": budget,
            "model_profile_id": profile_id,
            "source_page": source_page,
            "context_window": context_window,
            "creation_timestamp": creation_ts,
        }
        contract_hash = _compute_contract_hash(proto)

        # Determine format retries based on profile
        if strict_mode:
            max_format_retries = 0
        elif profile_id in ("native", "local_native"):
            max_format_retries = 1
        else:
            max_format_retries = 2

        # Resolve grammar profile
        grammar_profile_id: str | None = None
        try:
            from engine.protocol_adapter import get_grammar_profile
            gp = get_grammar_profile(profile_id, tool_policy.value)
            if gp is not None:
                grammar_profile_id = gp.profile_id
        except ImportError:
            pass

        return ExecutionContract(
            contract_id=contract_id,
            contract_hash=contract_hash,
            parent_contract_hash=parent_contract_hash,
            tool_policy=tool_policy,
            allowed_tools=tuple(sorted(allowed_tools)) if allowed_tools else None,
            strict_mode=strict_mode,
            max_inferences=budget,
            max_format_retries=max_format_retries,
            context_budget=ContextBudget(context_window=context_window),
            adapter_version="2a.1",
            model_profile_id=profile_id,
            contract_format_version="3.0",
            model_fingerprint=model_fingerprint,
            grammar_profile=grammar_profile_id,
            source_page=source_page,
            creation_timestamp=creation_ts,
        )


# ---------------------------------------------------------------------------
# OFAC v0.2 Contract Schemas (Hashed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvSnapshot:
    """
    Environment fingerprint — cached once at startup.

    All fields are strings/booleans (no floats/timestamps inside hash boundary).
    """
    workspace_root: str = ""
    python_version: str = ""
    platform: str = ""
    env_fingerprint: str = ""
    git_dirty: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_root": self.workspace_root,
            "python_version": self.python_version,
            "platform": self.platform,
            "env_fingerprint": self.env_fingerprint,
            "git_dirty": self.git_dirty,
        }


@dataclass(frozen=True)
class FitContract:
    """
    FIT (Feasibility + Intent + Trust) contract — OFAC v0.2 Section 6.

    Frozen after creation. No floats or timestamps inside hash boundary.
    """
    goal: str = ""
    success_criteria: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()
    stop_conditions: tuple[str, ...] = ()
    fit_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "success_criteria": list(self.success_criteria),
            "risk_flags": list(self.risk_flags),
            "stop_conditions": list(self.stop_conditions),
            "fit_hash": self.fit_hash,
        }


@dataclass(frozen=True)
class PlanStep:
    """Single step in a PLAN_SNAPSHOT."""
    id: str = ""
    tool: str = ""
    requires_ack: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "requires_ack": self.requires_ack,
        }


@dataclass(frozen=True)
class PlanSnapshot:
    """
    PLAN_SNAPSHOT — OFAC v0.2 Section 6.

    Immutable after creation. Integer version, no floats.
    """
    version: int = 1
    steps: tuple[PlanStep, ...] = ()
    plan_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "plan_hash": self.plan_hash,
        }
