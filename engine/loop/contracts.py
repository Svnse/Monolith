"""
Contracts: every data structure the loop touches.

RunPolicy, ToolSpec, Evidence, PreflightResult, Pad, Step, RunContext, RunResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPolicy:
    """Immutable budget + approval config for a single run."""

    max_cycles: int = 25
    max_tool_calls: int = 60
    max_elapsed_sec: float = 300.0
    max_retries: int = 2
    stall_window: int = 4
    repetition_window: int = 3

    auto_approve: frozenset[str] = frozenset({
        "read", "list", "search", "grep",
    })
    require_approval: frozenset[str] = frozenset({
        "write", "execute", "delete", "shell",
    })


# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """One tool available to the agent."""

    name: str
    description: str
    parameters: dict[str, Any]
    scope: str = "read"
    when_to_use: str = ""
    when_not_to_use: str = ""
    required_args: list[str] = field(default_factory=list)
    failure_recovery: str = ""
    example_calls: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """Normalized result from a single tool call."""

    tool: str
    args: dict[str, Any]
    output: str
    ok: bool
    cycle: int


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreflightResult:
    """Output of the preflight planning call."""

    variant_literal: str
    variant_constraint: str
    variant_intent: str
    variant_failure: str
    variant_minimal: str

    intent_type: str
    response_mode: str
    complexity_class: str
    primary_constraint: str
    implicit_assumptions: list[str] = field(default_factory=list)

    plan_granularity: int = 3
    execution_weight: int = 3

    vision_primary: str = ""
    vision_minimum: str = ""

    approach: str = ""
    risks: list[str] = field(default_factory=list)
    sequencing: str = ""
    verification_strategy: str = ""

    action_steps: list[dict[str, Any]] = field(default_factory=list)
    todo: list[dict[str, Any]] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pad
# ---------------------------------------------------------------------------

@dataclass
class Pad:
    """
    Working memory for the execution loop.

    Static fields are seeded once from preflight/environment.
    Dynamic fields are runtime-managed.
    """

    # Static fields.
    goal: str
    preflight: PreflightResult | None = None
    env_block: str = ""

    # Dynamic fields.
    plan: str = ""
    todo_state: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    last_check: dict[str, Any] | None = None
    tool_failures: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    progress: float = 0.0
    artifacts: dict[str, str] = field(default_factory=dict)
    invariants: list[str] = field(default_factory=list)

    # Internal bookkeeping.
    _progress_history: list[float] = field(default_factory=list, repr=False)
    _invariant_origins: dict[str, str] = field(default_factory=dict, repr=False)

    MAX_EVIDENCE: int = 20
    MAX_TOOL_FAILURES: int = 5
    MAX_OPEN_QUESTIONS: int = 3
    MAX_INVARIANTS: int = 16

    def trim_evidence(self) -> None:
        if len(self.evidence) > self.MAX_EVIDENCE:
            self.evidence = self.evidence[-self.MAX_EVIDENCE:]

    def record_progress(self) -> None:
        self._progress_history.append(self.progress)

    def record_check(self, *, cycle: int, ok: bool | None, note: str) -> None:
        note_s = str(note or "").strip()
        if not note_s and ok is None:
            return
        self.last_check = {
            "cycle": int(cycle),
            "ok": ok,
            "note": note_s,
        }

    def record_tool_failure(
        self,
        *,
        cycle: int,
        tool: str,
        signature: str,
        failure_class: str,
        error_code: str,
        note: str,
    ) -> None:
        self.tool_failures.append({
            "cycle": int(cycle),
            "tool": str(tool or ""),
            "signature": str(signature or ""),
            "failure_class": str(failure_class or "unknown"),
            "error_code": str(error_code or ""),
            "note": str(note or "").strip(),
        })
        if len(self.tool_failures) > self.MAX_TOOL_FAILURES:
            self.tool_failures = self.tool_failures[-self.MAX_TOOL_FAILURES:]

    def compute_progress(self) -> float:
        if not self.todo_state:
            total = len(self.evidence)
            ok = sum(1 for e in self.evidence if e.ok)
            return round(ok / max(1, total), 2)
        total = len(self.todo_state)
        done = sum(1 for t in self.todo_state if bool(t.get("crystallized")))
        return round(done / max(1, total), 2)

    def add_invariant(self, text: str, origin: str = "model") -> None:
        entry = str(text or "").strip()
        if not entry or entry in self._invariant_origins:
            return
        if len(self.invariants) >= self.MAX_INVARIANTS:
            for idx, inv in enumerate(self.invariants):
                if self._invariant_origins.get(inv) == "model":
                    self._invariant_origins.pop(inv, None)
                    self.invariants.pop(idx)
                    break
            else:
                return
        self.invariants.append(entry)
        self._invariant_origins[entry] = str(origin or "model")

    def render_todo_text(self) -> str:
        if not self.todo_state:
            return ""
        lines: list[str] = []
        for idx, item in enumerate(self.todo_state, start=1):
            mark = "x" if item.get("crystallized") else " "
            directive = str(item.get("directive") or "").strip() or "<todo>"
            tool_hint = str(item.get("tool_hint") or "").strip()
            done_cycle = item.get("cycle_crystallized")
            suffix = f" -> {tool_hint}" if tool_hint else ""
            done_note = f" (done cycle {done_cycle})" if done_cycle else ""
            lines.append(f"  [{mark}] {idx}. {directive}{suffix}{done_note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """Structured output from a single LLM cycle."""

    intent: str
    response: str = ""
    reasoning: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    self_check: str = ""
    step_ok: bool | None = None
    todo_update: str | None = None
    task_finished: bool | None = False
    finish_summary: str = ""


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    run_id: str
    goal: str
    policy: RunPolicy
    tools: list[ToolSpec]
    pad: Pad
    cycle: int = 0
    total_tool_calls: int = 0
    start_time: float = 0.0
    paused_sec: float = 0.0
    recent_intents: list[str] = field(default_factory=list)
    llm_call_count: int = 0
    action_failure_counts: dict[str, int] = field(default_factory=dict)
    action_failure_class: dict[str, str] = field(default_factory=dict)
    action_failure_code: dict[str, str] = field(default_factory=dict)
    action_success_cycle: dict[str, int] = field(default_factory=dict)
    attempted_installs: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_id: str
    success: bool
    summary: str
    pad: Pad
    cycles_used: int
    tool_calls_used: int
    wall_hit: str | None = None
