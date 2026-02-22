"""
Stepwise Agent Runtime — Async execution with signal propagation.

Refactors AgentRuntime.run() from blocking loop to resumable state machine.
Each step returns control to caller, allowing Qt signals to flush.

OFAC v0.2 WAIT_ACK behavior:
  - Enter WAIT_ACK: Stepper returns should_continue=False. Worker thread dies.
  - Resume (Approve): MonoGuard spawns new thread, FSM resumes to EXECUTE.
  - Resume (Deny):    MonoGuard spawns new thread, FSM transitions to TERMINATE.
  - Resume (Timeout): TTL timer fires, FSM transitions to TERMINATE.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal

from engine.agent_runtime import (
    AgentMessage,
    AgentRuntime,
    ToolCall,
    MAX_AGENT_STEPS,
)
from engine.capabilities import CapabilityManager, CapabilityScope, TOOL_SCOPE_MAP
from engine.contract import (
    AgentOutcome,
    AgentRunResult,
    RuntimeState,
    ToolPolicy,
)


@dataclass
class StepResult:
    """Result of a single step execution."""
    state: RuntimeState
    should_continue: bool
    result: AgentRunResult | None = None
    events_emitted: list[dict] = None
    # WAIT_ACK metadata — when should_continue is False and state is WAIT_ACK
    awaiting_ack: bool = False
    ack_tool_calls: list[dict] | None = None

    def __post_init__(self):
        if self.events_emitted is None:
            self.events_emitted = []


def _requires_ack(call: ToolCall) -> bool:
    """
    Determine if a tool call requires user acknowledgment.

    OFAC v0.2: EXEC-scope and WRITE-scope tools require ACK by default.
    """
    scope = TOOL_SCOPE_MAP.get(call.name)
    if scope in (CapabilityScope.EXEC, CapabilityScope.WRITE):
        return True
    return False


class StepwiseAgentRuntime:
    """
    Resumable agent runtime that executes one step at a time.

    Usage:
        stepper = StepwiseAgentRuntime(runtime, messages)
        stepper.initialize()

        while stepper.should_continue():
            result = stepper.step()
            # Signals flush here
            if not result.should_continue:
                if result.awaiting_ack:
                    # Worker thread should die; resume later via resume_from_ack()
                    break
                break

        final_result = stepper.get_result()
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        messages: list[dict[str, Any]],
        emit_event: Callable[[dict], None] | None = None,
    ):
        self._runtime = runtime
        self._messages = messages
        self._emit_event = emit_event or runtime._emit_event

        # Execution state (persisted across steps)
        self._history: list[AgentMessage] = []
        self._tools: list[dict] = []
        self._contract = runtime._contract
        self._max_inferences = self._contract.max_inferences if self._contract else MAX_AGENT_STEPS
        self._tool_policy = self._contract.tool_policy if self._contract else ToolPolicy.OPTIONAL
        self._max_format_retries = self._contract.max_format_retries if self._contract else 1
        self._cycle_forbid = set(self._contract.cycle_forbid) if self._contract and self._contract.cycle_forbid else set()
        self._run_id = str(uuid.uuid4())

        # Error flags
        self._had_protocol_error = False
        self._had_validation_error = False
        self._had_cycle_violation = False
        self._last_tool_name: str | None = None

        # Current execution state
        self._current_state = RuntimeState.PRECHECK
        self._initialized = False
        self._terminated = False
        self._final_result: AgentRunResult | None = None

        # Tool execution batch state
        self._pending_tool_calls: list[ToolCall] = []
        self._current_tool_index = 0

        # WAIT_ACK state
        self._ack_required: bool = False
        self._ack_tool_summaries: list[dict] = []

    def initialize(self) -> list[dict]:
        """Initialize the runtime. Call once before first step()."""
        if self._initialized:
            return []

        events = []

        # Reset runtime counters
        self._runtime._event_step_id = 0
        self._runtime._runtime_node_id = 0
        self._runtime._runtime_action_id = 0
        self._runtime._runtime_leaf_node_id = None
        self._runtime._fsm_state = None
        self._runtime._inferences = 0
        self._runtime._tokens_consumed = 0
        self._runtime._tools_executed = 0
        self._runtime._format_retries = 0
        self._runtime._run_start_time = time.time()
        self._runtime._force_terminate_after_next_inference = False
        self._runtime._last_outcome = None
        self._runtime._estimated_context_ratio = 0.0

        # Load history
        for msg in self._messages:
            role = msg.get("role")
            if role in {"system", "user", "assistant", "tool"}:
                self._history.append(AgentMessage(role=role, content=msg.get("content")))
                if role == "user":
                    self._runtime._ledger.append(
                        actor="user",
                        event_type="input",
                        payload={"content": msg.get("content", "")}
                    )
                    self._runtime._emit_runtime_node(role="user", content=str(msg.get("content") or ""))

        # Load capabilities
        self._tools = self._runtime._capability_manager.tool_schemas()
        self._runtime._ledger.append(
            actor="system",
            event_type="input",
            payload={
                "capability_profile": self._runtime._capability_manager.profile,
                "capability_digest": self._runtime._capability_manager.capability_digest,
                "tools": self._runtime._capability_manager.allowed_tools(),
            },
        )

        # Log contract
        if self._contract:
            self._runtime._ledger.append(
                actor="system",
                event_type="input",
                payload={"contract": self._contract.to_dict()},
            )
            events.append({
                "event": "CONTRACT_BOUND",
                "contract": self._contract.to_dict(),
                "timestamp": time.time(),
            })
        
        # OFAC v0.2: Emit ENV_SNAPSHOT (cached, computed once)
        self._runtime._emit_env_snapshot()
        events.append({"event": "ENV_SNAPSHOT", "timestamp": time.time()})
        
        # OFAC v0.2: Emit FIT_CONTRACT for this run
        # Extract goal from first user message if available
        goal = "Agent execution"
        for msg in self._messages:
            if msg.get("role") == "user":
                goal = msg.get("content", "")[:100]  # First 100 chars
                break
        self._runtime._emit_fit_contract(goal=goal, risk_flags=[])
        events.append({"event": "FIT_CONTRACT", "timestamp": time.time()})

        self._initialized = True
        return events

    def should_continue(self) -> bool:
        """Check if execution should continue."""
        return not self._terminated and self._current_state != RuntimeState.TERMINATE

    def step(self) -> StepResult:
        """
        Execute one step of the agent loop.
        Returns immediately after, allowing signals to propagate.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before step()")

        if self._terminated:
            return StepResult(
                state=RuntimeState.TERMINATE,
                should_continue=False,
                result=self._final_result,
            )

        # Check for STOP request
        if self._runtime._should_stop():
            self._current_state = RuntimeState.TERMINATE
            self._final_result = self._terminate(AgentOutcome.INTERRUPTED, "user_interrupt")
            self._terminated = True
            return StepResult(
                state=RuntimeState.TERMINATE,
                should_continue=False,
                result=self._final_result,
            )

        # Execute based on current state
        if self._current_state == RuntimeState.PRECHECK:
            return self._step_precheck()
        elif self._current_state == RuntimeState.INFER:
            return self._step_infer()
        elif self._current_state == RuntimeState.VALIDATE_CALLS:
            return self._step_validate_calls()
        elif self._current_state == RuntimeState.WAIT_ACK:
            return self._step_wait_ack()
        elif self._current_state == RuntimeState.EXECUTE:
            return self._step_execute()
        elif self._current_state == RuntimeState.OBSERVE:
            return self._step_observe()
        elif self._current_state == RuntimeState.COMMIT:
            return self._step_commit()
        else:
            # Unknown state, terminate
            self._current_state = RuntimeState.TERMINATE
            self._final_result = self._terminate(AgentOutcome.FAILED_PREFLIGHT, "unknown_state")
            self._terminated = True
            return StepResult(
                state=RuntimeState.TERMINATE,
                should_continue=False,
                result=self._final_result,
            )

    # ------------------------------------------------------------------
    # WAIT_ACK resume interface
    # ------------------------------------------------------------------

    def resume_from_ack(self, action: str) -> None:
        """
        Resume stepper after WAIT_ACK. Called when a new worker thread starts.

        action: "approve" → transition to EXECUTE
        action: "deny"    → transition to TERMINATE
        action: "timeout" → transition to TERMINATE
        """
        if self._current_state != RuntimeState.WAIT_ACK:
            raise RuntimeError(f"Cannot resume_from_ack in state {self._current_state}")

        self._ack_required = False

        if action == "approve":
            self._current_state = RuntimeState.EXECUTE
            self._runtime._transition_to(RuntimeState.EXECUTE)
            self._emit({
                "event": "ACK_APPROVED",
                "tools": [c.name for c in self._pending_tool_calls],
                "timestamp": time.time(),
            })
        elif action in ("deny", "timeout"):
            reason = "user_denied" if action == "deny" else "ack_timeout"
            self._emit({
                "event": "ACK_REJECTED",
                "reason": reason,
                "timestamp": time.time(),
            })
            self._current_state = RuntimeState.TERMINATE
            self._runtime._transition_to(RuntimeState.TERMINATE)
            self._final_result = self._terminate(AgentOutcome.INTERRUPTED, reason)
            self._terminated = True
        else:
            raise ValueError(f"Unknown ack action: {action}")

    # ------------------------------------------------------------------
    # State step implementations
    # ------------------------------------------------------------------

    def _step_precheck(self) -> StepResult:
        """Execute PRECHECK state."""
        if self._runtime._fsm_state != RuntimeState.PRECHECK:
            self._runtime._transition_to(RuntimeState.PRECHECK)

        preflight_failure = self._runtime._preflight_check()
        if preflight_failure is not None:
            self._emit({
                "event": "PREFLIGHT_FAILURE",
                "reason": preflight_failure["reason"],
                "timestamp": time.time(),
            })
            self._current_state = RuntimeState.TERMINATE
            self._final_result = self._terminate(
                preflight_failure["outcome"],
                preflight_failure["reason"]
            )
            self._terminated = True
            return StepResult(
                state=RuntimeState.TERMINATE,
                should_continue=False,
                result=self._final_result,
            )

        # Transition to first inference
        self._current_state = RuntimeState.INFER
        return StepResult(
            state=RuntimeState.PRECHECK,
            should_continue=True,
        )

    def _step_infer(self) -> StepResult:
        """Execute INFER state - LLM call."""
        # Budget check
        if self._runtime._inferences >= self._max_inferences:
            self._current_state = RuntimeState.TERMINATE
            self._final_result = self._terminate(
                AgentOutcome.FAILED_BUDGET_EXHAUSTED,
                "max_inferences_exceeded"
            )
            self._terminated = True
            return StepResult(
                state=RuntimeState.TERMINATE,
                should_continue=False,
                result=self._final_result,
            )

        if self._runtime._fsm_state != RuntimeState.INFER:
            self._runtime._transition_to(RuntimeState.INFER)

        llm_step = self._runtime._emit_step_start(label="LLM call", kind="llm_call")
        normalized_history = [self._runtime._serialize(m) for m in self._history]

        # LLM call (blocking, but now isolated to this step)
        if self._runtime._protocol_adapter is not None:
            raw_response = self._runtime._llm_call(normalized_history, self._tools)
            if isinstance(raw_response, dict):
                usage = raw_response.get("usage", {})
                if isinstance(usage, dict):
                    self._runtime._tokens_consumed += usage.get("total_tokens", 0)
            assistant = self._runtime._adapt_response(raw_response)
            if self._runtime._last_adapter_status in ("rejected", "recovered"):
                self._had_protocol_error = True
        else:
            assistant = self._runtime._llm_call(normalized_history, self._tools)

        self._runtime._inferences += 1

        # Log and emit
        self._runtime._ledger.append(
            actor="assistant",
            event_type="inference",
            payload=self._runtime._serialize(assistant),
            reasoning=assistant.content,
        )
        self._history.append(assistant)

        events = []

        if assistant.content:
            events.append({
                "event": "AGENT_THOUGHT",
                "step_id": llm_step,
                "thought": assistant.content,
                "timestamp": time.time(),
            })
            # Emit token chunks for UI streaming
            content = assistant.content
            chunk_size = max(1, len(content) // 10)  # ~10 chunks
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                events.append({"event": "LLM_TOKEN", "data": chunk})

        events.append({
            "event": "AGENT_MESSAGE",
            "message": self._runtime._serialize(assistant),
            "timestamp": time.time(),
        })

        self._runtime._emit_runtime_node(role="assistant", content=str(assistant.content or ""))
        self._runtime._emit_step_end(llm_step, status="ok")

        # Check for forced synthesis
        if self._runtime._force_terminate_after_next_inference and not assistant.tool_calls:
            output = assistant.content or ""
            outcome = (
                AgentOutcome.COMPLETED_WITH_TOOLS
                if self._runtime._tools_executed > 0
                else AgentOutcome.COMPLETED_CHAT_ONLY
            )
            self._current_state = RuntimeState.TERMINATE
            self._final_result = self._terminate(outcome, "forced_synthesis_complete", output)
            self._terminated = True
            return StepResult(
                state=RuntimeState.TERMINATE,
                should_continue=False,
                result=self._final_result,
                events_emitted=events,
            )

        # Check for no tool calls (potential termination)
        if not assistant.tool_calls:
            return self._handle_no_tool_calls(assistant, llm_step, events)

        # Have tool calls, proceed to validation
        self._pending_tool_calls = assistant.tool_calls
        self._current_tool_index = 0
        self._current_state = RuntimeState.VALIDATE_CALLS

        for ev in events:
            self._emit(ev)

        return StepResult(
            state=RuntimeState.INFER,
            should_continue=True,
            events_emitted=events,
        )

    def _handle_no_tool_calls(
        self,
        assistant: AgentMessage,
        llm_step: int,
        events: list[dict]
    ) -> StepResult:
        """Handle case where assistant returns no tool calls."""
        content = assistant.content or ""

        # Check for protocol error
        is_protocol_error = isinstance(content, str) and content.startswith("[PROTOCOL_ERROR:")

        if is_protocol_error and self._runtime._format_retries < self._max_format_retries:
            self._had_protocol_error = True
            self._runtime._format_retries += 1
            events.append({
                "event": "PARSE_ERROR",
                "retry": self._runtime._format_retries,
                "max_retries": self._max_format_retries,
                "timestamp": time.time(),
            })
            self._runtime._ledger.append(
                actor="system",
                event_type="error",
                payload={
                    "kind": "format_retry",
                    "retry": self._runtime._format_retries,
                    "content": content
                },
            )
            self._history.pop()  # Remove failed assistant message
            # Stay in INFER, will retry
            for ev in events:
                self._emit(ev)
            return StepResult(
                state=RuntimeState.INFER,
                should_continue=True,
                events_emitted=events,
            )

        if is_protocol_error and self._runtime._format_retries >= self._max_format_retries:
            self._had_protocol_error = True
            self._current_state = RuntimeState.TERMINATE
            self._final_result = self._terminate(
                AgentOutcome.FAILED_PROTOCOL_MALFORMED,
                "format_retries_exhausted"
            )
            self._terminated = True
            for ev in events:
                self._emit(ev)
            return StepResult(
                state=RuntimeState.TERMINATE,
                should_continue=False,
                result=self._final_result,
                events_emitted=events,
            )

        # Check tool policy
        if (
            self._tool_policy == ToolPolicy.REQUIRED
            and self._runtime._tools_executed == 0
            and self._runtime._format_retries < self._max_format_retries
        ):
            # Nudge required
            self._runtime._format_retries += 1
            events.append({
                "event": "TOOL_REQUIRED_NUDGE",
                "retry": self._runtime._format_retries,
                "max_retries": self._max_format_retries,
                "timestamp": time.time(),
            })
            nudge = AgentMessage(
                role="system",
                content="You MUST use a tool to complete this task. Output a tool call now.",
            )
            self._history.append(nudge)
            events.append({
                "event": "AGENT_NUDGE",
                "step_id": self._runtime._next_step_id(),
                "content": nudge.content,
                "timestamp": time.time(),
            })
            for ev in events:
                self._emit(ev)
            return StepResult(
                state=RuntimeState.INFER,
                should_continue=True,
                events_emitted=events,
            )

        # No tool calls, go to commit for termination evaluation
        self._current_state = RuntimeState.COMMIT
        for ev in events:
            self._emit(ev)
        return StepResult(
            state=RuntimeState.INFER,
            should_continue=True,
            events_emitted=events,
        )

    def _step_validate_calls(self) -> StepResult:
        """Execute VALIDATE_CALLS state — check calls and determine ACK requirement."""
        if self._runtime._fsm_state != RuntimeState.VALIDATE_CALLS:
            self._runtime._transition_to(RuntimeState.VALIDATE_CALLS)

        # Validate each pending tool call
        valid_calls = []
        needs_ack = False
        ack_summaries = []

        for call in self._pending_tool_calls:
            # Check cycle forbid
            if self._last_tool_name and (self._last_tool_name, call.name) in self._cycle_forbid:
                self._had_cycle_violation = True
                self._current_state = RuntimeState.TERMINATE
                self._final_result = self._terminate(
                    AgentOutcome.FAILED_CONTRACT_VIOLATION,
                    f"cycle_forbid_violation: {self._last_tool_name} -> {call.name}"
                )
                self._terminated = True
                return StepResult(
                    state=RuntimeState.TERMINATE,
                    should_continue=False,
                    result=self._final_result,
                )

            # Check if this call requires ACK
            if _requires_ack(call):
                needs_ack = True
                ack_summaries.append({
                    "tool": call.name,
                    "arguments": call.arguments,
                    "scope": TOOL_SCOPE_MAP.get(call.name, CapabilityScope.READ).value,
                })

            valid_calls.append(call)

        self._pending_tool_calls = valid_calls
        self._current_tool_index = 0

        # If any call needs ACK, transition to WAIT_ACK
        if needs_ack:
            self._ack_required = True
            self._ack_tool_summaries = ack_summaries
            self._current_state = RuntimeState.WAIT_ACK
            self._runtime._transition_to(RuntimeState.WAIT_ACK)

            self._emit({
                "event": "WAIT_ACK_ENTER",
                "tools": ack_summaries,
                "timestamp": time.time(),
            })

            return StepResult(
                state=RuntimeState.WAIT_ACK,
                should_continue=False,  # Worker thread should die here
                awaiting_ack=True,
                ack_tool_calls=ack_summaries,
            )

        # No ACK needed, proceed directly to EXECUTE
        self._current_state = RuntimeState.EXECUTE

        return StepResult(
            state=RuntimeState.VALIDATE_CALLS,
            should_continue=True,
        )

    def _step_wait_ack(self) -> StepResult:
        """
        WAIT_ACK state — should only be reached after resume_from_ack().

        If we arrive here and still need ACK, return should_continue=False
        to kill the worker thread again.
        """
        if self._ack_required:
            # Still waiting — this shouldn't happen in normal flow
            return StepResult(
                state=RuntimeState.WAIT_ACK,
                should_continue=False,
                awaiting_ack=True,
                ack_tool_calls=self._ack_tool_summaries,
            )

        # ACK was resolved via resume_from_ack(), state already updated
        # If we're now in EXECUTE, step will pick it up on next call
        # If we're in TERMINATE, step will pick it up on next call
        return StepResult(
            state=self._current_state,
            should_continue=self._current_state != RuntimeState.TERMINATE,
            result=self._final_result if self._terminated else None,
        )

    def _step_execute(self) -> StepResult:
        """Execute EXECUTE state - run one tool at a time."""
        # Only transition if not already in EXECUTE (handles resume_from_ack case)
        if self._runtime._fsm_state != RuntimeState.EXECUTE:
            self._runtime._transition_to(RuntimeState.EXECUTE)

        if self._current_tool_index >= len(self._pending_tool_calls):
            # All tools executed, move to observe
            self._current_state = RuntimeState.OBSERVE
            return StepResult(
                state=RuntimeState.EXECUTE,
                should_continue=True,
            )

        # OFAC: Take pre-execute manifest snapshot (first tool only)
        if self._current_tool_index == 0:
            self._runtime._snapshot_workspace_before()

        call = self._pending_tool_calls[self._current_tool_index]

        # Emit tool call start
        call_step = self._runtime._emit_step_start(
            label=f"Tool call: {call.name}",
            kind="tool_call",
            tool=call.name,
            arguments=call.arguments
        )

        action_id = self._runtime._next_runtime_action_id()
        action = {
            "action_id": action_id,
            "tool": call.name,
            "arguments": dict(call.arguments),
            "status": "pending",
            "step_id": call_step,
        }

        events = [
            {"event": "ACTION_QUEUED", "action": dict(action), "timestamp": time.time()},
            {
                "event": "TOOL_CALL_START",
                "step_id": call_step,
                "tool": call.name,
                "arguments": call.arguments,
                "tool_call_id": call.id,
                "timestamp": time.time(),
            },
        ]

        action["status"] = "running"
        events.append({"event": "ACTION_STARTED", "action": dict(action), "timestamp": time.time()})

        # Execute tool via AgentRuntime._execute_tool (routes through AgentBridge)
        observation = self._runtime._execute_tool(call)

        # Handle timeout/error
        if observation.role == "tool" and observation.content:
            try:
                parsed = json.loads(observation.content)
                if isinstance(parsed, dict) and parsed.get("status") == "error":
                    error_code = parsed.get("error_code", "")
                    if "timeout" in error_code.lower():
                        self._current_state = RuntimeState.TERMINATE
                        self._final_result = self._terminate(
                            AgentOutcome.FAILED_TIMEOUT,
                            f"step_timeout on tool {call.name}"
                        )
                        self._terminated = True
                        for ev in events:
                            self._emit(ev)
                        return StepResult(
                            state=RuntimeState.TERMINATE,
                            should_continue=False,
                            result=self._final_result,
                            events_emitted=events,
                        )
            except (json.JSONDecodeError, TypeError):
                pass

        self._runtime._tools_executed += 1
        self._last_tool_name = call.name

        # Process result
        self._history.append(observation)
        self._runtime._ledger.append(
            actor="tool",
            event_type="tool_result",
            payload=self._runtime._serialize(observation),
            execution={"tool_name": call.name, "arguments": call.arguments},
        )

        # Parse observation payload
        observation_payload = {}
        if isinstance(observation.content, str):
            try:
                parsed = json.loads(observation.content)
                if isinstance(parsed, dict):
                    observation_payload = parsed
            except Exception:
                observation_payload = {"raw": observation.content}

        call_status = "ok" if observation_payload.get("status") != "error" else "error"
        call_error = observation_payload.get("error") if isinstance(observation_payload.get("error"), str) else None

        action["status"] = "done" if call_status == "ok" else "error"
        action["result"] = observation_payload if observation_payload else self._runtime._serialize(observation)

        events.extend([
            {"event": "ACTION_FINISHED", "action": dict(action), "timestamp": time.time()},
            {"event": "TOOL_RESULT", "step_id": call_step, "tool": call.name, "result": observation_payload, "timestamp": time.time()},
            {"event": "TOOL_OBSERVATION", "message": self._runtime._serialize(observation), "timestamp": time.time()},
        ])

        self._runtime._emit_step_end(call_step, status=call_status, error=call_error)
        self._runtime._emit_runtime_node(role="tool", content=observation.content or "")

        # Validate envelope
        if observation_payload and not self._runtime._validate_tool_envelope(observation_payload):
            self._current_state = RuntimeState.TERMINATE
            self._final_result = self._terminate(
                AgentOutcome.FAILED_VALIDATION,
                f"malformed_tool_result_envelope from {call.name}"
            )
            self._terminated = True
            for ev in events:
                self._emit(ev)
            return StepResult(
                state=RuntimeState.TERMINATE,
                should_continue=False,
                result=self._final_result,
                events_emitted=events,
            )

        self._current_tool_index += 1

        # If more tools pending, stay in EXECUTE
        if self._current_tool_index < len(self._pending_tool_calls):
            for ev in events:
                self._emit(ev)
            return StepResult(
                state=RuntimeState.EXECUTE,
                should_continue=True,
                events_emitted=events,
            )

        # All tools done, move to OBSERVE
        self._current_state = RuntimeState.OBSERVE
        for ev in events:
            self._emit(ev)
        return StepResult(
            state=RuntimeState.EXECUTE,
            should_continue=True,
            events_emitted=events,
        )

    def _step_observe(self) -> StepResult:
        """Execute OBSERVE state."""
        if self._runtime._fsm_state != RuntimeState.OBSERVE:
            self._runtime._transition_to(RuntimeState.OBSERVE)
        
        # OFAC: Take post-execute snapshot and emit MUTATION_RECORD
        self._runtime._snapshot_workspace_after_and_emit()
        
        self._current_state = RuntimeState.COMMIT
        return StepResult(
            state=RuntimeState.OBSERVE,
            should_continue=True,
        )

    def _step_commit(self) -> StepResult:
        """Execute COMMIT state - evaluate continuation."""
        if self._runtime._fsm_state != RuntimeState.COMMIT:
            self._runtime._transition_to(RuntimeState.COMMIT)

        # Get last assistant message
        assistant = None
        for msg in reversed(self._history):
            if msg.role == "assistant":
                assistant = msg
                break

        commit = self._runtime._evaluate_commit(
            inferences=self._runtime._inferences,
            tools_executed=self._runtime._tools_executed,
            format_retries=self._runtime._format_retries,
            tokens_consumed=self._runtime._tokens_consumed,
            run_start_time=self._runtime._run_start_time,
            history=self._history,
            assistant=assistant,
        )

        if commit["action"] == "terminate":
            # Extract output from last assistant message
            output = ""
            if assistant and assistant.content:
                output = assistant.content

            self._current_state = RuntimeState.TERMINATE
            self._final_result = self._terminate(commit["outcome"], commit["reason"], output)
            self._terminated = True
            return StepResult(
                state=RuntimeState.TERMINATE,
                should_continue=False,
                result=self._final_result,
            )

        if commit["action"] == "force_synthesis":
            # Inject synthesis prompt
            synthesis_prompt = AgentMessage(
                role="system",
                content="Provide a final synthesis of your work so far.",
            )
            self._history.append(synthesis_prompt)
            self._runtime._force_terminate_after_next_inference = True
            self._current_state = RuntimeState.INFER
            return StepResult(
                state=RuntimeState.COMMIT,
                should_continue=True,
            )

        # Continue to next inference
        self._current_state = RuntimeState.INFER
        return StepResult(
            state=RuntimeState.COMMIT,
            should_continue=True,
        )

    def _terminate(self, outcome: AgentOutcome, reason: str, output: str = "") -> AgentRunResult:
        """Create termination result."""
        self._runtime._last_outcome = outcome
        run_end_time = time.time()

        self._runtime._ledger.append(
            actor="system",
            event_type="telemetry",
            payload={
                "reason": reason,
                "outcome": outcome.value,
                "tools_executed": self._runtime._tools_executed,
                "inferences": self._runtime._inferences,
            },
        )

        term_step = self._runtime._emit_step_start(label="Termination", kind="termination")

        events = [
            {
                "event": "TERMINATION",
                "step_id": term_step,
                "reason": reason,
                "outcome": outcome.value,
                "output": output if output else None,
                "timestamp": run_end_time,
            },
        ]

        if outcome.is_success:
            self._runtime._emit_step_end(term_step, status="ok")
            if output:
                # Chunk and emit final output
                chunk_size = max(1, len(output) // 20)
                for i in range(0, len(output), chunk_size):
                    chunk = output[i:i+chunk_size]
                    events.append({"event": "LLM_TOKEN", "data": chunk})
                events.append({"event": "FINAL_OUTPUT", "data": output})
        else:
            self._runtime._emit_step_end(term_step, status="error", error=reason)

        for ev in events:
            self._emit(ev)

        return AgentRunResult(
            outcome=outcome,
            output=output,
            history=[self._runtime._serialize(m) for m in self._history],
            contract=self._contract,
            steps_used=self._runtime._event_step_id,
            inferences_used=self._runtime._inferences,
            tools_executed=self._runtime._tools_executed,
            format_retries_used=self._runtime._format_retries,
            tokens_consumed=self._runtime._tokens_consumed,
            termination_reason=reason,
        )

    def _emit(self, event: dict):
        """Emit event through runtime's emitter."""
        self._emit_event(event)

    def get_result(self) -> AgentRunResult | None:
        """Get final result after termination."""
        return self._final_result
