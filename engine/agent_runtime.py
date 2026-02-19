"""
Agent Runtime — Phase 3 of Monolith Agent Contract V2.

Single-loop runtime with explicit FSM, typed AgentOutcome, unified budget
accounting, tool output budget enforcement, timeout enforcement, forced
synthesis, and state digest emission.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal

from core.event_ledger import AppendOnlyLedger
from engine.agent_bridge import AgentBridge
from engine.capabilities import CapabilityManager, extract_tool_path
from engine.tool_schema import TOOL_ARGUMENT_SCHEMAS

Role = Literal["system", "user", "assistant", "tool"]

MAX_AGENT_STEPS = 25


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentMessage:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class AgentRuntime:
    def __init__(
        self,
        llm_call: Callable[[list[dict[str, Any]], list[dict[str, Any]]], AgentMessage],
        bridge: AgentBridge | None = None,
        capability_manager: CapabilityManager | None = None,
        emit_event: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ):
        self._llm_call = llm_call
        self._capability_manager = capability_manager or CapabilityManager("code")
        self._bridge = bridge or AgentBridge(capability_manager=self._capability_manager)
        self._emit_event = emit_event or (lambda _: None)
        self._should_stop = should_stop or (lambda: False)
        self._ledger = AppendOnlyLedger()
        self._event_step_id = 0
        self._runtime_node_id = 0
        self._runtime_action_id = 0
        self._runtime_branch_id = "main"
        self._runtime_leaf_node_id: str | None = None

        # Protocol adapter — lazily set by worker or caller
        self._protocol_adapter: Any | None = None
        # Execution contract — lazily set by worker or caller
        self._contract: Any | None = None
        # Last adapter status for telemetry flag tracking
        self._last_adapter_status: str | None = None
        # Transcript chain (Phase 5)
        self._transcript_chain: Any | None = None

        # FSM state (Phase 3)
        self._fsm_state: Any | None = None  # RuntimeState, set at run() entry
        self._initial_contract_hash: str | None = None

        # Counters promoted to instance vars for state digest access (Phase 3)
        self._inferences = 0
        self._tokens_consumed = 0
        self._tools_executed = 0
        self._format_retries = 0
        self._run_start_time: float = 0.0
        self._force_terminate_after_next_inference = False
        self._last_outcome: Any | None = None  # AgentOutcome
        self._estimated_context_ratio: float = 0.0

    # ------------------------------------------------------------------
    # Event emission helpers
    # ------------------------------------------------------------------

    def _emit(self, payload: dict[str, Any]) -> None:
        self._emit_event(payload)

    def _next_step_id(self) -> int:
        self._event_step_id += 1
        return self._event_step_id

    def _next_runtime_node_id(self) -> str:
        self._runtime_node_id += 1
        return f"n{self._runtime_node_id}"

    def _next_runtime_action_id(self) -> str:
        self._runtime_action_id += 1
        return f"a{self._runtime_action_id}"

    def _emit_runtime_node(self, *, role: str, content: str) -> str:
        node_id = self._next_runtime_node_id()
        parent_node_id = self._runtime_leaf_node_id
        self._runtime_leaf_node_id = node_id
        self._emit(
            {
                "event": "NODE_CREATED",
                "created_node_id": node_id,
                "created_parent_node_id": parent_node_id,
                "created_branch_id": self._runtime_branch_id,
                "role": role,
                "content": content,
                "timestamp": time.time(),
            }
        )
        return node_id

    def _emit_step_start(self, *, label: str, kind: str, tool: str | None = None, arguments: dict[str, Any] | None = None) -> int:
        step_id = self._next_step_id()
        payload: dict[str, Any] = {
            "event": "STEP_START",
            "step_id": step_id,
            "label": label,
            "kind": kind,
            "timestamp": time.time(),
        }
        if tool:
            payload["tool"] = tool
        if arguments is not None:
            payload["arguments"] = arguments
        self._emit(payload)
        return step_id

    def _emit_step_end(self, step_id: int, *, status: str = "ok", error: str | None = None) -> None:
        payload: dict[str, Any] = {
            "event": "STEP_END",
            "step_id": step_id,
            "status": status,
            "timestamp": time.time(),
        }
        if error:
            payload["error"] = error
        self._emit(payload)

    # ------------------------------------------------------------------
    # FSM state transitions (Phase 3)
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: Any) -> None:
        """
        Enforce authoritative FSM transition. Raises RuntimeError on illegal
        transitions. Appends ledger entry and emits state digest.
        """
        from engine.contract import FSM_TRANSITIONS, RuntimeState

        if self._fsm_state is None:
            # First transition: entering PRECHECK
            self._fsm_state = new_state
            self._ledger.append(
                actor="system",
                event_type="state_transition",
                payload={
                    "fsm_transition": f"INIT -> {new_state.value}",
                    "contract_hash": self._contract.contract_hash if self._contract else "",
                },
            )
            self._chain_append_transition("INIT", new_state.value)
            self._emit_state_digest()
            return

        allowed = FSM_TRANSITIONS.get(self._fsm_state, frozenset())
        if new_state not in allowed:
            raise RuntimeError(
                f"Illegal FSM transition: {self._fsm_state.value} -> {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        old_state = self._fsm_state
        self._fsm_state = new_state
        self._ledger.append(
            actor="system",
            event_type="state_transition",
            payload={
                "fsm_transition": f"{old_state.value} -> {new_state.value}",
                "contract_hash": self._contract.contract_hash if self._contract else "",
            },
        )
        self._chain_append_transition(old_state.value, new_state.value)
        self._emit_state_digest()

    def _emit_state_digest(self) -> None:
        """Build and emit a StateDigest snapshot for UI consumption."""
        from engine.contract import StateDigest

        contract = self._contract
        max_inf = contract.max_inferences if contract else MAX_AGENT_STEPS
        max_tok = contract.max_tokens_consumed if contract else 0
        total_timeout = contract.total_timeout_ms if contract else 0
        elapsed = (time.time() - self._run_start_time) * 1000.0 if self._run_start_time else 0.0

        digest = StateDigest(
            fsm_state=self._fsm_state.value if self._fsm_state else "",
            inferences_used=self._inferences,
            inferences_remaining=max(0, max_inf - self._inferences),
            tokens_consumed=self._tokens_consumed,
            tokens_remaining=max(0, max_tok - self._tokens_consumed) if max_tok > 0 else 0,
            tools_executed=self._tools_executed,
            format_retries_used=self._format_retries,
            elapsed_ms=elapsed,
            timeout_remaining_ms=max(0.0, total_timeout - elapsed) if total_timeout > 0 else 0.0,
            context_ratio=self._estimated_context_ratio,
            force_synthesis_pending=self._force_terminate_after_next_inference,
            last_outcome=self._last_outcome.value if self._last_outcome else None,
            contract_id=contract.contract_id if contract else "",
        )
        self._emit({
            "event": "STATE_DIGEST",
            "digest": digest.to_dict(),
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------
    # Transcript chain helpers (Phase 5)
    # ------------------------------------------------------------------

    def _chain_append_transition(self, from_state: str, to_state: str) -> None:
        """Append an FSM transition entry to the transcript chain."""
        if self._transcript_chain is None:
            return
        action_hash = hashlib.sha256(
            f"transition:{from_state}->{to_state}".encode()
        ).hexdigest()
        empty_hash = hashlib.sha256(b"").hexdigest()
        self._transcript_chain.append(
            state=to_state,
            action_hash=action_hash,
            result_hash=empty_hash,
        )

    def _chain_append(self, *, state: str, action_hash: str, result_hash: str) -> None:
        """Append a generic entry to the transcript chain."""
        if self._transcript_chain is None:
            return
        self._transcript_chain.append(
            state=state,
            action_hash=action_hash,
            result_hash=result_hash,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _serialize(self, msg: AgentMessage) -> dict[str, Any]:
        data: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            data["content"] = msg.content
        if msg.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in msg.tool_calls
            ]
        if msg.tool_call_id:
            data["tool_call_id"] = msg.tool_call_id
        if msg.name:
            data["name"] = msg.name
        return data

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_call(self, call: ToolCall) -> dict[str, Any]:
        tool_gate = self._capability_manager.validate_tool_name(call.name)
        if not tool_gate.ok:
            return {"ok": False, "kind": "manifest_violation", "message": tool_gate.error}

        schema = TOOL_ARGUMENT_SCHEMAS.get(call.name, {"required": {}, "optional": {}})
        required = schema.get("required", {})
        optional = schema.get("optional", {})
        allowed_keys = set(required.keys()) | set(optional.keys())

        for key in call.arguments.keys():
            if key not in allowed_keys:
                return {"ok": False, "kind": "schema_violation", "message": f"Unexpected argument '{key}'."}

        for key, expected in required.items():
            if key not in call.arguments:
                return {"ok": False, "kind": "schema_violation", "message": f"Missing required argument '{key}'."}
            if not isinstance(call.arguments[key], expected):
                return {
                    "ok": False,
                    "kind": "schema_violation",
                    "message": f"Argument '{key}' must be {expected.__name__}.",
                }

        for key, expected in optional.items():
            if key in call.arguments and not isinstance(call.arguments[key], expected):
                return {
                    "ok": False,
                    "kind": "schema_violation",
                    "message": f"Argument '{key}' must be {expected.__name__}.",
                }

        policy_gate = self._capability_manager.authorize(tool=call.name, path=extract_tool_path(call.name, call.arguments))
        if not policy_gate.ok:
            return {"ok": False, "kind": "policy_violation", "message": policy_gate.error}

        return {"ok": True}

    def _validate_tool_envelope(self, envelope: dict[str, Any]) -> bool:
        """Validate the structure of a tool result envelope."""
        if not isinstance(envelope, dict):
            return False
        required_keys = {"status", "tool", "tool_call_id"}
        if not all(k in envelope for k in required_keys):
            return False
        if envelope.get("status") not in ("ok", "error"):
            return False
        return True

    def _virtual_tool_error(self, call: ToolCall, kind: str, message: str) -> AgentMessage:
        payload = {
            "status": "error",
            "kind": kind,
            "message": message,
            "retryable": True,
        }
        return AgentMessage(
            role="tool",
            name="system_validator",
            tool_call_id=call.id,
            content=json.dumps(payload),
        )

    def _execute_tool(self, call: ToolCall) -> AgentMessage:
        contract = self._contract
        step_timeout = contract.step_timeout_ms if contract else 0

        envelope: dict[str, Any] = {
            "status": "ok",
            "tool": call.name,
            "tool_call_id": call.id,
            "result": None,
            "error": None,
            "retryable": False,
        }
        try:
            result = self._bridge.execute(call.name, call.arguments, step_timeout_ms=step_timeout)

            # Step timeout handling
            if result.get("timeout"):
                envelope["status"] = "error"
                envelope["error"] = result.get("error", f"step_timeout_ms exceeded ({step_timeout}ms)")
                envelope["timeout"] = True
                envelope["retryable"] = False
                return AgentMessage(
                    role="tool",
                    name=call.name,
                    tool_call_id=call.id,
                    content=json.dumps(envelope),
                )

            envelope["result"] = result

            # Tool output budget enforcement
            if contract and contract.tool_output_budget:
                budget = contract.tool_output_budget
                result_str = json.dumps(result, default=str)
                result_bytes = len(result_str.encode("utf-8"))
                if result_bytes > budget.max_bytes_per_call:
                    truncated = result_str[:budget.max_bytes_per_call]
                    envelope["result"] = {
                        "ok": result.get("ok", True) if isinstance(result, dict) else True,
                        "content": truncated,
                        "truncated": True,
                        "original_bytes": result_bytes,
                        "marker": budget.truncation_marker,
                    }
                    self._ledger.append(
                        actor="system",
                        event_type="error",
                        payload={
                            "kind": "tool_output_truncated",
                            "tool": call.name,
                            "original_bytes": result_bytes,
                            "max_bytes": budget.max_bytes_per_call,
                        },
                    )

            if isinstance(result, dict) and not result.get("ok", False):
                envelope["status"] = "error"
                envelope["error"] = result.get("error")
                envelope["retryable"] = True
        except Exception as exc:
            envelope["status"] = "error"
            envelope["error"] = str(exc)
            envelope["retryable"] = True

        return AgentMessage(
            role="tool",
            name=call.name,
            tool_call_id=call.id,
            content=json.dumps(envelope),
        )

    # ------------------------------------------------------------------
    # Protocol adapter integration
    # ------------------------------------------------------------------

    def _adapt_response(self, raw_response: dict[str, Any]) -> AgentMessage:
        """
        Run raw LLM response through the protocol adapter.

        If adapter is available, uses it for native/recovered/rejected semantics
        with transcript logging. Falls back to direct AgentMessage construction
        if no adapter is set (backward compat).
        """
        if self._protocol_adapter is None:
            return raw_response  # type: ignore[return-value]

        from engine.protocol_adapter import AdapterStatus, ProtocolAdapterResult

        adapter_result: ProtocolAdapterResult = self._protocol_adapter.adapt(raw_response)

        self._ledger.append(
            actor="system",
            event_type="inference",
            payload={
                "adapter_status": adapter_result.status.value,
                "adapter_version": adapter_result.adapter_version,
                "raw_hash": adapter_result.raw_hash,
                "failure_code": adapter_result.failure_code,
                "recovery_detail": adapter_result.recovery_detail,
            },
        )

        self._emit({
            "event": "PROTOCOL_ADAPTER",
            "status": adapter_result.status.value,
            "raw_hash": adapter_result.raw_hash,
            "failure_code": adapter_result.failure_code,
            "recovery_detail": adapter_result.recovery_detail,
            "timestamp": time.time(),
        })

        self._last_adapter_status = adapter_result.status.value

        # Phase 5: feed adapter result into transcript chain
        reasoning_hash = hashlib.sha256(
            (adapter_result.canonical_message.content or "").encode()
        ).hexdigest() if adapter_result.canonical_message else hashlib.sha256(b"").hexdigest()
        self._chain_append(
            state="INFER",
            action_hash=adapter_result.raw_hash,
            result_hash=reasoning_hash,
        )

        if adapter_result.status == AdapterStatus.REJECTED:
            return AgentMessage(
                role="assistant",
                content=f"[PROTOCOL_ERROR: {adapter_result.failure_code}]",
            )

        if adapter_result.status == AdapterStatus.RECOVERED:
            self._emit({
                "event": "PROTOCOL_RECOVERY",
                "detail": adapter_result.recovery_detail,
                "timestamp": time.time(),
            })

        assert adapter_result.canonical_message is not None
        return adapter_result.canonical_message

    # ------------------------------------------------------------------
    # COMMIT transition predicates (V2 contract Section 5)
    # ------------------------------------------------------------------

    def _evaluate_commit(
        self,
        *,
        inferences: int,
        tools_executed: int,
        format_retries: int,
        tokens_consumed: int,
        run_start_time: float,
        history: list[AgentMessage],
        assistant: AgentMessage,
    ) -> dict[str, Any]:
        """
        Deterministic COMMIT transition predicates.

        Returns {"action": "terminate", "outcome": ..., "reason": ...}
        or     {"action": "continue"}
        or     {"action": "force_synthesis", "reason": ...}

        Order (per contract):
          1. Interrupt/timeout
          2. Budget exhaustion (inferences, tokens, time)
          3. Contract immutability check
          4. Context ratio / forced synthesis
          5. Policy violations (tool_policy enforcement)
          6. Format retry exhaustion
          7. Normal success checks
          8. Continue loop
        """
        from engine.contract import AgentOutcome, ToolPolicy

        contract = self._contract
        max_inferences = contract.max_inferences if contract else MAX_AGENT_STEPS
        tool_policy = contract.tool_policy if contract else ToolPolicy.OPTIONAL

        # 1. Interrupt
        if self._should_stop():
            return {"action": "terminate", "outcome": AgentOutcome.INTERRUPTED, "reason": "user_interrupt"}

        # 2a. Inference budget exhaustion
        if inferences >= max_inferences:
            return {"action": "terminate", "outcome": AgentOutcome.FAILED_BUDGET_EXHAUSTED, "reason": "max_inferences_exceeded"}

        # 2b. Token budget exhaustion
        if contract and contract.max_tokens_consumed > 0:
            if tokens_consumed >= contract.max_tokens_consumed:
                return {"action": "terminate", "outcome": AgentOutcome.FAILED_BUDGET_EXHAUSTED,
                        "reason": f"max_tokens_consumed exceeded: {tokens_consumed}/{contract.max_tokens_consumed}"}

        # 2c. Total timeout
        if contract and contract.total_timeout_ms > 0:
            elapsed_ms = (time.time() - run_start_time) * 1000.0
            if elapsed_ms >= contract.total_timeout_ms:
                return {"action": "terminate", "outcome": AgentOutcome.FAILED_TIMEOUT,
                        "reason": f"total_timeout_ms exceeded: {elapsed_ms:.0f}/{contract.total_timeout_ms}"}

        # 3. Contract immutability check
        if contract is not None and self._initial_contract_hash is not None:
            if contract.contract_hash != self._initial_contract_hash:
                return {"action": "terminate", "outcome": AgentOutcome.FAILED_CONTRACT_VIOLATION,
                        "reason": "contract_hash changed during execution (Invariant B violated)"}

        # 4. Context ratio / forced synthesis
        if contract and contract.context_budget.force_synthesis_at_ratio > 0:
            cb = contract.context_budget
            history_bytes = sum(len(json.dumps(self._serialize(m), default=str).encode("utf-8")) for m in history)
            estimated_tokens = history_bytes // 4
            available = cb.context_window - cb.reserved_system
            ratio = estimated_tokens / available if available > 0 else 1.0
            self._estimated_context_ratio = ratio

            if ratio >= cb.force_synthesis_at_ratio:
                return {"action": "force_synthesis",
                        "reason": f"context ratio {ratio:.2f} >= {cb.force_synthesis_at_ratio}"}

        # 5. No tool calls from this inference — evaluate policy
        if not assistant.tool_calls:
            if tool_policy == ToolPolicy.REQUIRED and tools_executed == 0:
                return {"action": "terminate", "outcome": AgentOutcome.FAILED_PROTOCOL_NO_TOOLS, "reason": "tool_policy_required_but_no_tools"}

            if tool_policy == ToolPolicy.FORBIDDEN:
                return {"action": "terminate", "outcome": AgentOutcome.COMPLETED_CHAT_ONLY, "reason": "completed_forbidden_policy"}

            if tools_executed > 0:
                return {"action": "terminate", "outcome": AgentOutcome.COMPLETED_WITH_TOOLS, "reason": "completed"}
            else:
                return {"action": "terminate", "outcome": AgentOutcome.COMPLETED_CHAT_ONLY, "reason": "completed_no_tools"}

        # 6. Has tool calls — continue loop
        return {"action": "continue"}

    # ------------------------------------------------------------------
    # PRECHECK — preflight validation (V2 contract Section 4)
    # ------------------------------------------------------------------

    def _preflight_check(self) -> dict[str, Any] | None:
        """
        Preflight validation before entering the main loop.

        Returns None if all checks pass, or a dict with
        {"outcome": AgentOutcome, "reason": str} on failure.
        """
        from engine.contract import AgentOutcome, ToolPolicy
        from engine.protocol_adapter import get_profile

        contract = self._contract
        if contract is None:
            return None

        if contract.max_inferences < 1:
            return {"outcome": AgentOutcome.FAILED_PREFLIGHT, "reason": "contract.max_inferences < 1"}
        if contract.max_format_retries < 0:
            return {"outcome": AgentOutcome.FAILED_PREFLIGHT, "reason": "contract.max_format_retries < 0"}

        if contract.allowed_tools is not None:
            available_tools = set(self._capability_manager.allowed_tools())
            contract_tools = set(contract.allowed_tools)
            missing = contract_tools - available_tools
            if missing and contract.tool_policy == ToolPolicy.REQUIRED:
                return {
                    "outcome": AgentOutcome.FAILED_PREFLIGHT,
                    "reason": f"contract requires tools not in capability manifest: {sorted(missing)}",
                }

        profile = get_profile(contract.model_profile_id)
        if contract.tool_policy == ToolPolicy.REQUIRED and profile.strict_mode and not profile.supports_native_tools:
            return {
                "outcome": AgentOutcome.FAILED_PREFLIGHT,
                "reason": f"tool_policy=required but profile '{profile.profile_id}' is strict with no native tool support",
            }

        cb = contract.context_budget
        minimum_loop_margin = 256
        required = cb.reserved_system + cb.reserved_synthesis + minimum_loop_margin
        if required > cb.context_window:
            return {
                "outcome": AgentOutcome.FAILED_PREFLIGHT,
                "reason": f"context infeasible: reserved({required}) > window({cb.context_window})",
            }

        if self._llm_call is None:
            return {"outcome": AgentOutcome.FAILED_PREFLIGHT, "reason": "llm_call not set"}

        # Phase 5: version compatibility check
        try:
            from engine.migration import check_compatibility
            adapter_ver = self._protocol_adapter.VERSION if self._protocol_adapter else "2a.1"
            contract_ver = getattr(contract, "contract_format_version", "3.0")
            compat, reason = check_compatibility(contract_ver, adapter_ver)
            if not compat:
                return {"outcome": AgentOutcome.FAILED_PREFLIGHT, "reason": reason}
        except ImportError:
            pass

        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, messages: list[dict[str, Any]]) -> "AgentRunResult":
        """
        Main agent execution loop.

        Returns AgentRunResult with typed outcome.
        """
        from engine.contract import (
            AgentOutcome, AgentRunResult, RuntimeState, RunSummary, ToolPolicy,
        )

        self._event_step_id = 0
        self._runtime_node_id = 0
        self._runtime_action_id = 0
        self._runtime_leaf_node_id = None

        # Reset FSM and counters
        self._fsm_state = None
        self._inferences = 0
        self._tokens_consumed = 0
        self._tools_executed = 0
        self._format_retries = 0
        self._run_start_time = time.time()
        self._force_terminate_after_next_inference = False
        self._last_outcome = None
        self._estimated_context_ratio = 0.0
        _tool_signature_set: set[str] = set()

        contract = self._contract
        max_inferences = contract.max_inferences if contract else MAX_AGENT_STEPS
        tool_policy = contract.tool_policy if contract else ToolPolicy.OPTIONAL
        max_format_retries = contract.max_format_retries if contract else 1
        cycle_forbid = set(contract.cycle_forbid) if contract and contract.cycle_forbid else set()
        run_id = str(uuid.uuid4())

        # Store initial contract hash for immutability check
        self._initial_contract_hash = contract.contract_hash if contract else None

        # Error flags
        had_protocol_error = False
        had_validation_error = False
        had_cycle_violation = False

        last_tool_name: str | None = None

        # ---- LOAD HISTORY ----
        history: list[AgentMessage] = []
        for msg in messages:
            role = msg.get("role")
            if role in {"system", "user", "assistant", "tool"}:
                history.append(AgentMessage(role=role, content=msg.get("content")))
                if role == "user":
                    self._ledger.append(actor="user", event_type="input", payload={"content": msg.get("content", "")})
                    self._emit_runtime_node(role="user", content=str(msg.get("content") or ""))

        # ---- CAPABILITY QUERY ----
        tools = self._capability_manager.tool_schemas()
        self._ledger.append(
            actor="system",
            event_type="input",
            payload={
                "capability_profile": self._capability_manager.profile,
                "capability_digest": self._capability_manager.capability_digest,
                "tools": self._capability_manager.allowed_tools(),
            },
        )

        # ---- CONTRACT LOGGING ----
        if contract is not None:
            self._ledger.append(
                actor="system",
                event_type="input",
                payload={"contract": contract.to_dict()},
            )
            self._emit({
                "event": "CONTRACT_BOUND",
                "contract": contract.to_dict(),
                "timestamp": time.time(),
            })

        # ---- TRANSCRIPT CHAIN (Phase 5) ----
        from core.event_ledger import TranscriptChain
        if contract is not None:
            self._transcript_chain = TranscriptChain(
                contract_hash=contract.contract_hash,
                adapter_version=contract.adapter_version,
                model_profile_id=contract.model_profile_id,
                model_fingerprint=contract.model_fingerprint,
            )
        else:
            self._transcript_chain = None

        # ---- TERMINATION HELPER ----
        def _terminate(outcome: AgentOutcome, reason: str, output: str = "") -> AgentRunResult:
            self._last_outcome = outcome
            run_end_time = time.time()
            self._ledger.append(
                actor="system",
                event_type="telemetry",
                payload={"reason": reason, "outcome": outcome.value, "tools_executed": self._tools_executed, "inferences": self._inferences},
            )
            term_step = self._emit_step_start(label="Termination", kind="termination")
            self._emit({
                "event": "TERMINATION",
                "step_id": term_step,
                "reason": reason,
                "outcome": outcome.value,
                "output": output if output else None,
                "timestamp": run_end_time,
            })
            if outcome.is_success:
                self._emit_step_end(term_step, status="ok")
                if output:
                    self._emit({"event": "FINAL_OUTPUT", "data": output})
            else:
                self._emit_step_end(term_step, status="error", error=reason)

            summary = RunSummary(
                contract_id=contract.contract_id if contract else "",
                run_id=run_id,
                termination_reason=reason,
                outcome=outcome.value,
                llm_calls=self._inferences,
                tool_calls=self._tools_executed,
                format_retries=self._format_retries,
                steps_used=self._event_step_id,
                max_inferences=max_inferences,
                budget_remaining=max(0, max_inferences - self._inferences),
                tokens_consumed=self._tokens_consumed,
                elapsed_ms=(run_end_time - self._run_start_time) * 1000.0,
                start_time=self._run_start_time,
                end_time=run_end_time,
                had_protocol_error=had_protocol_error,
                had_validation_error=had_validation_error,
                had_cycle_violation=had_cycle_violation,
                had_budget_exhaustion=(outcome == AgentOutcome.FAILED_BUDGET_EXHAUSTED),
                model_profile_id=contract.model_profile_id if contract else "",
                unique_tool_signatures=len(_tool_signature_set),
                total_tool_invocations=self._tools_executed,
                transcript_chain_head=self._transcript_chain.head_hash if self._transcript_chain else "",
                transcript_chain_length=self._transcript_chain.length if self._transcript_chain else 0,
            )
            self._emit({
                "event": "RUN_SUMMARY",
                "summary": summary.to_dict(),
                "timestamp": run_end_time,
            })
            self._ledger.append(
                actor="system",
                event_type="telemetry",
                payload=summary.to_dict(),
            )

            return AgentRunResult(
                outcome=outcome,
                output=output,
                history=[self._serialize(m) for m in history],
                contract=contract,
                steps_used=self._event_step_id,
                inferences_used=self._inferences,
                tools_executed=self._tools_executed,
                format_retries_used=self._format_retries,
                tokens_consumed=self._tokens_consumed,
                termination_reason=reason,
            )

        # ---- PRECHECK ----
        self._transition_to(RuntimeState.PRECHECK)

        preflight_failure = self._preflight_check()
        if preflight_failure is not None:
            self._emit({
                "event": "PREFLIGHT_FAILURE",
                "reason": preflight_failure["reason"],
                "timestamp": time.time(),
            })
            self._transition_to(RuntimeState.TERMINATE)
            return _terminate(preflight_failure["outcome"], preflight_failure["reason"])

        # ---- MAIN LOOP ----
        self._transition_to(RuntimeState.INFER)

        while True:
            # -- Pre-inference checks --
            if self._should_stop():
                self._transition_to(RuntimeState.TERMINATE)
                return _terminate(AgentOutcome.INTERRUPTED, "user_interrupt")

            if self._inferences >= max_inferences:
                self._transition_to(RuntimeState.TERMINATE)
                return _terminate(AgentOutcome.FAILED_BUDGET_EXHAUSTED, "max_inferences_exceeded")

            # -- INFER --
            llm_step = self._emit_step_start(label="LLM call", kind="llm_call")
            normalized_history = [self._serialize(m) for m in history]

            if self._protocol_adapter is not None:
                raw_response = self._llm_call(normalized_history, tools)
                # Token consumption tracking
                if isinstance(raw_response, dict):
                    usage = raw_response.get("usage", {})
                    if isinstance(usage, dict):
                        self._tokens_consumed += usage.get("total_tokens", 0)
                assistant = self._adapt_response(raw_response)
                if self._last_adapter_status in ("rejected", "recovered"):
                    had_protocol_error = True
            else:
                assistant = self._llm_call(normalized_history, tools)

            self._inferences += 1

            self._ledger.append(
                actor="assistant",
                event_type="inference",
                payload=self._serialize(assistant),
                reasoning=assistant.content,
            )
            history.append(assistant)

            if assistant.content:
                self._emit({"event": "AGENT_THOUGHT", "step_id": llm_step, "thought": assistant.content, "timestamp": time.time()})
            self._emit({"event": "AGENT_MESSAGE", "message": self._serialize(assistant), "timestamp": time.time()})
            self._emit_runtime_node(role="assistant", content=str(assistant.content or ""))
            self._emit_step_end(llm_step, status="ok")

            # -- Forced synthesis check (post-inference) --
            if self._force_terminate_after_next_inference and not assistant.tool_calls:
                output = assistant.content or ""
                outcome = AgentOutcome.COMPLETED_WITH_TOOLS if self._tools_executed > 0 else AgentOutcome.COMPLETED_CHAT_ONLY
                self._transition_to(RuntimeState.COMMIT)
                self._transition_to(RuntimeState.TERMINATE)
                return _terminate(outcome, "forced_synthesis_complete", output)

            # -- No tool calls path --
            if not assistant.tool_calls:
                is_protocol_error = (
                    isinstance(assistant.content, str)
                    and assistant.content.startswith("[PROTOCOL_ERROR:")
                )

                if is_protocol_error and self._format_retries < max_format_retries:
                    had_protocol_error = True
                    self._format_retries += 1
                    self._emit({
                        "event": "PARSE_ERROR",
                        "retry": self._format_retries,
                        "max_retries": max_format_retries,
                        "timestamp": time.time(),
                    })
                    self._ledger.append(
                        actor="system",
                        event_type="error",
                        payload={"kind": "format_retry", "retry": self._format_retries, "content": assistant.content},
                    )
                    history.pop()
                    # Stay in INFER state (retry)
                    continue

                if is_protocol_error and self._format_retries >= max_format_retries:
                    had_protocol_error = True
                    self._transition_to(RuntimeState.TERMINATE)
                    return _terminate(AgentOutcome.FAILED_PROTOCOL_MALFORMED, "format_retries_exhausted")

                # Tool-required nudge
                if (
                    tool_policy == ToolPolicy.REQUIRED
                    and self._tools_executed == 0
                    and self._format_retries < max_format_retries
                ):
                    self._format_retries += 1
                    self._emit({
                        "event": "TOOL_REQUIRED_NUDGE",
                        "retry": self._format_retries,
                        "max_retries": max_format_retries,
                        "timestamp": time.time(),
                    })
                    self._ledger.append(
                        actor="system",
                        event_type="error",
                        payload={
                            "kind": "tool_required_nudge",
                            "retry": self._format_retries,
                            "message": "Model answered with prose but tool_policy=required. Injecting nudge.",
                        },
                    )
                    nudge = AgentMessage(
                        role="system",
                        content=(
                            "You MUST use a tool to complete this task. "
                            "Do NOT answer with prose or code blocks. "
                            "Use the write_file tool to create the file. "
                            "Output a tool call now."
                        ),
                    )
                    history.append(nudge)
                    self._emit({
                        "event": "AGENT_NUDGE",
                        "step_id": self._next_step_id(),
                        "content": nudge.content,
                        "timestamp": time.time(),
                    })
                    # Stay in INFER state (retry)
                    continue

                # Normal no-tool-calls completion
                output = assistant.content or ""
                if not output.strip():
                    for msg in reversed(history):
                        if msg.role == "assistant" and msg.content and msg.content.strip():
                            if msg.content.startswith("[PROTOCOL_ERROR:"):
                                continue
                            output = msg.content.strip()
                            break

                if not output.strip() and self._tools_executed > 0:
                    output = f"[Agent completed — {self._tools_executed} tool(s) executed]"

                self._transition_to(RuntimeState.COMMIT)

                commit = self._evaluate_commit(
                    inferences=self._inferences,
                    tools_executed=self._tools_executed,
                    format_retries=self._format_retries,
                    tokens_consumed=self._tokens_consumed,
                    run_start_time=self._run_start_time,
                    history=history,
                    assistant=assistant,
                )

                if commit["action"] == "force_synthesis":
                    # Inject synthesis prompt
                    synthesis_prompt = AgentMessage(
                        role="system",
                        content=(
                            "You have used most of the available context window. "
                            "Provide a final synthesis of your work so far. "
                            "Summarize what was accomplished and any remaining steps."
                        ),
                    )
                    history.append(synthesis_prompt)
                    self._ledger.append(
                        actor="system",
                        event_type="state_transition",
                        payload={"kind": "forced_synthesis", "reason": commit["reason"]},
                    )
                    self._emit({"event": "FORCED_SYNTHESIS", "reason": commit["reason"], "timestamp": time.time()})
                    self._force_terminate_after_next_inference = True
                    self._transition_to(RuntimeState.INFER)
                    continue

                self._transition_to(RuntimeState.TERMINATE)
                return _terminate(commit["outcome"], commit["reason"], output)

            # -- VALIDATE_CALLS --
            self._transition_to(RuntimeState.VALIDATE_CALLS)

            # -- EXECUTE --
            self._transition_to(RuntimeState.EXECUTE)

            completed_in_batch = 0
            for call in assistant.tool_calls:
                # Cycle detection
                if last_tool_name is not None and (last_tool_name, call.name) in cycle_forbid:
                    had_cycle_violation = True
                    # Record partial execution
                    if completed_in_batch > 0:
                        self._ledger.append(
                            actor="system",
                            event_type="error",
                            payload={
                                "kind": "partial_batch_execution",
                                "completed_tools": completed_in_batch,
                                "total_tools_in_batch": len(assistant.tool_calls),
                                "failed_tool": call.name,
                                "failure_reason": f"cycle_forbid_violation: {last_tool_name} -> {call.name}",
                            },
                        )
                    self._transition_to(RuntimeState.TERMINATE)
                    return _terminate(
                        AgentOutcome.FAILED_CONTRACT_VIOLATION,
                        f"cycle_forbid_violation: {last_tool_name} -> {call.name}",
                    )

                call_step = self._emit_step_start(label=f"Tool call: {call.name}", kind="tool_call", tool=call.name, arguments=call.arguments)
                action = {
                    "action_id": self._next_runtime_action_id(),
                    "tool": call.name,
                    "arguments": dict(call.arguments),
                    "status": "pending",
                    "step_id": call_step,
                }
                self._emit({"event": "ACTION_QUEUED", "action": dict(action), "timestamp": time.time()})
                self._ledger.append(
                    actor="assistant",
                    event_type="tool_invocation",
                    payload={"tool_name": call.name, "arguments": call.arguments, "tool_call_id": call.id},
                    execution={"tool_name": call.name, "arguments": call.arguments},
                )
                self._emit({
                    "event": "TOOL_CALL_START",
                    "step_id": call_step,
                    "tool": call.name,
                    "arguments": call.arguments,
                    "tool_call_id": call.id,
                    "timestamp": time.time(),
                })
                action["status"] = "running"
                self._emit({"event": "ACTION_STARTED", "action": dict(action), "timestamp": time.time()})

                # Tool policy: FORBIDDEN
                if tool_policy == ToolPolicy.FORBIDDEN:
                    observation = self._virtual_tool_error(call, "policy_violation", "tool_policy is FORBIDDEN; tool calls not allowed")
                else:
                    validation = self._validate_call(call)
                    if not validation.get("ok", False):
                        had_validation_error = True
                        observation = self._virtual_tool_error(call, validation.get("kind", "validation_error"), validation.get("message", "validation failed"))
                    else:
                        observation = self._execute_tool(call)

                        # Check for step timeout in the result
                        if observation.content:
                            try:
                                obs_parsed = json.loads(observation.content)
                                if isinstance(obs_parsed, dict) and obs_parsed.get("timeout"):
                                    # Record partial execution
                                    if completed_in_batch > 0:
                                        self._ledger.append(
                                            actor="system",
                                            event_type="error",
                                            payload={
                                                "kind": "partial_batch_execution",
                                                "completed_tools": completed_in_batch,
                                                "total_tools_in_batch": len(assistant.tool_calls),
                                                "failed_tool": call.name,
                                                "failure_reason": f"step_timeout on {call.name}",
                                            },
                                        )
                                    # Record the observation in history before terminating
                                    history.append(observation)
                                    self._transition_to(RuntimeState.TERMINATE)
                                    return _terminate(AgentOutcome.FAILED_TIMEOUT, f"step_timeout on tool {call.name}")
                            except (json.JSONDecodeError, TypeError):
                                pass

                        self._tools_executed += 1
                        sig = hashlib.sha256(
                            f"{call.name}:{json.dumps(call.arguments, sort_keys=True, separators=(',', ':'))}".encode()
                        ).hexdigest()
                        _tool_signature_set.add(sig)

                        # Phase 5: feed tool execution into transcript chain
                        tool_result_hash = hashlib.sha256(
                            (observation.content or "").encode()
                        ).hexdigest()
                        self._chain_append(
                            state="EXECUTE",
                            action_hash=sig,
                            result_hash=tool_result_hash,
                        )

                observation_payload: dict[str, Any] = {}
                if isinstance(observation.content, str):
                    try:
                        parsed = json.loads(observation.content)
                        if isinstance(parsed, dict):
                            observation_payload = parsed
                    except Exception:
                        observation_payload = {"raw": observation.content}

                # Malformed envelope detection
                if observation_payload and not self._validate_tool_envelope(observation_payload):
                    had_validation_error = True
                    self._ledger.append(
                        actor="system",
                        event_type="error",
                        payload={
                            "kind": "malformed_tool_envelope",
                            "tool": call.name,
                            "envelope_keys": list(observation_payload.keys()),
                        },
                    )
                    if completed_in_batch > 0:
                        self._ledger.append(
                            actor="system",
                            event_type="error",
                            payload={
                                "kind": "partial_batch_execution",
                                "completed_tools": completed_in_batch,
                                "total_tools_in_batch": len(assistant.tool_calls),
                                "failed_tool": call.name,
                                "failure_reason": "malformed_tool_envelope",
                            },
                        )
                    history.append(observation)
                    self._transition_to(RuntimeState.TERMINATE)
                    return _terminate(AgentOutcome.FAILED_VALIDATION, f"malformed_tool_result_envelope from {call.name}")

                history.append(observation)
                self._ledger.append(
                    actor="tool",
                    event_type="tool_result",
                    payload=self._serialize(observation),
                    execution={"tool_name": call.name, "arguments": call.arguments},
                )

                call_status = "ok" if observation_payload.get("status") != "error" else "error"
                call_error = observation_payload.get("error")
                if not isinstance(call_error, str):
                    call_error = None
                action["status"] = "done" if call_status == "ok" else "error"
                action["result"] = observation_payload if observation_payload else self._serialize(observation)
                self._emit({"event": "ACTION_FINISHED", "action": dict(action), "timestamp": time.time()})
                self._emit_step_end(call_step, status=call_status, error=call_error)

                result_step = self._emit_step_start(label=f"Tool result: {call.name}", kind="tool_result", tool=call.name)
                self._emit({
                    "event": "TOOL_RESULT",
                    "step_id": result_step,
                    "tool": call.name,
                    "result": observation_payload if observation_payload else self._serialize(observation),
                    "timestamp": time.time(),
                })
                self._emit_step_end(result_step, status=call_status, error=call_error)
                self._emit({"event": "TOOL_OBSERVATION", "message": self._serialize(observation), "timestamp": time.time()})
                self._emit_runtime_node(role="tool", content=observation.content or "")

                last_tool_name = call.name
                completed_in_batch += 1

            # -- OBSERVE --
            self._transition_to(RuntimeState.OBSERVE)

            # -- COMMIT (post-tool) --
            self._transition_to(RuntimeState.COMMIT)

            # Check budget after tool execution
            commit = self._evaluate_commit(
                inferences=self._inferences,
                tools_executed=self._tools_executed,
                format_retries=self._format_retries,
                tokens_consumed=self._tokens_consumed,
                run_start_time=self._run_start_time,
                history=history,
                assistant=assistant,
            )
            if commit["action"] == "terminate":
                self._transition_to(RuntimeState.TERMINATE)
                return _terminate(commit["outcome"], commit["reason"])

            if commit["action"] == "force_synthesis":
                synthesis_prompt = AgentMessage(
                    role="system",
                    content=(
                        "You have used most of the available context window. "
                        "Provide a final synthesis of your work so far. "
                        "Summarize what was accomplished and any remaining steps."
                    ),
                )
                history.append(synthesis_prompt)
                self._ledger.append(
                    actor="system",
                    event_type="state_transition",
                    payload={"kind": "forced_synthesis", "reason": commit["reason"]},
                )
                self._emit({"event": "FORCED_SYNTHESIS", "reason": commit["reason"], "timestamp": time.time()})
                self._force_terminate_after_next_inference = True
                self._transition_to(RuntimeState.INFER)
                continue

            # Continue loop
            self._transition_to(RuntimeState.INFER)

    # ------------------------------------------------------------------
    # Runtime commands
    # ------------------------------------------------------------------

    def runtime_command(self, command: str, payload: dict | None = None) -> dict:
        if command == "ledger":
            return {"ok": True, "ledger": [event.__dict__ for event in self._ledger.snapshot()]}
        if command == "capability_manifest":
            return {
                "ok": True,
                "profile": self._capability_manager.profile,
                "tools": self._capability_manager.allowed_tools(),
                "capability_digest": self._capability_manager.capability_digest,
            }
        if command == "contract":
            if self._contract is not None:
                return {"ok": True, "contract": self._contract.to_dict()}
            return {"ok": True, "contract": None}
        return {"ok": False, "error": f"unknown runtime action: {command}"}
