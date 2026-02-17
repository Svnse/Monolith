from __future__ import annotations

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
        envelope: dict[str, Any] = {
            "status": "ok",
            "tool": call.name,
            "tool_call_id": call.id,
            "result": None,
            "error": None,
            "retryable": False,
        }
        try:
            result = self._bridge.execute(call.name, call.arguments)
            envelope["result"] = result
            if not result.get("ok", False):
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

    def run(self, messages: list[dict[str, Any]]) -> tuple[bool, str, list[dict[str, Any]]]:
        self._event_step_id = 0
        self._runtime_node_id = 0
        self._runtime_action_id = 0
        self._runtime_leaf_node_id = None
        history: list[AgentMessage] = []
        for msg in messages:
            role = msg.get("role")
            if role in {"system", "user", "assistant", "tool"}:
                history.append(AgentMessage(role=role, content=msg.get("content")))
                if role == "user":
                    self._ledger.append(actor="user", event_type="input", payload={"content": msg.get("content", "")})
                    self._emit_runtime_node(role="user", content=str(msg.get("content") or ""))

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

        steps = 0
        while True:
            if self._should_stop():
                self._ledger.append(actor="system", event_type="yield", payload={"reason": "user_interrupt"})
                term_step = self._emit_step_start(label="Termination", kind="termination")
                self._emit({"event": "TERMINATION", "step_id": term_step, "reason": "user_interrupt", "timestamp": time.time()})
                self._emit_step_end(term_step, status="error", error="Generation interrupted by user")
                return False, "", [self._serialize(m) for m in history]
            if steps >= MAX_AGENT_STEPS:
                self._ledger.append(actor="system", event_type="yield", payload={"reason": "max_steps_exceeded"})
                term_step = self._emit_step_start(label="Termination", kind="termination")
                self._emit({"event": "TERMINATION", "step_id": term_step, "reason": "max_steps_exceeded", "timestamp": time.time()})
                self._emit_step_end(term_step, status="error", error="Maximum agent step count exceeded")
                return False, "max_steps exceeded", [self._serialize(m) for m in history]

            # INFERENCE
            llm_step = self._emit_step_start(label="LLM call", kind="llm_call")
            normalized_history = [self._serialize(m) for m in history]
            assistant = self._llm_call(normalized_history, tools)
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

            # ROUTER
            if not assistant.tool_calls:
                output = assistant.content or ""
                self._ledger.append(actor="assistant", event_type="yield", payload={"content": output}, reasoning=output)
                term_step = self._emit_step_start(label="Termination", kind="termination")
                self._emit({"event": "TERMINATION", "step_id": term_step, "reason": "completed", "output": output, "timestamp": time.time()})
                self._emit_step_end(term_step, status="ok")
                self._emit({"event": "FINAL_OUTPUT", "data": output})
                return True, output, [self._serialize(m) for m in history]

            # VALIDATE -> EXECUTE -> OBSERVE
            for call in assistant.tool_calls:
                steps += 1
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

                validation = self._validate_call(call)
                if not validation.get("ok", False):
                    observation = self._virtual_tool_error(call, validation.get("kind", "validation_error"), validation.get("message", "validation failed"))
                else:
                    observation = self._execute_tool(call)

                observation_payload: dict[str, Any] = {}
                if isinstance(observation.content, str):
                    try:
                        parsed = json.loads(observation.content)
                        if isinstance(parsed, dict):
                            observation_payload = parsed
                    except Exception:
                        observation_payload = {"raw": observation.content}

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
        return {"ok": False, "error": f"unknown runtime action: {command}"}
