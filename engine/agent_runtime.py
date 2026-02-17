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

    def _emit(self, payload: dict[str, Any]) -> None:
        self._emit_event(payload)

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
        history: list[AgentMessage] = []
        for msg in messages:
            role = msg.get("role")
            if role in {"system", "user", "assistant", "tool"}:
                history.append(AgentMessage(role=role, content=msg.get("content")))
                if role == "user":
                    self._ledger.append(actor="user", event_type="input", payload={"content": msg.get("content", "")})

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
                return False, "", [self._serialize(m) for m in history]
            if steps >= MAX_AGENT_STEPS:
                self._ledger.append(actor="system", event_type="yield", payload={"reason": "max_steps_exceeded"})
                return False, "max_steps exceeded", [self._serialize(m) for m in history]

            # INFERENCE
            normalized_history = [self._serialize(m) for m in history]
            assistant = self._llm_call(normalized_history, tools)
            self._ledger.append(
                actor="assistant",
                event_type="inference",
                payload=self._serialize(assistant),
                reasoning=assistant.content,
            )
            history.append(assistant)
            self._emit({"event": "AGENT_MESSAGE", "message": self._serialize(assistant), "timestamp": time.time()})

            # ROUTER
            if not assistant.tool_calls:
                output = assistant.content or ""
                self._ledger.append(actor="assistant", event_type="yield", payload={"content": output}, reasoning=output)
                self._emit({"event": "FINAL_OUTPUT", "data": output})
                return True, output, [self._serialize(m) for m in history]

            # VALIDATE -> EXECUTE -> OBSERVE
            for call in assistant.tool_calls:
                steps += 1
                self._ledger.append(
                    actor="assistant",
                    event_type="tool_invocation",
                    payload={"tool_name": call.name, "arguments": call.arguments, "tool_call_id": call.id},
                    execution={"tool_name": call.name, "arguments": call.arguments},
                )

                validation = self._validate_call(call)
                if not validation.get("ok", False):
                    observation = self._virtual_tool_error(call, validation.get("kind", "validation_error"), validation.get("message", "validation failed"))
                else:
                    observation = self._execute_tool(call)

                history.append(observation)
                self._ledger.append(
                    actor="tool",
                    event_type="tool_result",
                    payload=self._serialize(observation),
                    execution={"tool_name": call.name, "arguments": call.arguments},
                )
                self._emit({"event": "TOOL_OBSERVATION", "message": self._serialize(observation), "timestamp": time.time()})

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
