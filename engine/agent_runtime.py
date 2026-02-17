from __future__ import annotations

import json
import time
from enum import Enum
from typing import Callable

from engine.agent_bridge import AgentBridge

MAX_AGENT_STEPS = 25
MAX_AGENT_TIMEOUT = 120
MAX_INVALID_RETRIES = 2

STRUCTURED_OUTPUT_PROMPT = (
    "Return ONLY valid JSON. "
    "If you need a tool, return {\"tool\": \"<registered_tool_name>\", \"arguments\": {...}, \"thought\": \"<optional summary>\"}. "
    "If done, return {\"final\": \"<text>\", \"thought\": \"<optional summary>\"}. "
    "Any other shape is invalid."
)


class AgentState(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    FINAL = "FINAL"
    ERROR = "ERROR"


class AgentRuntime:
    def __init__(
        self,
        llm_call: Callable[[list[dict]], str],
        bridge: AgentBridge | None = None,
        emit_event: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ):
        self._llm_call = llm_call
        self._bridge = bridge or AgentBridge()
        self._emit_event = emit_event or (lambda _: None)
        self._should_stop = should_stop or (lambda: False)
        self._state = AgentState.IDLE
        self._step_id = 0

    def _next_step_id(self) -> int:
        self._step_id += 1
        return self._step_id

    def _emit_step_start(self, step_id: int, label: str, kind: str, **extra) -> None:
        payload = {
            "event": "STEP_START",
            "step_id": step_id,
            "label": label,
            "kind": kind,
            "timestamp": time.time(),
        }
        payload.update(extra)
        self._emit_event(payload)

    def _emit_step_end(self, step_id: int, status: str, **extra) -> None:
        payload = {
            "event": "STEP_END",
            "step_id": step_id,
            "status": status,
            "timestamp": time.time(),
        }
        payload.update(extra)
        self._emit_event(payload)

    def run(self, messages: list[dict]) -> tuple[bool, str, list[dict]]:
        loop_messages = list(messages)
        loop_messages.append({"role": "system", "content": STRUCTURED_OUTPUT_PROMPT})
        self._state = AgentState.IDLE
        self._step_id = 0
        started = time.monotonic()
        steps = 0
        invalid_retries = 0

        while True:
            if self._should_stop():
                return False, "", loop_messages
            if time.monotonic() - started > MAX_AGENT_TIMEOUT:
                return self._error("agent timeout reached", loop_messages)
            if steps >= MAX_AGENT_STEPS:
                return self._error("agent step limit reached", loop_messages)

            self._state = AgentState.THINKING
            thinking_step_id = self._next_step_id()
            self._emit_step_start(thinking_step_id, "Thinking", "llm")
            self._emit_event({"event": "LLM_THINKING_START", "step_id": thinking_step_id, "timestamp": time.time()})
            raw = self._llm_call(loop_messages)
            self._emit_event({"event": "LLM_TOKEN", "data": raw, "step_id": thinking_step_id, "timestamp": time.time()})

            parsed = self._parse_structured(raw)
            if "error" in parsed:
                invalid_retries += 1
                self._emit_event(
                    {
                        "event": "PARSE_INVALID",
                        "step_id": thinking_step_id,
                        "error": parsed["error"],
                        "retry": invalid_retries,
                        "timestamp": time.time(),
                    }
                )
                self._emit_step_end(thinking_step_id, "error", error=parsed["error"])
                if invalid_retries > MAX_INVALID_RETRIES:
                    return self._error(parsed["error"], loop_messages)
                loop_messages.append(
                    {
                        "role": "system",
                        "content": "Previous response invalid. Reply with ONLY one JSON object following the contract.",
                    }
                )
                continue

            invalid_retries = 0
            thought = parsed.get("thought")
            if isinstance(thought, str) and thought.strip():
                self._emit_event(
                    {
                        "event": "AGENT_THOUGHT",
                        "step_id": thinking_step_id,
                        "thought": thought.strip(),
                        "timestamp": time.time(),
                    }
                )
            if "final" in parsed:
                final_text = parsed["final"]
                self._state = AgentState.FINAL
                self._emit_step_end(thinking_step_id, "ok")
                loop_messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
                final_step_id = self._next_step_id()
                self._emit_step_start(final_step_id, "Final Output", "final")
                self._emit_event({"event": "FINAL_OUTPUT", "data": final_text, "step_id": final_step_id, "timestamp": time.time()})
                self._emit_step_end(final_step_id, "ok")
                return True, final_text, loop_messages

            tool = parsed["tool"]
            arguments = parsed["arguments"]
            self._state = AgentState.TOOL_CALL
            self._emit_step_end(thinking_step_id, "ok")
            tool_step_id = self._next_step_id()
            self._emit_step_start(tool_step_id, f"Tool: {tool}", "tool", tool=tool, arguments=arguments)
            self._emit_event({"event": "TOOL_CALL_START", "step_id": tool_step_id, "tool": tool, "arguments": arguments, "timestamp": time.time()})
            bridge_result = self._bridge.execute(tool, arguments)
            self._state = AgentState.TOOL_RESULT
            result_payload = bridge_result.get("result", {"ok": False, "content": "", "error": bridge_result.get("error")})
            self._emit_event({"event": "TOOL_RESULT", "step_id": tool_step_id, "tool": tool, "result": result_payload, "timestamp": time.time()})
            self._emit_step_end(tool_step_id, "ok" if result_payload.get("ok", False) else "error")

            loop_messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            loop_messages.append(
                {
                    "role": "tool",
                    "name": tool,
                    "content": json.dumps(result_payload, ensure_ascii=False),
                }
            )
            steps += 1

    def _error(self, message: str, loop_messages: list[dict]) -> tuple[bool, str, list[dict]]:
        self._state = AgentState.ERROR
        self._emit_event({"event": "FINAL_OUTPUT", "data": ""})
        return False, message, loop_messages

    def _parse_structured(self, raw: str) -> dict:
        try:
            obj = json.loads(raw)
        except Exception:
            return {"error": "invalid structured output: must be valid JSON object"}

        if not isinstance(obj, dict):
            return {"error": "invalid structured output: must be a JSON object"}

        thought = obj.get("thought")
        if thought is not None and not isinstance(thought, str):
            return {"error": "invalid structured output: thought must be a string"}

        allowed = {"tool", "arguments", "final", "thought"}
        keys = set(obj.keys())
        if not keys.issubset(allowed):
            return {"error": "invalid structured output: unknown keys"}

        if "final" in obj and isinstance(obj.get("final"), str) and keys.issubset({"final", "thought"}):
            payload = {"final": obj["final"]}
            if isinstance(thought, str):
                payload["thought"] = thought
            return payload
        if "tool" in obj and "arguments" in obj and isinstance(obj.get("tool"), str) and isinstance(obj.get("arguments"), dict) and keys.issubset({"tool", "arguments", "thought"}):
            payload = {"tool": obj["tool"], "arguments": obj["arguments"]}
            if isinstance(thought, str):
                payload["thought"] = thought
            return payload

        return {"error": "invalid structured output: expected {tool,arguments,thought?} or {final,thought?}"}
