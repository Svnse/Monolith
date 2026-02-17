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
    "If you need a tool, return {\"tool\": \"<registered_tool_name>\", \"arguments\": {...}}. "
    "If done, return {\"final\": \"<text>\"}. "
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

    def run(self, messages: list[dict]) -> tuple[bool, str, list[dict]]:
        loop_messages = list(messages)
        loop_messages.append({"role": "system", "content": STRUCTURED_OUTPUT_PROMPT})
        self._state = AgentState.IDLE
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
            self._emit_event({"event": "LLM_THINKING_START"})
            raw = self._llm_call(loop_messages)
            self._emit_event({"event": "LLM_TOKEN", "data": raw})

            parsed = self._parse_structured(raw)
            if "error" in parsed:
                invalid_retries += 1
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
            if "final" in parsed:
                final_text = parsed["final"]
                self._state = AgentState.FINAL
                loop_messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
                self._emit_event({"event": "FINAL_OUTPUT", "data": final_text})
                return True, final_text, loop_messages

            tool = parsed["tool"]
            arguments = parsed["arguments"]
            self._state = AgentState.TOOL_CALL
            self._emit_event({"event": "TOOL_CALL_START", "tool": tool})
            bridge_result = self._bridge.execute(tool, arguments)
            self._state = AgentState.TOOL_RESULT
            result_payload = bridge_result.get("result", {"ok": False, "content": "", "error": bridge_result.get("error")})
            self._emit_event({"event": "TOOL_RESULT", "tool": tool, "result": result_payload})

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

        keys = set(obj.keys())
        if keys == {"final"} and isinstance(obj.get("final"), str):
            return {"final": obj["final"]}
        if keys == {"tool", "arguments"} and isinstance(obj.get("tool"), str) and isinstance(obj.get("arguments"), dict):
            return {"tool": obj["tool"], "arguments": obj["arguments"]}

        return {"error": "invalid structured output: expected {tool,arguments} or {final}"}
