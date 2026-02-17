from __future__ import annotations

import json
import re
import time
from enum import Enum
from typing import Callable

from engine.agent_bridge import AgentBridge

MAX_AGENT_STEPS = 25
MAX_AGENT_TIMEOUT = 120
MAX_PARSE_RETRIES = 2

FLEXIBLE_OUTPUT_PROMPT = (
    "You may answer normally in freeform text. "
    "If you want a tool, include exactly one JSON object with shape "
    '{"tool": "<registered_tool_name>", "arguments": {...}}.'
)

PROTOCOL_BLOCK_PATTERN = re.compile(
    r"<(REASONING|ACTION|CAPABILITY_REQUEST)>(.*?)</\\1>",
    flags=re.DOTALL,
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
        loop_messages.append({"role": "system", "content": FLEXIBLE_OUTPUT_PROMPT})
        self._state = AgentState.IDLE
        self._step_id = 0
        started = time.monotonic()
        steps = 0
        parse_retries = 0

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
            self._emit_event(
                {
                    "event": "AGENT_THOUGHT",
                    "step_id": thinking_step_id,
                    "thought": raw,
                    "timestamp": time.time(),
                }
            )

            blocks = self._extract_protocol_blocks(raw)
            protocol_actions: list[dict] = []
            protocol_compliant = True
            protocol_error: str | None = None
            for block in blocks:
                if block["type"] == "REASONING":
                    self._emit_event(
                        {
                            "event": "AGENT_REASONING",
                            "step_id": thinking_step_id,
                            "reasoning": block["content"],
                            "timestamp": time.time(),
                        }
                    )
                    continue

                if block["type"] == "CAPABILITY_REQUEST":
                    self._emit_event(
                        {
                            "event": "CAPABILITY_REQUEST",
                            "step_id": thinking_step_id,
                            "request": block["content"],
                            "timestamp": time.time(),
                        }
                    )
                    continue

                try:
                    action_obj = json.loads(block["content"])
                except Exception:
                    protocol_error = "invalid <ACTION> JSON block: expected {'tool': <str>, 'arguments': <object>}"
                    break

                if (
                    not isinstance(action_obj, dict)
                    or "tool" not in action_obj
                    or "arguments" not in action_obj
                    or not isinstance(action_obj["tool"], str)
                    or not isinstance(action_obj["arguments"], dict)
                ):
                    protocol_error = "invalid <ACTION> JSON block: expected {'tool': <str>, 'arguments': <object>}"
                    break

                protocol_actions.append({"tool": action_obj["tool"], "arguments": action_obj["arguments"]})

            if protocol_error is not None:
                parse_retries += 1
                self._emit_event(
                    {
                        "event": "PARSE_ERROR",
                        "step_id": thinking_step_id,
                        "error": protocol_error,
                        "retry": parse_retries,
                        "timestamp": time.time(),
                    }
                )
                self._emit_step_end(thinking_step_id, "error", error=protocol_error, protocol_compliant=protocol_compliant)
                if parse_retries > MAX_PARSE_RETRIES:
                    return self._error(protocol_error, loop_messages)
                loop_messages.append(
                    {
                        "role": "system",
                        "content": "Parse error: put executable tool calls only in <ACTION>{\"tool\":\"...\",\"arguments\":{...}}</ACTION> blocks.",
                    }
                )
                continue

            actions_to_execute = protocol_actions
            if not actions_to_execute:
                parsed = self._extract_first_json_block(raw)
                if parsed is not None:
                    protocol_compliant = False
                    actions_to_execute = [
                        {
                            "tool": parsed["tool"],
                            "arguments": parsed["arguments"],
                            "synthetic": True,
                        }
                    ]
                    self._emit_event(
                        {
                            "event": "PROTOCOL_COMPLIANCE_WARNING",
                            "step_id": thinking_step_id,
                            "warning": "No <ACTION> block found. Falling back to legacy first-JSON extraction.",
                            "timestamp": time.time(),
                        }
                    )

            if not actions_to_execute:
                if self._contains_json_candidate(raw):
                    parse_retries += 1
                    error = "invalid tool JSON block: expected {'tool': <str>, 'arguments': <object>}"
                    self._emit_event(
                        {
                            "event": "PARSE_ERROR",
                            "step_id": thinking_step_id,
                            "error": error,
                            "retry": parse_retries,
                            "timestamp": time.time(),
                        }
                    )
                    self._emit_step_end(thinking_step_id, "error", error=error, protocol_compliant=protocol_compliant)
                    if parse_retries > MAX_PARSE_RETRIES:
                        return self._error(error, loop_messages)
                    loop_messages.append(
                        {
                            "role": "system",
                            "content": "Parse error: if you need a tool, include one valid JSON object with tool and arguments keys.",
                        }
                    )
                    continue

                self._state = AgentState.FINAL
                self._emit_step_end(thinking_step_id, "ok", protocol_compliant=protocol_compliant)
                loop_messages.append({"role": "assistant", "content": raw})
                final_step_id = self._next_step_id()
                self._emit_step_start(final_step_id, "Final Output", "final")
                self._emit_event({"event": "FINAL_OUTPUT", "data": raw, "step_id": final_step_id, "timestamp": time.time()})
                self._emit_step_end(final_step_id, "ok")
                return True, raw, loop_messages

            parse_retries = 0
            self._state = AgentState.TOOL_CALL
            self._emit_step_end(thinking_step_id, "ok", protocol_compliant=protocol_compliant)
            loop_messages.append({"role": "assistant", "content": raw})
            for action in actions_to_execute:
                tool = action["tool"]
                arguments = action["arguments"]
                tool_step_id = self._next_step_id()
                self._emit_step_start(tool_step_id, f"Tool: {tool}", "tool", tool=tool, arguments=arguments)
                self._emit_event(
                    {
                        "event": "TOOL_CALL_START",
                        "step_id": tool_step_id,
                        "tool": tool,
                        "arguments": arguments,
                        "synthetic": action.get("synthetic", False),
                        "timestamp": time.time(),
                    }
                )
                bridge_result = self._bridge.execute(tool, arguments)
                self._state = AgentState.TOOL_RESULT
                result_payload = bridge_result.get("result", {"ok": False, "content": "", "error": bridge_result.get("error")})
                self._emit_event(
                    {
                        "event": "TOOL_RESULT",
                        "step_id": tool_step_id,
                        "tool": tool,
                        "result": result_payload,
                        "timestamp": time.time(),
                    }
                )
                self._emit_step_end(tool_step_id, "ok" if result_payload.get("ok", False) else "error")

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

    def _contains_json_candidate(self, text: str) -> bool:
        return bool(re.search(r"\{.*?\}", text, flags=re.DOTALL))

    def _extract_protocol_blocks(self, text: str) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = []
        for match in PROTOCOL_BLOCK_PATTERN.finditer(text):
            blocks.append({"type": match.group(1), "content": match.group(2).strip()})
        return blocks

    def _extract_first_json_block(self, text: str) -> dict | None:
        starts = [i for i, ch in enumerate(text) if ch == "{"]
        for start in starts:
            depth = 0
            for idx in range(start, len(text)):
                ch = text[idx]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        block = text[start : idx + 1]
                        try:
                            obj = json.loads(block)
                        except Exception:
                            break
                        if (
                            isinstance(obj, dict)
                            and "tool" in obj
                            and "arguments" in obj
                            and isinstance(obj["tool"], str)
                            and isinstance(obj["arguments"], dict)
                        ):
                            return {"tool": obj["tool"], "arguments": obj["arguments"]}
                        break
        return None
