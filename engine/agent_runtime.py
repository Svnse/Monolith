from __future__ import annotations

import json
import re
import time
from enum import Enum
from typing import Callable

from engine.agent_bridge import AgentBridge
from engine.capabilities import CapabilityManager, CapabilityScope
from engine.checkpointing import get_checkpoint_store
from engine.execution_tree import ExecutionTree
from engine.pty_runtime import get_pty_session_manager
from engine.tools import WORKSPACE_ROOT

MAX_AGENT_STEPS = 25
MAX_AGENT_TIMEOUT = 120
MAX_PARSE_RETRIES = 2
CAPABILITY_DECISION_TIMEOUT = 300

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
        capability_manager: CapabilityManager | None = None,
        emit_event: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ):
        self._llm_call = llm_call
        self._capability_manager = capability_manager or CapabilityManager()
        self._bridge = bridge or AgentBridge(capability_manager=self._capability_manager)
        self._emit_event = emit_event or (lambda _: None)
        self._should_stop = should_stop or (lambda: False)
        self._state = AgentState.IDLE
        self._step_id = 0
        self._tree = ExecutionTree()
        self._active_branch_id = "main"
        self._active_leaf_node_id: str | None = None
        self._capability_decisions: dict[str, dict] = {}

    def _next_step_id(self) -> int:
        self._step_id += 1
        return self._step_id

    def _lineage_payload(self, node_id: str | None = None) -> dict:
        effective_node_id = node_id or self._active_leaf_node_id
        if effective_node_id is None:
            return {"branch_id": self._active_branch_id, "node_id": None, "parent_node_id": None}
        node = self._tree.get_node(effective_node_id)
        return {"branch_id": node.branch_id, "node_id": node.node_id, "parent_node_id": node.parent_node_id}

    def _emit(self, payload: dict, node_id: str | None = None) -> None:
        event = dict(payload)
        event.update(self._lineage_payload(node_id=node_id))
        self._emit_event(event)

    def _append_node(self, role: str, content: str, **extra):
        node = self._tree.append_node(self._active_branch_id, role, content, **extra)
        self._active_leaf_node_id = node.node_id
        return node

    def fork(self, node_id: str) -> str:
        parent_branch_id = self._tree.get_node(node_id).branch_id
        branch = self._tree.fork(node_id)
        self._capability_manager.fork_branch(parent_branch_id, branch.branch_id)
        try:
            get_pty_session_manager(workspace_root=WORKSPACE_ROOT).fork_branch(parent_branch_id, branch.branch_id)
        except Exception:
            pass
        return branch.branch_id

    def resume(self, leaf_node_id: str) -> None:
        self._active_leaf_node_id = leaf_node_id
        self._active_branch_id = self._tree.branch_from_leaf(leaf_node_id)

    def prune(self, branch_id: str) -> None:
        self._tree.prune(branch_id)
        try:
            get_pty_session_manager(workspace_root=WORKSPACE_ROOT).destroy_branch(branch_id)
        except Exception:
            pass
        try:
            get_checkpoint_store().mark_branch_pruned(branch_id)
        except Exception:
            pass

    def compare(self, branch_a: str, branch_b: str) -> dict:
        return self._tree.compare(branch_a, branch_b)

    def apply_capability_decision(
        self,
        request_id: str,
        *,
        approved: bool,
        scope: str | None = None,
        path_pattern: str = "**",
        ttl_seconds: int | None = None,
        constraints: dict | None = None,
        reason: str | None = None,
    ) -> None:
        self._capability_decisions[request_id] = {
            "approved": approved,
            "scope": scope,
            "path_pattern": path_pattern,
            "ttl_seconds": ttl_seconds,
            "constraints": constraints or {},
            "reason": reason,
        }

    def revoke_capability(self, token_id: str, branch_id: str | None = None) -> bool:
        target_branch = branch_id or self._active_branch_id
        revoked = self._capability_manager.revoke(branch_id=target_branch, token_id=token_id)
        self._emit(
            {
                "event": "CAPABILITY_REVOKED",
                "token_id": token_id,
                "revoked": revoked,
                "timestamp": time.time(),
            }
        )
        return revoked

    def _emit_step_start(self, step_id: int, label: str, kind: str, **extra) -> None:
        payload = {
            "event": "STEP_START",
            "step_id": step_id,
            "label": label,
            "kind": kind,
            "timestamp": time.time(),
        }
        payload.update(extra)
        self._emit(payload)

    def _emit_step_end(self, step_id: int, status: str, **extra) -> None:
        payload = {
            "event": "STEP_END",
            "step_id": step_id,
            "status": status,
            "timestamp": time.time(),
        }
        payload.update(extra)
        self._emit(payload)

    def run(self, messages: list[dict]) -> tuple[bool, str, list[dict]]:
        self._tree.init_from_messages(messages)
        if self._active_leaf_node_id is None:
            self._active_leaf_node_id = self._tree.branch_leaf(self._active_branch_id)

        prompt_node = self._append_node("system", FLEXIBLE_OUTPUT_PROMPT, compliance={"prompt_injected": True})
        loop_messages = self._tree.build_messages_to_node(prompt_node.node_id)

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
            self._emit({"event": "LLM_THINKING_START", "step_id": thinking_step_id, "timestamp": time.time()})
            raw = self._llm_call(loop_messages)
            self._emit({"event": "LLM_TOKEN", "data": raw, "step_id": thinking_step_id, "timestamp": time.time()})
            self._emit(
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
                    self._emit(
                        {
                            "event": "AGENT_REASONING",
                            "step_id": thinking_step_id,
                            "reasoning": block["content"],
                            "timestamp": time.time(),
                        }
                    )
                    continue

                if block["type"] == "CAPABILITY_REQUEST":
                    capability_result = self._handle_capability_request(
                        block["content"],
                        step_id=thinking_step_id,
                        loop_messages=loop_messages,
                    )
                    if not capability_result["ok"]:
                        protocol_error = capability_result["error"]
                        protocol_compliant = False
                        break
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
                self._emit(
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
                retry_node = self._append_node(
                    "system",
                    "Parse error: put executable tool calls only in <ACTION>{\"tool\":\"...\",\"arguments\":{...}}</ACTION> blocks.",
                    compliance={"parse_retry": parse_retries},
                )
                loop_messages = self._tree.build_messages_to_node(retry_node.node_id)
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
                    self._emit(
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
                    self._emit(
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
                    retry_node = self._append_node(
                        "system",
                        "Parse error: if you need a tool, include one valid JSON object with tool and arguments keys.",
                        compliance={"parse_retry": parse_retries},
                    )
                    loop_messages = self._tree.build_messages_to_node(retry_node.node_id)
                    continue

                self._state = AgentState.FINAL
                self._emit_step_end(thinking_step_id, "ok", protocol_compliant=protocol_compliant)
                assistant_node = self._append_node(
                    "assistant",
                    raw,
                    reasoning=raw,
                    compliance={"protocol_compliant": protocol_compliant, "terminal": True},
                )
                loop_messages = self._tree.build_messages_to_node(assistant_node.node_id)
                final_step_id = self._next_step_id()
                self._emit_step_start(final_step_id, "Final Output", "final")
                self._emit({"event": "FINAL_OUTPUT", "data": raw, "step_id": final_step_id, "timestamp": time.time()})
                self._emit_step_end(final_step_id, "ok")
                return True, raw, loop_messages

            parse_retries = 0
            self._state = AgentState.TOOL_CALL
            self._emit_step_end(thinking_step_id, "ok", protocol_compliant=protocol_compliant)
            assistant_node = self._append_node(
                "assistant",
                raw,
                reasoning=raw,
                compliance={"protocol_compliant": protocol_compliant, "terminal": False},
            )
            loop_messages = self._tree.build_messages_to_node(assistant_node.node_id)
            for action in actions_to_execute:
                tool = action["tool"]
                arguments = action["arguments"]
                tool_step_id = self._next_step_id()
                self._checkpoint_before_tool(tool, arguments, loop_messages)
                self._emit_step_start(tool_step_id, f"Tool: {tool}", "tool", tool=tool, arguments=arguments)
                self._emit(
                    {
                        "event": "TOOL_CALL_START",
                        "step_id": tool_step_id,
                        "tool": tool,
                        "arguments": arguments,
                        "synthetic": action.get("synthetic", False),
                        "timestamp": time.time(),
                    }
                )
                bridge_result = self._bridge.execute(tool, arguments, branch_id=self._active_branch_id)
                self._state = AgentState.TOOL_RESULT
                result_payload = bridge_result.get("result", {"ok": False, "content": "", "error": bridge_result.get("error")})
                self._emit(
                    {
                        "event": "TOOL_RESULT",
                        "step_id": tool_step_id,
                        "tool": tool,
                        "result": result_payload,
                        "timestamp": time.time(),
                    }
                )
                self._emit_step_end(tool_step_id, "ok" if result_payload.get("ok", False) else "error")

                tool_node = self._append_node(
                    "tool",
                    json.dumps(result_payload, ensure_ascii=False),
                    action={"name": tool, "arguments": arguments},
                    result=result_payload,
                    compliance={"tool_success": result_payload.get("ok", False)},
                )
                loop_messages = self._tree.build_messages_to_node(tool_node.node_id)
                steps += 1

    def _handle_capability_request(self, content: str, *, step_id: int, loop_messages: list[dict]) -> dict:
        parsed = self._parse_capability_request(content)
        if not parsed["ok"]:
            return parsed

        request = parsed["request"]
        request_id = request["request_id"]
        self._create_checkpoint(
            pending_action={"type": "capability_escalation", "request": request},
            capabilities={"requested": request},
            message_history=loop_messages,
        )
        self._emit(
            {
                "event": "CAPABILITY_REQUEST",
                "step_id": step_id,
                "request_id": request_id,
                "request": request,
                "timestamp": time.time(),
            }
        )

        started = time.monotonic()
        while request_id not in self._capability_decisions:
            if self._should_stop():
                return {"ok": False, "error": "capability request interrupted"}
            if time.monotonic() - started > CAPABILITY_DECISION_TIMEOUT:
                return {"ok": False, "error": "capability decision timeout"}
            time.sleep(0.1)

        decision = self._capability_decisions.pop(request_id)
        if not decision.get("approved", False):
            self._emit(
                {
                    "event": "CAPABILITY_DECISION",
                    "step_id": step_id,
                    "request_id": request_id,
                    "approved": False,
                    "reason": decision.get("reason"),
                    "timestamp": time.time(),
                }
            )
            return {"ok": True}

        scope_value = decision.get("scope") or request["scope"]
        try:
            scope = CapabilityScope(scope_value)
        except Exception:
            return {"ok": False, "error": f"invalid capability scope: {scope_value}"}

        token = self._capability_manager.issue_token(
            branch_id=self._active_branch_id,
            scope=scope,
            path_pattern=str(decision.get("path_pattern") or request.get("path_pattern") or "**"),
            ttl_seconds=decision.get("ttl_seconds") if decision.get("ttl_seconds") is not None else request.get("ttl_seconds"),
            constraints=decision.get("constraints") if isinstance(decision.get("constraints"), dict) else request.get("constraints", {}),
        )
        self._emit(
            {
                "event": "CAPABILITY_DECISION",
                "step_id": step_id,
                "request_id": request_id,
                "approved": True,
                "token_id": token.token_id,
                "scope": token.scope.value,
                "path_pattern": token.path_pattern,
                "expires_at": token.expires_at,
                "timestamp": time.time(),
            }
        )
        return {"ok": True}

    def _parse_capability_request(self, content: str) -> dict:
        request_id = f"req_{int(time.time() * 1000)}_{self._step_id}"
        default_request = {
            "request_id": request_id,
            "scope": CapabilityScope.READ.value,
            "path_pattern": "**",
            "ttl_seconds": 300,
            "constraints": {},
            "reason": content,
        }
        try:
            payload = json.loads(content)
        except Exception:
            return {"ok": True, "request": default_request}

        if not isinstance(payload, dict):
            return {"ok": False, "error": "invalid <CAPABILITY_REQUEST> payload: expected JSON object"}

        scope_value = payload.get("scope", CapabilityScope.READ.value)
        try:
            scope = CapabilityScope(scope_value)
        except Exception:
            return {"ok": False, "error": f"invalid capability scope: {scope_value}"}

        request = {
            "request_id": str(payload.get("request_id") or request_id),
            "scope": scope.value,
            "path_pattern": str(payload.get("path_pattern", "**")),
            "ttl_seconds": payload.get("ttl_seconds", 300),
            "constraints": payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {},
            "reason": str(payload.get("reason", "")),
        }
        return {"ok": True, "request": request}

    def _error(self, message: str, loop_messages: list[dict]) -> tuple[bool, str, list[dict]]:
        self._state = AgentState.ERROR
        self._emit({"event": "FINAL_OUTPUT", "data": ""})
        return False, message, loop_messages

    def _checkpoint_before_tool(self, tool: str, arguments: dict, message_history: list[dict]) -> None:
        if tool not in {"write_file", "apply_patch", "run_cmd", "delete_file", "move_file"}:
            return
        capabilities = arguments.get("capabilities") if isinstance(arguments, dict) else {}
        pty_state_ref = arguments.get("pty_state_ref") if isinstance(arguments, dict) else None
        self._create_checkpoint(
            pending_action={"type": "tool_call", "tool": tool, "arguments": arguments},
            capabilities=capabilities if isinstance(capabilities, dict) else {},
            pty_state_ref=pty_state_ref if isinstance(pty_state_ref, str) else None,
            message_history=message_history,
        )

        if isinstance(arguments, dict):
            checkpoint_context = {
                "branch_id": self._active_branch_id,
                "node_id": self._active_leaf_node_id,
                "message_history": message_history,
            }
            if isinstance(capabilities, dict):
                checkpoint_context["capabilities"] = capabilities
            if isinstance(pty_state_ref, str):
                checkpoint_context["pty_state_ref"] = pty_state_ref
            arguments["_checkpoint"] = checkpoint_context

    def _create_checkpoint(
        self,
        *,
        pending_action: dict,
        capabilities: dict[str, str] | dict[str, object] | None = None,
        pty_state_ref: str | None = None,
        message_history: list[dict] | None = None,
    ) -> None:
        try:
            get_checkpoint_store().create_checkpoint(
                branch_id=self._active_branch_id,
                node_id=self._active_leaf_node_id,
                message_history=message_history or [],
                pending_action=pending_action,
                capabilities=capabilities or {},
                pty_state_ref=pty_state_ref,
            )
        except Exception:
            return

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
