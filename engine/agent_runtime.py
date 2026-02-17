from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
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
PROTOCOL_COMPLIANCE_THRESHOLD = float(os.getenv("MONOLITH_PROTOCOL_COMPLIANCE_THRESHOLD", "0.8"))

V4_PROTOCOL_PROMPT = (
    "Respond only using the v4 protocol blocks. "
    "Put private reasoning in <REASONING>...</REASONING>. "
    "Put each tool call in <ACTION>{\"tool\":\"<registered_tool_name>\",\"arguments\":{...}}</ACTION>. "
    "Put capability escalations in <CAPABILITY_REQUEST>{...}</CAPABILITY_REQUEST>. "
    "Raw JSON outside these blocks is non-compliant. "
    "If no tool is needed, return a final user-facing response and do not emit <ACTION>."
)

PROTOCOL_BLOCK_PATTERN = re.compile(
    r"<(REASONING|ACTION|CAPABILITY_REQUEST)>(.*?)</\1>",
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
        self._tree_path = Path(WORKSPACE_ROOT) / ".monolith" / "execution_tree.json"
        self._active_branch_id = "main"
        self._active_leaf_node_id: str | None = None
        self._capability_decisions: dict[str, dict] = {}
        self._capability_waiters: dict[str, threading.Event] = {}
        self._pending_actions: list[dict] = []
        self._action_counter = 0
        try:
            loaded = ExecutionTree.load(self._tree_path)
            self._tree = loaded
            self._active_leaf_node_id = self._tree.branch_leaf(self._active_branch_id)
        except Exception:
            pass


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

    def _save_tree(self) -> None:
        try:
            self._tree.save(self._tree_path)
        except Exception:
            return

    def _append_node(self, role: str, content: str, **extra):
        payload = dict(extra)
        node = self._tree.append_node(self._active_branch_id, role, content, **payload)
        self._active_leaf_node_id = node.node_id
        self._emit(
            {
                "event": "NODE_CREATED",
                "created_node_id": node.node_id,
                "created_parent_node_id": node.parent_node_id,
                "created_branch_id": node.branch_id,
                "role": role,
                "content": content,
                "timestamp": time.time(),
            },
            node_id=node.node_id,
        )
        self._save_tree()
        return node

    def fork(self, node_id: str) -> str:
        parent_branch_id = self._tree.get_node(node_id).branch_id
        branch = self._tree.fork(node_id)
        self._capability_manager.fork_branch(parent_branch_id, branch.branch_id)
        try:
            get_pty_session_manager(workspace_root=WORKSPACE_ROOT).fork_branch(parent_branch_id, branch.branch_id)
        except Exception:
            pass
        self._emit(
            {
                "event": "BRANCH_FORKED",
                "source_node_id": node_id,
                "parent_branch_id": parent_branch_id,
                "branch_id": branch.branch_id,
                "timestamp": time.time(),
            },
            node_id=node_id,
        )
        self._save_tree()
        return branch.branch_id

    def resume(self, leaf_node_id: str) -> None:
        self._active_leaf_node_id = leaf_node_id
        self._active_branch_id = self._tree.branch_from_leaf(leaf_node_id)
        self._emit(
            {
                "event": "BRANCH_RESUMED",
                "branch_id": self._active_branch_id,
                "node_id": leaf_node_id,
                "timestamp": time.time(),
            },
            node_id=leaf_node_id,
        )
        self._save_tree()

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
        self._emit(
            {
                "event": "BRANCH_PRUNED",
                "branch_id": branch_id,
                "timestamp": time.time(),
            }
        )
        self._save_tree()

    def compare(self, branch_a: str, branch_b: str) -> dict:
        result = self._tree.compare(branch_a, branch_b)
        self._emit({"event": "BRANCH_COMPARED", "comparison": result, "timestamp": time.time()})
        return result

    def queue_snapshot(self) -> list[dict]:
        return [dict(item) for item in self._pending_actions]

    def action_queue_update(self, payload: dict) -> list[dict]:
        action = payload.get("op") if isinstance(payload, dict) else None
        if action == "cancel":
            action_id = str(payload.get("action_id", ""))
            for item in self._pending_actions:
                if item.get("action_id") == action_id and item.get("status") == "pending":
                    item["status"] = "cancelled"
                    self._emit({"event": "ACTION_CANCELLED", "action": dict(item), "timestamp": time.time()})
                    break
        elif action == "edit":
            action_id = str(payload.get("action_id", ""))
            for item in self._pending_actions:
                if item.get("action_id") == action_id and item.get("status") == "pending":
                    if isinstance(payload.get("tool"), str) and payload.get("tool"):
                        item["tool"] = payload.get("tool")
                    if isinstance(payload.get("arguments"), dict):
                        item["arguments"] = payload.get("arguments")
                    self._emit({"event": "ACTION_UPDATED", "action": dict(item), "timestamp": time.time()})
                    break
        elif action == "reorder":
            order = payload.get("order") if isinstance(payload.get("order"), list) else []
            index = {item.get("action_id"): item for item in self._pending_actions}
            reordered: list[dict] = []
            for action_id in order:
                if action_id in index:
                    reordered.append(index.pop(action_id))
            reordered.extend(index.values())
            self._pending_actions = reordered
            self._emit(
                {
                    "event": "ACTION_REORDERED",
                    "order": [item.get("action_id") for item in self._pending_actions],
                    "timestamp": time.time(),
                }
            )
        return self.queue_snapshot()

    def update_capability(self, payload: dict) -> dict:
        token_id = str(payload.get("token_id", ""))
        token = self._capability_manager.update_token(
            branch_id=str(payload.get("branch_id") or self._active_branch_id),
            token_id=token_id,
            path_pattern=payload.get("path_pattern") if isinstance(payload.get("path_pattern"), str) else None,
            ttl_seconds=payload.get("ttl_seconds") if isinstance(payload.get("ttl_seconds"), int) else None,
            constraints=payload.get("constraints") if isinstance(payload.get("constraints"), dict) else None,
        )
        serialized = self._serialize_token(token) if token is not None else None
        self._emit(
            {
                "event": "CAPABILITY_UPDATED",
                "token": serialized,
                "ok": token is not None,
                "timestamp": time.time(),
            }
        )
        return {"ok": token is not None, "token": serialized}

    def capability_ledger(self) -> dict:
        return {
            "branch_id": self._active_branch_id,
            "tokens": [
                self._serialize_token(token)
                for token in self._capability_manager.active_tokens(self._active_branch_id)
            ],
        }

    def _serialize_token(self, token) -> dict | None:
        if token is None:
            return None
        return {
            "token_id": token.token_id,
            "scope": token.scope.value,
            "path_pattern": token.path_pattern,
            "constraints": dict(token.constraints),
            "issued_at": token.issued_at,
            "expires_at": token.expires_at,
            "branch_id": token.branch_id,
            "revoked": token.revoked,
            "inherited_from": token.inherited_from,
        }

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
        waiter = self._capability_waiters.get(request_id)
        if waiter is not None:
            waiter.set()

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

    def compliance_rate(self) -> float:
        return self._tree.compliance_rate()

    def runtime_command(self, command: str, payload: dict | None = None) -> dict:
        req = payload or {}
        try:
            if command == "fork":
                return {"ok": True, "branch_id": self.fork(str(req.get("node_id")))}
            if command == "resume":
                self.resume(str(req.get("node_id")))
                return {"ok": True}
            if command == "prune":
                self.prune(str(req.get("branch_id")))
                return {"ok": True}
            if command == "compare":
                return {"ok": True, "comparison": self.compare(str(req.get("branch_a")), str(req.get("branch_b")))}
            if command == "diff_nodes":
                return {"ok": True, "diff": self._tree.diff_nodes(str(req.get("node_a")), str(req.get("node_b")))}
            if command == "action_queue":
                return {"ok": True, "queue": self.action_queue_update(req)}
            if command == "capability_update":
                return self.update_capability(req)
            if command == "capability_revoke":
                return {"ok": self.revoke_capability(str(req.get("token_id")), req.get("branch_id"))}
            if command == "capability_decision":
                request_id = str(req.get("request_id") or req.get("token_id") or "")
                if not request_id:
                    return {"ok": False, "error": "request_id is required"}
                self.apply_capability_decision(
                    request_id,
                    approved=bool(req.get("approved", False)),
                    scope=req.get("scope") if isinstance(req.get("scope"), str) else None,
                    path_pattern=str(req.get("path_pattern") or "**"),
                    ttl_seconds=req.get("ttl_seconds") if isinstance(req.get("ttl_seconds"), int) else None,
                    constraints=req.get("constraints") if isinstance(req.get("constraints"), dict) else None,
                    reason=req.get("reason") if isinstance(req.get("reason"), str) else None,
                )
                return {"ok": True}
            if command == "ledger":
                return {"ok": True, "ledger": self.capability_ledger()}
            if command == "compliance_rate":
                return {"ok": True, "rate": self.compliance_rate()}
            return {"ok": False, "error": f"unknown runtime action: {command}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

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

        prompt_node = self._append_node("system", V4_PROTOCOL_PROMPT, compliance={"prompt_injected": True})
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
            capability_granted = False
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
                    capability_granted = True
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

            if capability_granted and not protocol_actions and protocol_error is None:
                self._emit_step_end(thinking_step_id, "ok", protocol_compliant=protocol_compliant)
                cap_node = self._append_node(
                    "system",
                    "Capability granted. Proceed with the tool call using <ACTION> blocks.",
                    compliance={"capability_granted": True},
                )
                loop_messages = self._tree.build_messages_to_node(cap_node.node_id)
                steps += 1
                continue

            actions_to_execute = protocol_actions
            if not blocks and not actions_to_execute:
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
                self._emit_protocol_metrics()
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
            self._emit_protocol_metrics()
            loop_messages = self._tree.build_messages_to_node(assistant_node.node_id)

            for action in actions_to_execute:
                self._action_counter += 1
                queued = {
                    "action_id": f"a{self._action_counter}",
                    "tool": action["tool"],
                    "arguments": dict(action["arguments"]),
                    "synthetic": action.get("synthetic", False),
                    "status": "pending",
                    "step_id": thinking_step_id,
                }
                self._pending_actions.append(queued)
                self._emit({"event": "ACTION_QUEUED", "action": dict(queued), "timestamp": time.time()})

            for queued in list(self._pending_actions):
                if queued.get("status") != "pending":
                    continue
                tool = queued["tool"]
                arguments = queued["arguments"]
                tool_step_id = self._next_step_id()
                queued["status"] = "running"
                self._emit({"event": "ACTION_STARTED", "action": dict(queued), "timestamp": time.time()})
                self._checkpoint_before_tool(tool, arguments, loop_messages)
                self._emit_step_start(tool_step_id, f"Tool: {tool}", "tool", tool=tool, arguments=arguments)
                self._emit(
                    {
                        "event": "TOOL_CALL_START",
                        "step_id": tool_step_id,
                        "tool": tool,
                        "arguments": arguments,
                        "synthetic": queued.get("synthetic", False),
                        "timestamp": time.time(),
                    }
                )
                bridge_result = self._bridge.execute(tool, arguments, branch_id=self._active_branch_id)
                self._state = AgentState.TOOL_RESULT
                result_payload = bridge_result.get("result", {"ok": False, "content": "", "error": bridge_result.get("error")})
                queued["status"] = "done" if result_payload.get("ok", False) else "error"
                queued["result"] = result_payload
                self._emit(
                    {
                        "event": "TOOL_RESULT",
                        "step_id": tool_step_id,
                        "tool": tool,
                        "result": result_payload,
                        "timestamp": time.time(),
                    }
                )
                self._emit({"event": "ACTION_FINISHED", "action": dict(queued), "timestamp": time.time()})
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

        waiter = threading.Event()
        self._capability_waiters[request_id] = waiter
        if not waiter.wait(timeout=CAPABILITY_DECISION_TIMEOUT):
            self._capability_waiters.pop(request_id, None)
            return {"ok": False, "error": "capability decision timeout"}

        decision = self._capability_decisions.pop(request_id, None)
        self._capability_waiters.pop(request_id, None)
        if decision is None:
            return {"ok": False, "error": "capability request interrupted"}
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
                "event": "CAPABILITY_ISSUED",
                "step_id": step_id,
                "request_id": request_id,
                "token": self._serialize_token(token),
                "timestamp": time.time(),
            }
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
            checkpoint_result = get_checkpoint_store().create_checkpoint(
                branch_id=self._active_branch_id,
                node_id=self._active_leaf_node_id,
                message_history=message_history or [],
                pending_action=pending_action,
                capabilities=capabilities or {},
                pty_state_ref=pty_state_ref,
            )
            self._emit(
                {
                    "event": "CHECKPOINT_CREATED",
                    "checkpoint": checkpoint_result,
                    "pending_action": pending_action,
                    "timestamp": time.time(),
                }
            )
        except Exception:
            return

    def _emit_protocol_metrics(self) -> None:
        rate = self.compliance_rate()
        self._emit({"event": "PROTOCOL_COMPLIANCE_RATE", "rate": rate, "threshold": PROTOCOL_COMPLIANCE_THRESHOLD, "timestamp": time.time()})
        if rate < PROTOCOL_COMPLIANCE_THRESHOLD:
            self._emit({"event": "PROTOCOL_REGRESSION_WARNING", "rate": rate, "threshold": PROTOCOL_COMPLIANCE_THRESHOLD, "timestamp": time.time()})

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
