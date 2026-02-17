from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionNode:
    node_id: str
    branch_id: str
    parent_node_id: str | None
    role: str
    content: str
    reasoning: str | None = None
    action: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    compliance: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionBranch:
    branch_id: str
    parent_branch_id: str | None
    forked_from_node_id: str | None
    leaf_node_id: str | None
    pruned: bool = False


class ExecutionTree:
    def __init__(self) -> None:
        self.nodes: dict[str, ExecutionNode] = {}
        self.branches: dict[str, ExecutionBranch] = {
            "main": ExecutionBranch(
                branch_id="main",
                parent_branch_id=None,
                forked_from_node_id=None,
                leaf_node_id=None,
            )
        }
        self._node_counter = 0
        self._branch_counter = 0

    def has_nodes(self) -> bool:
        return bool(self.nodes)

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter}"

    def _next_branch_id(self) -> str:
        self._branch_counter += 1
        return f"b{self._branch_counter}"

    def append_node(
        self,
        branch_id: str,
        role: str,
        content: str,
        *,
        reasoning: str | None = None,
        action: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        compliance: dict[str, Any] | None = None,
    ) -> ExecutionNode:
        branch = self.branches[branch_id]
        node = ExecutionNode(
            node_id=self._next_node_id(),
            branch_id=branch_id,
            parent_node_id=branch.leaf_node_id,
            role=role,
            content=content,
            reasoning=reasoning,
            action=action,
            result=result,
            compliance=compliance or {},
        )
        self.nodes[node.node_id] = node
        branch.leaf_node_id = node.node_id
        return node

    def init_from_messages(self, messages: list[dict], branch_id: str = "main") -> None:
        if self.has_nodes():
            return
        for msg in messages:
            self.append_node(
                branch_id,
                msg.get("role", "assistant"),
                msg.get("content", ""),
                action={"name": msg.get("name")} if msg.get("name") else None,
            )

    def get_node(self, node_id: str) -> ExecutionNode:
        return self.nodes[node_id]

    def branch_leaf(self, branch_id: str) -> str | None:
        return self.branches[branch_id].leaf_node_id

    def branch_from_leaf(self, leaf_node_id: str) -> str:
        return self.get_node(leaf_node_id).branch_id

    def build_messages_to_node(self, node_id: str | None) -> list[dict]:
        if node_id is None:
            return []
        ordered: list[ExecutionNode] = []
        cursor = node_id
        while cursor is not None:
            node = self.nodes[cursor]
            ordered.append(node)
            cursor = node.parent_node_id
        ordered.reverse()

        messages: list[dict] = []
        for node in ordered:
            msg = {"role": node.role, "content": node.content}
            if node.action and node.action.get("name"):
                msg["name"] = node.action["name"]
            messages.append(msg)
        return messages

    def fork(self, node_id: str) -> ExecutionBranch:
        node = self.get_node(node_id)
        branch_id = self._next_branch_id()
        branch = ExecutionBranch(
            branch_id=branch_id,
            parent_branch_id=node.branch_id,
            forked_from_node_id=node_id,
            leaf_node_id=node_id,
        )
        self.branches[branch_id] = branch
        return branch

    def prune(self, branch_id: str) -> None:
        if branch_id == "main":
            raise ValueError("cannot prune main branch")
        self.branches[branch_id].pruned = True

    def compare(self, branch_a: str, branch_b: str) -> dict[str, Any]:
        leaf_a = self.branches[branch_a].leaf_node_id
        leaf_b = self.branches[branch_b].leaf_node_id
        msgs_a = self.build_messages_to_node(leaf_a)
        msgs_b = self.build_messages_to_node(leaf_b)
        return {
            "branch_a": branch_a,
            "branch_b": branch_b,
            "leaf_a": leaf_a,
            "leaf_b": leaf_b,
            "length_a": len(msgs_a),
            "length_b": len(msgs_b),
            "diverges": msgs_a != msgs_b,
            "preview_a": msgs_a[-3:],
            "preview_b": msgs_b[-3:],
        }
