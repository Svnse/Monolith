from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import json
from pathlib import Path
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
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint_ref: str | None = None
    capability_decisions: list[dict[str, Any]] = field(default_factory=list)


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
        metadata: dict[str, Any] | None = None,
        checkpoint_ref: str | None = None,
        capability_decisions: list[dict[str, Any]] | None = None,
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
            metadata=metadata or {},
            checkpoint_ref=checkpoint_ref,
            capability_decisions=list(capability_decisions or []),
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

    def compliance_rate(self) -> float:
        steps = [n for n in self.nodes.values() if n.role == "assistant"]
        if not steps:
            return 1.0
        compliant = sum(1 for n in steps if bool(n.compliance.get("protocol_compliant", False)))
        return compliant / len(steps)

    def diff_nodes(self, node_a: str, node_b: str) -> dict[str, Any]:
        meta_a = self.get_node(node_a).metadata if node_a in self.nodes else {}
        meta_b = self.get_node(node_b).metadata if node_b in self.nodes else {}
        manifest_a = meta_a.get("workspace_manifest") if isinstance(meta_a.get("workspace_manifest"), dict) else {}
        manifest_b = meta_b.get("workspace_manifest") if isinstance(meta_b.get("workspace_manifest"), dict) else {}

        files_a = set(manifest_a.keys())
        files_b = set(manifest_b.keys())
        added = sorted(files_b - files_a)
        removed = sorted(files_a - files_b)
        modified = sorted(path for path in (files_a & files_b) if manifest_a.get(path) != manifest_b.get(path))

        modified_diffs: dict[str, str] = {}
        for path in modified:
            content_a = str(meta_a.get("workspace_files", {}).get(path, ""))
            content_b = str(meta_b.get("workspace_files", {}).get(path, ""))
            diff = difflib.unified_diff(
                content_a.splitlines(),
                content_b.splitlines(),
                fromfile=f"{node_a}/{path}",
                tofile=f"{node_b}/{path}",
                lineterm="",
            )
            modified_diffs[path] = "\n".join(diff)

        return {
            "node_a": node_a,
            "node_b": node_b,
            "added_files": added,
            "removed_files": removed,
            "modified_files": modified,
            "modified_diffs": modified_diffs,
        }

    def to_dict(self) -> dict:
        return {
            "node_counter": self._node_counter,
            "branch_counter": self._branch_counter,
            "nodes": {
                node_id: {
                    "node_id": node.node_id,
                    "branch_id": node.branch_id,
                    "parent_node_id": node.parent_node_id,
                    "role": node.role,
                    "content": node.content,
                    "reasoning": node.reasoning,
                    "action": node.action,
                    "result": node.result,
                    "compliance": node.compliance,
                    "metadata": node.metadata,
                    "checkpoint_ref": node.checkpoint_ref,
                    "capability_decisions": node.capability_decisions,
                }
                for node_id, node in self.nodes.items()
            },
            "branches": {
                branch_id: {
                    "branch_id": branch.branch_id,
                    "parent_branch_id": branch.parent_branch_id,
                    "forked_from_node_id": branch.forked_from_node_id,
                    "leaf_node_id": branch.leaf_node_id,
                    "pruned": branch.pruned,
                }
                for branch_id, branch in self.branches.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict):
        tree = cls()
        tree.nodes = {}
        for node_id, raw in data.get("nodes", {}).items():
            tree.nodes[node_id] = ExecutionNode(
                node_id=str(raw.get("node_id", node_id)),
                branch_id=str(raw.get("branch_id", "main")),
                parent_node_id=raw.get("parent_node_id"),
                role=str(raw.get("role", "assistant")),
                content=str(raw.get("content", "")),
                reasoning=raw.get("reasoning"),
                action=raw.get("action") if isinstance(raw.get("action"), dict) else None,
                result=raw.get("result") if isinstance(raw.get("result"), dict) else None,
                compliance=raw.get("compliance") if isinstance(raw.get("compliance"), dict) else {},
                metadata=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
                checkpoint_ref=raw.get("checkpoint_ref"),
                capability_decisions=raw.get("capability_decisions") if isinstance(raw.get("capability_decisions"), list) else [],
            )
        tree.branches = {}
        for branch_id, raw in data.get("branches", {}).items():
            tree.branches[branch_id] = ExecutionBranch(
                branch_id=str(raw.get("branch_id", branch_id)),
                parent_branch_id=raw.get("parent_branch_id"),
                forked_from_node_id=raw.get("forked_from_node_id"),
                leaf_node_id=raw.get("leaf_node_id"),
                pruned=bool(raw.get("pruned", False)),
            )
        if "main" not in tree.branches:
            tree.branches["main"] = ExecutionBranch("main", None, None, None)
        tree._node_counter = int(data.get("node_counter", 0))
        tree._branch_counter = int(data.get("branch_counter", 0))
        return tree

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)
