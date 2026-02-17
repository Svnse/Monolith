from __future__ import annotations

from typing import Callable

from engine.tools import TOOL_REGISTRY, ToolResult


class ToolDispatcher:
    def __init__(self, registry: dict[str, Callable[[dict], ToolResult]] | None = None):
        self._registry = registry or TOOL_REGISTRY

    def has_tool(self, name: str) -> bool:
        return name in self._registry

    def dispatch(self, name: str, arguments: dict) -> dict:
        if name not in self._registry:
            return {"ok": False, "content": "", "error": f"tool not found: {name}"}
        try:
            result = self._registry[name](arguments)
            return result.to_message()
        except Exception as exc:
            return {"ok": False, "content": "", "error": f"tool execution failed: {exc}"}
