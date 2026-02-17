from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from engine.tool_dispatcher import ToolDispatcher
from engine.tool_schema import TOOL_ARGUMENT_SCHEMAS


class AgentBridge:
    def __init__(self, dispatcher: ToolDispatcher | None = None):
        self._dispatcher = dispatcher or ToolDispatcher()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-tool")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def execute(self, tool: str, arguments: dict) -> dict:
        validation = self._validate(tool, arguments)
        if not validation["ok"]:
            return validation

        future = self._executor.submit(self._dispatcher.dispatch, tool, arguments)
        try:
            result = future.result()
            return {"ok": True, "tool": tool, "result": result}
        except Exception as exc:
            return {
                "ok": False,
                "tool": tool,
                "error": f"tool execution failed: {exc}",
                "result": {"ok": False, "content": "", "error": str(exc)},
            }

    def _validate(self, tool: str, arguments: dict) -> dict:
        if not isinstance(tool, str) or not tool:
            return {"ok": False, "tool": tool, "error": "tool name must be a non-empty string"}
        if not self._dispatcher.has_tool(tool):
            return {"ok": False, "tool": tool, "error": f"tool not found: {tool}"}
        if not isinstance(arguments, dict):
            return {"ok": False, "tool": tool, "error": "tool arguments must be an object"}

        schema = TOOL_ARGUMENT_SCHEMAS.get(tool)
        if not schema:
            return {"ok": True, "tool": tool}

        required = schema.get("required", {})
        optional = schema.get("optional", {})
        allowed_keys = set(required.keys()) | set(optional.keys())

        for key in arguments.keys():
            if key not in allowed_keys:
                return {"ok": False, "tool": tool, "error": f"unexpected argument: {key}"}

        for key, expected_type in required.items():
            if key not in arguments:
                return {"ok": False, "tool": tool, "error": f"missing required argument: {key}"}
            if not isinstance(arguments[key], expected_type):
                return {
                    "ok": False,
                    "tool": tool,
                    "error": f"argument '{key}' must be {expected_type.__name__}",
                }

        for key, expected_type in optional.items():
            if key in arguments and not isinstance(arguments[key], expected_type):
                return {
                    "ok": False,
                    "tool": tool,
                    "error": f"argument '{key}' must be {expected_type.__name__}",
                }

        return {"ok": True, "tool": tool}
