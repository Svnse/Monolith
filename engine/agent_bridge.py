from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from engine.capabilities import CapabilityManager, extract_tool_path
from engine.pty_runtime import get_pty_session_manager
from engine.tool_dispatcher import ToolDispatcher
from engine.tool_schema import TOOL_ARGUMENT_SCHEMAS
from engine.tools import WORKSPACE_ROOT


class AgentBridge:
    def __init__(self, dispatcher: ToolDispatcher | None = None, capability_manager: CapabilityManager | None = None):
        self._dispatcher = dispatcher or ToolDispatcher()
        self._capability_manager = capability_manager or CapabilityManager()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-tool")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
        try:
            get_pty_session_manager(workspace_root=WORKSPACE_ROOT).destroy_all()
        except Exception:
            pass

    def execute(self, tool: str, arguments: dict, *, branch_id: str = "main") -> dict:
        validation = self._validate(tool, arguments, branch_id=branch_id)
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

    def _validate(self, tool: str, arguments: dict, *, branch_id: str) -> dict:
        if not isinstance(tool, str) or not tool:
            return {"ok": False, "tool": tool, "error": "tool name must be a non-empty string"}
        if not self._dispatcher.has_tool(tool):
            return {"ok": False, "tool": tool, "error": f"tool not found: {tool}"}
        if not isinstance(arguments, dict):
            return {"ok": False, "tool": tool, "error": "tool arguments must be an object"}

        schema = TOOL_ARGUMENT_SCHEMAS.get(tool)
        required = schema.get("required", {}) if schema else {}
        optional = schema.get("optional", {}) if schema else {}
        allowed_keys = set(required.keys()) | set(optional.keys())

        for key in arguments.keys():
            if key.startswith("_"):
                continue
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

        requested_path = extract_tool_path(tool, arguments)
        auth_result = self._capability_manager.authorize(branch_id=branch_id, tool=tool, path=requested_path)
        if not auth_result.ok:
            return {
                "ok": False,
                "tool": tool,
                "error": auth_result.error,
            }

        return {"ok": True, "tool": tool, "capability_token": auth_result.token.token_id if auth_result.token else None}
