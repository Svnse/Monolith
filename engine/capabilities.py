from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from engine.tool_schema import TOOL_ARGUMENT_SCHEMAS


class CapabilityScope(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    EXEC = "EXEC"
    NETWORK = "NETWORK"


CAPABILITY_MANIFEST: dict[str, list[str]] = {
    "code": [
        "read_file",
        "write_file",
        "list_dir",
        "grep_search",
        "glob_files",
        "mkdir",
        "move_path",
        "copy_path",
        "delete_path",
        "zip_path",
        "unzip_archive",
        "run_cmd",
        "run_tests",
        "apply_patch",
        "run_python",
        "git_status",
        "git_diff",
        "http_fetch",
    ],
    "analysis": [
        "read_file",
        "list_dir",
        "grep_search",
        "glob_files",
        "run_python",
        "git_status",
        "git_diff",
        "http_fetch",
    ],
    "secure": ["read_file"],
}


TOOL_SCOPE_MAP: dict[str, CapabilityScope] = {
    "read_file": CapabilityScope.READ,
    "list_dir": CapabilityScope.READ,
    "grep_search": CapabilityScope.READ,
    "glob_files": CapabilityScope.READ,
    "git_status": CapabilityScope.READ,
    "git_diff": CapabilityScope.READ,
    "write_file": CapabilityScope.WRITE,
    "apply_patch": CapabilityScope.WRITE,
    "mkdir": CapabilityScope.WRITE,
    "move_path": CapabilityScope.WRITE,
    "copy_path": CapabilityScope.WRITE,
    "delete_path": CapabilityScope.WRITE,
    "zip_path": CapabilityScope.WRITE,
    "unzip_archive": CapabilityScope.WRITE,
    "run_cmd": CapabilityScope.EXEC,
    "run_python": CapabilityScope.EXEC,
    "run_tests": CapabilityScope.EXEC,
    "http_fetch": CapabilityScope.NETWORK,
}


@dataclass
class AuthorizationResult:
    ok: bool
    error: str | None = None


class CapabilityManager:
    def __init__(self, profile: str = "code") -> None:
        self._profile = profile if profile in CAPABILITY_MANIFEST else "code"
        self._manifest_tools = set(CAPABILITY_MANIFEST[self._profile])
        self._capability_digest = hashlib.sha256(
            json.dumps(sorted(self._manifest_tools), separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def capability_digest(self) -> str:
        return self._capability_digest

    def allowed_tools(self) -> list[str]:
        return sorted(self._manifest_tools)

    def tool_schemas(self) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for tool_name in self.allowed_tools():
            spec = TOOL_ARGUMENT_SCHEMAS.get(tool_name, {"required": {}, "optional": {}})
            properties: dict[str, Any] = {}
            required_keys: list[str] = []
            for key, py_type in spec.get("required", {}).items():
                properties[key] = {"type": _json_type(py_type)}
                required_keys.append(key)
            for key, py_type in spec.get("optional", {}).items():
                properties[key] = {"type": _json_type(py_type)}

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Execute the {tool_name} tool.",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required_keys,
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return schemas

    def validate_tool_name(self, name: str) -> AuthorizationResult:
        if name not in self._manifest_tools:
            return AuthorizationResult(False, f"tool '{name}' is not in capability profile '{self._profile}'")
        return AuthorizationResult(True)

    def authorize(self, *, tool: str, path: str | None) -> AuthorizationResult:
        _ = path
        return self.validate_tool_name(tool)


def extract_tool_path(tool: str, arguments: dict[str, Any]) -> str | None:
    if not isinstance(arguments, dict):
        return None
    if tool in {
        "read_file",
        "write_file",
        "list_dir",
        "grep_search",
        "glob_files",
        "apply_patch",
        "delete_path",
        "mkdir",
        "git_status",
        "git_diff",
    }:
        raw = arguments.get("path", ".")
        if isinstance(raw, str):
            return PurePosixPath(raw).as_posix()
    if tool == "unzip_archive":
        raw = arguments.get("dst", ".")
        if isinstance(raw, str):
            return PurePosixPath(raw).as_posix()
    if tool == "zip_path":
        raw = arguments.get("dst")
        if isinstance(raw, str):
            return PurePosixPath(raw).as_posix()
    if tool in {"move_path", "copy_path"}:
        raw = arguments.get("src")
        if isinstance(raw, str):
            return PurePosixPath(raw).as_posix()
    return "."


def _json_type(py_type: type[Any]) -> str:
    if py_type is str:
        return "string"
    if py_type is int:
        return "integer"
    if py_type is float:
        return "number"
    if py_type is bool:
        return "boolean"
    if py_type is list:
        return "array"
    if py_type is dict:
        return "object"
    return "string"
