from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
import json
import re
import subprocess
import os
import sys
from typing import Callable

from core.config import DEFAULT_WORKSPACE_ROOT
from engine.pty_runtime import get_pty_session_manager
from engine.process_controller import ProcessGroupController

# Path to standalone Python runtime for run_python tool
_PYTHON_RUNTIME_PATH = Path(__file__).parent / "python_runtime.py"

WORKSPACE_ROOT: Path = DEFAULT_WORKSPACE_ROOT

# Singleton process controller — shared across all tool executions
_PROCESS_CONTROLLER: ProcessGroupController | None = None


def get_process_controller() -> ProcessGroupController:
    """Return the singleton ProcessGroupController."""
    global _PROCESS_CONTROLLER
    if _PROCESS_CONTROLLER is None:
        _PROCESS_CONTROLLER = ProcessGroupController()
    return _PROCESS_CONTROLLER


def set_workspace_root(_: str | Path | None = None) -> None:
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = DEFAULT_WORKSPACE_ROOT


def resolve_path(user_path: str) -> Path:
    # Strip workspace prefix if the model redundantly includes it.
    # e.g. "workspace/star.py" → "star.py" when WORKSPACE_ROOT already ends in /workspace
    cleaned = user_path.strip().replace("\\", "/")
    ws_name = WORKSPACE_ROOT.name  # e.g. "workspace"
    if cleaned.startswith(f"{ws_name}/"):
        cleaned = cleaned[len(ws_name) + 1:]
    elif cleaned == ws_name:
        cleaned = "."
    target = (WORKSPACE_ROOT / cleaned).expanduser().resolve()
    if not str(target).startswith(str(WORKSPACE_ROOT)):
        raise ValueError(f"path outside workspace boundary: {target}")
    return target


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: str | None = None

    def to_message(self) -> dict:
        return {
            "ok": self.ok,
            "content": self.content,
            "error": self.error,
        }


def _resolve_path(args: dict, default: str = ".") -> Path:
    raw = str(args.get("path", default))
    return resolve_path(raw)


def read_file(args: dict) -> ToolResult:
    try:
        path = _resolve_path(args)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))
    if not path.exists() or not path.is_file():
        return ToolResult(False, "", f"File not found: {path}. Use list_dir to check available files.")

    offset = int(args.get("offset", 0) or 0)
    limit = args.get("limit")
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return ToolResult(False, "", f"failed to read file: {exc}")

    lines = text.splitlines()
    if offset < 0:
        offset = 0
    if limit is not None:
        try:
            limit = max(0, int(limit))
            lines = lines[offset : offset + limit]
        except Exception:
            lines = lines[offset:]
    else:
        lines = lines[offset:]
        total_lines = len(lines)
        if total_lines > 200:
            lines = lines[:200]
            lines.append(
                f"[OUTPUT TRUNCATED — showing first 200 of {total_lines} lines. Use offset/limit for specific ranges, or grep_search to find what you need.]"
            )

    return ToolResult(True, "\n".join(lines))


def write_file(args: dict) -> ToolResult:
    try:
        path = _resolve_path(args)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))
    content = args.get("content")
    if not isinstance(content, str):
        return ToolResult(False, "", "content must be a string")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except PermissionError:
        return ToolResult(False, "", f"Permission denied: {path}. Check workspace boundary.")
    except Exception as exc:
        return ToolResult(False, "", f"failed to write file: {exc}")
    return ToolResult(True, f"wrote {len(content)} chars to {path}")


def list_dir(args: dict) -> ToolResult:
    try:
        path = _resolve_path(args)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))
    pattern = args.get("pattern")
    if not path.exists() or not path.is_dir():
        return ToolResult(False, "", f"directory not found: {path}")
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
        out = []
        for entry in entries:
            name = entry.name + ("/" if entry.is_dir() else "")
            if pattern and not fnmatch(name, str(pattern)):
                continue
            out.append(name)
        return ToolResult(True, "\n".join(out))
    except Exception as exc:
        return ToolResult(False, "", f"failed to list directory: {exc}")


def grep_search(args: dict) -> ToolResult:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return ToolResult(False, "", "pattern is required")

    try:
        root = _resolve_path(args)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = [p for p in root.rglob("*") if p.is_file()]
    else:
        return ToolResult(False, "", f"path not found: {root}")

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return ToolResult(False, "", f"invalid regex: {exc}")

    matches = []
    for file_path in files:
        try:
            for idx, line in enumerate(file_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{file_path}:{idx}:{line}")
        except Exception:
            continue

    if not matches:
        return ToolResult(True, f"No matches found for pattern '{pattern}' in {root}. Try a broader pattern or different path.")

    total_matches = len(matches)
    if total_matches > 50:
        matches = matches[:50]
        matches.append(
            f"[RESULTS TRUNCATED — showing first 50 of {total_matches} matches. Narrow your pattern.]"
        )
    return ToolResult(True, "\n".join(matches))


_BLOCKED_CMD_PATTERNS = (
    "rm -rf /",
    "mkfs",
    "shutdown",
    "reboot",
    "poweroff",
)


def run_cmd(args: dict) -> ToolResult:
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return ToolResult(False, "", "command is required")

    lowered = command.lower()
    if any(pat in lowered for pat in _BLOCKED_CMD_PATTERNS):
        return ToolResult(False, "", "Command blocked (matches dangerous pattern). Try a safer alternative.")

    timeout = args.get("timeout", 30)
    try:
        timeout = max(1, int(timeout))
    except Exception:
        timeout = 30

    checkpoint_ctx = args.get("_checkpoint") if isinstance(args.get("_checkpoint"), dict) else {}
    branch_id = str(checkpoint_ctx.get("branch_id", "main"))
    pty_enabled = bool(args.get("pty_enabled", True)) and os.environ.get("MONOLITH_DISABLE_PTY_RUNTIME", "0") != "1"

    if pty_enabled:
        idle_timeout = int(os.environ.get("MONOLITH_PTY_IDLE_TIMEOUT", "300") or "300")
        try:
            manager = get_pty_session_manager(workspace_root=WORKSPACE_ROOT, idle_timeout_seconds=idle_timeout)
            exit_code, stdout, error = manager.run(branch_id=branch_id, command=command, timeout=timeout)
            if error is None:
                payload = f"exit_code: {exit_code}\nstdout:\n{stdout}\nstderr:\n"
                return ToolResult(exit_code == 0, payload)
        except RuntimeError:
            pass

    # Use ProcessGroupController for managed execution with threaded stream readers
    controller = get_process_controller()
    try:
        handle = controller.start(command, cwd=str(WORKSPACE_ROOT))
    except Exception as exc:
        return ToolResult(False, "", f"command failed to start: {exc}")

    try:
        exit_code = controller.wait(handle, timeout=float(timeout))
    except subprocess.TimeoutExpired:
        controller.terminate(handle)
        try:
            controller.wait(handle, timeout=2.0)
        except Exception:
            controller.force_kill(handle)
        return ToolResult(False, "", f"command timed out after {timeout}s")
    except Exception as exc:
        controller.force_kill(handle)
        return ToolResult(False, "", f"command execution error: {exc}")

    stdout_text, stderr_text = controller.get_output(handle)

    stdout_lines = stdout_text.splitlines()
    if len(stdout_lines) > 150:
        truncated_stdout = "\n".join(stdout_lines[:150])
        truncated_stdout += f"\n[OUTPUT TRUNCATED — showing first 150 of {len(stdout_lines)} lines.]"
    else:
        truncated_stdout = stdout_text

    payload = (
        f"exit_code: {exit_code}\n"
        f"stdout:\n{truncated_stdout}\n"
        f"stderr:\n{stderr_text}"
    )
    return ToolResult(exit_code == 0, payload)


def apply_patch(args: dict) -> ToolResult:
    try:
        path = _resolve_path(args)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))
    old = args.get("old")
    new = args.get("new")
    if not isinstance(old, str) or not isinstance(new, str):
        return ToolResult(False, "", "old and new must be strings")
    if not path.exists() or not path.is_file():
        return ToolResult(False, "", f"file not found: {path}")


    try:
        current = path.read_text(encoding="utf-8")
    except Exception as exc:
        return ToolResult(False, "", f"failed to read file: {exc}")

    if old not in current:
        return ToolResult(False, "", f"Text to replace not found in {path}. Use read_file to verify current contents.")

    updated = current.replace(old, new, 1)
    try:
        path.write_text(updated, encoding="utf-8")
    except Exception as exc:
        return ToolResult(False, "", f"failed to write patched file: {exc}")

    return ToolResult(True, f"applied patch in {path}")


def run_python(args: dict) -> ToolResult:
    """
    Execute Python code in subprocess — fully privileged OS-level execution.

    Contract: JSON in, JSON out. Structured result for agent self-repair.
    - code: str - Python code to execute
    - timeout: int - seconds (default 30)

    Convention: assign to 'result' variable for structured return value.
    Workspace root available as 'workspace_root' variable.
    Capability: EXEC scope. Flagged as destructive/high-risk in OFAC contract.
    """
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return ToolResult(False, "", "code is required and must be non-empty string")

    timeout = args.get("timeout", 30)
    try:
        timeout = max(1, min(int(timeout), 300))  # Clamp 1-300 seconds
    except (TypeError, ValueError):
        timeout = 30

    contract = {
        "code": code,
        "workspace_root": str(WORKSPACE_ROOT),
        "timeout": timeout,
    }

    # Use ProcessGroupController for managed subprocess execution
    controller = get_process_controller()
    python_cmd = f'"{sys.executable}" "{_PYTHON_RUNTIME_PATH}"'

    try:
        handle = controller.start(python_cmd, cwd=str(WORKSPACE_ROOT), stdin_pipe=True)
    except Exception as exc:
        return ToolResult(False, "", f"failed to spawn runtime: {exc}")

    # Feed contract JSON to stdin
    try:
        if handle.proc.stdin:
            handle.proc.stdin.write(json.dumps(contract))
            handle.proc.stdin.close()
    except Exception as exc:
        controller.force_kill(handle)
        return ToolResult(False, "", f"failed to write to runtime stdin: {exc}")

    try:
        controller.wait(handle, timeout=float(timeout))
    except subprocess.TimeoutExpired:
        controller.terminate(handle)
        try:
            controller.wait(handle, timeout=2.0)
        except Exception:
            controller.force_kill(handle)
        return ToolResult(False, "", f"execution timed out after {timeout}s and was terminated")
    except Exception as exc:
        controller.force_kill(handle)
        return ToolResult(False, "", f"runtime error: {exc}")

    stdout_text, stderr_text = controller.get_output(handle)

    # Parse structured output from runtime
    try:
        result = json.loads(stdout_text)
    except json.JSONDecodeError:
        raw_stdout = stdout_text[:2000] if stdout_text else ""
        raw_stderr = stderr_text[:500] if stderr_text else ""
        return ToolResult(
            False,
            raw_stdout,
            f"runtime error (exit {handle.proc.returncode}): {raw_stderr}",
        )

    # Build human-readable content from structured result
    status = result.get("status", "error")
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    return_value = result.get("return_value")
    exception = result.get("exception")
    exec_time = result.get("execution_time_ms", 0)

    # Format content for agent consumption
    content_parts = []
    if stdout:
        content_parts.append(f"[stdout]\n{stdout}")
    if return_value is not None:
        content_parts.append(f"[return_value]\n{json.dumps(return_value, indent=2, default=str)}")
    content = "\n\n".join(content_parts) if content_parts else "(no output)"

    # Format error if applicable
    error = None
    if status != "ok":
        if exception:
            err_parts = [f"{exception.get('type', 'Error')}: {exception.get('message', 'unknown')}"]
            if exception.get('line'):
                err_parts.append(f"at line {exception['line']}")
            error = " ".join(err_parts)
        else:
            error = stderr if stderr else "runtime error"
    elif stderr:
        content += f"\n\n[stderr]\n{stderr}"

    return ToolResult(
        ok=(status == "ok"),
        content=content,
        error=error,
    )


TOOL_REGISTRY: dict[str, Callable[[dict], ToolResult]] = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "grep_search": grep_search,
    "run_cmd": run_cmd,
    "apply_patch": apply_patch,
    "run_python": run_python,
}
