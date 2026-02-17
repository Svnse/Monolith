from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
import re
import subprocess
import os
from typing import Callable

from core.config import DEFAULT_WORKSPACE_ROOT
from engine.checkpointing import get_checkpoint_store
from engine.pty_runtime import get_pty_session_manager

WORKSPACE_ROOT: Path = DEFAULT_WORKSPACE_ROOT


def _capture_checkpoint(args: dict, action_type: str, action_payload: dict) -> None:
    context = args.get("_checkpoint") if isinstance(args.get("_checkpoint"), dict) else {}
    message_history = context.get("message_history")
    if not isinstance(message_history, list):
        message_history = []

    try:
        get_checkpoint_store().create_checkpoint(
            branch_id=str(context.get("branch_id", "main")),
            node_id=context.get("node_id"),
            message_history=message_history,
            pending_action={"type": action_type, **action_payload},
            capabilities=context.get("capabilities") if isinstance(context.get("capabilities"), dict) else {},
            pty_state_ref=context.get("pty_state_ref"),
        )
    except Exception:
        return


def set_workspace_root(_: str | Path | None = None) -> None:
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = DEFAULT_WORKSPACE_ROOT


def resolve_path(user_path: str) -> Path:
    target = (WORKSPACE_ROOT / user_path).expanduser().resolve()
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
    _capture_checkpoint(args, "write_file", {"path": args.get("path")})
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

    _capture_checkpoint(args, "run_cmd", {"command": command})

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
        manager = get_pty_session_manager(workspace_root=WORKSPACE_ROOT, idle_timeout_seconds=idle_timeout)
        exit_code, stdout, error = manager.run(branch_id=branch_id, command=command, timeout=timeout)
        if error is not None:
            return ToolResult(False, "", error)
        payload = f"exit_code: {exit_code}\nstdout:\n{stdout}\nstderr:\n"
        return ToolResult(exit_code == 0, payload)

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKSPACE_ROOT),
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, "", f"command timed out after {timeout}s")
    except Exception as exc:
        return ToolResult(False, "", f"command failed to start: {exc}")

    stdout_lines = completed.stdout.splitlines()
    if len(stdout_lines) > 150:
        truncated_stdout = "\n".join(stdout_lines[:150])
        truncated_stdout += f"\n[OUTPUT TRUNCATED — showing first 150 of {len(stdout_lines)} lines.]"
    else:
        truncated_stdout = completed.stdout

    payload = (
        f"exit_code: {completed.returncode}\n"
        f"stdout:\n{truncated_stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return ToolResult(completed.returncode == 0, payload)


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

    _capture_checkpoint(args, "apply_patch", {"path": args.get("path")})

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


TOOL_REGISTRY: dict[str, Callable[[dict], ToolResult]] = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "grep_search": grep_search,
    "run_cmd": run_cmd,
    "apply_patch": apply_patch,
}
