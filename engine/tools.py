from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Callable

from core.config import DEFAULT_WORKSPACE_ROOT
from engine.pty_runtime import get_pty_session_manager
from engine.process_controller import ProcessGroupController

# Path to standalone Python runtime for run_python tool
_PYTHON_RUNTIME_PATH = Path(__file__).parent / "python_runtime.py"

WORKSPACE_ROOT: Path = DEFAULT_WORKSPACE_ROOT
BOUNDARY_MODE: str = str(os.getenv("MONOLITH_BOUNDARY_MODE") or "strict_workspace").strip().lower()
ALLOWED_PATH_ROOTS: list[Path] = []


def _refresh_allowed_roots_from_env() -> None:
    global ALLOWED_PATH_ROOTS
    raw = str(os.getenv("MONOLITH_ALLOWED_PATHS") or "").strip()
    roots: list[Path] = []
    if raw:
        for part in raw.split(os.pathsep):
            p = part.strip()
            if not p:
                continue
            try:
                roots.append(Path(p).expanduser().resolve())
            except Exception:
                continue
    ALLOWED_PATH_ROOTS = roots


_refresh_allowed_roots_from_env()

# Singleton process controller — shared across all tool executions
_PROCESS_CONTROLLER: ProcessGroupController | None = None


def get_process_controller() -> ProcessGroupController:
    """Return the singleton ProcessGroupController."""
    global _PROCESS_CONTROLLER
    if _PROCESS_CONTROLLER is None:
        _PROCESS_CONTROLLER = ProcessGroupController()
    return _PROCESS_CONTROLLER


def stop_active_process_groups(*, grace_timeout_s: float = 1.0) -> dict[str, int]:
    """
    Terminate all active subprocess groups started by tool execution.

    Used by STOP handling to ensure no long-running child process outlives
    generation cancellation.
    """
    controller = get_process_controller()
    handles = list(controller.active_handles)
    active_before = len(handles)
    if active_before == 0:
        return {"active_before": 0, "terminated": 0, "force_killed": 0, "active_after": 0}

    terminated = 0
    force_killed = 0
    controller.terminate_all()

    pending = list(handles)
    deadline = time.monotonic() + max(0.0, float(grace_timeout_s))
    while pending and time.monotonic() < deadline:
        remaining: list = []
        for handle in pending:
            try:
                controller.wait(handle, timeout=0.05)
                terminated += 1
            except subprocess.TimeoutExpired:
                remaining.append(handle)
            except Exception:
                remaining.append(handle)
        pending = remaining

    for handle in pending:
        try:
            controller.force_kill(handle)
            force_killed += 1
        except Exception:
            pass

    # Also tear down PTY-backed branch sessions (POSIX shells).
    try:
        manager = get_pty_session_manager(workspace_root=WORKSPACE_ROOT, idle_timeout_seconds=300)
        if hasattr(manager, "destroy_all"):
            manager.destroy_all()
    except Exception:
        pass

    return {
        "active_before": active_before,
        "terminated": terminated,
        "force_killed": force_killed,
        "active_after": len(controller.active_handles),
    }


def _detect_project_root() -> Path:
    override = str(os.getenv("MONOLITH_PROJECT_ROOT") or "").strip()
    if override:
        try:
            return Path(override).expanduser().resolve()
        except Exception:
            pass
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    return WORKSPACE_ROOT


def _is_within_root(path: Path, root: Path) -> bool:
    p = str(path.resolve())
    r = str(root.resolve())
    return p == r or p.startswith(r + os.sep)


def get_boundary_config() -> dict[str, object]:
    return {
        "mode": BOUNDARY_MODE,
        "workspace_root": str(WORKSPACE_ROOT),
        "project_root": str(_detect_project_root()),
        "allowed_roots": [str(p) for p in ALLOWED_PATH_ROOTS],
    }


def set_boundary_config(
    *,
    mode: str | None = None,
    allowed_roots: list[str] | None = None,
) -> dict[str, object]:
    global BOUNDARY_MODE, ALLOWED_PATH_ROOTS
    if mode is not None:
        m = str(mode or "").strip().lower()
        if m not in {"strict_workspace", "project_root", "allowlist", "unrestricted"}:
            raise ValueError(f"invalid boundary mode: {mode}")
        BOUNDARY_MODE = m
    if allowed_roots is not None:
        parsed: list[Path] = []
        for item in allowed_roots:
            s = str(item or "").strip()
            if not s:
                continue
            parsed.append(Path(s).expanduser().resolve())
        ALLOWED_PATH_ROOTS = parsed
    return get_boundary_config()


def set_workspace_root(_: str | Path | None = None) -> None:
    global WORKSPACE_ROOT
    if _ is None:
        WORKSPACE_ROOT = DEFAULT_WORKSPACE_ROOT
        return
    try:
        WORKSPACE_ROOT = Path(_).expanduser().resolve()
    except Exception:
        WORKSPACE_ROOT = DEFAULT_WORKSPACE_ROOT


def resolve_path(user_path: str, *, allow_outside_boundary: bool = False) -> Path:
    # Strip workspace prefix if the model redundantly includes it.
    # e.g. "workspace/star.py" → "star.py" when WORKSPACE_ROOT already ends in /workspace
    raw = str(user_path or "").strip() or "."
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        target = expanded.resolve()
    else:
        cleaned = raw.replace("\\", "/")
        for alias in ("workspace_root", "$workspace_root", "${workspace_root}", "{workspace_root}"):
            if cleaned == alias:
                cleaned = "."
                break
            if cleaned.startswith(f"{alias}/"):
                cleaned = cleaned[len(alias) + 1:]
                break
        ws_name = WORKSPACE_ROOT.name  # e.g. "workspace"
        if cleaned.startswith(f"{ws_name}/"):
            cleaned = cleaned[len(ws_name) + 1:]
        elif cleaned == ws_name:
            cleaned = "."
        target = (WORKSPACE_ROOT / cleaned).expanduser().resolve()
    mode = BOUNDARY_MODE
    if mode == "unrestricted" or allow_outside_boundary:
        return target
    if mode == "project_root":
        root = _detect_project_root()
        if not _is_within_root(target, root):
            raise ValueError(f"path outside project_root boundary: {target}")
        return target
    if mode == "allowlist":
        if any(_is_within_root(target, root) for root in ALLOWED_PATH_ROOTS):
            return target
        raise ValueError(f"path outside allowlist boundary: {target}")
    # strict_workspace (default)
    if not _is_within_root(target, WORKSPACE_ROOT):
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


def _resolve_path(args: dict, default: str = ".", *, allow_outside_boundary: bool = False) -> Path:
    raw = str(args.get("path", default))
    return resolve_path(raw, allow_outside_boundary=allow_outside_boundary)


def _to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _to_int(value: object, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _workspace_relative(path: Path) -> str:
    try:
        return path.relative_to(WORKSPACE_ROOT).as_posix()
    except Exception:
        return str(path)


def _remove_target(path: Path, *, recursive: bool) -> None:
    if path.is_dir() and not path.is_symlink():
        if recursive:
            shutil.rmtree(path)
        else:
            path.rmdir()
    else:
        path.unlink()


def _run_capture(command: list[str], *, cwd: Path, timeout: int) -> tuple[int | None, str, str, str | None]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=float(timeout),
        )
    except subprocess.TimeoutExpired:
        return None, "", "", f"command timed out after {timeout}s"
    except Exception as exc:
        return None, "", "", f"command failed: {exc}"

    return completed.returncode, completed.stdout or "", completed.stderr or "", None


def read_file(args: dict) -> ToolResult:
    try:
        path = _resolve_path(args, allow_outside_boundary=True)
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
        path = _resolve_path(args, allow_outside_boundary=True)
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
        root = _resolve_path(args, allow_outside_boundary=True)
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


def mkdir(args: dict) -> ToolResult:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return ToolResult(False, "", "path is required")

    try:
        path = resolve_path(raw_path)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))

    parents = _to_bool(args.get("parents", True), default=True)
    exist_ok = _to_bool(args.get("exist_ok", True), default=True)

    if path.exists() and path.is_file():
        return ToolResult(False, "", f"path exists and is a file: {path}")

    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except FileExistsError:
        return ToolResult(False, "", f"directory already exists: {path}")
    except Exception as exc:
        return ToolResult(False, "", f"failed to create directory: {exc}")

    return ToolResult(True, f"directory ready: {_workspace_relative(path)}")


def move_path(args: dict) -> ToolResult:
    src_raw = args.get("src")
    dst_raw = args.get("dst")
    if not isinstance(src_raw, str) or not src_raw.strip():
        return ToolResult(False, "", "src is required")
    if not isinstance(dst_raw, str) or not dst_raw.strip():
        return ToolResult(False, "", "dst is required")

    try:
        src = resolve_path(src_raw)
        dst = resolve_path(dst_raw)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))

    overwrite = _to_bool(args.get("overwrite", False), default=False)

    if not src.exists():
        return ToolResult(False, "", f"source not found: {src}")
    if src == dst:
        return ToolResult(True, f"source and destination are identical: {_workspace_relative(src)}")

    if dst.exists():
        if not overwrite:
            return ToolResult(False, "", f"destination already exists: {dst}")
        try:
            _remove_target(dst, recursive=True)
        except Exception as exc:
            return ToolResult(False, "", f"failed to overwrite destination: {exc}")

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except Exception as exc:
        return ToolResult(False, "", f"failed to move path: {exc}")

    return ToolResult(True, f"moved {_workspace_relative(src)} -> {_workspace_relative(dst)}")


def copy_path(args: dict) -> ToolResult:
    src_raw = args.get("src")
    dst_raw = args.get("dst")
    if not isinstance(src_raw, str) or not src_raw.strip():
        return ToolResult(False, "", "src is required")
    if not isinstance(dst_raw, str) or not dst_raw.strip():
        return ToolResult(False, "", "dst is required")

    try:
        src = resolve_path(src_raw)
        dst = resolve_path(dst_raw)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))

    overwrite = _to_bool(args.get("overwrite", False), default=False)
    recursive = _to_bool(args.get("recursive", True), default=True)

    if not src.exists():
        return ToolResult(False, "", f"source not found: {src}")

    if src.is_dir() and not recursive:
        return ToolResult(False, "", "source is a directory; set recursive=true to copy directories")

    if dst.exists():
        if not overwrite:
            return ToolResult(False, "", f"destination already exists: {dst}")
        try:
            _remove_target(dst, recursive=True)
        except Exception as exc:
            return ToolResult(False, "", f"failed to overwrite destination: {exc}")

    try:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except Exception as exc:
        return ToolResult(False, "", f"failed to copy path: {exc}")

    return ToolResult(True, f"copied {_workspace_relative(src)} -> {_workspace_relative(dst)}")


def delete_path(args: dict) -> ToolResult:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return ToolResult(False, "", "path is required")

    try:
        path = resolve_path(raw_path)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))

    recursive = _to_bool(args.get("recursive", False), default=False)
    missing_ok = _to_bool(args.get("missing_ok", False), default=False)
    trash = _to_bool(args.get("trash", True), default=True)

    if not path.exists():
        if missing_ok:
            return ToolResult(True, f"path already absent: {_workspace_relative(path)}")
        return ToolResult(False, "", f"path not found: {path}")

    if trash:
        trash_root = WORKSPACE_ROOT / ".monolith_trash"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = trash_root / f"{stamp}_{path.name}"
        suffix = 1
        while destination.exists():
            destination = trash_root / f"{stamp}_{suffix}_{path.name}"
            suffix += 1
        try:
            trash_root.mkdir(parents=True, exist_ok=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
        except Exception as exc:
            return ToolResult(False, "", f"failed to move path to trash: {exc}")
        return ToolResult(True, f"moved {_workspace_relative(path)} to trash: {_workspace_relative(destination)}")

    try:
        _remove_target(path, recursive=recursive)
    except OSError as exc:
        return ToolResult(False, "", f"failed to delete path: {exc}. Use recursive=true for non-empty directories.")
    except Exception as exc:
        return ToolResult(False, "", f"failed to delete path: {exc}")
    return ToolResult(True, f"deleted {_workspace_relative(path)}")


def zip_path(args: dict) -> ToolResult:
    src_raw = args.get("src")
    dst_raw = args.get("dst")
    if not isinstance(src_raw, str) or not src_raw.strip():
        return ToolResult(False, "", "src is required")
    if not isinstance(dst_raw, str) or not dst_raw.strip():
        return ToolResult(False, "", "dst is required")

    try:
        src = resolve_path(src_raw)
        dst = resolve_path(dst_raw)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))

    recursive = _to_bool(args.get("recursive", True), default=True)
    overwrite = _to_bool(args.get("overwrite", False), default=False)

    if not src.exists():
        return ToolResult(False, "", f"source not found: {src}")
    if dst.exists():
        if not overwrite:
            return ToolResult(False, "", f"destination already exists: {dst}")
        try:
            _remove_target(dst, recursive=True)
        except Exception as exc:
            return ToolResult(False, "", f"failed to overwrite destination: {exc}")

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return ToolResult(False, "", f"failed to create destination directory: {exc}")

    archived_files = 0
    try:
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if src.is_file():
                archive.write(src, arcname=src.name)
                archived_files = 1
            elif src.is_dir():
                file_iter = src.rglob("*") if recursive else src.iterdir()
                for file_path in file_iter:
                    if not file_path.is_file():
                        continue
                    arcname = file_path.relative_to(src.parent).as_posix()
                    archive.write(file_path, arcname=arcname)
                    archived_files += 1
                if archived_files == 0:
                    archive.writestr(f"{src.name}/", "")
            else:
                return ToolResult(False, "", f"source is not a file or directory: {src}")
    except Exception as exc:
        return ToolResult(False, "", f"failed to create zip archive: {exc}")

    return ToolResult(
        True,
        f"archived {archived_files} file(s) from {_workspace_relative(src)} to {_workspace_relative(dst)}",
    )


def unzip_archive(args: dict) -> ToolResult:
    src_raw = args.get("src")
    dst_raw = args.get("dst")
    if not isinstance(src_raw, str) or not src_raw.strip():
        return ToolResult(False, "", "src is required")
    if not isinstance(dst_raw, str) or not dst_raw.strip():
        return ToolResult(False, "", "dst is required")

    try:
        src = resolve_path(src_raw)
        dst = resolve_path(dst_raw)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))

    overwrite = _to_bool(args.get("overwrite", False), default=False)

    if not src.exists() or not src.is_file():
        return ToolResult(False, "", f"archive not found: {src}")
    if dst.exists() and dst.is_file():
        return ToolResult(False, "", f"destination exists and is a file: {dst}")

    try:
        dst.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return ToolResult(False, "", f"failed to create destination directory: {exc}")

    extracted_files = 0
    try:
        with zipfile.ZipFile(src, "r") as archive:
            planned: list[tuple[zipfile.ZipInfo, Path, bool]] = []
            for info in archive.infolist():
                raw_name = (info.filename or "").strip()
                if not raw_name:
                    continue
                normalized = PurePosixPath(raw_name.replace("\\", "/"))
                if normalized.is_absolute() or any(part == ".." for part in normalized.parts):
                    return ToolResult(False, "", f"unsafe archive entry path: {raw_name}")
                target = (dst / normalized.as_posix()).resolve()
                if not str(target).startswith(str(dst)):
                    return ToolResult(False, "", f"archive entry escapes destination: {raw_name}")
                is_dir = info.is_dir() or raw_name.endswith("/")
                if target.exists() and not overwrite and not is_dir:
                    return ToolResult(False, "", f"destination file exists: {_workspace_relative(target)}")
                planned.append((info, target, is_dir))

            for info, target, is_dir in planned:
                if is_dir:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source_fp, target.open("wb") as target_fp:
                    shutil.copyfileobj(source_fp, target_fp)
                extracted_files += 1
    except zipfile.BadZipFile:
        return ToolResult(False, "", f"invalid zip archive: {src}")
    except Exception as exc:
        return ToolResult(False, "", f"failed to unzip archive: {exc}")

    return ToolResult(
        True,
        f"extracted {extracted_files} file(s) from {_workspace_relative(src)} to {_workspace_relative(dst)}",
    )


def glob_files(args: dict) -> ToolResult:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return ToolResult(False, "", "pattern is required")

    try:
        root = _resolve_path(args, allow_outside_boundary=True)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))

    if not root.exists() or not root.is_dir():
        return ToolResult(False, "", f"directory not found: {root}")

    include_dirs = _to_bool(args.get("include_dirs", False), default=False)
    limit = _to_int(args.get("limit", 500), default=500, min_value=1, max_value=5000)

    try:
        entries = sorted(root.glob(pattern), key=lambda p: str(p).lower())
    except Exception as exc:
        return ToolResult(False, "", f"invalid glob pattern: {exc}")

    output: list[str] = []
    for entry in entries:
        if entry.is_dir() and not include_dirs:
            continue
        rel = _workspace_relative(entry)
        if entry.is_dir():
            rel = f"{rel}/"
        output.append(rel)

    total = len(output)
    if total == 0:
        return ToolResult(True, f"No matches found for '{pattern}' in {_workspace_relative(root)}")

    if total > limit:
        output = output[:limit]
        output.append(f"[RESULTS TRUNCATED -- showing first {limit} of {total} matches]")

    return ToolResult(True, "\n".join(output))


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


def git_status(args: dict) -> ToolResult:
    try:
        target = _resolve_path(args, allow_outside_boundary=True)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))

    if target.is_file():
        target = target.parent

    timeout = _to_int(args.get("timeout", 20), default=20, min_value=1, max_value=120)
    code, stdout, stderr, run_error = _run_capture(
        ["git", "-C", str(target), "status", "--short", "--branch"],
        cwd=WORKSPACE_ROOT,
        timeout=timeout,
    )
    if run_error:
        return ToolResult(False, "", run_error)
    if code != 0:
        msg = stderr.strip() or stdout.strip() or "git status failed"
        return ToolResult(False, "", msg)

    content = stdout.strip() if stdout.strip() else "(working tree clean)"
    return ToolResult(True, content)


def git_diff(args: dict) -> ToolResult:
    try:
        target = _resolve_path(args, allow_outside_boundary=True)
    except ValueError as exc:
        return ToolResult(False, "", str(exc))

    if target.is_file():
        target = target.parent

    timeout = _to_int(args.get("timeout", 20), default=20, min_value=1, max_value=120)
    context = _to_int(args.get("context", 3), default=3, min_value=0, max_value=20)
    max_lines = _to_int(args.get("max_lines", 500), default=500, min_value=20, max_value=5000)
    staged = _to_bool(args.get("staged", False), default=False)
    base = args.get("base")
    file_path = args.get("file")

    command = ["git", "-C", str(target), "diff", f"--unified={context}"]
    if staged:
        command.append("--staged")
    if isinstance(base, str) and base.strip():
        command.append(base.strip())

    if isinstance(file_path, str) and file_path.strip():
        try:
            resolved_file = resolve_path(file_path)
            rel = resolved_file.relative_to(target)
        except Exception:
            return ToolResult(False, "", f"file must be inside {target}")
        command.extend(["--", rel.as_posix()])

    code, stdout, stderr, run_error = _run_capture(command, cwd=WORKSPACE_ROOT, timeout=timeout)
    if run_error:
        return ToolResult(False, "", run_error)
    if code != 0:
        msg = stderr.strip() or stdout.strip() or "git diff failed"
        return ToolResult(False, "", msg)

    lines = stdout.splitlines()
    if not lines:
        return ToolResult(True, "(no diff)")

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"[OUTPUT TRUNCATED -- showing first {max_lines} lines]")

    return ToolResult(True, "\n".join(lines))


def run_tests(args: dict) -> ToolResult:
    command = args.get("command", "pytest -q")
    if not isinstance(command, str) or not command.strip():
        return ToolResult(False, "", "command must be a non-empty string")

    timeout = _to_int(args.get("timeout", 120), default=120, min_value=1, max_value=3600)
    pty_enabled = _to_bool(args.get("pty_enabled", False), default=False)

    result = run_cmd({
        "command": command,
        "timeout": timeout,
        "pty_enabled": pty_enabled,
    })

    status = "passed" if result.ok else "failed"
    summary = f"command: {command}\nstatus: {status}"
    content = f"{summary}\n{result.content}" if result.content else summary
    return ToolResult(result.ok, content, result.error)


def _is_private_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _is_blocked_host(hostname: str) -> bool:
    host = hostname.strip().lower().rstrip(".")
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}:
        return True
    if host.endswith(".local") or host.endswith(".internal"):
        return True
    if _is_private_ip(host):
        return True

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    except Exception:
        return False

    for info in infos:
        address = info[4][0]
        if _is_private_ip(address):
            return True
    return False


def _allowlisted_host(hostname: str) -> bool:
    raw = os.environ.get("MONOLITH_HTTP_FETCH_ALLOWLIST", "")
    entries = [part.strip().lower().lstrip(".") for part in raw.split(",") if part.strip()]
    if not entries:
        return True

    host = hostname.strip().lower().rstrip(".")
    for entry in entries:
        if host == entry or host.endswith(f".{entry}"):
            return True
    return False


def _validate_fetch_host(hostname: str) -> str | None:
    host = str(hostname or "").strip()
    if not host:
        return "url is missing hostname"
    if _is_blocked_host(host):
        return f"blocked host: {host}"
    if not _allowlisted_host(host):
        return f"host not in MONOLITH_HTTP_FETCH_ALLOWLIST: {host}"
    return None


def _decode_http_body(data: bytes, content_type: str) -> str:
    charset = "utf-8"
    if isinstance(content_type, str):
        match = re.search(r"charset=([A-Za-z0-9_\-]+)", content_type, flags=re.IGNORECASE)
        if match:
            charset = match.group(1)
    try:
        return data.decode(charset, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")


def _classify_waf_challenge(text: str, headers: dict[str, str], status_code: int) -> str | None:
    lowered = text.lower()
    markers = (
        "just a moment",
        "cf-challenge",
        "cdn-cgi/challenge-platform",
        "<title>attention required",
        "captcha",
        "turnstile",
    )

    server = headers.get("server", "").lower()
    is_cloudflare = "cloudflare" in server or "cf-ray" in {k.lower() for k in headers.keys()}

    if any(marker in lowered for marker in markers):
        if is_cloudflare:
            return "cloudflare_js"
        if "captcha" in lowered:
            return "captcha"
        return "unknown"

    if status_code in {403, 429, 503} and is_cloudflare and "/cdn-cgi/" in lowered:
        return "cloudflare_js"
    return None


def http_fetch(args: dict) -> ToolResult:
    raw_url = args.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        return ToolResult(False, "", "url is required")

    parsed = urllib.parse.urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"}:
        return ToolResult(False, "", "url must use http or https")
    if not parsed.hostname:
        return ToolResult(False, "", "url is missing hostname")

    host_error = _validate_fetch_host(str(parsed.hostname or ""))
    if host_error:
        return ToolResult(False, "", host_error)

    method = str(args.get("method", "GET")).strip().upper()
    if method not in {"GET", "HEAD"}:
        return ToolResult(False, "", "method must be GET or HEAD")

    timeout = _to_int(args.get("timeout", 15), default=15, min_value=1, max_value=60)
    max_bytes = _to_int(args.get("max_bytes", 200_000), default=200_000, min_value=1024, max_value=2_000_000)
    follow_redirects = _to_bool(args.get("follow_redirects", True), default=True)

    request_headers: dict[str, str] = {
        "User-Agent": "Monolith/1.0 (+local-agent)",
        "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.8",
    }
    raw_headers = args.get("headers")
    allowed_header_names = {"accept", "accept-language", "user-agent"}
    if isinstance(raw_headers, dict):
        for key, value in raw_headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            if key.lower() in allowed_header_names:
                request_headers[key] = value

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            return None

    class _SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            target = urllib.parse.urlparse(str(newurl or ""))
            host = str(target.hostname or "").strip()
            host_err = _validate_fetch_host(host)
            if host_err:
                raise urllib.error.URLError(f"redirect blocked: {host_err}")
            return super().redirect_request(req, fp, code, msg, hdrs, newurl)

    opener = urllib.request.build_opener(_SafeRedirect()) if follow_redirects else urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(raw_url.strip(), headers=request_headers, method=method)

    status_code = 0
    final_url = raw_url.strip()
    headers: dict[str, str] = {}
    body_bytes = b""

    try:
        with opener.open(request, timeout=float(timeout)) as response:
            status_code = int(getattr(response, "status", response.getcode() or 0) or 0)
            final_url = response.geturl() or final_url
            headers = dict(response.headers.items())
            if method == "GET":
                body_bytes = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code or 0)
        final_url = exc.geturl() or final_url
        headers = dict(exc.headers.items()) if exc.headers else {}
        if method == "GET":
            body_bytes = exc.read(max_bytes + 1)
    except Exception as exc:
        return ToolResult(False, "", f"http_fetch failed: {exc}")

    final_host = urllib.parse.urlparse(final_url).hostname
    final_host_err = _validate_fetch_host(str(final_host or ""))
    if final_host_err:
        return ToolResult(False, "", f"redirect blocked: {final_host_err}")

    truncated = len(body_bytes) > max_bytes
    if truncated:
        body_bytes = body_bytes[:max_bytes]

    content_type = headers.get("Content-Type", "")
    text = _decode_http_body(body_bytes, content_type)

    challenge_type = _classify_waf_challenge(text, headers, status_code)
    kind = "page"
    suggestion = None
    if challenge_type:
        kind = "waf_challenge"
        suggestion = "Challenge detected. Prefer API/RSS/direct source links."
    elif status_code >= 400:
        kind = "http_error"

    payload = {
        "kind": kind,
        "url": raw_url.strip(),
        "final_url": final_url,
        "status_code": status_code,
        "content_type": content_type,
        "text": text,
        "truncated": truncated,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    if challenge_type:
        payload["challenge_type"] = challenge_type
    if suggestion:
        payload["suggestion"] = suggestion

    return ToolResult(True, json.dumps(payload, ensure_ascii=False, indent=2))


TOOL_REGISTRY: dict[str, Callable[[dict], ToolResult]] = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "grep_search": grep_search,
    "glob_files": glob_files,
    "mkdir": mkdir,
    "move_path": move_path,
    "copy_path": copy_path,
    "delete_path": delete_path,
    "zip_path": zip_path,
    "unzip_archive": unzip_archive,
    "run_cmd": run_cmd,
    "run_tests": run_tests,
    "apply_patch": apply_patch,
    "run_python": run_python,
    "git_status": git_status,
    "git_diff": git_diff,
    "http_fetch": http_fetch,
}
