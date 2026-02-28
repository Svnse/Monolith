from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


FAILURE_UNKNOWN = "unknown"
FAILURE_MISSING_DEPENDENCY = "missing_dependency"
FAILURE_PERMISSION = "permission"
FAILURE_PATH_NOT_FOUND = "path_not_found"
FAILURE_SYNTAX = "syntax_error"
FAILURE_TIMEOUT = "timeout"
FAILURE_RUNTIME = "runtime_error"


@dataclass(frozen=True)
class FailureClassification:
    failure_class: str
    error_code: str
    detail: str
    directive: str


_RE_MODULE = re.compile(r"(?:No module named|ModuleNotFoundError:\s*No module named)\s+['\"]?([A-Za-z0-9_.\-]+)['\"]?", re.IGNORECASE)
_RE_PERMISSION = re.compile(r"permission denied|access is denied|operation not permitted", re.IGNORECASE)
_RE_NOT_FOUND = re.compile(r"no such file|file not found|path not found|cannot find the path", re.IGNORECASE)
_RE_SYNTAX = re.compile(r"syntaxerror|indentationerror|nameerror", re.IGNORECASE)
_RE_TIMEOUT = re.compile(r"timed out|timeout", re.IGNORECASE)


def classify_tool_failure(tool: str, args: dict[str, Any], output: str) -> FailureClassification:
    txt = str(output or "").strip()
    lower = txt.lower()
    tname = str(tool or "").strip().lower()

    mod_match = _RE_MODULE.search(txt)
    if mod_match:
        dep = mod_match.group(1).strip()
        return FailureClassification(
            failure_class=FAILURE_MISSING_DEPENDENCY,
            error_code=f"missing_dependency:{dep}",
            detail=f"Missing dependency '{dep}'",
            directive=f"Dependency missing: {dep}. Use run_cmd to install it, or use read_file to inspect code without executing.",
        )
    if _RE_PERMISSION.search(lower):
        return FailureClassification(
            failure_class=FAILURE_PERMISSION,
            error_code="permission_denied",
            detail="Permission denied",
            directive="Permission issue detected. Correct path/scope/approval or choose a non-mutating alternative.",
        )
    if _RE_NOT_FOUND.search(lower):
        return FailureClassification(
            failure_class=FAILURE_PATH_NOT_FOUND,
            error_code="path_not_found",
            detail="Path or file not found",
            directive="Missing path/file. Re-list directory and correct path before retrying.",
        )
    if _RE_SYNTAX.search(lower):
        return FailureClassification(
            failure_class=FAILURE_SYNTAX,
            error_code="syntax_error",
            detail="Syntax/runtime code error detected",
            directive="Use read_file to inspect source, then apply_patch/write_file to fix it, then run_cmd for targeted verification.",
        )
    if _RE_TIMEOUT.search(lower):
        return FailureClassification(
            failure_class=FAILURE_TIMEOUT,
            error_code="timeout",
            detail="Execution timed out",
            directive="Execution timeout. Use run_cmd with a narrower command, or use read_file for static verification.",
        )
    if tname in {"run_cmd", "run_tests", "run_python"}:
        return FailureClassification(
            failure_class=FAILURE_RUNTIME,
            error_code="runtime_error",
            detail="Execution failed",
            directive="Execution failed. Use run_cmd to capture stderr, then use read_file and apply_patch to correct the issue.",
        )
    return FailureClassification(
        failure_class=FAILURE_UNKNOWN,
        error_code="unknown",
        detail="Tool failed",
        directive="Tool failed. Use list_dir and read_file to re-check assumptions, then choose a concrete tool action.",
    )


def normalize_action_signature(tool: str, args: dict[str, Any] | None) -> str:
    tname = str(tool or "").strip().lower()
    a = args if isinstance(args, dict) else {}

    if tname == "run_cmd":
        cmd = _normalize_ws(str(a.get("command") or ""))
        return f"run_cmd:command={cmd}"
    if tname in {"write_file", "read_file", "list_dir", "grep_search", "glob_files", "git_status", "git_diff"}:
        path = _normalize_ws(str(a.get("path") or ""))
        if tname == "grep_search":
            pattern = _normalize_ws(str(a.get("pattern") or ""))
            return f"{tname}:path={path}|pattern={pattern}"
        if tname == "glob_files":
            pattern = _normalize_ws(str(a.get("pattern") or ""))
            return f"{tname}:path={path}|glob={pattern}"
        return f"{tname}:path={path}"
    if tname in {"move_path", "copy_path"}:
        src = _normalize_ws(str(a.get("src") or ""))
        dst = _normalize_ws(str(a.get("dst") or ""))
        return f"{tname}:src={src}|dst={dst}"
    return f"{tname}:{_normalize_ws(str(sorted(a.items())))}"


def should_trip_circuit(
    *,
    signature: str,
    failure_counts: dict[str, int],
    threshold: int = 2,
) -> bool:
    count = int(failure_counts.get(signature, 0))
    return count >= int(max(1, threshold))


def pivot_directive_for_class(failure_class: str) -> str:
    fc = str(failure_class or "").strip().lower()
    if fc == FAILURE_MISSING_DEPENDENCY:
        return "Missing dependency path blocked. Use run_cmd to install dependency, or use read_file for static verification."
    if fc in {FAILURE_PERMISSION, FAILURE_PATH_NOT_FOUND}:
        return "Environment/path issue. Correct boundary/path assumptions before retrying execution."
    if fc in {FAILURE_SYNTAX, FAILURE_RUNTIME}:
        return "Execution path unstable. Use read_file and apply_patch for targeted fixes, then use run_cmd for verification."
    if fc == FAILURE_TIMEOUT:
        return "Execution timed out. Narrow command or avoid full runtime execution."
    return "Repeated tool failure. Use list_dir/read_file to gather evidence, then choose a different concrete tool call."


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def extract_missing_dependency_from_code(error_code: str) -> str:
    code = str(error_code or "").strip().lower()
    prefix = "missing_dependency:"
    if code.startswith(prefix):
        return code[len(prefix):].strip()
    return ""


def extract_install_target_from_command(command: str) -> str:
    cmd = _normalize_ws(command)
    # Supports common shapes:
    #   pip install pygame
    #   python -m pip install pygame
    m = re.search(r"(?:^|\s)(?:python -m )?pip install\s+([a-z0-9_.\-]+)", cmd)
    if not m:
        return ""
    return str(m.group(1) or "").strip()
