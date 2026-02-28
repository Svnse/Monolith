"""
Monolith integration helpers for engine.loop.

This module keeps `engine.loop` itself framework-agnostic while providing
small adapters to:
  - expose Monolith tools as `ToolSpec`s
  - execute Monolith tools via `engine.tools.TOOL_REGISTRY`
  - bridge loop events to trace logging
  - wrap a loaded `LLMEngine` instance as `infer_fn(messages) -> str`
"""

from __future__ import annotations

import json
from threading import Lock
from typing import Any, Callable

from engine.loop.contracts import ToolSpec
from engine.tools import TOOL_REGISTRY


_TOOL_PARAM_HINTS: dict[str, dict[str, Any]] = {
    "read_file": {"path": "string", "offset": "int?", "limit": "int?"},
    "write_file": {"path": "string", "content": "string"},
    "list_dir": {"path": "string?", "pattern": "string?"},
    "grep_search": {"pattern": "regex string", "path": "string?"},
    "glob_files": {"pattern": "glob string", "path": "string?"},
    "mkdir": {"path": "string", "parents": "bool?", "exist_ok": "bool?"},
    "move_path": {"src": "string", "dst": "string", "overwrite": "bool?"},
    "copy_path": {"src": "string", "dst": "string", "overwrite": "bool?", "recursive": "bool?"},
    "delete_path": {"path": "string", "recursive": "bool?", "trash": "bool?"},
    "run_cmd": {"command": "string", "timeout": "int?", "pty_enabled": "bool?"},
    "run_tests": {"command": "string?", "timeout": "int?"},
    "apply_patch": {"path": "string", "old": "string", "new": "string"},
    "run_python": {"code": "string", "timeout": "int?"},
    "git_status": {"path": "string?"},
    "git_diff": {"path": "string?", "base": "string?", "staged": "bool?"},
    "http_fetch": {"url": "string", "method": "GET|HEAD?", "timeout": "int?"},
}

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": "Read a text file (supports offset/limit).",
    "write_file": "Write a text file, creating directories as needed.",
    "list_dir": "List files/directories under a path.",
    "grep_search": "Regex search across a file or directory tree.",
    "glob_files": "Find files using a glob pattern.",
    "mkdir": "Create a directory.",
    "move_path": "Move/rename a file or directory.",
    "copy_path": "Copy a file or directory.",
    "delete_path": "Delete a file or directory (optionally to trash).",
    "run_cmd": "Run a shell command inside the workspace.",
    "run_tests": "Run tests via a shell command (defaults to pytest -q).",
    "apply_patch": "Replace a string in a file (simple patch operation).",
    "run_python": "Execute Python code in a controlled runtime.",
    "git_status": "Show git status for the workspace/repo.",
    "git_diff": "Show git diff (optionally base/staged/file path).",
    "http_fetch": "Fetch a public HTTP/HTTPS URL with safety restrictions.",
}

_TOOL_WHEN_TO_USE: dict[str, str] = {
    "read_file": "Inspect exact file contents before editing.",
    "write_file": "Create or fully replace a file's contents.",
    "list_dir": "Discover files/directories when path or structure is uncertain.",
    "grep_search": "Find text/regex matches across files quickly.",
    "glob_files": "Find files by pattern without opening contents.",
    "mkdir": "Create required directory structure before writing.",
    "move_path": "Rename or relocate existing files/directories.",
    "copy_path": "Duplicate files/directories while preserving source.",
    "delete_path": "Remove files/directories only when explicitly intended.",
    "run_cmd": "Execute shell commands for build/run/verification tasks.",
    "run_tests": "Execute test command to validate behavior.",
    "apply_patch": "Perform a targeted single text replacement in a file.",
    "run_python": "Run non-trivial Python logic where shell is insufficient.",
    "git_status": "Check working tree state and branch status.",
    "git_diff": "Inspect exact source changes.",
    "http_fetch": "Fetch a public URL for external content when needed.",
}

_TOOL_WHEN_NOT_TO_USE: dict[str, str] = {
    "write_file": "Do not use for tiny edits when apply_patch is safer.",
    "run_cmd": "Do not use as first choice for simple file reads/writes.",
    "delete_path": "Do not use without explicit deletion intent.",
    "run_python": "Do not use for simple command execution; prefer run_cmd.",
    "http_fetch": "Do not use for local files or private/internal hosts.",
}

_TOOL_REQUIRED_ARGS: dict[str, list[str]] = {
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "list_dir": [],
    "grep_search": ["pattern"],
    "glob_files": ["pattern"],
    "mkdir": ["path"],
    "move_path": ["src", "dst"],
    "copy_path": ["src", "dst"],
    "delete_path": ["path"],
    "run_cmd": ["command"],
    "run_tests": [],
    "apply_patch": ["path", "old", "new"],
    "run_python": ["code"],
    "git_status": [],
    "git_diff": [],
    "http_fetch": ["url"],
}

_TOOL_FAILURE_RECOVERY: dict[str, str] = {
    "read_file": "Use list_dir to confirm path, then retry with correct path/offset.",
    "write_file": "Check parent path and boundary, then retry with corrected path/content.",
    "list_dir": "Verify directory exists and is within workspace boundary.",
    "grep_search": "Broaden pattern/path or validate regex syntax.",
    "glob_files": "Try a broader pattern or confirm root path.",
    "mkdir": "Check boundary and parent permissions; retry with parents=true.",
    "move_path": "Verify src exists and dst overwrite policy.",
    "copy_path": "Verify src exists and recursive/overwrite flags.",
    "delete_path": "Re-check intent and recursive/trash flags before retry.",
    "run_cmd": "Inspect stderr/exit code with run_cmd. If repeated failure, use read_file to inspect code/output instead of rerunning.",
    "run_tests": "Adjust command/path and rerun a narrower test subset.",
    "apply_patch": "Re-read file and patch exact current text.",
    "run_python": "Capture traceback and simplify script or add guards.",
    "git_status": "Verify repository path and git availability.",
    "git_diff": "Verify repository path and optional file/base args.",
    "http_fetch": "Validate URL/host policy; prefer direct stable source.",
}

_TOOL_EXAMPLE_CALLS: dict[str, list[dict[str, Any]]] = {
    "read_file": [{"path": "src/app.py"}],
    "write_file": [{"path": "src/app.py", "content": "print('hi')\n"}],
    "list_dir": [{"path": "."}],
    "grep_search": [{"pattern": "TODO", "path": "."}],
    "glob_files": [{"pattern": "**/*.py", "path": "."}],
    "mkdir": [{"path": "src/new"}],
    "run_cmd": [{"command": "python -m pytest -q"}],
    "apply_patch": [{"path": "src/app.py", "old": "foo", "new": "bar"}],
    "run_tests": [{"command": "pytest -q"}],
}


def _infer_scope(tool_name: str) -> str:
    name = str(tool_name or "").lower()
    if name == "list_dir":
        return "list"
    if name == "grep_search":
        return "grep"
    if name == "glob_files":
        return "search"
    if name in {"read_file", "git_status", "git_diff", "http_fetch"}:
        return "read"
    if name in {"write_file", "apply_patch", "mkdir", "move_path", "copy_path", "zip_path", "unzip_archive"}:
        return "write"
    if name == "delete_path":
        return "delete"
    if name == "run_cmd":
        return "shell"
    if name in {"run_tests", "run_python"}:
        return "execute"
    return "read"


def build_monolith_tool_specs(
    *,
    tool_names: list[str] | None = None,
    registry: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> list[ToolSpec]:
    """
    Convert Monolith tool registry entries into loop `ToolSpec`s.

    Unknown tools still get a usable fallback spec so the loop prompt can list
    them, but descriptions/parameters may be generic.
    """
    reg = registry or TOOL_REGISTRY
    names = tool_names or sorted(reg.keys())
    specs: list[ToolSpec] = []
    for name in names:
        if name not in reg:
            continue
        specs.append(
            ToolSpec(
                name=name,
                description=_TOOL_DESCRIPTIONS.get(name, f"Monolith tool: {name}"),
                parameters=dict(_TOOL_PARAM_HINTS.get(name, {})),
                scope=_infer_scope(name),
                when_to_use=_TOOL_WHEN_TO_USE.get(name, ""),
                when_not_to_use=_TOOL_WHEN_NOT_TO_USE.get(name, ""),
                required_args=list(_TOOL_REQUIRED_ARGS.get(name, [])),
                failure_recovery=_TOOL_FAILURE_RECOVERY.get(name, ""),
                example_calls=list(_TOOL_EXAMPLE_CALLS.get(name, [])),
            )
        )
    return specs


def execute_monolith_tool(
    tool_name: str,
    args: dict[str, Any] | None,
    *,
    registry: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> str:
    """
    Loop-compatible tool executor wrapper around Monolith `TOOL_REGISTRY`.

    Returns the successful content string.
    Raises `RuntimeError` on tool failure so `LoopRuntime` records failed evidence.
    """
    reg = registry or TOOL_REGISTRY
    fn = reg.get(tool_name)
    if fn is None:
        raise RuntimeError(f"unknown tool: {tool_name}")

    payload = args if isinstance(args, dict) else {}
    result = fn(payload)

    ok = bool(getattr(result, "ok", False))
    content = str(getattr(result, "content", "") or "")
    error = getattr(result, "error", None)
    error_text = str(error or "").strip()

    if not ok:
        raise RuntimeError(error_text or content or f"{tool_name} failed")

    return content


def make_loop_trace_emitter(trace_fn: Callable[[str], None]) -> Callable[[str, dict[str, Any]], None]:
    """
    Adapt LoopRuntime `on_event(kind, data)` to a plain trace logger callback.
    """
    def _emit(kind: str, data: dict[str, Any]) -> None:
        try:
            if kind == "cycle_start":
                trace_fn(f"[LOOP] cycle={data.get('cycle')} start")
            elif kind == "step_parsed":
                step = data.get("step") or {}
                trace_fn(
                    f"[LOOP] step intent={step.get('intent')} actions={step.get('actions_count')} task_finished={step.get('task_finished')}"
                )
            elif kind == "action_result":
                tool = data.get("tool", "?")
                ok = bool(data.get("ok"))
                trace_fn(f"[LOOP] tool {tool}: {'ok' if ok else 'fail'}")
            elif kind == "approval_required":
                trace_fn(f"[LOOP] approval required: tool={data.get('tool')} scope={data.get('scope')}")
            elif kind == "retry":
                preview = str(data.get("raw_preview") or "").strip().replace("\n", " ")
                if len(preview) > 120:
                    preview = preview[:120] + "..."
                suffix = f" | raw={preview!r}" if preview else ""
                trace_fn(
                    f"[LOOP] retry cycle={data.get('cycle')} attempt={data.get('attempt')}: {data.get('error')}{suffix}"
                )
            elif kind == "circuit_breaker":
                trace_fn(
                    "[LOOP] circuit_breaker: "
                    f"tool={data.get('tool')} class={data.get('failure_class')} "
                    f"signature={data.get('signature')}"
                )
            elif kind == "noop_blocked":
                trace_fn(
                    "[LOOP] noop_blocked: "
                    f"tool={data.get('tool')} signature={data.get('signature')}"
                )
            elif kind == "routing_blocked":
                trace_fn(
                    "[LOOP] routing_blocked: "
                    f"tool={data.get('tool')} dep={data.get('dependency')} signature={data.get('signature')}"
                )
            elif kind == "wall_hit":
                trace_fn(f"[LOOP] wall hit: {data.get('wall')} (cycle={data.get('cycle')})")
            elif kind == "finish":
                result = data.get("result") or {}
                trace_fn(
                    f"[LOOP] finish success={result.get('success')} cycles={result.get('cycles_used')} tools={result.get('tool_calls_used')}"
                )
            elif kind == "error":
                trace_fn(f"[LOOP] error: {data}")
            else:
                trace_fn(f"[LOOP] {kind}: {data}")
        except Exception:
            # Trace sink must not break the runtime.
            pass
    return _emit


class LLMEngineInferAdapter:
    """
    Duck-typed adapter that wraps a loaded `engine.llm.LLMEngine` as `infer_fn`.

    This is a synchronous call path meant for an external loop runner.
    It intentionally avoids touching `GeneratorWorker` internals.
    """

    _STOP_SEQUENCES = ["</response>", "</answer>", "</output>", "<|end|>", "<|im_end|>"]

    def __init__(
        self,
        llm_engine: Any,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stream_tokens: bool = True,
        on_token: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        enforce_json_object: bool = False,
    ) -> None:
        self.engine = llm_engine
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream_tokens = bool(stream_tokens)
        self.on_token = on_token
        self.should_stop = should_stop
        self.enforce_json_object = bool(enforce_json_object)
        self._emit_lock = Lock()

    def __call__(self, messages: list[dict[str, str]]) -> str:
        eng = self.engine
        if getattr(eng, "llm", None) is None or not bool(getattr(eng, "model_loaded", False)):
            raise RuntimeError("LLMEngine model is not loaded")

        worker = getattr(eng, "worker", None)
        if worker is not None and hasattr(worker, "isRunning") and worker.isRunning():
            raise RuntimeError("LLMEngine is busy with an active generation")

        llm = eng.llm
        kwargs_base = {
            "messages": messages,
            "temperature": float(self.temperature if self.temperature is not None else getattr(eng, "temp", 0.7)),
            "top_p": float(self.top_p if self.top_p is not None else getattr(eng, "top_p", 0.9)),
            "max_tokens": int(self.max_tokens if self.max_tokens is not None else getattr(eng, "max_tokens", 1024)),
            "stop": list(self._STOP_SEQUENCES),
        }
        if self.enforce_json_object:
            kwargs_base["response_format"] = {"type": "json_object"}
        if self.stream_tokens:
            return self._call_streaming(llm, eng, kwargs_base)
        return self._call_non_streaming(llm, eng, kwargs_base)

    def _invoke_with_fallback(self, llm: Any, kwargs: dict[str, Any]) -> Any:
        """
        Call backend and gracefully retry without response_format if unsupported.
        """
        try:
            return llm.create_chat_completion(**kwargs)
        except TypeError:
            # Some llama.cpp builds/providers don't accept response_format.
            if "response_format" in kwargs:
                stripped = dict(kwargs)
                stripped.pop("response_format", None)
                return llm.create_chat_completion(**stripped)
            raise
        except Exception as exc:
            if "response_format" in kwargs:
                msg = str(exc).lower()
                unsupported = any(s in msg for s in (
                    "response_format",
                    "json_schema",
                    "unsupported",
                    "unexpected keyword",
                ))
                if unsupported:
                    stripped = dict(kwargs)
                    stripped.pop("response_format", None)
                    return llm.create_chat_completion(**stripped)
            raise

    def _call_non_streaming(self, llm: Any, eng: Any, kwargs_base: dict[str, Any]) -> str:
        kwargs = dict(kwargs_base)
        kwargs["stream"] = False
        lock = getattr(eng, "_runtime_lock", None)
        if lock is not None:
            with lock:
                response = self._invoke_with_fallback(llm, kwargs)
        else:
            response = self._invoke_with_fallback(llm, kwargs)
        text = self._extract_text(response)
        if text and self.on_token is not None:
            try:
                self.on_token(text)
            except Exception:
                pass
        return text

    def _call_streaming(self, llm: Any, eng: Any, kwargs_base: dict[str, Any]) -> str:
        kwargs = dict(kwargs_base)
        kwargs["stream"] = True
        parts: list[str] = []
        lock = getattr(eng, "_runtime_lock", None)
        if lock is not None:
            with lock:
                stream = self._invoke_with_fallback(llm, kwargs)
                self._consume_stream(stream, parts)
        else:
            stream = self._invoke_with_fallback(llm, kwargs)
            self._consume_stream(stream, parts)
        return "".join(parts)

    def _consume_stream(self, stream: Any, parts: list[str]) -> None:
        for chunk in stream:
            if self.should_stop is not None and bool(self.should_stop()):
                break
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice0 = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice0.get("delta") if isinstance(choice0.get("delta"), dict) else {}
            token = delta.get("content")
            if not isinstance(token, str) or not token:
                continue
            parts.append(token)
            if self.on_token is not None:
                try:
                    # Serialize callback if llama stream invokes from unusual context.
                    with self._emit_lock:
                        self.on_token(token)
                except Exception:
                    pass

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(str(c) for c in chunks)
        # Fallback for odd providers
        try:
            return json.dumps(content)
        except Exception:
            return str(content or "")
