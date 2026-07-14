"""
core/cmd_parser.py - Parse and execute <tool_call> envelopes emitted by
the LLM inside its response text.

Wire format (Hermes/Qwen native, single tool call):
  <tool_call>{"name":"<tool>","arguments":{...}}</tool_call>

Batch / chain envelope (bespoke orchestration wrapper, no native LLM equivalent):
  <tool_call>{"calls":[...],"mode":"parallel"}</tool_call>
  <tool_call>{"calls":[...],"mode":"chain"}</tool_call>

Inside batch/chain `calls` lists, each call uses the flat shape
`{"tool":"<tool>", **args}` because the chain executor normalizes there.

In chain mode each step runs sequentially.  String values in a step's
parameters may reference prior results via placeholders:

  $prev        — full text output of the immediately preceding step
  $prev.line1  — first line of the previous result
  $<id>        — full text output of the step with that id
  $<id>.line1  — first line of that step's result

If any step in a chain fails (result starts with "[error" or "[<tool>: error"),
execution halts and the chain returns a summary up to that point.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from core.skill_registry import canonical_tool_name
from core.skill_runtime import ToolExecutionContext, execute_tool_call_enveloped, L1_PRINCIPAL_TOOLS

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"

# Env name kept for backwards-compatibility with operators who set it.
# Despite the legacy "CMD" in the name, this gates the <tool_call> strict parser.
_STRICT_PARSE_ENV = "MONOLITH_CMD_PARSER_STRICT"

_PARSER_METRICS_TEMPLATE = {
    "extract_calls": 0,
    "extract_strict_calls": 0,
    "extract_compat_calls": 0,
    "candidate_blocks": 0,
    "parsed_commands": 0,
    "parse_errors": 0,
    "dangling_tag_repair_attempts": 0,
    "dangling_tag_repair_success": 0,
    "dangling_tag_repair_failures": 0,
    "json_backslash_repair_attempts": 0,
    "json_backslash_repair_success": 0,
    "json_backslash_repair_failures": 0,
    "strict_missing_close_tag_errors": 0,
}
_PARSER_METRICS: dict[str, int] = dict(_PARSER_METRICS_TEMPLATE)

# Matches $prev, $prev.line3, $prev.path1, $step.data.path, $step.data.matches.0.path, etc.
_VAR_RE = re.compile(r"\$(\w+)(?:\.([\w.]+))?")


def _env_flag_enabled(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_strict_cmd_parser_enabled() -> bool:
    return _env_flag_enabled(_STRICT_PARSE_ENV)


def reset_cmd_parser_metrics() -> None:
    _PARSER_METRICS.clear()
    _PARSER_METRICS.update(_PARSER_METRICS_TEMPLATE)


def get_cmd_parser_metrics() -> dict[str, int]:
    return dict(_PARSER_METRICS)


def _metric_inc(key: str, delta: int = 1) -> None:
    _PARSER_METRICS[key] = int(_PARSER_METRICS.get(key, 0)) + int(delta)


# ---------------------------------------------------------------------------
# Chain variable substitution
# ---------------------------------------------------------------------------


def _escape_invalid_backslashes(raw: str) -> str:
    """Escape invalid JSON backslashes inside quoted strings.

    Example: C:\\Users -> C:\\\\Users
    """
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\":
            nxt = raw[i + 1] if (i + 1) < len(raw) else ""
            if nxt in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
                out.append(ch)
                escape = True
            elif nxt == "u" and (i + 5) < len(raw):
                hex_candidate = raw[i + 2 : i + 6]
                if re.fullmatch(r"[0-9a-fA-F]{4}", hex_candidate):
                    out.append(ch)
                    escape = True
                else:
                    out.append("\\\\")
            else:
                out.append("\\\\")
            i += 1
            continue

        out.append(ch)
        if ch == '"':
            in_string = False
        i += 1
    return "".join(out)


_WINDOWS_PATH_KEYS = {
    "path",
    "cwd",
    "output",
    "archive",
    "output_dir",
    "base_dir",
}


def _repair_windows_path_control_chars(value: str) -> str:
    replacements = {
        "\n": r"\n",
        "\r": r"\r",
        "\t": r"\t",
        "\f": r"\f",
        "\b": r"\b",
    }
    out = value
    for bad, replacement in replacements.items():
        out = out.replace(bad, replacement)
    return out


def _normalize_windows_paths(obj: Any) -> Any:
    if isinstance(obj, dict):
        fixed: dict[str, Any] = {}
        for key, value in obj.items():
            if (
                isinstance(value, str)
                and key in _WINDOWS_PATH_KEYS
                and (re.match(r"^[A-Za-z]:", value) or value.startswith("\\\\"))
                and any(ch in value for ch in ("\n", "\r", "\t", "\f", "\b"))
            ):
                fixed[key] = _repair_windows_path_control_chars(value)
            else:
                fixed[key] = _normalize_windows_paths(value)
        return fixed
    if isinstance(obj, list):
        return [_normalize_windows_paths(item) for item in obj]
    return obj


def normalize_tool_call_to_cmd(parsed: dict) -> dict | None:
    """Normalize a parsed <tool_call> JSON payload to internal flat cmd dict.

    Accepts:
      - Hermes/Qwen native:    {"name": "X", "arguments": {...}}
      - Batch/chain envelope:  {"calls": [...], "mode": "parallel|chain"}
      - Legacy flat shape:     {"tool": "X", ...}  (transitional tolerance)

    Returns flat cmd dict or {"calls":[...], "mode":...}, or None if no tool
    call recognized. Windows path normalization happens upstream in
    _parse_command_json.
    """
    if not isinstance(parsed, dict):
        return None

    # Batch / chain envelope — passed through; chain executor reads flat shape per inner call.
    if isinstance(parsed.get("calls"), list):
        return dict(parsed)

    # Hermes/Qwen native single-call format.
    name = parsed.get("name")
    if isinstance(name, str) and name.strip():
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        cmd: dict[str, Any] = {"tool": name.strip()}
        cmd.update(arguments)
        call_id = parsed.get("id") or parsed.get("call_id")
        if call_id:
            cmd["id"] = str(call_id)
        return cmd

    # Legacy flat shape — model sometimes still emits {"tool":"X", ...} inside <tool_call>.
    # Tolerated for transition; new prompts ask for Hermes shape.
    if "tool" in parsed or "skill" in parsed or "op" in parsed:
        return dict(parsed)

    return None


def _find_dangling_tool_call_payload(text: str) -> str | None:
    """Return raw payload after a dangling <tool_call> open tag, if any."""
    if _TOOL_CALL_OPEN not in text:
        return None
    last_open = text.rfind(_TOOL_CALL_OPEN)
    last_close = text.rfind(_TOOL_CALL_CLOSE)
    if last_open <= last_close:
        return None
    raw = text[last_open + len(_TOOL_CALL_OPEN):].strip()
    return raw or None


def _iter_tool_call_blocks_strict(text: str) -> list[tuple[str, bool]]:
    """Strict mode: only fully-closed <tool_call>...</tool_call> blocks.

    Peels double-nested envelopes. Observed in the wild: some models echo
    the literal ``<tool_call>...</tool_call>`` markup from the retry-prompt
    instructions, wrapping their actual JSON envelope inside an extra outer
    ``<tool_call>`` pair. The non-greedy ``_TOOL_CALL_RE`` matches OUTER-open
    to INNER-close, so the captured group looks like ``<tool_call>{json}``
    — the inner OPEN tag is a prefix on the content, the inner CLOSE has
    already been consumed by the outer match. Recursion can't find anything
    (no close tag remains), so peel the open prefix instead. Loop to handle
    arbitrary nesting depth.
    """
    out: list[tuple[str, bool]] = []
    for match in _TOOL_CALL_RE.finditer(text):
        inner = match.group(1).strip()
        # Peel any number of nested open-tag prefixes left on the content.
        while inner.startswith(_TOOL_CALL_OPEN):
            inner = inner[len(_TOOL_CALL_OPEN):].lstrip()
        if inner:
            out.append((inner, False))
    return out


def _iter_tool_call_blocks_compat(text: str) -> list[tuple[str, bool]]:
    """Compat mode: include dangling open-tag payload (for streaming/truncated output)."""
    blocks = _iter_tool_call_blocks_strict(text)
    raw = _find_dangling_tool_call_payload(text)
    if raw and raw.lstrip().startswith("{"):
        blocks.append((raw, True))
    return blocks


def _iter_tool_call_blocks(text: str, *, strict: bool = False) -> list[tuple[str, bool]]:
    if strict:
        return _iter_tool_call_blocks_strict(text)
    return _iter_tool_call_blocks_compat(text)


def _parse_command_json_strict(raw: str) -> tuple[dict | None, bool]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return _normalize_windows_paths(parsed), False
        return None, False
    except (json.JSONDecodeError, ValueError):
        return None, False


def _parse_command_json_compat(raw: str) -> tuple[dict | None, bool]:
    parsed, repaired = _parse_command_json_strict(raw)
    if parsed is not None:
        return parsed, repaired
    repaired = _escape_invalid_backslashes(raw)
    if repaired == raw:
        return None, False
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return _normalize_windows_paths(parsed), True
    except (json.JSONDecodeError, ValueError):
        return None, True
    return None, True


def _parse_command_json(raw: str, *, strict: bool = False) -> tuple[dict | None, bool]:
    if strict:
        return _parse_command_json_strict(raw)
    return _parse_command_json_compat(raw)

def _extract_line(text: str, line_no: int) -> str:
    """Return a 1-indexed line from *text*, or '' if out of range."""
    lines = text.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1]
    return ""


def _get_match_lines(text: str) -> list[str]:
    """Return only the data lines from a tool result, skipping headers like [grep: ...]."""
    lines = text.splitlines()
    return [ln for ln in lines if ln and not ln.startswith("[")]


def _extract_match(text: str, n: int) -> str:
    """Return the Nth (1-indexed) non-header line from a tool result."""
    matches = _get_match_lines(text)
    if 1 <= n <= len(matches):
        return matches[n - 1]
    return ""


def _extract_path(text: str, n: int) -> str:
    """Return just the file path from the Nth match line.

    Handles grep-style output (path:lineno: content) and plain paths.
    """
    line = _extract_match(text, n)
    if not line:
        return ""
    # grep output: C:/path/file.py:42: def foo(...)
    # Try to extract the path before the first :linenum: pattern
    m = re.match(r"^(.+?\.\w+):\d+:", line)
    if m:
        return m.group(1).strip()
    # list_files output: "  filename.py (1,234 B)" — less useful standalone
    # Fall back to the whole line stripped
    return line.strip()


def _source_text(source: Any) -> str:
    if isinstance(source, dict):
        return str(source.get("display_text", source.get("text", "")))
    return str(source or "")


def _lookup_field(source: Any, accessor: str) -> str:
    text = _source_text(source)
    legacy = re.fullmatch(r"(line|match|path)(\d+)", accessor)
    if legacy:
        key = legacy.group(1)
        n = int(legacy.group(2))
        if key == "line":
            return _extract_line(text, n)
        if key == "match":
            return _extract_match(text, n)
        return _extract_path(text, n)

    if not isinstance(source, dict):
        return ""

    # Convenience aliases
    if accessor in {"path", "response"}:
        data = source.get("data")
        if isinstance(data, dict) and accessor in data:
            return str(data.get(accessor, ""))

    current: Any = source
    for token in accessor.split("."):
        if isinstance(current, dict):
            if token in current:
                current = current[token]
                continue
            data = current.get("data")
            if isinstance(data, dict) and token in data:
                current = data[token]
                continue
            return ""
        if isinstance(current, list):
            if token.isdigit():
                idx = int(token)
                if 0 <= idx < len(current):
                    current = current[idx]
                    continue
            return ""
        return ""

    if isinstance(current, (dict, list)):
        return json.dumps(current, ensure_ascii=False)
    if current is None:
        return ""
    return str(current)


def _substitute_vars(value: str, results_by_id: dict[str, dict[str, Any]], prev: dict[str, Any] | None) -> str:
    """Replace $prev / $<id> / legacy accessors / typed accessors."""

    def _replacer(m: re.Match) -> str:
        ref_name = m.group(1)
        accessor = m.group(2)

        if ref_name == "prev":
            source: Any = prev or {}
        else:
            source = results_by_id.get(ref_name)
            if source is None:
                # Unknown step ID — return a clear error marker instead of
                # silently passing the literal placeholder to downstream tools.
                return f"[unresolved: {m.group(0)} — step '{ref_name}' not found]"

        if accessor:
            resolved = _lookup_field(source, accessor)
            if resolved == "":
                return f"[unresolved: {m.group(0)} — field '{accessor}' is empty or missing]"
            return resolved
        return _source_text(source)

    return _VAR_RE.sub(_replacer, value)


def _substitute_call(call: dict, results_by_id: dict[str, dict[str, Any]], prev: dict[str, Any] | None) -> dict:
    """Return a shallow copy of *call* with all string values interpolated."""
    out: dict = {}
    for key, value in call.items():
        if isinstance(value, str):
            out[key] = _substitute_vars(value, results_by_id, prev)
        else:
            out[key] = value
    return out


def _normalize_call_for_execution(call: dict[str, Any]) -> dict[str, Any]:
    """Apply lightweight compatibility/safety normalization before execution.

    The goal is to keep tool behavior stable even when the model emits stale or
    deprecated arguments.
    """
    out = dict(call)
    tool_name = canonical_tool_name(out.get("tool", out.get("skill", out.get("op", ""))))
    if tool_name == "get_budget_score" and not bool(out.get("evaluate_message", False)):
        # Current default mode is snapshot-only; ignore probe args unless explicitly enabled.
        out.pop("message", None)
        out.pop("message_count", None)
    return out


def _normalize_command_for_execution(cmd: dict[str, Any]) -> dict[str, Any]:
    out = dict(cmd)
    calls = out.get("calls")
    if isinstance(calls, list):
        normalized_calls: list[Any] = []
        for call in calls:
            if isinstance(call, dict):
                normalized_calls.append(_normalize_call_for_execution(call))
            else:
                normalized_calls.append(call)
        out["calls"] = normalized_calls
        return out
    if "tool" in out or "skill" in out or "op" in out:
        return _normalize_call_for_execution(out)
    return out


def _is_cancellation_requested(ctx: ToolExecutionContext) -> bool:
    checker = getattr(ctx, "should_cancel", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _build_cancellation_artifact(*, scope: str, step: int | None = None, total: int | None = None) -> dict:
    if scope == "chain" and step is not None and total is not None:
        msg = f"[chain: cancelled before step {step}/{total}]"
        data = {"cancelled": True, "scope": scope, "step": step, "total_steps": total}
    elif scope == "batch" and step is not None and total is not None:
        msg = f"[batch: cancelled before call {step}/{total}]"
        data = {"cancelled": True, "scope": scope, "step": step, "total_calls": total}
    else:
        msg = "[tools: cancelled]"
        data = {"cancelled": True, "scope": scope}
    return {
        "kind": "tool_result",
        "tool": scope,
        "call": {"scope": scope, "step": step, "total": total},
        "result": msg,
        "envelope": {
            "type": "tool_result",
            "tool": scope,
            "status": "error",
            "ok": False,
            "call_id": None,
            "text": msg,
            "display_text": msg,
            "data": data,
        },
    }


# ---------------------------------------------------------------------------
# Chain execution
# ---------------------------------------------------------------------------

_MAX_CHAIN_STEPS = 10


def _execute_chain(
    calls: list[dict],
    ctx: ToolExecutionContext,
    artifacts: list[dict],
) -> None:
    """Execute *calls* sequentially, piping results via variable substitution.

    Populates *artifacts* in-place with tool_call / tool_result entries.
    """
    results_by_id: dict[str, dict[str, Any]] = {}
    prev: dict[str, Any] | None = None

    for step_index, raw_call in enumerate(calls[:_MAX_CHAIN_STEPS]):
        if _is_cancellation_requested(ctx):
            artifacts.append(
                _build_cancellation_artifact(
                    scope="chain",
                    step=step_index + 1,
                    total=min(len(calls), _MAX_CHAIN_STEPS),
                )
            )
            break
        if not isinstance(raw_call, dict):
            continue

        call = dict(raw_call)
        call.setdefault("id", f"chain_{step_index + 1}")
        step_id = str(call["id"])

        # Interpolate placeholders from prior results
        call = _substitute_call(call, results_by_id, prev)
        call = _normalize_call_for_execution(call)

        envelope = execute_tool_call_enveloped(call, ctx)
        result = envelope.display_text
        tool_name = call.get("tool", call.get("skill", call.get("op", "")))

        artifacts.append(
            {
            "kind": "tool_call",
            "command": call,
            "chain_step": step_index,
            }
        )
        artifacts.append(
            {
            "kind": "tool_result",
            "tool": tool_name,
            "call": call,
            "result": result,
            "envelope": envelope.as_dict(),
            "chain_step": step_index,
            }
        )

        # Store for downstream references
        results_by_id[step_id] = envelope.as_dict()
        prev = envelope.as_dict()

        # Halt on explicit envelope error state.
        if not bool(envelope.ok):
            halt_msg = f"[chain: halted at step {step_index + 1}/{len(calls)} ({tool_name}) due to error]"
            artifacts.append(
                {
                    "kind": "tool_result",
                    "tool": "chain",
                    "call": {"mode": "chain", "halt_step": step_index + 1, "failed_tool": tool_name},
                    "result": halt_msg,
                    "envelope": {
                        "type": "tool_result",
                        "tool": "chain",
                        "status": "error",
                        "ok": False,
                        "call_id": None,
                        "text": halt_msg,
                        "display_text": halt_msg,
                        "data": {"halted": True, "step": step_index + 1, "total_steps": len(calls)},
                    },
                }
            )
            break


def _build_tool_result_bundle(artifacts: list[dict]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for artifact in artifacts:
        if str(artifact.get("kind", "")).strip().lower() != "tool_result":
            continue
        result = str(artifact.get("result", "") or "").strip()
        if not result:
            continue

        envelope = artifact.get("envelope", {})
        call_id: str | None = None
        ok = not bool(artifact.get("parse_error"))
        status = "ok" if ok else "error"
        if isinstance(envelope, dict):
            if envelope.get("call_id") is not None:
                call_id = str(envelope.get("call_id") or "").strip() or None
            if envelope.get("ok") is not None:
                ok = bool(envelope.get("ok"))
                status = "ok" if ok else "error"
            raw_status = str(envelope.get("status", "")).strip().lower()
            if raw_status in {"ok", "error"}:
                status = raw_status
                ok = raw_status == "ok"

        entries.append(
            {
                "index": len(entries) + 1,
                "tool": str(artifact.get("tool", "") or "").strip(),
                "call_id": call_id,
                "ok": ok,
                "status": status,
                "result": result,
            }
        )

    error_count = sum(1 for item in entries if not bool(item.get("ok")))
    return {
        "entry_count": len(entries),
        "ok_count": len(entries) - error_count,
        "error_count": error_count,
        "entries": entries,
    }


def _render_tool_result_bundle(bundle: dict[str, Any]) -> str | None:
    entries = bundle.get("entries", [])
    if not isinstance(entries, list) or not entries:
        return None
    if len(entries) == 1:
        return str(entries[0].get("result", "")).strip() or None

    blocks: list[str] = []
    for entry in entries:
        index = int(entry.get("index", len(blocks) + 1))
        tool = str(entry.get("tool", "") or "").strip() or "unknown"
        status = str(entry.get("status", "") or "").strip().lower() or "ok"
        call_id = str(entry.get("call_id", "") or "").strip()
        header = f"[tool_result_{index}: tool={tool}, status={status}"
        if call_id:
            header += f", call_id={call_id}"
        header += "]"
        result = str(entry.get("result", "") or "").strip()
        blocks.append(f"{header}\n{result}" if result else header)
    return "\n\n".join(blocks).strip() or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_commands(text: str, *, strict: bool | None = None) -> list[dict]:
    """Return all parsed command dicts found in <tool_call> envelopes.

    Wire format is Hermes/Qwen native: {"name":"X","arguments":{...}}
    Batch/chain envelope: {"calls":[...],"mode":"parallel|chain"} (bespoke).
    Legacy flat shape {"tool":"X",...} is tolerated for transition.

    All forms are normalized to internal flat dicts before entering the pipeline.
    Malformed JSON is reported as a parse-error placeholder.
    """
    strict_mode = is_strict_cmd_parser_enabled() if strict is None else bool(strict)
    _metric_inc("extract_calls")
    _metric_inc("extract_strict_calls" if strict_mode else "extract_compat_calls")

    results: list[dict] = []

    for raw, repaired_close_tag in _iter_tool_call_blocks(text, strict=strict_mode):
        _metric_inc("candidate_blocks")
        if repaired_close_tag:
            _metric_inc("dangling_tag_repair_attempts")

        # Ignore non-JSON payloads to avoid false-positive parsing on literal
        # mentions like "use <tool_call> in docs".
        if not raw.lstrip().startswith("{"):
            continue
        parsed, repaired_json = _parse_command_json(raw, strict=strict_mode)
        if repaired_json:
            _metric_inc("json_backslash_repair_attempts")
        if parsed is None:
            _metric_inc("parse_errors")
            if repaired_close_tag:
                _metric_inc("dangling_tag_repair_failures")
            if repaired_json:
                _metric_inc("json_backslash_repair_failures")
            results.append(
                {
                    "_parse_error": True,
                    "_raw": raw[:300],
                    "_repair_close_tag": repaired_close_tag,
                    "_repair_json": repaired_json,
                }
            )
            continue
        cmd = normalize_tool_call_to_cmd(parsed)
        if cmd is None:
            # Valid JSON that matched no known tool/skill/op shape. Pre-fix this
            # command silently vanished (MODEL_OUTPUT_BLINDSPOT_MAP.md, the
            # critical_silent parse-drop). Capture the why + the offending JSON
            # so the model learns its shape was unrecognized (outside-in
            # feedback) instead of thinking nothing happened. Best-effort,
            # flag-gated, isolated — never affects parsing.
            try:
                from core import command_feedback
                command_feedback.record_failure(
                    kind="tool_call",
                    failed_rules=["unrecognized_command_shape"],
                    detail="The JSON parsed but matched no known tool/skill/op "
                           "shape — it needs one of: name, tool, skill, op, or calls.",
                    offending=raw[:400],
                )
            except Exception:
                pass
            continue
        _metric_inc("parsed_commands")
        if repaired_close_tag:
            _metric_inc("dangling_tag_repair_success")
            cmd["_repair_close_tag"] = True
        if repaired_json:
            _metric_inc("json_backslash_repair_success")
            cmd["_repair_json"] = True
        results.append(cmd)

    if strict_mode:
        dangling_raw = _find_dangling_tool_call_payload(text)
        if dangling_raw and dangling_raw.lstrip().startswith("{"):
            _metric_inc("parse_errors")
            _metric_inc("strict_missing_close_tag_errors")
            results.append(
                {
                    "_parse_error": True,
                    "_raw": dangling_raw[:300],
                    "_strict_missing_close_tag": True,
                    "_repair_close_tag": False,
                    "_repair_json": False,
                }
            )

    return results


def expand_calls(cmd: dict) -> list[dict]:
    if not isinstance(cmd, dict):
        return []
    calls = cmd.get("calls")
    if isinstance(calls, list):
        expanded: list[dict] = []
        for index, call in enumerate(calls, start=1):
            if isinstance(call, dict):
                item = dict(call)
                item.setdefault("id", f"call_{index}")
                expanded.append(item)
        return expanded
    if "tool" in cmd or "skill" in cmd or "op" in cmd:
        return [cmd]
    return []


def strip_commands(text: str) -> str:
    """Remove all <tool_call> blocks and any orphan tag fragments.

    After regex-stripping matched pairs, two kinds of orphans can survive:

    * **Truncated open** — model generation cut off mid-envelope, leaving
      ``<tool_call>{json...`` with no close. Drop from the opener onward.
    * **Orphan close** — non-greedy regex captured a double-nested envelope's
      outer-open through inner-close, leaving the outer ``</tool_call>`` as a
      bare tag. Strip every bare close tag.

    A bare prose mention like ``"use <tool_call> in docs"`` (open tag with no
    JSON content following) is preserved — we discriminate truncation from
    prose by whether JSON-shaped content (``{...``) follows the opener.
    """
    cleaned = _TOOL_CALL_RE.sub("", text)
    open_pos = cleaned.rfind(_TOOL_CALL_OPEN)
    close_pos = cleaned.rfind(_TOOL_CALL_CLOSE)
    # Truncated-open case (open with no matching close, JSON content trailing).
    if open_pos > close_pos:
        after = cleaned[open_pos + len(_TOOL_CALL_OPEN):].lstrip()
        if after.startswith("{"):
            cleaned = cleaned[:open_pos]
            open_pos = cleaned.rfind(_TOOL_CALL_OPEN)
            close_pos = cleaned.rfind(_TOOL_CALL_CLOSE)
    # Orphan-close case (close tag with no remaining open before it).
    if close_pos >= 0 and open_pos < 0:
        cleaned = cleaned.replace(_TOOL_CALL_CLOSE, "")
    return cleaned.strip()


def process_response(
    text: str,
    archive_dir: Path,
    on_open_addon: Callable[[str], None] | None = None,
    bridge: object | None = None,
    guard: object | None = None,
    world_state: object | None = None,
    on_generate_audio: Callable[[dict], None] | None = None,
    on_soundtrap: Callable[[dict], str] | None = None,
    on_set_session_meta: Callable[[dict], str | None] | None = None,
    on_ask_user: Callable[[dict], object] | None = None,
    strict: bool | None = None,
    should_cancel: Callable[[], bool] | None = None,
    result_cache: object | None = None,
    vision_artifact_bridge: object | None = None,
    level: int = 1,
    allowed_tools: object | None = None,
    parent_turn_id: str | None = None,
    spawn_budget: object | None = None,
    on_spawn_subagent: object | None = None,
    on_run_workshop: object | None = None,
    on_author_workshop_card: object | None = None,
) -> tuple[str, str | None, list[dict]]:
    """
    Parse and execute all skill commands embedded in a response.

    Returns:
        (clean_text, tool_result, artifacts)
        - clean_text: response with all <tool_call> envelopes stripped
        - tool_result: result text to feed back into the model, or None
        - artifacts: ordered tool_call/tool_result payloads for UI/session rendering
    """
    commands = extract_commands(text, strict=strict)
    if not commands:
        return text, None, []

    clean_text = strip_commands(text)
    ctx = ToolExecutionContext(
        archive_dir=archive_dir,
        on_open_addon=on_open_addon,
        bridge=bridge,
        guard=guard,
        world_state=world_state,
        on_generate_audio=on_generate_audio,
        on_soundtrap=on_soundtrap,
        on_set_session_meta=on_set_session_meta,
        on_ask_user=on_ask_user,
        should_cancel=should_cancel,
        result_cache=result_cache,
        vision_artifact_bridge=vision_artifact_bridge,
        level=level,
        allowed_tools=(allowed_tools if allowed_tools is not None else L1_PRINCIPAL_TOOLS),
        parent_turn_id=parent_turn_id,
        spawn_budget=spawn_budget,
        on_spawn_subagent=on_spawn_subagent,
        on_run_workshop=on_run_workshop,
        on_author_workshop_card=on_author_workshop_card,
    )
    artifacts: list[dict] = []

    for cmd in commands:
        if _is_cancellation_requested(ctx):
            artifacts.append(_build_cancellation_artifact(scope="tools"))
            break
        if cmd.get("_parse_error"):
            snippet = cmd.get("_raw", "")[:160]
            flags: list[str] = []
            if cmd.get("_repair_close_tag"):
                flags.append("dangling_tag_repair_attempted")
            if cmd.get("_repair_json"):
                flags.append("json_backslash_repair_attempted")
            if cmd.get("_strict_missing_close_tag"):
                flags.append("strict_mode_missing_close_tag")
            suffix = f" ({', '.join(flags)})" if flags else ""
            error_msg = (
                f"[error: malformed tool call{suffix} - could not parse JSON. Raw: {snippet}]"
            )
            artifacts.append({"kind": "tool_call", "command": {"tool": "error", "raw": snippet}})
            artifacts.append(
                {
                    "kind": "tool_result",
                    "tool": "error",
                    "call": {},
                    "result": error_msg,
                    "parse_error": True,
                    "raw": snippet,
                    "repair_close_tag": bool(cmd.get("_repair_close_tag")),
                    "repair_json": bool(cmd.get("_repair_json")),
                }
            )
            continue

        normalized_cmd = _normalize_command_for_execution(cmd)

        # --- Chain mode: sequential execution with variable piping ---
        if normalized_cmd.get("mode") == "chain" and isinstance(normalized_cmd.get("calls"), list):
            artifacts.append({"kind": "tool_call", "command": normalized_cmd})
            _execute_chain(
                normalized_cmd["calls"], ctx, artifacts,
            )
            continue

        # --- Parallel / single mode (existing behaviour) ---
        artifacts.append({"kind": "tool_call", "command": normalized_cmd})
        calls = expand_calls(normalized_cmd)
        for index, call in enumerate(calls, start=1):
            if _is_cancellation_requested(ctx):
                artifacts.append(
                    _build_cancellation_artifact(scope="batch", step=index, total=len(calls))
                )
                break
            envelope = execute_tool_call_enveloped(call, ctx)
            result = envelope.display_text
            artifacts.append(
                {
                    "kind": "tool_result",
                    "tool": call.get("tool", call.get("skill", call.get("op", ""))),
                    "call": call,
                    "result": result,
                    "envelope": envelope.as_dict(),
                }
            )
    bundle = _build_tool_result_bundle(artifacts)
    tool_result = _render_tool_result_bundle(bundle)
    return clean_text, tool_result, artifacts
