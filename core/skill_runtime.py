from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import math
import operator
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from core.acu_store import ACUStore
from core.adaptive_budget import evaluate_budget_for_message, get_last_budget_snapshot
from core.context_refresh import get_last_context_refresh
from core.file_readers import open_file as open_local_file
from core.history_search import SearchResult, search_archives
from core.llm_config import load_config
from core.mononote import store as mononote_store
from core.paths import SKILLS_DIR
from core.skill_registry import canonical_tool_name, clear_skill_cache, get_tool, list_tools
from core.tool_validation import validate_tool_arguments
from engine.sync_bridge import generate_sync_from_config

if TYPE_CHECKING:
    from core.world_state import WorldStateStore
    from monokernel.bridge import MonoBridge
    from monokernel.guard import MonoGuard

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


# Tools safe to cache — deterministic given same inputs and filesystem state.
_CACHEABLE_TOOLS: frozenset[str] = frozenset({
    "open_file", "read_file", "grep", "find_files", "list_files", "calculate",
})
# Tools where cache key includes the target path's mtime for invalidation.
_FILE_KEYED_TOOLS: frozenset[str] = frozenset({
    "open_file", "read_file", "grep", "find_files", "list_files",
})
# Tools safe to pre-execute on the main thread mid-stream (fast, no side effects).
STREAMING_PREEXEC_TOOLS: frozenset[str] = frozenset({
    "read_file", "grep", "find_files", "list_files", "calculate", "recall",
})


# --- Governance ladder (Phase A subagent substrate) ---
# Every name below MUST resolve via get_tool() at build time (T11), and every
# registered executor must stay reachable at L1 (reachability audit). llm_call
# lives in the L3 floor: it is the safest degenerate inference and the live turn
# calls it, so it must be permitted at every level.
MAX_SPAWN_LEVEL = 3

L3_LEAF_TOOLS: frozenset[str] = frozenset({
    "open_file", "read_file", "grep", "find_files", "list_files", "calculate",
    "search_history", "inspect_trace", "inspect_pipeline",
    "monosearch", "monopulse", "get_budget_score", "get_context_summary",
    "llm_call",
})
L2_WORKER_TOOLS: frozenset[str] = L3_LEAF_TOOLS | frozenset({
    "write_file", "edit_file", "save_note", "create_tool", "reload_skills",
    "zip_files", "unzip_file", "generate_image", "generate_audio",
    "spawn_subagent",   # holds the spawn tool -- but the cap forbids L2->L2.
})
L1_PRINCIPAL_TOOLS: frozenset[str] = L2_WORKER_TOOLS | frozenset({
    "run_command", "run_tests",                       # shell / world
    "open_addon", "ask_user", "set_session_meta",     # user-facing / session authority
    "soundtrap",                                      # local loop/project library writes
    "run_workshop",   # L1-ONLY: the principal may run a Workshop workflow. Deliberately ABSENT
                      # from L2/L3 sets, so a flow's L3 blocks can't call it -> no unbounded
                      # workflow -> run_workshop -> workflow recursion (the governance gate).
    "author_workshop_card",  # L1-ONLY: the principal may author (write) a Workshop card.
                             # Writes a file -> side-effecting; absent from L2/L3 so worker
                             # blocks cannot author cards (no untrusted file-write path).
})
_LEVEL_DEFAULT_TOOLS: dict[int, frozenset[str]] = {
    1: L1_PRINCIPAL_TOOLS, 2: L2_WORKER_TOOLS, 3: L3_LEAF_TOOLS,
}


@dataclass
class SpawnBudget:
    """Per-L1-turn hard ceiling, shared by reference down the whole tree so
    fan-out and depth both hit the same limit."""
    max_total_spawns: int = 12
    _used: int = 0

    def can_spawn(self) -> bool:
        return self._used < self.max_total_spawns

    def charge(self) -> None:
        self._used += 1


class ToolResultCache:
    """Session-scoped tool result cache.

    Provides two services:
      1. File-mtime cache (feature 4): read_file/grep/etc. results keyed by
         path + mtime — invalidated automatically when files change.
      2. Session deduplication (feature 5): any tool called with the same
         arguments within a session returns the cached result instantly,
         avoiding redundant work and context bloat.
    """

    def __init__(self) -> None:
        # key -> (envelope, mtime_at_cache_time | None)
        self._store: dict[str, tuple[Any, float | None]] = {}

    # ------------------------------------------------------------------
    def _path_mtime(self, path: str) -> float | None:
        try:
            return Path(path).stat().st_mtime
        except Exception:
            return None

    def _make_key(self, tool: str, cmd: dict) -> str:
        relevant: dict[str, Any] = {"tool": tool}
        if tool in _FILE_KEYED_TOOLS:
            for k in ("path", "pattern", "glob", "max_results", "recursive", "limit", "offset"):
                v = cmd.get(k)
                if v is not None:
                    relevant[k] = v
        elif tool == "calculate":
            relevant["expr"] = str(cmd.get("expr", ""))
        else:
            for k, v in sorted(cmd.items()):
                if not k.startswith("_") and k not in ("id", "call_id"):
                    relevant[k] = v
        raw = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode()).hexdigest()

    def _current_mtime(self, tool: str, cmd: dict) -> float | None:
        if tool not in _FILE_KEYED_TOOLS:
            return None
        path = str(cmd.get("path", "") or "")
        return self._path_mtime(path) if path else None

    # ------------------------------------------------------------------
    def get(self, tool: str, cmd: dict) -> Any | None:
        """Return cached ToolResultEnvelope or None on miss/stale."""
        if tool not in _CACHEABLE_TOOLS:
            return None
        key = self._make_key(tool, cmd)
        entry = self._store.get(key)
        if entry is None:
            return None
        envelope, cached_mtime = entry
        if tool in _FILE_KEYED_TOOLS and cached_mtime is not None:
            current = self._current_mtime(tool, cmd)
            if current != cached_mtime:
                del self._store[key]
                return None
        return envelope

    def set(self, tool: str, cmd: dict, envelope: Any) -> None:
        """Store a successful result; errors are never cached."""
        if tool not in _CACHEABLE_TOOLS:
            return
        if not getattr(envelope, "ok", False):
            return
        key = self._make_key(tool, cmd)
        mtime = self._current_mtime(tool, cmd)
        self._store[key] = (envelope, mtime)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


@dataclass(frozen=True)
class ToolExecutionContext:
    archive_dir: Path
    on_open_addon: Callable[[str], None] | None = None
    bridge: Any = None          # MonoBridge — dispatches vision/engine commands
    guard: Any = None           # MonoGuard  — check engine state
    world_state: Any = None     # WorldStateStore — read engine status
    on_generate_audio: Callable[[dict], None] | None = None
    on_soundtrap: Callable[[dict], str] | None = None
    on_set_session_meta: Callable[[dict], str | None] | None = None
    on_ask_user: Callable[[dict], Any] | None = None  # ask_user tool: host renders question panel; returns True on accept, False if busy, str starting "error" on failure
    should_cancel: Callable[[], bool] | None = None
    result_cache: Any = None    # ToolResultCache — cross-turn caching + dedup
    vision_artifact_bridge: Any = None  # VisionArtifactBridge - async image artifact path
    # --- governance ladder (Phase A subagent substrate) ---
    level: int = 1                                      # 1=Principal, 2=Worker, 3=Leaf
    allowed_tools: frozenset[str] = L1_PRINCIPAL_TOOLS  # capability of THIS ctx
    spawn_depth: int = 0                                # L1=0, L2=1, L3=2
    subagent_label: str | None = None                  # trace attribution + UI
    parent_turn_id: str | None = None                  # observability spine -> turn_trace
    spawn_budget: Any = None                           # shared-by-reference per-turn budget
    on_spawn_subagent: Callable[[dict], str] | None = None  # L1 async hand-off; None => inline
    on_run_workshop: Callable[[dict], str] | None = None    # L1 async hand-off: run a Workshop workflow
    on_author_workshop_card: Callable[[dict], str] | None = None  # L1 sync: author (write) a Workshop card


SkillExecutionContext = ToolExecutionContext


def derive_child_context(parent: ToolExecutionContext, child_level: int,
                         *, label: str | None = None,
                         child_turn_id: str | None = None) -> ToolExecutionContext:
    """The ONE and ONLY way to mint a child context. Privilege is monotonically
    NON-INCREASING BY CONSTRUCTION: child surface = parent.allowed_tools & child floor.
    Nothing else constructs a non-L1 context (INV-B)."""
    child_floor = _LEVEL_DEFAULT_TOOLS.get(child_level, L3_LEAF_TOOLS)
    child_tools = parent.allowed_tools & child_floor          # <= both bounds
    return dataclasses.replace(
        parent,
        level=child_level,
        allowed_tools=child_tools,
        spawn_depth=parent.spawn_depth + 1,
        subagent_label=label,
        parent_turn_id=child_turn_id or parent.parent_turn_id,
        result_cache=None,                  # clean cache (no cross-level dedup bleed)
        should_cancel=parent.should_cancel, # cancellation propagates DOWN
        spawn_budget=parent.spawn_budget,   # SAME object, shared by reference
        on_spawn_subagent=None,             # children spawn INLINE (only L1 has the host hook)
        on_soundtrap=None,                  # Soundtrap mutates local user library: L1 only
        on_run_workshop=None,               # only L1 may run Workshop workflows
        on_author_workshop_card=None,       # only L1 may author (write) Workshop cards
    )


@dataclass(frozen=True)
class ToolResultEnvelope:
    tool: str
    text: str
    display_text: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool": self.tool,
            "status": "ok" if self.ok else "error",
            "ok": self.ok,
            "call_id": self.call_id,
            "text": self.text,
            "display_text": self.display_text,
            "data": dict(self.data),
        }


_ERROR_MARKERS = (
    ": error",
    "does not exist",
    ": denied",
    "permission denied",
    "invalid path",
    "timed out",
    "not a file",
    "not a directory",
    "path traversal",
    ": no path provided",
    ": no pattern provided",
    ": no command provided",
    ": no expression provided",
    ": no prompt provided",
    ": no addon provided",
    ": no title provided",
    ": no content provided",
)

_DYNAMIC_EXECUTOR_CACHE: dict[str, tuple[float, Callable[[dict, ToolExecutionContext], str]]] = {}
_MAX_DYNAMIC_EXECUTOR_CACHE = 256
_MAX_ZIP_ENTRY_COUNT = 10_000
_MAX_ZIP_TOTAL_UNCOMPRESSED = 512 * 1024 * 1024  # 512 MiB
_MAX_ZIP_ENTRY_UNCOMPRESSED = 128 * 1024 * 1024  # 128 MiB
_MAX_ZIP_COMPRESSION_RATIO = 200


def _coerce_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _coerce_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _is_cancellation_requested(ctx: ToolExecutionContext | None) -> bool:
    if ctx is None:
        return False
    checker = getattr(ctx, "should_cancel", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _format_call_result(cmd: dict, result: str) -> str:
    call_id = str(cmd.get("id", "")).strip()
    if not call_id:
        return result
    return f"[call:{call_id}]\n{result}"


def _is_error_result(result: str) -> bool:
    head = str(result or "").strip().lower()[:220]
    if head.startswith("[error"):
        return True
    return any(marker in head for marker in _ERROR_MARKERS)


def clear_dynamic_executor_cache() -> None:
    _DYNAMIC_EXECUTOR_CACHE.clear()


def _refresh_skill_discovery() -> None:
    clear_skill_cache()
    clear_dynamic_executor_cache()


def _maybe_refresh_skill_cache_for_path(path: Path) -> None:
    try:
        rel = path.resolve().relative_to(SKILLS_DIR.resolve())
    except Exception:
        return
    if len(rel.parts) < 2:
        return
    if rel.name in {"SKILL.md", "executor.py"}:
        _refresh_skill_discovery()


def _resolve_tool_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_WORKSPACE_ROOT / path).resolve()


def _path_is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def _load_dynamic_executor(spec_path: Path) -> Callable[[dict, ToolExecutionContext], str] | None:
    executor_path = spec_path.parent / "executor.py"
    if not executor_path.exists() or not executor_path.is_file():
        return None
    cache_key = str(executor_path)
    try:
        mtime = executor_path.stat().st_mtime
    except Exception:
        return None
    cached = _DYNAMIC_EXECUTOR_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    module_name = f"monolith_skill_executor_{abs(hash(cache_key))}_{int(mtime)}"
    spec = importlib.util.spec_from_file_location(module_name, executor_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "run", None)
    if not callable(fn):
        return None
    if cache_key not in _DYNAMIC_EXECUTOR_CACHE and len(_DYNAMIC_EXECUTOR_CACHE) >= _MAX_DYNAMIC_EXECUTOR_CACHE:
        oldest_key = next(iter(_DYNAMIC_EXECUTOR_CACHE))
        _DYNAMIC_EXECUTOR_CACHE.pop(oldest_key, None)
    _DYNAMIC_EXECUTOR_CACHE[cache_key] = (mtime, fn)
    return fn


def _parse_tool_data(tool: str, text: str, cmd: dict) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if tool == "open_file":
        m = re.match(r"^\[open_file: [^(]*\(([^)]+)\)[^\]]*\]\n?(.*)$", text, re.DOTALL)
        if m:
            data["path"] = m.group(1)
            data["content"] = m.group(2)
        kind_m = re.match(r"^\[open_file: .*?\| kind=([^|\]]+)", text)
        if kind_m:
            data["kind"] = kind_m.group(1).strip()
        data["truncated"] = "[PARTIAL -" in text
    elif tool == "read_file":
        m = re.match(r"^\[read_file: [^(]*\(([^)]+)\)[^\]]*\]\n?(.*)$", text, re.DOTALL)
        if m:
            data["path"] = m.group(1).strip()
            raw_body = m.group(2)
            # Strip the trailing [PARTIAL...] line from content so data["content"] is clean
            partial_marker = "\n[PARTIAL"
            if partial_marker in raw_body:
                data["content"] = raw_body[:raw_body.index(partial_marker)]
            else:
                data["content"] = raw_body
        # Range header: chars 0–4000 of 18420 | 22% shown
        m_range = re.search(r"chars (\d+)–(\d+) of (\d+)", text)
        if m_range:
            data["offset"] = int(m_range.group(1))
            data["end"] = int(m_range.group(2))
            data["total_chars"] = int(m_range.group(3))
        # PARTIAL footer
        m_partial = re.search(
            r"\[PARTIAL — ([\d,]+) chars remaining\. To continue: read_file\(path=\"([^\"]+)\", offset=(\d+)\)\]",
            text,
        )
        if m_partial:
            data["completeness"] = "partial"
            data["remaining_chars"] = int(m_partial.group(1).replace(",", ""))
            data["continuation"] = {
                "tool": "read_file",
                "path": m_partial.group(2),
                "offset": int(m_partial.group(3)),
            }
        else:
            data["completeness"] = "full"
    elif tool == "grep":
        matches: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("["):
                continue
            m = re.match(r"^(.+?):(\d+):\s?(.*)$", line)
            if m:
                matches.append(
                    {"path": m.group(1).strip(), "line": int(m.group(2)), "text": m.group(3)}
                )
            else:
                matches.append({"path": line, "line": None, "text": ""})
        data["matches"] = matches
        if matches:
            data["path"] = matches[0].get("path")
    elif tool == "list_files":
        m = re.match(r"^\[list_files: (.+?) \(", text)
        if m:
            data["path"] = m.group(1).strip()
        files: list[str] = []
        dirs: list[str] = []
        for line in text.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            m_file = re.match(r"^(.+?) \((?:\?|[\d,]+(?: B| KB))\)$", line)
            if m_file:
                files.append(m_file.group(1).strip())
                continue
            if line.endswith("/"):
                dirs.append(line[:-1].strip())
        if files:
            data["files"] = files
        if dirs:
            data["dirs"] = dirs
    elif tool == "find_files":
        m = re.match(r"^\[find_files: (.+?) \(", text)
        if m:
            data["root"] = m.group(1).strip()
        matches: list[str] = []
        for line in text.splitlines()[1:]:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            if line.endswith("/"):
                line = line[:-1]
            matches.append(line)
        if matches:
            data["matches"] = matches
            data["path"] = matches[0]
    elif tool in {"write_file", "edit_file"}:
        m = re.search(r" to (.+?)\]?$", text)
        if m:
            data["path"] = m.group(1).strip()
    elif tool in {"run_command", "run_tests"}:
        m = re.search(r"exit_code=(\d+)", text)
        if m:
            data["exit_code"] = int(m.group(1))
    elif tool == "calculate":
        m = re.match(r"^\[calculate: (.+?) = (.+?)\]$", text.strip())
        if m:
            data["expr"] = m.group(1)
            data["value"] = m.group(2)
    elif tool == "llm_call":
        marker = "[llm_call:"
        if text.startswith(marker):
            header, _, rest = text.partition("\n")
            m_chars = re.search(r"chars=(\d+)", header)
            if m_chars:
                data["chars"] = int(m_chars.group(1))
            m_max_tokens = re.search(r"max_tokens=(\d+)", header)
            if m_max_tokens:
                data["max_tokens"] = int(m_max_tokens.group(1))
            m_source = re.search(r"max_tokens_source=([a-z_]+)", header)
            if m_source:
                data["max_tokens_source"] = m_source.group(1)
            m_truncated = re.search(r"truncated=(true|false)", header)
            if m_truncated:
                data["truncated"] = m_truncated.group(1) == "true"
            if rest:
                data["response"] = rest
    elif tool == "generate_image":
        # The "submitted" path carries artifact_id=X; chat session uses
        # this to find the pending bubble when sig_artifact_ready fires.
        m_aid = re.search(r"artifact_id=([A-Za-z0-9_]+)", text)
        if m_aid:
            data["artifact_id"] = m_aid.group(1)
            data["status"] = "pending"
            data["batch_size"] = _coerce_int(cmd.get("batch_size", 1), 1, 1, 16) if cmd else 1
            if cmd:
                data["prompt"] = str(cmd.get("prompt", ""))
    elif tool == "soundtrap":
        if cmd:
            data["op"] = str(cmd.get("op") or cmd.get("verb") or "state")
            for key in ("project_id", "clip_id", "placement_id", "path", "prompt", "track"):
                if cmd.get(key) is not None:
                    data[key] = cmd.get(key)
    elif tool == "web_search":
        if cmd:
            data["query"] = str(cmd.get("query") or "")
        urls: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("url: "):
                urls.append(stripped[5:].strip())
        if urls:
            data["urls"] = urls
            data["url"] = urls[0]
    if cmd:
        data.setdefault("input", dict(cmd))
    return data


def execute_search_history(cmd: dict, ctx: ToolExecutionContext) -> str:
    query = str(cmd.get("query", "")).strip()
    max_results = _coerce_int(cmd.get("max_results", 6), 6, 1, 8)
    if not query:
        return "[search_history: no query provided]"

    results: list[SearchResult] = search_archives(
        query=query,
        archive_dir=ctx.archive_dir,
        max_results=max_results,
    )
    if not results:
        return f"[search_history: no results found for '{query}']"

    lines = [f"[search_history results for '{query}']"]
    for idx, result in enumerate(results, start=1):
        lines.append(f"{idx}. {result.to_context_block()}")
    return "\n".join(lines)


def execute_recall(cmd: dict, ctx: ToolExecutionContext) -> str:
    query = str(cmd.get("query", "")).strip()
    max_results = _coerce_int(cmd.get("max_results", 10), 10, 1, 20)
    if not query:
        return "[recall: no query provided]"
    store = ACUStore()
    try:
        results = store.search(query, limit=max_results)
    finally:
        store.close()
    if not results:
        return f"[recall: no memories found for '{query}']"
    from core.irp import label_text

    lines = [f"[recall results for '{query}']"]
    for idx, acu in enumerate(results, start=1):
        lines.append(f"{idx}. {label_text(acu['canonical'], scope='claim', row=acu)}")
    return "\n".join(lines)


def execute_open_addon(cmd: dict, ctx: ToolExecutionContext) -> str:
    addon_id = str(cmd.get("addon", "")).strip()
    if not addon_id:
        return "[open_addon: no addon provided]"
    if ctx.on_open_addon is None:
        return f"[open_addon: UI bridge unavailable for '{addon_id}']"
    try:
        ctx.on_open_addon(addon_id)
        return f"[open_addon: launched '{addon_id}']"
    except Exception as exc:
        return f"[open_addon: error launching '{addon_id}' - {exc}]"


def execute_open_file(cmd: dict, _ctx: ToolExecutionContext) -> str:
    path_str = str(cmd.get("path", "")).strip()
    if not path_str:
        return "[open_file: no path provided]"
    try:
        path = _resolve_tool_path(path_str)
    except Exception as exc:
        return f"[open_file: invalid path - {exc}]"

    return open_local_file(
        path,
        max_chars=_coerce_int(cmd.get("max_chars", 8000), 8000, 200, 50000),
        offset=_coerce_int(cmd.get("offset", 0), 0, 0, 10 ** 9),
        member=str(cmd.get("member", "")).strip() or None,
        sheet=str(cmd.get("sheet", "")).strip() or None,
        max_rows=_coerce_int(cmd.get("max_rows", 80), 80, 1, 1000),
        max_members=_coerce_int(cmd.get("max_members", 80), 80, 1, 500),
    )


def execute_read_file(cmd: dict, _ctx: ToolExecutionContext) -> str:
    path_str = str(cmd.get("path", "")).strip()
    max_chars = _coerce_int(cmd.get("max_chars", 4000), 4000, 200, 8000)
    offset = _coerce_int(cmd.get("offset", 0), 0, 0, 10 ** 9)
    if not path_str:
        return "[read_file: no path provided]"

    try:
        path = _resolve_tool_path(path_str)
    except Exception as exc:
        return f"[read_file: invalid path - {exc}]"

    if not path.exists():
        return f"[read_file: path does not exist - {path}]"
    if not path.is_file():
        return f"[read_file: path is not a file - {path}]"

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"[read_file: permission denied - {path}]"
    except Exception as exc:
        return f"[read_file: error reading file - {exc}]"

    total_chars = len(content)

    # Apply char offset before windowing
    if offset > 0:
        content = content[offset:]

    truncated = False
    if len(content) > max_chars:
        content = content[:max_chars]
        truncated = True

    end = offset + len(content)
    remaining = total_chars - end

    if total_chars <= max_chars and offset == 0:
        # Full file shown — simple header
        header = f"[read_file: {path.name} ({path})]"
    else:
        pct = int(end / total_chars * 100) if total_chars else 100
        header = (
            f"[read_file: {path.name} ({path})"
            f" | chars {offset}–{end} of {total_chars} | {pct}% shown]"
        )

    result = f"{header}\n{content}"
    if truncated:
        result += (
            f"\n[PARTIAL — {remaining:,} chars remaining."
            f' To continue: read_file(path="{path_str}", offset={end})]'
        )
    return result


def execute_list_files(cmd: dict, _ctx: ToolExecutionContext) -> str:
    path_str = str(cmd.get("path", "")).strip()
    pattern = str(cmd.get("pattern", "*")).strip() or "*"

    if not path_str:
        return "[list_files: no path provided]"

    try:
        path = _resolve_tool_path(path_str)
    except Exception as exc:
        return f"[list_files: invalid path - {exc}]"

    if not path.exists():
        filename_hint = Path(path_str).name.strip()
        hint = ""
        if "." in filename_hint and not filename_hint.startswith(".") and not filename_hint.endswith("."):
            hint = (
                f"\nhint: list_files expects a directory. For a file request, use read_file with an "
                f"absolute file path. If location is unknown, use find_files with pattern='{filename_hint}'."
            )
        return f"[list_files: path does not exist - {path}]{hint}"
    if not path.is_dir():
        if path.is_file():
            return (
                f"[list_files: path is a file, not a directory - {path}]"
                f"\nhint: use read_file with this path."
            )
        return f"[list_files: path is not a directory - {path}]"

    try:
        entries = sorted(path.glob(pattern))
    except Exception as exc:
        return f"[list_files: glob error - {exc}]"

    files = [entry for entry in entries if entry.is_file()][:100]
    dirs = [entry for entry in entries if entry.is_dir()][:100]
    if not files and not dirs:
        return f"[list_files: {path} (0 files, 0 dirs, pattern='{pattern}')]"

    n_files = len(files)
    n_dirs = len(dirs)
    lines = [
        f"[list_files: {path} ({n_files} file{'s' if n_files != 1 else ''}, "
        f"{n_dirs} dir{'s' if n_dirs != 1 else ''}, pattern='{pattern}')]"
    ]
    for dir_path in dirs:
        lines.append(f"  {dir_path.name}/")
    for file_path in files:
        try:
            size = file_path.stat().st_size
            size_str = f"{size:,} B" if size < 1024 else f"{size // 1024:,} KB"
        except Exception:
            size_str = "?"
        lines.append(f"  {file_path.name} ({size_str})")
    return "\n".join(lines)


def execute_find_files(cmd: dict, _ctx: ToolExecutionContext) -> str:
    path_str = str(cmd.get("path", "")).strip()
    pattern = str(cmd.get("pattern", "")).strip()
    recursive = bool(cmd.get("recursive", True))
    include_dirs = bool(cmd.get("include_dirs", False))
    max_results = _coerce_int(cmd.get("max_results", 100), 100, 1, 500)

    if not path_str:
        return "[find_files: no path provided]"
    if not pattern:
        return "[find_files: no pattern provided]"

    try:
        root = _resolve_tool_path(path_str)
    except Exception as exc:
        return f"[find_files: invalid path - {exc}]"

    if not root.exists():
        return f"[find_files: path does not exist - {root}]"
    if not root.is_dir():
        return f"[find_files: path is not a directory - {root}]"

    try:
        walker = root.rglob(pattern) if recursive else root.glob(pattern)
    except Exception as exc:
        return f"[find_files: glob error - {exc}]"

    matches: list[Path] = []
    try:
        for entry in walker:
            if entry.is_file():
                pass
            elif entry.is_dir() and include_dirs:
                pass
            else:
                continue
            matches.append(entry.resolve())
            if len(matches) >= max_results:
                break
    except Exception as exc:
        return f"[find_files: scan error - {exc}]"

    kind = "matches" if include_dirs else "files"
    if not matches:
        return f"[find_files: no {kind} matching '{pattern}' in {root}]"

    lines = [
        (
            f"[find_files: {root} ({len(matches)} match{'es' if len(matches) != 1 else ''}, "
            f"pattern='{pattern}', recursive={str(recursive).lower()}, "
            f"include_dirs={str(include_dirs).lower()})]"
        )
    ]
    for path in matches:
        suffix = "/" if path.is_dir() else ""
        lines.append(f"  {path}{suffix}")
    return "\n".join(lines)


def execute_grep(cmd: dict, _ctx: ToolExecutionContext) -> str:
    pattern = str(cmd.get("pattern", "")).strip()
    path_str = str(cmd.get("path", "")).strip()
    glob_pattern = str(cmd.get("glob", "*")).strip() or "*"
    max_results = _coerce_int(cmd.get("max_results", 20), 20, 1, 200)
    case_sensitive = bool(cmd.get("case_sensitive", False))
    recursive = bool(cmd.get("recursive", True))

    if not pattern:
        return "[grep: no pattern provided]"
    if not path_str:
        return "[grep: no path provided]"

    try:
        root = _resolve_tool_path(path_str)
    except Exception as exc:
        return f"[grep: invalid path - {exc}]"

    if not root.exists():
        return f"[grep: path does not exist - {root}]"

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"[grep: invalid pattern - {exc}]"

    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        try:
            walker = root.rglob(glob_pattern) if recursive else root.glob(glob_pattern)
            candidates = [entry for entry in walker if entry.is_file()]
        except Exception as exc:
            return f"[grep: glob error - {exc}]"
    else:
        return f"[grep: unsupported path type - {root}]"

    _GREP_MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB — skip large/binary files
    matches: list[str] = []
    skipped_large = 0
    for file_path in candidates:
        try:
            if file_path.stat().st_size > _GREP_MAX_FILE_SIZE:
                skipped_large += 1
                continue
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line_no, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append(f"{file_path}:{line_no}: {line.strip()}")
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    skip_suffix = ""
    if skipped_large:
        skip_suffix = f"; skipped {skipped_large} file{'s' if skipped_large != 1 else ''} > {_GREP_MAX_FILE_SIZE // (1024 * 1024)}MB"

    if not matches:
        return f"[grep: no matches for '{pattern}' in {root}{skip_suffix}]"

    header = (
        f"[grep: {len(matches)} match{'es' if len(matches) != 1 else ''} "
        f"for '{pattern}' in {root}{skip_suffix}]"
    )
    return "\n".join([header, *matches])

def execute_save_note(cmd: dict, _ctx: ToolExecutionContext) -> str:
    title = str(cmd.get("title", "")).strip()
    content = str(cmd.get("content", "")).strip()
    if not title:
        return "[save_note: no title provided]"
    if not content:
        return "[save_note: no content provided]"

    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        body = f"# {title}\n_saved {timestamp}_\n\n{content}\n"
        note = mononote_store.write_note(title, body)
        return f"[save_note: saved to {note.path}]"
    except Exception as exc:
        return f"[save_note: error - {exc}]"


_CALC_BINOPS: dict[type[ast.operator], Callable[[object, object], object]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_CALC_UNOPS: dict[type[ast.unaryop], Callable[[object], object]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_CALC_FUNCS: dict[str, Callable[..., object]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
}

_CALC_MAX_AST_NODES = 256
_CALC_MAX_DEPTH = 64
_CALC_MAX_CALL_ARGS = 8
_CALC_MAX_POW_ABS_EXPONENT = 1000
_CALC_MAX_POW_RESULT_BITS = 4096


def _safe_pow(base: object, exponent: object) -> object:
    if isinstance(exponent, bool):
        raise ValueError("boolean exponent not allowed")
    if isinstance(exponent, (int, float)) and abs(exponent) > _CALC_MAX_POW_ABS_EXPONENT:
        raise ValueError("power exponent too large")
    if isinstance(base, int) and isinstance(exponent, int) and exponent >= 0 and abs(base) > 1:
        estimated_bits = exponent * math.log2(abs(base))
        if estimated_bits > _CALC_MAX_POW_RESULT_BITS:
            raise ValueError("power expression too large")
    return operator.pow(base, exponent)


_CALC_BINOPS[ast.Pow] = _safe_pow


def _eval_node(node: ast.AST, depth: int = 0) -> float | int:
    if depth > _CALC_MAX_DEPTH:
        raise ValueError("expression nested too deeply")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"non-numeric constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op = _CALC_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.left, depth + 1), _eval_node(node.right, depth + 1))  # type: ignore[return-value]

    if isinstance(node, ast.UnaryOp):
        op = _CALC_UNOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.operand, depth + 1))  # type: ignore[return-value]

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only simple function names are allowed")
        fn = _CALC_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"function not allowed: '{node.func.id}'")
        if node.keywords:
            raise ValueError("keyword arguments not allowed")
        if len(node.args) > _CALC_MAX_CALL_ARGS:
            raise ValueError("too many function arguments")
        return fn(*[_eval_node(arg, depth + 1) for arg in node.args])  # type: ignore[return-value]

    raise ValueError(f"expression type not allowed: {type(node).__name__}")


def execute_calculate(cmd: dict, _ctx: ToolExecutionContext) -> str:
    expr = str(cmd.get("expr", "")).strip()
    if not expr:
        return "[calculate: no expression provided]"
    try:
        tree = ast.parse(expr, mode="eval")
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > _CALC_MAX_AST_NODES:
            return f"[calculate: error - expression too complex ({node_count} nodes)]"
        result = _eval_node(tree.body)
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            result_str = str(int(result))
        elif isinstance(result, float):
            result_str = f"{result:g}"
        else:
            result_str = str(result)
        return f"[calculate: {expr} = {result_str}]"
    except SyntaxError:
        return f"[calculate: syntax error in expression - {expr!r}]"
    except Exception as exc:
        return f"[calculate: error - {exc}]"


def execute_generate_image(cmd: dict, ctx: ToolExecutionContext) -> str:
    prompt = str(cmd.get("prompt", "")).strip()
    if not prompt:
        return "[generate_image: no prompt provided]"

    if ctx.bridge is None:
        return "[generate_image: vision engine unavailable]"

    from core.vision_models import scan_model_root, fuzzy_match_model, DEFAULT_MODEL_ROOT

    # Check vision engine status via world state
    vision_status = "UNKNOWN"
    if ctx.world_state is not None:
        snapshot = ctx.world_state.snapshot()
        engines = snapshot.get("engines", {})
        vision_meta = engines.get("vision", {})
        vision_status = str(vision_meta.get("status", "UNKNOWN")).upper()

    model_param = str(cmd.get("model", "")).strip()

    # If no explicit model arg, fall back to the VISION tab's saved default
    # (config.vision.model_path). This is what the SKILL.md contract promises.
    if not model_param:
        try:
            from core.config import get_config
            cfg_default = str(getattr(get_config().vision, "model_path", "") or "").strip()
            if cfg_default and Path(cfg_default).exists():
                model_param = cfg_default
        except Exception:
            pass

    # If model param is an exact absolute path that exists, use directly
    if model_param:
        model_path_candidate = Path(model_param)
        if model_path_candidate.is_absolute() and model_path_candidate.exists():
            artifact_id = _dispatch_vision(ctx, model_param, cmd)
            return _submit_text(prompt, cmd, artifact_id, model_label=Path(model_param).name)

        # Try fuzzy match against the scanned model root
        entries = scan_model_root()
        match = fuzzy_match_model(model_param, entries)
        if match:
            artifact_id = _dispatch_vision(ctx, match.path, cmd)
            return _submit_text(prompt, cmd, artifact_id, model_label=match.label)

        model_list = ", ".join(e.label for e in entries[:10]) if entries else "none found"
        return f"[generate_image: model \"{model_param}\" not found. Available: {model_list}]"

    # No model param and no config default — check if engine is already loaded
    if vision_status == "READY":
        artifact_id = _dispatch_vision(ctx, None, cmd)
        return _submit_text(prompt, cmd, artifact_id)

    # No model anywhere — list available so the LLM can retry with model=name
    entries = scan_model_root()
    if not entries:
        return (
            f"[generate_image: no vision model loaded and no models found in {DEFAULT_MODEL_ROOT}. "
            f"If the user provided a model path, retry with model=path. Otherwise ask them for one.]"
        )
    model_list = "\n".join(f"  - {e.label}" for e in entries[:15])
    return f"[generate_image: no vision model loaded. Available models:\n{model_list}\nRetry with model=name to load one.]"


def _submit_text(prompt: str, cmd: dict, artifact_id: str | None, model_label: str = "") -> str:
    """Compose the synchronous tool-result text. The artifact_id is the
    handle the chat session uses to find the pending bubble when sig_image
    arrives. If artifact_id is None the bridge was unavailable and we fall
    back to the old text-only behavior."""
    w = _coerce_int(cmd.get("width", 512), 512, 64, 4096)
    h = _coerce_int(cmd.get("height", 512), 512, 64, 4096)
    parts = [f'submitted - "{prompt[:60]}"', f"{w}x{h}"]
    if model_label:
        parts.append(f"model={model_label}")
    if artifact_id:
        parts.append(f"artifact_id={artifact_id}")
    return f"[generate_image: {', '.join(parts)}]"


def _dispatch_vision(ctx: ToolExecutionContext, model_path: str | None, cmd: dict) -> str | None:
    """Submit set_path + load + generate to the vision engine via bridge.

    Returns the artifact_id minted by VisionArtifactBridge if both the
    vision engine and the bridge are available. Returns None on the
    text-only fallback path (e.g. running with no bridge wired yet)."""
    if model_path:
        ctx.bridge.submit(ctx.bridge.wrap("skill", "set_path", "vision", payload={"path": model_path}))
        ctx.bridge.submit(ctx.bridge.wrap("skill", "load", "vision"))

    width = _coerce_int(cmd.get("width", 512), 512, 64, 4096)
    height = _coerce_int(cmd.get("height", 512), 512, 64, 4096)
    steps = _coerce_int(cmd.get("steps", 25), 25, 1, 200)
    seed = _coerce_int(cmd.get("seed", -1), -1, -1, 2**31)
    batch_size = _coerce_int(cmd.get("batch_size", 1), 1, 1, 16)
    guidance = 7.5
    try:
        guidance = float(cmd.get("guidance_scale", 7.5))
    except (TypeError, ValueError):
        pass

    payload: dict = {
        "prompt": str(cmd.get("prompt", "")),
        "negative_prompt": str(cmd.get("negative_prompt", "")),
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
        "scheduler": "dpm++",
        "batch_size": batch_size,
        "model_path": model_path or "",
    }

    # Reserve gen_id + register pending so sig_image arrivals can be routed
    # back to this call's tool-result bubble. Skip if either the engine or
    # the bridge isn't available — fall back to text-only behavior.
    artifact_id: str | None = None
    vision_engine = None
    if ctx.guard is not None and hasattr(ctx.guard, "engines"):
        vision_engine = ctx.guard.engines.get("vision")
    bridge_obj = getattr(ctx, "vision_artifact_bridge", None)

    if vision_engine is not None and bridge_obj is not None \
       and hasattr(vision_engine, "reserve_gen_id") \
       and hasattr(bridge_obj, "register_pending"):
        try:
            gen_id = int(vision_engine.reserve_gen_id())
            call_id = str(cmd.get("id", "")).strip() or ""
            artifact_id = bridge_obj.register_pending(gen_id, call_id, payload)
            payload["gen_id"] = gen_id
        except Exception:
            artifact_id = None

    ctx.bridge.submit(ctx.bridge.wrap("skill", "generate", "vision", payload=payload))
    return artifact_id


def execute_generate_audio(cmd: dict, ctx: ToolExecutionContext) -> str:
    prompt = str(cmd.get("prompt", "")).strip()
    if not prompt:
        return "[generate_audio: no prompt provided]"

    if ctx.on_generate_audio is None:
        return "[generate_audio: audio engine unavailable - open the AUDIO module first]"

    duration = 5.0
    try:
        duration = max(1.0, min(30.0, float(cmd.get("duration", 5.0))))
    except (TypeError, ValueError):
        pass

    sample_rate = _coerce_int(cmd.get("sample_rate", 32000), 32000, 8000, 48000)
    if sample_rate not in (32000, 44100, 48000):
        sample_rate = 32000

    try:
        result = ctx.on_generate_audio({"prompt": prompt, "duration": duration, "sample_rate": sample_rate})
        if isinstance(result, str) and result:
            return f"[generate_audio: {result}]"
        return f"[generate_audio: submitted - \"{prompt[:60]}\", duration={duration}s]"
    except Exception as exc:
        return f"[generate_audio: error - {exc}]"


def execute_soundtrap(cmd: dict, ctx: ToolExecutionContext) -> str:
    if ctx.on_soundtrap is not None:
        try:
            return str(ctx.on_soundtrap(dict(cmd)) or "[soundtrap: no result]")
        except Exception as exc:
            return f"[soundtrap: error - {exc}]"
    from core.soundtrap import execute_soundtrap_command

    return execute_soundtrap_command(dict(cmd))


def execute_write_file(cmd: dict, _ctx: ToolExecutionContext) -> str:
    path_str = str(cmd.get("path", "")).strip()
    content = str(cmd.get("content", ""))
    if not path_str:
        return "[write_file: no path provided]"

    # Safety: check for traversal on the raw path BEFORE resolve collapses it
    if ".." in path_str:
        return "[write_file: path traversal not allowed]"

    try:
        path = _resolve_tool_path(path_str)
    except Exception as exc:
        return f"[write_file: invalid path - {exc}]"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        _maybe_refresh_skill_cache_for_path(path)
        return f"[write_file: written {len(content)} chars to {path}]"
    except PermissionError:
        return f"[write_file: permission denied - {path}]"
    except Exception as exc:
        return f"[write_file: error - {exc}]"


def execute_edit_file(cmd: dict, _ctx: ToolExecutionContext) -> str:
    path_str = str(cmd.get("path", "")).strip()
    find_str = str(cmd.get("find", ""))
    replace_str = str(cmd.get("replace", ""))
    if not path_str:
        return "[edit_file: no path provided]"
    if not find_str:
        return "[edit_file: no find string provided]"

    if ".." in path_str:
        return "[edit_file: path traversal not allowed]"

    try:
        path = _resolve_tool_path(path_str)
    except Exception as exc:
        return f"[edit_file: invalid path - {exc}]"

    if not path.exists():
        return f"[edit_file: file not found - {path}]"
    if not path.is_file():
        return f"[edit_file: not a file - {path}]"

    try:
        original = path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"[edit_file: read error - {exc}]"

    count = None
    if cmd.get("count") is not None:
        count = _coerce_int(cmd.get("count"), -1, 1, 10000)

    if find_str not in original:
        return f"[edit_file: text not found in {path.name}]"

    if count is not None:
        new_content = original.replace(find_str, replace_str, count)
        replacements = count
    else:
        replacements = original.count(find_str)
        new_content = original.replace(find_str, replace_str)

    try:
        path.write_text(new_content, encoding="utf-8")
        _maybe_refresh_skill_cache_for_path(path)
        return f"[edit_file: {replacements} replacement(s) in {path.name}]"
    except Exception as exc:
        return f"[edit_file: write error - {exc}]"


def _run_shell_with_cancellation(
    command: str,
    *,
    cwd: str | None,
    timeout: int,
    ctx: ToolExecutionContext,
    tool_name: str,
) -> tuple[int | None, str, str]:
    """Run a shell command and cooperatively terminate when cancellation is requested.

    Returns (exit_code, output, status):
      - status in {"ok", "timeout", "cancelled", "error"}
    """
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
        start = time.monotonic()
        while True:
            if _is_cancellation_requested(ctx):
                if proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=2)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                return None, "", "cancelled"

            if proc.poll() is not None:
                break

            elapsed = time.monotonic() - start
            if elapsed >= float(timeout):
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                return None, "", "timeout"
            time.sleep(0.05)

        stdout, stderr = proc.communicate()
        output = ""
        if stdout:
            output += stdout
        if stderr:
            if output:
                output += "\n"
            output += stderr
        return int(proc.returncode or 0), output, "ok"
    except Exception as exc:
        return None, f"[{tool_name}: error - {exc}]", "error"
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass


def execute_run_command(cmd: dict, _ctx: ToolExecutionContext) -> str:
    command = str(cmd.get("command", "")).strip()
    if not command:
        return "[run_command: no command provided]"
    if _is_cancellation_requested(_ctx):
        return "[run_command: cancelled]"

    cwd = str(cmd.get("cwd", "")).strip() or None
    timeout = _coerce_int(cmd.get("timeout", 30), 30, 1, 120)

    exit_code, output, status = _run_shell_with_cancellation(
        command, cwd=cwd, timeout=timeout, ctx=_ctx, tool_name="run_command"
    )
    if status == "cancelled":
        return "[run_command: cancelled]"
    if status == "timeout":
        return f"[run_command: timed out after {timeout}s]"
    if status == "error":
        return output or "[run_command: error - unknown failure]"

    if len(output) > 4000:
        output = output[:4000] + "\n... [truncated]"
    if not output.strip():
        return f"[run_command: exit_code={exit_code}, no output]"
    return f"[run_command: exit_code={exit_code}]\n{output}"


def execute_run_tests(cmd: dict, _ctx: ToolExecutionContext) -> str:
    runner = str(cmd.get("runner", "pytest")).strip() or "pytest"
    target = str(cmd.get("target", "")).strip()
    cwd = str(cmd.get("cwd", "")).strip() or None
    timeout = _coerce_int(cmd.get("timeout", 120), 120, 1, 1800)
    if _is_cancellation_requested(_ctx):
        return "[run_tests: cancelled]"

    command = runner if not target else f"{runner} {target}"
    exit_code, output, status = _run_shell_with_cancellation(
        command, cwd=cwd, timeout=timeout, ctx=_ctx, tool_name="run_tests"
    )
    if status == "cancelled":
        return "[run_tests: cancelled]"
    if status == "timeout":
        return f"[run_tests: timed out after {timeout}s]"
    if status == "error":
        return output or "[run_tests: error - unknown failure]"

    if len(output) > 8000:
        output = output[:8000] + "\n... [truncated]"

    status_label = "passed" if exit_code == 0 else "failed"
    if not output.strip():
        return f"[run_tests: {status_label}, exit_code={exit_code}, no output]"
    return f"[run_tests: {status_label}, exit_code={exit_code}]\n{output}"


def execute_zip_files(cmd: dict, _ctx: ToolExecutionContext) -> str:
    raw_paths = cmd.get("paths")
    output_str = str(cmd.get("output", "")).strip()
    base_dir_str = str(cmd.get("base_dir", "")).strip()
    if not isinstance(raw_paths, list) or not raw_paths:
        return "[zip_files: paths must be a non-empty list]"
    if not output_str:
        return "[zip_files: no output provided]"

    try:
        output_path = Path(output_str).resolve()
    except Exception as exc:
        return f"[zip_files: invalid output path - {exc}]"

    base_dir: Path | None = None
    if base_dir_str:
        try:
            base_dir = Path(base_dir_str).resolve()
        except Exception as exc:
            return f"[zip_files: invalid base_dir - {exc}]"
        if not base_dir.exists() or not base_dir.is_dir():
            return f"[zip_files: base_dir not found - {base_dir}]"

    sources: list[Path] = []
    for item in raw_paths:
        value = str(item).strip()
        if not value:
            continue
        try:
            source = Path(value).resolve()
        except Exception as exc:
            return f"[zip_files: invalid source path '{item}' - {exc}]"
        if not source.exists():
            return f"[zip_files: source path does not exist - {source}]"
        sources.append(source)

    if not sources:
        return "[zip_files: no valid source paths]"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    try:
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in sources:
                if source.is_file():
                    if base_dir is not None:
                        try:
                            arcname = source.relative_to(base_dir).as_posix()
                        except Exception:
                            arcname = source.name
                    else:
                        arcname = source.name
                    archive.write(source, arcname=arcname)
                    file_count += 1
                    continue

                for child in source.rglob("*"):
                    if not child.is_file():
                        continue
                    if base_dir is not None:
                        try:
                            arcname = child.relative_to(base_dir).as_posix()
                        except Exception:
                            arcname = f"{source.name}/{child.relative_to(source).as_posix()}"
                    else:
                        arcname = f"{source.name}/{child.relative_to(source).as_posix()}"
                    archive.write(child, arcname=arcname)
                    file_count += 1
    except Exception as exc:
        return f"[zip_files: error - {exc}]"

    return f"[zip_files: created {output_path} with {file_count} file(s)]"


def execute_unzip_file(cmd: dict, _ctx: ToolExecutionContext) -> str:
    archive_str = str(cmd.get("archive", "")).strip()
    output_dir_str = str(cmd.get("output_dir", "")).strip()
    if not archive_str:
        return "[unzip_file: no archive provided]"
    if not output_dir_str:
        return "[unzip_file: no output_dir provided]"

    try:
        archive_path = Path(archive_str).resolve()
        output_dir = Path(output_dir_str).resolve()
    except Exception as exc:
        return f"[unzip_file: invalid path - {exc}]"

    if not archive_path.exists() or not archive_path.is_file():
        return f"[unzip_file: archive not found - {archive_path}]"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            entries = archive.infolist()
            if len(entries) > _MAX_ZIP_ENTRY_COUNT:
                return (
                    f"[unzip_file: blocked - archive has {len(entries)} items "
                    f"(limit {_MAX_ZIP_ENTRY_COUNT})]"
                )
            total_uncompressed = 0
            # Zip-slip protection: reject entries that resolve outside output_dir
            for info in entries:
                normalized_name = str(info.filename or "").replace("\\", "/")
                if not normalized_name:
                    continue
                target = (output_dir / normalized_name).resolve()
                if not _path_is_within(output_dir, target):
                    return f"[unzip_file: blocked - entry '{info.filename}' escapes output directory (zip-slip)]"
                if info.file_size > _MAX_ZIP_ENTRY_UNCOMPRESSED:
                    return (
                        f"[unzip_file: blocked - entry '{info.filename}' is too large "
                        f"({info.file_size} bytes)]"
                    )
                total_uncompressed += int(info.file_size)
                if total_uncompressed > _MAX_ZIP_TOTAL_UNCOMPRESSED:
                    return (
                        f"[unzip_file: blocked - archive expands to {total_uncompressed} bytes "
                        f"(limit {_MAX_ZIP_TOTAL_UNCOMPRESSED})]"
                    )
                if not info.is_dir() and info.compress_size > 0:
                    ratio = info.file_size / info.compress_size
                    if ratio > _MAX_ZIP_COMPRESSION_RATIO:
                        return (
                            f"[unzip_file: blocked - suspicious compression ratio in "
                            f"'{info.filename}' ({ratio:.1f}x)]"
                        )
            archive.extractall(output_dir)
    except zipfile.BadZipFile:
        return f"[unzip_file: invalid zip archive - {archive_path}]"
    except Exception as exc:
        return f"[unzip_file: error - {exc}]"

    return f"[unzip_file: extracted {len(entries)} item(s) to {output_dir}]"


def execute_set_session_meta(cmd: dict, ctx: ToolExecutionContext) -> str:
    if ctx.on_set_session_meta is None:
        return "[set_session_meta: session bridge unavailable]"

    changes: dict[str, Any] = {}
    if "title" in cmd:
        title = str(cmd.get("title", "")).strip()
        if title:
            changes["title"] = title
    if "summary" in cmd:
        summary = cmd.get("summary")
        if isinstance(summary, list):
            changes["summary"] = [str(item) for item in summary if str(item).strip()]
        elif isinstance(summary, str) and summary.strip():
            changes["summary"] = [line.strip() for line in summary.splitlines() if line.strip()]

    if not changes:
        return "[set_session_meta: no valid fields provided]"

    try:
        result = ctx.on_set_session_meta(changes)
    except Exception as exc:
        return f"[set_session_meta: error - {exc}]"

    if isinstance(result, str) and result.strip():
        return f"[set_session_meta: {result.strip()}]"
    return "[set_session_meta: updated]"


def execute_llm_call(cmd: dict, _ctx: ToolExecutionContext) -> str:
    prompt = str(cmd.get("prompt", "")).strip()
    raw_messages = cmd.get("messages")
    messages: list[dict[str, str]] = []
    if isinstance(raw_messages, list):
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", ""))
            if role in {"system", "user", "assistant"} and content.strip():
                messages.append({"role": role, "content": content})
    if not messages:
        if not prompt:
            return "[llm_call: provide prompt or messages]"
        system = str(cmd.get("system", "")).strip()
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    # Rule-6 deletion: llm_call no longer calls the inference primitive directly -- it
    # routes through the gated atom (level 3, no tools, single shot). Its max_tokens
    # resolution is PRESERVED and passed to the atom as llm_config; inference is reached
    # ONLY via the atom.
    base_config = load_config()
    call_config: dict[str, Any] = {}
    placeholder_tokens = {
        "", "auto", "default", "max_tokens", "$max_tokens", "<max_tokens>", "{max_tokens}",
    }
    raw_max_tokens = cmd.get("max_tokens")
    explicit_max_tokens: int | None = None
    if isinstance(raw_max_tokens, str):
        token = raw_max_tokens.strip().lower()
        if token not in placeholder_tokens:
            explicit_max_tokens = _coerce_positive_int(raw_max_tokens)
    else:
        explicit_max_tokens = _coerce_positive_int(raw_max_tokens)
    if explicit_max_tokens is not None:
        effective_max_tokens = explicit_max_tokens
        max_tokens_source = "explicit"
    else:
        cfg_max = _coerce_positive_int(base_config.get("llm_call_max_tokens"))
        if cfg_max is None:
            cfg_max = _coerce_positive_int(base_config.get("max_tokens"))
        effective_max_tokens = cfg_max or 2048
        max_tokens_source = "placeholder"
    call_config["max_tokens"] = effective_max_tokens
    if cmd.get("temp") is not None:
        try:
            call_config["temp"] = max(0.0, min(float(cmd.get("temp")), 2.0))
        except (TypeError, ValueError):
            pass
    if cmd.get("top_p") is not None:
        try:
            call_config["top_p"] = max(0.0, min(float(cmd.get("top_p")), 1.0))
        except (TypeError, ValueError):
            pass
    max_chars = _coerce_int(cmd.get("max_chars", 6000), 6000, 200, 12000)

    from core.subagent import run_subagent
    res = run_subagent(
        messages, base_config, level=3, frame="llm_call",
        parent_turn_id=getattr(_ctx, "parent_turn_id", None), allowed_tools=frozenset(),
        should_cancel=getattr(_ctx, "should_cancel", None), max_followups=0,
        spawn_budget=getattr(_ctx, "spawn_budget", None), llm_config=call_config or None)
    # Advisor-flagged behavior change: llm_call now serializes via the atom's
    # non-blocking generation lock. If contended (e.g. an expedition tick mid-run),
    # surface it gracefully -- never block, never crash.
    if res.halt_reason == "busy":
        return "[llm_call: generator busy - another generation is running; retry shortly]"
    text = str(res.text or "")
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    header = (
        f"[llm_call: chars={len(text)}, "
        f"max_tokens={effective_max_tokens}, "
        f"max_tokens_source={max_tokens_source}"
    )
    if truncated:
        header += ", truncated=true"
    header += "]"
    if text:
        return f"{header}\n{text}"
    return f"{header}\n(no output)"


def execute_reload_skills(cmd: dict, _ctx: ToolExecutionContext) -> str:
    _refresh_skill_discovery()
    specs = list_tools()
    max_names = _coerce_int(cmd.get("max_names", 12), 12, 1, 50)
    names = [spec.name for spec in specs[:max_names]]
    joined = ", ".join(names) if names else "none"
    return f"[reload_skills: discovered {len(specs)} tool(s)]\n{joined}"


def execute_create_tool(cmd: dict, _ctx: ToolExecutionContext) -> str:
    raw_name = str(cmd.get("name", "")).strip()
    description = str(cmd.get("description", "")).strip() or "Custom tool."
    if not raw_name:
        return "[create_tool: no name provided]"

    canonical = canonical_tool_name(raw_name)
    if not canonical:
        return f"[create_tool: invalid name '{raw_name}']"
    folder_name = canonical.replace("_", "-")
    tool_dir = (SKILLS_DIR / folder_name).resolve()
    if tool_dir.exists() and not bool(cmd.get("overwrite", False)):
        return f"[create_tool: skill already exists - {tool_dir}]"

    try:
        tool_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return f"[create_tool: could not create skill directory - {exc}]"

    skill_md = tool_dir / "SKILL.md"
    executor_py = tool_dir / "executor.py"
    skill_body = (
        "---\n"
        f"name: {folder_name}\n"
        f"description: {description}\n"
        "---\n\n"
        f'{{"tool":"{canonical}"}}\n'
    )
    executor_body = (
        "from __future__ import annotations\n\n"
        "def run(cmd: dict, ctx) -> str:\n"
        f"    return \"[{canonical}: TODO implement executor.py]\"\n"
    )
    try:
        skill_md.write_text(skill_body, encoding="utf-8")
        if not executor_py.exists() or bool(cmd.get("overwrite_executor", False)):
            executor_py.write_text(executor_body, encoding="utf-8")
    except Exception as exc:
        return f"[create_tool: write error - {exc}]"

    _refresh_skill_discovery()
    return (
        f"[create_tool: created {canonical}]"
        f"\nskill={skill_md}"
        f"\nexecutor={executor_py}"
    )


def execute_get_budget_score(cmd: dict, ctx: ToolExecutionContext) -> str:
    current = get_last_budget_snapshot()
    evaluate_message = bool(cmd.get("evaluate_message", False))
    message = str(cmd.get("message", "")).strip() if evaluate_message else ""
    message_count = _coerce_int(cmd.get("message_count", 1), 1, 1, 200)
    if evaluate_message and not message and ctx.world_state is not None:
        snapshot = ctx.world_state.snapshot()
        message = str(snapshot.get("session", {}).get("last_user_prompt", "")).strip()

    evaluated: dict[str, Any] = {}
    if evaluate_message and message:
        evaluated = evaluate_budget_for_message(message, message_count=message_count)

    payload = {
        "current": current,
        "mode": "evaluate_message" if evaluate_message else "snapshot_only",
        "evaluated": evaluated,
    }
    return f"[get_budget_score]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def execute_get_context_summary(cmd: dict, ctx: ToolExecutionContext) -> str:
    snapshot = get_last_context_refresh()
    include_prompt = bool(cmd.get("include_last_prompt", False))
    if include_prompt and ctx.world_state is not None:
        ws = ctx.world_state.snapshot()
        snapshot["last_user_prompt"] = ws.get("session", {}).get("last_user_prompt", "")
    if not snapshot:
        return "[get_context_summary: no refresh snapshot available]"
    return f"[get_context_summary]\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}"


def execute_inspect_trace(cmd: dict, ctx: ToolExecutionContext) -> str:
    """Verb-dispatched read-only inspector for the turn-trace store.

    Verbs (Q6, "deliberate use" — caller picks one):
      recent  : last N turns (limit param, default 5, max 50)
      errors  : recent turns that have at least one errored stage
      one     : full joined record for a specific turn_id
    """
    from core import turn_trace as _tt

    verb = str(cmd.get("verb") or cmd.get("op") or "").strip().lower()
    if not verb:
        return (
            "[inspect_trace: 'verb' is required. Use one of: recent, errors, one. "
            "For 'one', also pass 'turn_id'.]"
        )
    limit = cmd.get("limit", 5)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 50))

    if verb == "recent":
        rows = _tt.list_recent_turns(limit=limit)
        if not rows:
            return "[inspect_trace: no turns recorded]"
        payload = [
            {
                "turn_id": r.turn_id,
                "parent_turn_id": r.parent_turn_id,
                "captured_at": r.captured_at,
                "backend": r.backend,
                "stage_count": r.stage_count,
                "errored_stage_count": r.errored_stage_count,
                "total_chars": r.total_chars,
            }
            for r in rows
        ]
        return f"[inspect_trace:recent count={len(payload)}]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

    if verb == "errors":
        rows = _tt.search_turns(has_errored_stage=True, limit=limit)
        if not rows:
            return "[inspect_trace: no errored turns]"
        payload = [
            {
                "turn_id": r.turn_id,
                "parent_turn_id": r.parent_turn_id,
                "captured_at": r.captured_at,
                "errored_stage_count": r.errored_stage_count,
                "stage_count": r.stage_count,
                "backend": r.backend,
            }
            for r in rows
        ]
        return f"[inspect_trace:errors count={len(payload)}]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

    if verb == "one":
        turn_id = str(cmd.get("turn_id") or "").strip()
        if not turn_id:
            return "[inspect_trace:one requires 'turn_id']"
        joined = _tt.get_turn_trace(turn_id)
        if joined is None:
            return f"[inspect_trace: turn '{turn_id}' not found]"
        payload = {
            "turn_id": joined.turn_id,
            "parent_turn_id": joined.parent_turn_id,
            "summary": joined.summary,
            "stages": [s.to_payload() for s in joined.stages],
            "frame": joined.frame.to_payload() if joined.frame is not None else None,
        }
        return f"[inspect_trace:one turn={turn_id}]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

    return (
        f"[inspect_trace: unknown verb '{verb}'. "
        f"Use one of: recent, errors, one]"
    )


def execute_inspect_pipeline(cmd: dict, ctx: ToolExecutionContext) -> str:
    """Verb-dispatched read-only inspector for the Turn Pipeline (Layer E).

    Verbs:
      events  : all pipeline events for a turn (requires turn_id)
      faults  : recent FaultDetectedEvent rows (newest-first; optional fault_kind filter, since)
      last    : pipeline events for the most recent turn that has any
      one     : alias for events (matches inspect_trace verb naming)

    Dual-consumer contract: this is the same data the local Pipeline
    Inspector renders. CONNECT-side callers (Claude via agent_server.py)
    query this tool; local UI surfaces query list_pipeline_events directly.
    Both reach the same fault_traces table.
    """
    from datetime import datetime, timedelta, timezone
    from core import turn_trace as _tt

    verb = str(cmd.get("verb") or cmd.get("op") or "").strip().lower()
    if not verb:
        return (
            "[inspect_pipeline: 'verb' is required. Use one of: events, faults, last, one. "
            "For events/one, pass 'turn_id'.]"
        )

    if verb in ("events", "one"):
        turn_id = str(cmd.get("turn_id") or "").strip()
        if not turn_id:
            return f"[inspect_pipeline:{verb} requires 'turn_id']"
        rows = _tt.list_pipeline_events(turn_id)
        if not rows:
            return f"[inspect_pipeline: turn '{turn_id}' has no pipeline events]"
        payload = [r.to_dict() for r in rows]
        return (
            f"[inspect_pipeline:events turn={turn_id} count={len(payload)}]\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    if verb == "last":
        latest = _tt.most_recent_pipeline_turn_id()
        if latest is None:
            return "[inspect_pipeline: no turns with pipeline events recorded]"
        evs = _tt.list_pipeline_events(latest)
        payload = [e.to_dict() for e in evs]
        return (
            f"[inspect_pipeline:last turn={latest} count={len(payload)}]\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    if verb == "faults":
        limit = cmd.get("limit", 20)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, 200))
        since_raw = str(cmd.get("since") or "").strip()
        if since_raw:
            since_iso = since_raw
        else:
            since_iso = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        fault_kind = str(cmd.get("fault_kind") or "").strip() or None
        rows = _tt.list_faults_since(since_iso, fault_kind=fault_kind, limit=limit)
        if not rows:
            return f"[inspect_pipeline:faults since={since_iso} no faults]"
        payload = [r.to_dict() for r in rows]
        return (
            f"[inspect_pipeline:faults since={since_iso} count={len(payload)}]\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    return (
        f"[inspect_pipeline: unknown verb '{verb}'. "
        f"Use one of: events, faults, last, one]"
    )


def _source_tier_marker(rec) -> str:
    """Non-performative constraint marker for an un-established (generation-tier)
    monosearch hit. The raw tier name is NEVER rendered to the model — only the
    actionable constraint ``[unverified]`` (the SKILL.md contract says: re-ground
    before citing). Empty unless the feature flag is on AND the hit is generation.
    """
    try:
        from core.source_tier import source_tier_enabled
        if not source_tier_enabled():
            return ""
        meta = getattr(rec, "metadata", None)
        if isinstance(meta, dict) and meta.get("source_tier") == "generation":
            return " [unverified]"
    except Exception:
        return ""
    return ""


_MONOSEARCH_VERBS = "failing, recurring/recur, pulling, unresolved, search/find, get"
_MONOSEARCH_META_SOURCES: dict[str, tuple[str, ...]] = {
    "tool": ("tools",),
    "tools": ("tools",),
    "skill": ("skills",),
    "skills": ("skills",),
    "capability": ("tools", "skills"),
    "capabilities": ("tools", "skills"),
    "debug": ("faults", "turns", "stages", "ratings", "health"),
    "memory": ("memory", "knowledge", "bearing", "identity", "conversation"),
    "workflow": ("tools", "skills"),
    "workflows": ("tools", "skills"),
}
_MONOSEARCH_META_DEFAULT_QUERY: dict[str, str] = {
    "workflow": "workflow workshop monoline card run author",
    "workflows": "workflow workshop monoline card run author",
}
_MONOSEARCH_META_HINT = "tools, skills, capabilities, debug, memory, workflows"
_MONOSEARCH_SOURCE_HINT = (
    "tools, skills, faults, ratings, knowledge, warrants, claim_graph, "
    "conversation, turns, memory, bearing, identity, curiosity, stages, "
    "reminders, investigations, lag, health"
)


def _trim_monosearch_text(text: str, limit: int = 220) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _format_monosearch_records(verb: str, recs: list, *, scope: str = "") -> str:
    lines = [
        (
            f"  {r.namespaced_id} [{r.source}/{r.evidence_tier.name.lower()}] "
            f"{_trim_monosearch_text(r.text)}{_source_tier_marker(r)}"
        )
        for r in recs
    ]
    return f"[monosearch:{verb}{scope} count={len(recs)}]\n" + "\n".join(lines)


def _resolve_monosearch_meta(meta: object) -> tuple[str | None, tuple[str, ...] | None]:
    key = str(meta or "").strip().lower()
    if not key:
        return None, None
    return key, _MONOSEARCH_META_SOURCES.get(key)


def _normalize_monosearch_verb(cmd: dict) -> str:
    """Accept the forms the model naturally emits.

    ``verb`` stays explicit when provided, but a query/source/meta call is
    obviously a search and an id-only call is obviously a get. This keeps the
    tool usable when the catalog has been compressed down to examples.
    """
    verb = str(cmd.get("verb", "") or "").strip().lower()
    aliases = {
        "search/find": "search",
        "search_find": "search",
        "lookup": "search",
        "recur": "recurring",
        "recurrence": "recurring",
    }
    if verb:
        return aliases.get(verb, verb)
    # Search signals win over a bare correlation id. The parallel envelope
    # auto-stamps `id` onto every sub-call (cmd_parser.expand_calls), so checking
    # id first forced every verb-less batched search into a get-by-label that
    # fails ([monosearch:get 'a' not found]). Only a *bare* id is a get.
    if cmd.get("source") or cmd.get("meta") or cmd.get("query") is not None:
        return "search"
    if cmd.get("id"):
        return "get"
    return ""


def execute_monosearch(cmd: dict, ctx: ToolExecutionContext) -> str:
    """Read-only federated search over Monolith's own stores. Verb-dispatched:
    failing / recurring / pulling / unresolved / search/find / get. `failing` and
    `recurring` POPULATE the salience ledger (rebuild from registered adapters)
    before reading, so the selector is live, not a dark store. `pulling`/
    `unresolved` read the identity_signals adapter directly (current-state signals,
    not ledger-ranked). Read-only — never writes.
    """
    from core.monosearch import registry, service

    if not registry.all_adapters():
        from core.monosearch.bootstrap import init_monosearch
        init_monosearch()

    verb = _normalize_monosearch_verb(cmd)
    try:
        if cmd.get("limit") is None:
            from core.context_profiles import active_context_profile
            default_limit = active_context_profile().monosearch_result_count
        else:
            default_limit = cmd.get("limit")
        limit = max(1, min(int(default_limit or 10), 50))
    except (TypeError, ValueError):
        limit = 10
    if not verb:
        return f"[monosearch: 'verb' is required. Use one of: {_MONOSEARCH_VERBS}]"
    try:
        if verb == "failing":
            rows = service.failing(limit=limit)
            if not rows:
                return "[monosearch:failing none — no recurring faults recorded yet]"
            lines = [
                f"  {r['recurrence_key']}  (x{r['count']}, salience {r.get('salience', 0):.2f})"
                for r in rows
            ]
            return f"[monosearch:failing count={len(rows)}]\n" + "\n".join(lines)
        if verb == "recurring":
            rows = service.recurring(limit=limit)
            if not rows:
                return "[monosearch:recurring none]"
            lines = [
                f"  {r['recurrence_key']}  [{r['source']}] (x{r['count']}, salience {r.get('salience', 0):.2f})"
                for r in rows
            ]
            return f"[monosearch:recurring count={len(rows)}]\n" + "\n".join(lines)
        if verb in ("pulling", "unresolved"):
            recs = service.pulling(limit) if verb == "pulling" else service.unresolved(limit)
            if not recs:
                return f"[monosearch:{verb} none — no current {'curiosity' if verb == 'pulling' else 'emergence'} signal]"
            lines = [f"  {r.namespaced_id}  {r.text[:140]}{_source_tier_marker(r)}" for r in recs]
            return f"[monosearch:{verb} count={len(recs)}]\n" + "\n".join(lines)
        if verb in {"search", "find"}:
            meta_key, meta_sources = _resolve_monosearch_meta(cmd.get("meta"))
            meta_redirect: str | None = None
            if meta_key and meta_sources is None:
                from core.monosearch.router import resolve_source as _resolve_source
                if _resolve_source(meta_key) is not None:
                    # meta is not a bucket but IS a store (e.g. 'warrants',
                    # 'knowledge') — redirect to source= and teach, don't wall.
                    meta_redirect = meta_key
                    meta_key, meta_sources = None, None
                else:
                    return f"[monosearch:{verb} unknown meta {meta_key!r}. Try: {_MONOSEARCH_META_HINT}]"
            query = str(cmd.get("query", "") or "")
            if meta_key and not query.strip():
                query = _MONOSEARCH_META_DEFAULT_QUERY.get(meta_key, "")
            filters: dict = {}
            since = cmd.get("since")
            if since:
                filters["since"] = str(since)
            source = cmd.get("source") or meta_redirect
            if source:
                from core.monosearch.router import resolve_source, source_usage_hint
                guidance = source_usage_hint(source)
                if meta_redirect and not guidance:
                    guidance = (
                        f"meta={meta_redirect!r} is a store, not a bucket — "
                        f"use source='{meta_redirect}'. (meta is for buckets: "
                        f"{_MONOSEARCH_META_HINT}.)"
                    )
                if resolve_source(source) is None:
                    if guidance:
                        return f"[monosearch:{verb} source guidance: {guidance}]"
                    return f"[monosearch:{verb} unknown source {source!r}. Try: {_MONOSEARCH_SOURCE_HINT}]"
                filters["source"] = str(source)
                recs = service.search(query, filters, limit)
                scope = f" source={source}"
            elif meta_sources:
                recs = []
                seen: set[str] = set()
                for source_name in meta_sources:
                    scoped = dict(filters)
                    scoped["source"] = source_name
                    for rec in service.search(query, scoped, limit):
                        if rec.namespaced_id in seen:
                            continue
                        seen.add(rec.namespaced_id)
                        recs.append(rec)
                        if len(recs) >= limit:
                            break
                    if len(recs) >= limit:
                        break
                scope = f" meta={meta_key}"
            else:
                recs = service.search(query, filters, limit)
                scope = ""
            if not recs:
                if source and guidance:
                    return (
                        f"[monosearch:{verb} guidance: {guidance}]\n"
                        f"[monosearch:{verb} no matches for {query!r}{scope}]"
                    )
                return f"[monosearch:{verb} no matches for {query!r}{scope}]"
            out = _format_monosearch_records(verb, recs, scope=scope)
            if source and guidance:
                return f"[monosearch:{verb} guidance: {guidance}]\n{out}"
            return out
        if verb == "get":
            nsid = str(cmd.get("id", "") or "").strip()
            if not nsid:
                return "[monosearch:get requires 'id' (e.g. tool:edit_file, skill:monosearch, fault:991, clog:1840)]"
            rec = service.get(nsid)
            if rec is None:
                return f"[monosearch:get '{nsid}' not found]"
            return (
                f"[monosearch:get {rec.namespaced_id}]{_source_tier_marker(rec)} source={rec.source} "
                f"tier={rec.evidence_tier.name.lower()} provenance={rec.provenance.value}\n{rec.text}"
            )
    except Exception as exc:  # never break the tool loop
        return f"[monosearch:{verb} error: {exc}]"
    return f"[monosearch: unknown verb '{verb}'. Use one of: {_MONOSEARCH_VERBS}]"


def execute_monopulse(cmd: dict, ctx: ToolExecutionContext) -> str:
    """Pull-only runtime attention view over MonoSearch and related stores.

    MonoPulse does not write and does not inject prompt context. It answers one
    compact question: what should the model/operator attend to now?
    """
    from core import monopulse

    verb = str(cmd.get("verb") or "pulse").strip().lower()
    default_limit = 12 if verb == "pulse" else 10
    try:
        limit = max(1, min(int(cmd.get("limit") or default_limit), 50))
    except (TypeError, ValueError):
        limit = default_limit
    if not verb:
        return "[monopulse: 'verb' is required. Use one of: pulse, hotspots, stalled, drift, changed]"
    try:
        report = monopulse.run(verb, limit=limit)
    except ValueError:
        return (
            f"[monopulse: unknown verb '{verb}'. "
            "Use one of: pulse, hotspots, stalled, drift, changed]"
        )
    except Exception as exc:
        return f"[monopulse:{verb} error: {exc}]"
    return monopulse.format_report(report)


def execute_spawn_subagent(cmd: dict, ctx: ToolExecutionContext) -> str:
    """Spawn executor. Reached only AFTER the gate (Guard B) validated + clamped the
    level into cmd['_validated_child_level'] and charged the budget. L1 hands off to
    the host worker (off the Qt thread); below L1 runs the atom inline."""
    from core.subagent import run_subagent
    child_level = int(cmd.get("_validated_child_level", ctx.level + 1))
    label = str(cmd.get("frame", cmd.get("label", f"L{child_level}"))).strip() or f"L{child_level}"

    # L1 async hand-off: the host runs the atom off-thread and folds back later.
    if ctx.on_spawn_subagent is not None:
        try:
            return str(ctx.on_spawn_subagent(dict(cmd)) or "[spawn_subagent: PENDING]")
        except Exception as exc:
            return f"[spawn_subagent: error - {exc}]"

    # Below L1 (already off-thread) => run inline, synchronously.
    child = derive_child_context(ctx, child_level, label=label)
    prompt = str(cmd.get("prompt", "")).strip()
    raw_messages = cmd.get("messages")
    messages: list[dict[str, str]] = []
    if isinstance(raw_messages, list):
        for item in raw_messages:
            if isinstance(item, dict):
                role = str(item.get("role", "")).strip().lower()
                content = str(item.get("content", ""))
                if role in {"system", "user", "assistant"} and content.strip():
                    messages.append({"role": role, "content": content})
    if not messages and prompt:
        messages.append({"role": "user", "content": prompt})
    if not messages:
        return "[spawn_subagent: provide prompt or messages]"

    res = run_subagent(
        messages, load_config(), level=child_level, frame=label,
        parent_turn_id=child.parent_turn_id, allowed_tools=child.allowed_tools,
        should_cancel=child.should_cancel, max_followups=0,
        spawn_budget=child.spawn_budget)
    return res.fenced


def execute_run_workshop(cmd: dict, ctx: ToolExecutionContext) -> str:
    """Run a named Workshop workflow (a Monoline pipeline) and fold its OUTPUT back into the turn.
    L1-ONLY by construction: 'run_workshop' is absent from the L2/L3 tool sets, so Guard A denies
    it below L1 -> a workflow's L3 blocks cannot recursively run a workflow. At L1 it hands off to
    the host (ctx.on_run_workshop), which runs the flow OFF the Qt thread and folds back later."""
    if ctx.on_run_workshop is not None:
        try:
            return str(ctx.on_run_workshop(dict(cmd)) or "[run_workshop: PENDING]")
        except Exception as exc:
            return f"[run_workshop: error - {exc}]"
    # Unreachable below L1 (the gate denies it); explicit guard for offline/headless callers.
    return "[run_workshop: only the principal (L1) may run workflows]"


def execute_author_workshop_card(cmd: dict, ctx: ToolExecutionContext) -> str:
    """Author a new Workshop card (Monoline blueprint) and save it under WORKFLOWS_DIR.
    L1-ONLY by construction: 'author_workshop_card' is absent from the L2/L3 tool sets, so
    Guard A denies it below L1. Authoring is synchronous (build + validate + write-file, no
    inference) so the fix loop closes within the SAME turn — the model sees validation errors
    immediately and can re-author. At L1 it hands off to the host (ctx.on_author_workshop_card)."""
    if ctx.on_author_workshop_card is not None:
        try:
            return str(ctx.on_author_workshop_card(dict(cmd)) or "[author_workshop_card: no result]")
        except Exception as exc:
            return f"[author_workshop_card: error - {exc}]"
    # Unreachable below L1 (the gate denies it); explicit guard for offline/headless callers.
    return "[author_workshop_card: only the principal (L1) may author cards]"


_TOOL_EXECUTORS: dict[str, Callable[[dict, ToolExecutionContext], str]] = {
    "calculate": execute_calculate,
    "create_tool": execute_create_tool,
    "find_files": execute_find_files,
    "get_budget_score": execute_get_budget_score,
    "get_context_summary": execute_get_context_summary,
    "grep": execute_grep,
    "inspect_pipeline": execute_inspect_pipeline,
    "inspect_trace": execute_inspect_trace,
    "llm_call": execute_llm_call,
    "list_files": execute_list_files,
    "monopulse": execute_monopulse,
    "monosearch": execute_monosearch,
    "open_addon": execute_open_addon,
    "open_file": execute_open_file,
    "recall": execute_recall,
    "read_file": execute_read_file,
    "reload_skills": execute_reload_skills,
    "save_note": execute_save_note,
    "search_history": execute_search_history,
    "generate_image": execute_generate_image,
    "generate_audio": execute_generate_audio,
    "soundtrap": execute_soundtrap,
    "write_file": execute_write_file,
    "edit_file": execute_edit_file,
    "run_command": execute_run_command,
    "run_tests": execute_run_tests,
    "zip_files": execute_zip_files,
    "unzip_file": execute_unzip_file,
    "set_session_meta": execute_set_session_meta,
    "spawn_subagent": execute_spawn_subagent,
    "run_workshop": execute_run_workshop,
    "author_workshop_card": execute_author_workshop_card,
}


def execute_tool_call_enveloped(cmd: dict, ctx: ToolExecutionContext) -> ToolResultEnvelope:
    tool_token = cmd.get("tool", cmd.get("skill", cmd.get("op", "")))
    tool_name = canonical_tool_name(tool_token)

    # Cache check — return immediately on hit (features 4+5)
    cache: ToolResultCache | None = getattr(ctx, "result_cache", None)
    if cache is not None and tool_name:
        cached = cache.get(tool_name, cmd)
        if cached is not None:
            cached_data = dict(getattr(cached, "data", {}) or {})
            cached_data["cached"] = True
            return ToolResultEnvelope(
                tool=cached.tool,
                text=cached.text,
                display_text=cached.display_text,
                ok=cached.ok,
                data=cached_data,
                call_id=str(cmd.get("id", "")).strip() or cached.call_id,
            )
    if not tool_name:
        text = "[tool: no tool provided]"
        return ToolResultEnvelope(
            tool="tool",
            text=text,
            display_text=_format_call_result(cmd, text),
            ok=False,
            call_id=str(cmd.get("id", "")).strip() or None,
        )
    if _is_cancellation_requested(ctx):
        text = f"[{tool_name}: cancelled]"
        return ToolResultEnvelope(
            tool=tool_name,
            text=text,
            display_text=_format_call_result(cmd, text),
            ok=False,
            call_id=str(cmd.get("id", "")).strip() or None,
            data={"input": dict(cmd), "cancelled": True},
        )

    # ---- Governance gate (Phase A subagent substrate) ----
    # Guard A: capability. L1 (the principal / sole authority) is UNRESTRICTED so no
    # existing L1 caller -- nor a dynamically-loaded skill outside L1_PRINCIPAL_TOOLS --
    # can ever be denied. Only spawned L2/L3 children are gated against their
    # (monotonically shrinking) allowed_tools.
    if ctx.level > 1 and tool_name not in ctx.allowed_tools:
        text = (f"[{tool_name}: denied - not permitted at level L{ctx.level} "
                f"(allowed: {len(ctx.allowed_tools)} tools)]")
        return ToolResultEnvelope(
            tool=tool_name, text=text, display_text=_format_call_result(cmd, text), ok=False,
            call_id=str(cmd.get("id", "")).strip() or None,
            data={"input": dict(cmd), "denied": True, "reason": "capability",
                  "level": ctx.level, "label": ctx.subagent_label})

    # Guard B: spawn-cap. cmd.get("level") is a REQUEST, clamped vs the trusted ctx.level.
    if tool_name == "spawn_subagent":
        try:
            req_level = int(cmd.get("level", ctx.level + 1))
        except (TypeError, ValueError):
            req_level = ctx.level + 1
        child_level = max(ctx.level + 1, min(req_level, MAX_SPAWN_LEVEL))  # clamp; never trust model
        deny = None
        if ctx.level >= MAX_SPAWN_LEVEL:
            deny = f"L{ctx.level} is terminal (the cap) - leaves cannot spawn"
        elif req_level > MAX_SPAWN_LEVEL:
            deny = f"requested L{req_level} exceeds the L{MAX_SPAWN_LEVEL} cap"
        elif ctx.level == 2 and req_level == 2:
            deny = "L2 Worker may spawn L3 Leaf only (never another L2)"
        elif req_level <= ctx.level:
            deny = f"child L{req_level} must be deeper than caller L{ctx.level}"
        elif ctx.spawn_budget is not None and not ctx.spawn_budget.can_spawn():
            deny = "per-turn spawn budget exhausted"
        if deny is not None:
            # The gate is the only code that runs on a denied spawn, so it writes the
            # authoritative deny fault. Deferred import (avoids any import cycle AND lets
            # tests monkeypatch turn_trace.record_fault). Best-effort -- never break a turn.
            try:
                from datetime import datetime, timezone
                from core.turn_trace import FaultTraceRecord, record_fault
                budget_exhausted = (ctx.spawn_budget is not None
                                    and not ctx.spawn_budget.can_spawn())
                used = getattr(ctx.spawn_budget, "_used", 0) if ctx.spawn_budget else 0
                mx = getattr(ctx.spawn_budget, "max_total_spawns", 0) if ctx.spawn_budget else 0
                record_fault(FaultTraceRecord(
                    turn_id=str(ctx.parent_turn_id or ""),
                    parent_turn_id=ctx.parent_turn_id,
                    seq=0,
                    emitted_at=datetime.now(timezone.utc).isoformat(),
                    event_kind=("spawn_budget_exhausted" if budget_exhausted
                                else "spawn_denied"),
                    source_kind="policy",
                    source_name="subagent_gate",
                    authority_tier="dispatch",
                    fault_kind="spawn_denied",
                    severity="hard",
                    payload={"level": ctx.level, "requested_level": req_level,
                             "child_level": child_level, "label": ctx.subagent_label,
                             "deny_reason": deny, "used": used, "max": mx},
                ))
            except Exception:
                pass
            text = f"[spawn_subagent: denied - {deny}]"
            return ToolResultEnvelope(
                tool="spawn_subagent", text=text,
                display_text=_format_call_result(cmd, text), ok=False,
                call_id=str(cmd.get("id", "")).strip() or None,
                data={"input": dict(cmd), "denied": True, "reason": "spawn_cap",
                      "caller_level": ctx.level, "requested_level": req_level,
                      "child_level": child_level, "label": ctx.subagent_label})
        # Granted: charge the shared budget HERE (synchronously, at the gate) -- not in
        # the atom. The L1 async path runs the atom later on a worker thread, so charging
        # there would let N spawns in one generation all pass (count still 0) before any
        # charge landed, blowing the budget in a single fan-out (R5).
        if ctx.spawn_budget is not None:
            try:
                ctx.spawn_budget.charge()
            except Exception:
                pass
        cmd["_validated_child_level"] = child_level

    spec = get_tool(tool_name)
    if spec is None:
        text = f"[tool: unknown tool '{tool_name}']"
        return ToolResultEnvelope(
            tool=tool_name,
            text=text,
            display_text=_format_call_result(cmd, text),
            ok=False,
            call_id=str(cmd.get("id", "")).strip() or None,
            data={"input": dict(cmd)},
        )

    executor = _TOOL_EXECUTORS.get(spec.name)
    if executor is None:
        try:
            executor = _load_dynamic_executor(spec.path)
        except Exception as exc:
            text = f"[tool: failed to load runtime for '{spec.name}' - {exc}]"
            return ToolResultEnvelope(
                tool=spec.name,
                text=text,
                display_text=_format_call_result(cmd, text),
                ok=False,
                call_id=str(cmd.get("id", "")).strip() or None,
                data={"input": dict(cmd)},
            )
    if executor is None:
        text = f"[tool: no runtime available for '{spec.name}']"
        return ToolResultEnvelope(
            tool=spec.name,
            text=text,
            display_text=_format_call_result(cmd, text),
            ok=False,
            call_id=str(cmd.get("id", "")).strip() or None,
            data={"input": dict(cmd)},
        )

    normalized = dict(cmd)
    normalized["tool"] = spec.name
    validation_errors = validate_tool_arguments(spec.name, normalized)
    if validation_errors:
        detail = "; ".join(validation_errors)
        text = f"[tool: invalid arguments for '{spec.name}' - {detail}]"
        return ToolResultEnvelope(
            tool=spec.name,
            text=text,
            display_text=_format_call_result(cmd, text),
            ok=False,
            call_id=str(cmd.get("id", "")).strip() or None,
            data={"input": dict(cmd), "validation_errors": validation_errors},
        )
    if _is_cancellation_requested(ctx):
        text = f"[{spec.name}: cancelled]"
        return ToolResultEnvelope(
            tool=spec.name,
            text=text,
            display_text=_format_call_result(cmd, text),
            ok=False,
            call_id=str(cmd.get("id", "")).strip() or None,
            data={"input": dict(cmd), "cancelled": True},
        )

    try:
        result = executor(normalized, ctx)
    except Exception as exc:
        result = f"[{spec.name}: error - {exc}]"
    result_text = str(result or "")
    display_text = _format_call_result(cmd, result_text)
    envelope = ToolResultEnvelope(
        tool=spec.name,
        text=result_text,
        display_text=display_text,
        ok=not _is_error_result(result_text),
        data=_parse_tool_data(spec.name, result_text, normalized),
        call_id=str(cmd.get("id", "")).strip() or None,
    )
    # Store in cache on success (features 4+5)
    if cache is not None:
        cache.set(spec.name, cmd, envelope)
    return envelope


def execute_tool_call(cmd: dict, ctx: ToolExecutionContext) -> str:
    return execute_tool_call_enveloped(cmd, ctx).display_text


def execute_skill_call(cmd: dict, ctx: ToolExecutionContext) -> str:
    return execute_tool_call(cmd, ctx)

