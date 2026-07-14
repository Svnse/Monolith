from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from core.paths import SKILLS_DIR
from core.tool_validation import BUILTIN_TOOL_SCHEMAS


@dataclass(frozen=True)
class ToolParamSpec:
    name: str
    required: bool
    detail: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    path: Path
    params: tuple[ToolParamSpec, ...] = ()
    example_call: str | None = None
    legacy_ops: tuple[str, ...] = ()
    json_schema: dict | None = None


SkillParamSpec = ToolParamSpec
SkillSpec = ToolSpec

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

_TOOL_RUNTIME_META: dict[str, dict] = {
    "calculate": {
        "params": (
            ToolParamSpec("expr", True, "Python-compatible arithmetic expression."),
        ),
        "example_call": '{"name":"calculate","arguments":{"expr":"(1024 ** 2) * 4 / 1e6"}}',
        "legacy_ops": ("calculate",),
    },
    "grep": {
        "params": (
            ToolParamSpec("pattern", True, "Regex or literal text to search for."),
            ToolParamSpec("path", True, "Absolute file or directory path."),
            ToolParamSpec("glob", False, "Glob filter such as *.py or *.md."),
            ToolParamSpec("max_results", False, "Maximum number of matches to return."),
        ),
        "example_call": '{"name":"grep","arguments":{"pattern":"TODO","path":"C:/Users/name/projects","glob":"*.py","max_results":20}}',
        "legacy_ops": (),
    },
    "find_files": {
        "params": (
            ToolParamSpec("path", True, "Absolute directory path to search from."),
            ToolParamSpec("pattern", True, "Filename/glob pattern such as skills.md or *.md."),
            ToolParamSpec("recursive", False, "Search subdirectories when true (default true)."),
            ToolParamSpec("max_results", False, "Maximum number of file paths to return."),
        ),
        "example_call": '{"name":"find_files","arguments":{"path":"C:/Users/name/projects","pattern":"skills.md","recursive":true,"max_results":20}}',
        "legacy_ops": ("find-files",),
    },
    "list_files": {
        "params": (
            ToolParamSpec("path", True, "Absolute directory path."),
            ToolParamSpec("pattern", False, "Glob filter such as *.py."),
        ),
        "example_call": '{"name":"list_files","arguments":{"path":"C:/Users/name/projects","pattern":"*.py"}}',
        "legacy_ops": ("list-files", "list_dir", "ls"),
    },
    "open_addon": {
        "params": (
            ToolParamSpec("addon", True, "Addon id such as databank or terminal."),
        ),
        "example_call": '{"name":"open_addon","arguments":{"addon":"databank"}}',
        "legacy_ops": ("open-addon",),
    },
    "open_file": {
        "params": (
            ToolParamSpec("path", True, "Absolute or workspace-relative local file path."),
            ToolParamSpec("max_chars", False, "Character cap from 200 to 50000 (default 8000)."),
            ToolParamSpec("offset", False, "Character offset for continuing partial text output."),
            ToolParamSpec("member", False, "Archive member path to preview inside .zip/.tar/.tgz."),
            ToolParamSpec("sheet", False, "Worksheet name for .xlsx/.xlsm files."),
            ToolParamSpec("max_rows", False, "Spreadsheet rows to preview (default 80)."),
            ToolParamSpec("max_members", False, "Archive entries to list (default 80)."),
        ),
        "example_call": '{"name":"open_file","arguments":{"path":"C:/Users/name/document.pdf","max_chars":8000}}',
        "legacy_ops": ("open-file", "read_any_file", "read-document", "read_document"),
    },
    "read_file": {
        "params": (
            ToolParamSpec("path", True, "Absolute file path."),
            ToolParamSpec("max_chars", False, "Character cap from 200 to 8000."),
        ),
        "example_call": '{"name":"read_file","arguments":{"path":"C:/Users/name/notes.txt","max_chars":3000}}',
        "legacy_ops": ("read-file",),
    },
    "save_note": {
        "params": (
            ToolParamSpec("title", True, "Note title; letters, digits, underscores, or hyphens."),
            ToolParamSpec("content", True, "Note body to save."),
        ),
        "example_call": '{"name":"save_note","arguments":{"title":"project-ideas","content":"1. Semantic search\\n2. Voice input"}}',
        "legacy_ops": ("save-note",),
    },
    "list_notes": {
        "params": (
            ToolParamSpec("pattern", False, "Case-insensitive substring filter on note titles."),
            ToolParamSpec("max_results", False, "Maximum number of titles (default 50, max 200)."),
        ),
        "example_call": '{"name":"list_notes","arguments":{"max_results":10}}',
        "legacy_ops": ("list-notes",),
    },
    "read_note": {
        "params": (
            ToolParamSpec("title", True, "MonoNote title, matched against the normalized filename stem."),
            ToolParamSpec("max_chars", False, "Character cap from 1 to 50000 (default 8000)."),
            ToolParamSpec("offset", False, "Character offset for continuing partial note output."),
            ToolParamSpec("selection_start", False, "Optional selection start offset for a note excerpt."),
            ToolParamSpec("selection_end", False, "Optional selection end offset for a note excerpt."),
        ),
        "example_call": '{"name":"read_note","arguments":{"title":"project-ideas","max_chars":8000}}',
        "legacy_ops": ("read-note",),
    },
    "session": {
        "params": (
            ToolParamSpec("verb", True, "One of: state, recent, search."),
            ToolParamSpec("pattern", False, "Substring match (verb=search)."),
            ToolParamSpec("limit", False, "Event count (recent: 1-100 default 10; search: 1-100 default 20)."),
        ),
        "example_call": '{"name":"session","arguments":{"verb":"recent","limit":5}}',
        "legacy_ops": (),
    },
    "web": {
        "params": (
            ToolParamSpec("verb", False, "text (default; HTML stripped) or fetch (raw decoded body)."),
            ToolParamSpec("url", True, "Absolute http/https URL; private/loopback IPs blocked."),
            ToolParamSpec("max_chars", False, "Output cap (200-50000, default 4000)."),
            ToolParamSpec("timeout", False, "Request timeout in seconds (1-60, default 15)."),
        ),
        "example_call": '{"name":"web","arguments":{"verb":"text","url":"https://example.com/"}}',
        "legacy_ops": (),
    },
    "web_search": {
        "params": (
            ToolParamSpec("query", True, "Live web/internet query for current, latest, news, or unknown-URL information."),
            ToolParamSpec("max_results", False, "Ranked result count, 1-20, default 5."),
            ToolParamSpec("search_depth", False, "ultra-fast, fast, basic, or advanced."),
            ToolParamSpec("topic", False, "general, news, or finance."),
            ToolParamSpec("time_range", False, "day/week/month/year or d/w/m/y."),
            ToolParamSpec("start_date", False, "Optional ISO start date for dated searches."),
            ToolParamSpec("end_date", False, "Optional ISO end date for dated searches."),
            ToolParamSpec("country", False, "Optional country hint for localized results."),
            ToolParamSpec("include_domains", False, "Optional list of domains to restrict results to."),
            ToolParamSpec("exclude_domains", False, "Optional list of domains to exclude."),
            ToolParamSpec("include_answer", False, "Include Tavily answer: true/basic/advanced."),
            ToolParamSpec("include_raw_content", False, "Include raw content: true/markdown/text."),
            ToolParamSpec("include_images", False, "Include image results when useful."),
            ToolParamSpec("include_favicon", False, "Include result favicons when useful."),
            ToolParamSpec("include_usage", False, "Include Tavily usage/credit metadata."),
            ToolParamSpec("max_chars", False, "Output cap, 1000-50000, default 8000."),
            ToolParamSpec("timeout", False, "Request timeout in seconds, 1-60, default 20."),
        ),
        "example_call": '{"name":"web_search","arguments":{"query":"latest AI audio tools","max_results":5}}',
        "legacy_ops": ("search_web",),
    },
    "recall": {
        "params": (
            ToolParamSpec("query", True, "Keywords to search stored memories."),
            ToolParamSpec("max_results", False, "Maximum results from 1 to 20 (default 10)."),
        ),
        "example_call": '{"name":"recall","arguments":{"query":"python preferences","max_results":5}}',
        "legacy_ops": (),
    },
    "search_history": {
        "params": (
            ToolParamSpec("query", True, "Keywords to search for."),
            ToolParamSpec("max_results", False, "Maximum result count from 1 to 8."),
        ),
        "example_call": '{"name":"search_history","arguments":{"query":"python venv setup","max_results":4}}',
        "legacy_ops": ("search-history",),
    },
    "generate_image": {
        "params": (
            ToolParamSpec("prompt", True, "Image description."),
            ToolParamSpec("negative_prompt", False, "What to avoid."),
            ToolParamSpec("width", False, "Width in px (default 512)."),
            ToolParamSpec("height", False, "Height in px (default 512)."),
            ToolParamSpec("steps", False, "Diffusion steps (default 25)."),
            ToolParamSpec("seed", False, "Seed (-1 = random)."),
            ToolParamSpec("model", False, "Model name or path if not already loaded."),
            ToolParamSpec("batch_size", False, "Number of images (1-16, default 1)."),
            ToolParamSpec("guidance_scale", False, "CFG scale (default 7.5)."),
        ),
        "example_call": '{"name":"generate_image","arguments":{"prompt":"a cat in a forest","steps":30}}',
        "legacy_ops": ("generate-image",),
    },
    "generate_audio": {
        "params": (
            ToolParamSpec("prompt", True, "Audio/music description."),
            ToolParamSpec("duration", False, "Duration in seconds (1-30, default 5)."),
            ToolParamSpec("sample_rate", False, "32000, 44100, or 48000."),
        ),
        "example_call": '{"name":"generate_audio","arguments":{"prompt":"upbeat electronic music","duration":8}}',
        "legacy_ops": ("generate-audio",),
    },
    "soundtrap": {
        "params": (
            ToolParamSpec("op", True, "Operation: state, create_project, set_active_project, set_bpm, list_projects, list_clips, add_track, add_clip, place_clip, move_placement, remove_placement, remove_clip, or generate_clip."),
            ToolParamSpec("name", False, "Project or clip name."),
            ToolParamSpec("path", False, "Audio file path for add_clip."),
            ToolParamSpec("project_id", False, "Target Soundtrap project id."),
            ToolParamSpec("clip_id", False, "Target clip id for place_clip/remove_clip."),
            ToolParamSpec("placement_id", False, "Target arrangement placement id for move_placement/remove_placement."),
            ToolParamSpec("track", False, "Track name for place_clip."),
            ToolParamSpec("start_beat", False, "Placement start beat."),
            ToolParamSpec("length_beats", False, "Placement length in beats."),
            ToolParamSpec("prompt", False, "Audio description for generate_clip."),
            ToolParamSpec("duration", False, "Generated clip duration in seconds."),
        ),
        "example_call": '{"name":"soundtrap","arguments":{"op":"place_clip","clip_id":"clip_123","track":"drums","start_beat":0}}',
        "legacy_ops": (),
    },
    "write_file": {
        "params": (
            ToolParamSpec("path", True, "Absolute file path."),
            ToolParamSpec("content", True, "Content to write."),
        ),
        "example_call": '{"name":"write_file","arguments":{"path":"C:/Users/name/hello.txt","content":"Hello world"}}',
        "legacy_ops": ("write-file",),
    },
    "edit_file": {
        "params": (
            ToolParamSpec("path", True, "Absolute file path."),
            ToolParamSpec("find", True, "Exact text to find."),
            ToolParamSpec("replace", True, "Replacement text."),
            ToolParamSpec("count", False, "Max replacements (default all)."),
        ),
        "example_call": '{"name":"edit_file","arguments":{"path":"C:/Users/name/hello.txt","find":"Hello","replace":"Hi"}}',
        "legacy_ops": ("edit-file",),
    },
    "run_command": {
        "params": (
            ToolParamSpec("command", True, "Shell command to execute."),
            ToolParamSpec("cwd", False, "Working directory."),
            ToolParamSpec("timeout", False, "Timeout in seconds (default 30, max 120)."),
        ),
        "example_call": '{"name":"run_command","arguments":{"command":"python --version"}}',
        "legacy_ops": ("run-command",),
    },
    "run_tests": {
        "params": (
            ToolParamSpec("target", False, "Optional test target file/folder/node."),
            ToolParamSpec("runner", False, "Test runner command (default: pytest)."),
            ToolParamSpec("cwd", False, "Working directory."),
            ToolParamSpec("timeout", False, "Timeout in seconds (default 120, max 1800)."),
        ),
        "example_call": '{"name":"run_tests","arguments":{"target":"tests","runner":"pytest","timeout":300}}',
        "legacy_ops": ("run-tests",),
    },
    "git": {
        "params": (
            ToolParamSpec("verb", True, "One of: status, diff, log, branch."),
            ToolParamSpec("cwd", False, "Absolute directory path; defaults to process cwd."),
            ToolParamSpec("limit", False, "Commit count for verb=log (1-100, default 10)."),
            ToolParamSpec("full", False, "When verb=diff, return full diff instead of --stat (default false)."),
        ),
        "example_call": '{"name":"git","arguments":{"verb":"status"}}',
        "legacy_ops": (),
    },
    "zip_files": {
        "params": (
            ToolParamSpec("paths", True, "List of absolute files/directories to include."),
            ToolParamSpec("output", True, "Output .zip path."),
            ToolParamSpec("base_dir", False, "Optional base directory for relative archive names."),
        ),
        "example_call": '{"name":"zip_files","arguments":{"paths":["C:/project/src","C:/project/README.md"],"output":"C:/project/archive.zip"}}',
        "legacy_ops": ("zip-files",),
    },
    "unzip_file": {
        "params": (
            ToolParamSpec("archive", True, "Input .zip archive path."),
            ToolParamSpec("output_dir", True, "Directory to extract into."),
        ),
        "example_call": '{"name":"unzip_file","arguments":{"archive":"C:/tmp/archive.zip","output_dir":"C:/tmp/unpacked"}}',
        "legacy_ops": ("unzip-file",),
    },
    "set_session_meta": {
        "params": (
            ToolParamSpec("title", False, "New chat title."),
            ToolParamSpec("summary", False, "Optional summary lines (list or string)."),
        ),
        "example_call": '{"name":"set_session_meta","arguments":{"title":"Monolith capability sprint"}}',
        "legacy_ops": ("set-session-meta", "rename-chat"),
    },
    "llm_call": {
        "params": (
            ToolParamSpec("prompt", False, "Prompt for the nested LLM call."),
            ToolParamSpec("messages", False, "Optional explicit message list."),
            ToolParamSpec("system", False, "Optional system instruction."),
            ToolParamSpec("max_tokens", False, "Completion tokens (int) or <MAX_TOKENS> placeholder."),
        ),
        "example_call": '{"name":"llm_call","arguments":{"prompt":"Summarize this file in 5 bullets.","max_tokens":"<MAX_TOKENS>"}}',
        "legacy_ops": ("llm-call",),
    },
    "reload_skills": {
        "params": (
            ToolParamSpec("max_names", False, "Max skill names to list in the response."),
        ),
        "example_call": '{"name":"reload_skills","arguments":{}}',
        "legacy_ops": ("reload-skills",),
    },
    "create_tool": {
        "params": (
            ToolParamSpec("name", True, "Tool name (letters, numbers, hyphen/underscore)."),
            ToolParamSpec("description", False, "Short human-readable description."),
            ToolParamSpec("overwrite", False, "Overwrite existing SKILL.md if true."),
        ),
        "example_call": '{"name":"create_tool","arguments":{"name":"summarize_notes","description":"Summarize local note files."}}',
        "legacy_ops": ("create-tool",),
    },
    "get_budget_score": {
        "params": (
            ToolParamSpec("evaluate_message", False, "If true, evaluate the supplied message."),
            ToolParamSpec("message", False, "Optional message probe (used only when evaluate_message=true)."),
            ToolParamSpec("message_count", False, "Synthetic history depth hint for message probe."),
        ),
        "example_call": '{"name":"get_budget_score","arguments":{}}',
        "legacy_ops": ("get-budget-score",),
    },
    "get_context_summary": {
        "params": (
            ToolParamSpec("include_last_prompt", False, "Include last user prompt from world state."),
        ),
        "example_call": '{"name":"get_context_summary","arguments":{"include_last_prompt":true}}',
        "legacy_ops": ("get-context-summary",),
    },
    "inspect_trace": {
        "params": (
            ToolParamSpec("verb", True, "Operation: recent, errors, or one."),
            ToolParamSpec("limit", False, "Result count for recent/errors (default 5, max 50)."),
            ToolParamSpec("turn_id", False, "Turn id (required for verb=one)."),
        ),
        "example_call": '{"name":"inspect_trace","arguments":{"verb":"recent","limit":5}}',
        "legacy_ops": (),
    },
    "inspect_pipeline": {
        "params": (
            ToolParamSpec("verb", True, "Operation: events, faults, last, or one."),
            ToolParamSpec("turn_id", False, "Turn id (required for verb=events/one)."),
            ToolParamSpec("limit", False, "Result count for verb=faults (default 20, max 200)."),
            ToolParamSpec("fault_kind", False, "Optional filter for verb=faults (e.g. 'tool_no_fire')."),
            ToolParamSpec("since", False, "ISO-8601 cutoff for verb=faults (default: 24h ago)."),
        ),
        "example_call": '{"name":"inspect_pipeline","arguments":{"verb":"events","turn_id":"<id>"}}',
        "legacy_ops": (),
    },
    "monosearch": {
        "params": (
            ToolParamSpec("verb", False, "Operation: failing, recurring, pulling, unresolved, search/find, or get. Defaults to search when query/source/meta is present, get when id is present."),
            ToolParamSpec("query", False, "Keyword for verb=search/find."),
            ToolParamSpec("meta", False, "High-level search target: tools, skills, capabilities, debug, memory, workflows."),
            ToolParamSpec("source", False, "Scope verb=search/find to one store: tools, skills, faults, ratings, knowledge, warrants or claim_graph, history or canonical_log, turns, stages, memory, bearing, identity, curiosity, reminders, investigations, lag, health."),
            ToolParamSpec("id", False, "Namespaced id for verb=get (e.g. tool:edit_file, skill:monosearch, fault:991, outcome:42, clog:1840)."),
            ToolParamSpec("limit", False, "Result count (default 10, max 50)."),
            ToolParamSpec("since", False, "ISO-8601 cutoff for verb=search/find."),
        ),
        "example_call": '{"name":"monosearch","arguments":{"verb":"find","meta":"tools","query":"edit file","limit":5}}',
        "legacy_ops": (),
    },
    "monopulse": {
        "params": (
            ToolParamSpec("verb", True, "Operation: pulse, hotspots, stalled, drift, or changed."),
            ToolParamSpec("limit", False, "Result count (default 12 for pulse, 10 for other modes; max 50)."),
        ),
        "example_call": '{"name":"monopulse","arguments":{"verb":"pulse","limit":12}}',
        "legacy_ops": (),
    },
    "spawn_subagent": {
        "params": (
            ToolParamSpec("prompt", True, "The focused task for the sub-agent (or provide 'messages' instead)."),
            ToolParamSpec("level", False, "Authority tier: 2=Worker, 3=Leaf. Defaults to one deeper than the caller."),
            ToolParamSpec("frame", False, "Short label naming the sub-agent's scope, e.g. 'read-only view'."),
            ToolParamSpec("messages", False, "Alternative to prompt: explicit [{\"role\":...,\"content\":...}] turns."),
        ),
        "example_call": '{"name":"spawn_subagent","arguments":{"level":3,"frame":"read-only view","prompt":"Read the file and summarize its public API.","messages":[]}}',
        "legacy_ops": (),
    },
    "scratchpad": {
        "params": (
            ToolParamSpec("op", True, "Operation: pin, retire, read, working_memory_*, propose_amendment, list_proposals, record_confidence, review_read, review_mark, or observe."),
            ToolParamSpec("text", False, "Pin body for op=pin (max 200 chars for lesson|pending, 500 for anchor)."),
            ToolParamSpec("category", False, "Pin category for op=pin: anchor, pending, or lesson (default lesson)."),
            ToolParamSpec("source", False, "Pin source for op=pin: user_said, i_inferred, or evidence (default i_inferred)."),
            ToolParamSpec("evidence", False, "Concrete instance backing a lesson (op=pin; max 120 chars)."),
            ToolParamSpec("supersedes", False, "Pin id this pin replaces (op=pin); predecessor auto-retires."),
            ToolParamSpec("id", False, "Pin id (required for op=retire)."),
            ToolParamSpec("reason", False, "Retirement reason (op=retire): resolved, wrong, stale, or user_retired."),
            ToolParamSpec("include_retired", False, "Include last 5 retired pins (op=read; default false)."),
            ToolParamSpec("item_id", False, "Review item id (op=review_mark), e.g. acu:25 or proposal:1."),
            ToolParamSpec("action", False, "Review action (op=review_mark): resolve, dismiss, snooze, or escalate."),
            ToolParamSpec("kind", False, "Review kind filter (op=review_read)."),
            ToolParamSpec("subkind", False, "Review subkind filter or observation subkind."),
            ToolParamSpec("limit", False, "Max review items to return (op=review_read)."),
            ToolParamSpec("summary", False, "Observation summary (op=observe)."),
            ToolParamSpec("note", False, "Optional review note (op=review_mark)."),
            ToolParamSpec("severity", False, "Observation severity 1-5 (op=observe)."),
            ToolParamSpec("snooze_hours", False, "Hours to snooze (op=review_mark action=snooze)."),
        ),
        "example_call": '{"name":"scratchpad","arguments":{"op":"pin","category":"lesson","source":"i_inferred","text":"User wants terse responses when in flow."}}',
        "legacy_ops": (),
    },
}


def canonical_tool_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def canonical_skill_name(value: str) -> str:
    return canonical_tool_name(value)


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_skill_frontmatter(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"Skill missing YAML frontmatter: {path}")
    fields: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        fields[key.strip()] = _unquote(raw_value)
    name = canonical_tool_name(fields.get("name", ""))
    description = fields.get("description", "").strip()
    if not name or not description:
        raise ValueError(f"Skill frontmatter missing name or description: {path}")
    return name, description


@lru_cache(maxsize=1)
def list_tools() -> tuple[ToolSpec, ...]:
    tools: list[ToolSpec] = []
    if not SKILLS_DIR.exists():
        return tuple()
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        try:
            name, description = _parse_skill_frontmatter(skill_md)
        except Exception:
            continue
        meta = _TOOL_RUNTIME_META.get(name, {})
        tools.append(
            ToolSpec(
                name=name,
                description=description,
                path=skill_md,
                params=tuple(meta.get("params", ())),
                example_call=meta.get("example_call"),
                legacy_ops=tuple(meta.get("legacy_ops", ())),
                json_schema=BUILTIN_TOOL_SCHEMAS.get(name),
            )
        )
    return tuple(tools)


@lru_cache(maxsize=1)
def _tool_lookup() -> dict[str, ToolSpec]:
    lookup: dict[str, ToolSpec] = {}
    for spec in list_tools():
        lookup[spec.name] = spec
        for alias in spec.legacy_ops:
            lookup[canonical_tool_name(alias)] = spec
    return lookup


def get_tool(name_or_alias: str) -> ToolSpec | None:
    return _tool_lookup().get(canonical_tool_name(name_or_alias))


def get_skill(name_or_alias: str) -> ToolSpec | None:
    return get_tool(name_or_alias)


def build_tool_catalog() -> str:
    lines = ["Available tools:"]
    for spec in list_tools():
        params_str = ""
        if spec.params:
            parts = []
            for p in spec.params:
                parts.append(f"{p.name}{'?' if not p.required else ''}")
            params_str = f" ({', '.join(parts)})"
        lines.append(f"- {spec.name}{params_str}: {spec.description}")
    return "\n".join(lines).strip()


def build_skill_catalog() -> str:
    return build_tool_catalog()


def clear_skill_cache() -> None:
    """Invalidate cached skill/tool lookups so new skill directories are discovered."""
    list_tools.cache_clear()
    _tool_lookup.cache_clear()


def list_skills() -> tuple[ToolSpec, ...]:
    return list_tools()
