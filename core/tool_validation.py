from __future__ import annotations

import re
from typing import Any

_INT_RE = re.compile(r"^-?\d+$")
_NUM_RE = re.compile(r"^-?(?:\d+(?:\.\d+)?|\.\d+)$")

_ENVELOPE_KEYS = {"tool", "skill", "op", "id"}

# JSON-schema-like contracts for built-in tool argument payloads.
BUILTIN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "calculate": {
        "type": "object",
        "properties": {"expr": {"type": "string", "minLength": 1}},
        "required": ["expr"],
        "additionalProperties": False,
    },
    "grep": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
            "glob": {"type": "string"},
            "max_results": {"type": "integer"},
            "case_sensitive": {"type": "boolean"},
            "recursive": {"type": "boolean"},
        },
        "required": ["pattern", "path"],
        "additionalProperties": False,
    },
    "find_files": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "pattern": {"type": "string", "minLength": 1},
            "recursive": {"type": "boolean"},
            "max_results": {"type": "integer"},
        },
        "required": ["path", "pattern"],
        "additionalProperties": False,
    },
    "list_files": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "pattern": {"type": "string"},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "open_addon": {
        "type": "object",
        "properties": {"addon": {"type": "string", "minLength": 1}},
        "required": ["addon"],
        "additionalProperties": False,
    },
    "open_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "max_chars": {"type": "integer"},
            "offset": {"type": "integer"},
            "member": {"type": "string"},
            "sheet": {"type": "string"},
            "max_rows": {"type": "integer"},
            "max_members": {"type": "integer"},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "read_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "max_chars": {"type": "integer"},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "save_note": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "content": {"type": "string", "minLength": 1},
        },
        "required": ["title", "content"],
        "additionalProperties": False,
    },
    "list_notes": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": [],
        "additionalProperties": False,
    },
    "read_note": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "max_chars": {"type": "integer"},
            "offset": {"type": "integer"},
            "selection_start": {"type": "integer"},
            "selection_end": {"type": "integer"},
        },
        "required": ["title"],
        "additionalProperties": False,
    },
    "session": {
        "type": "object",
        "properties": {
            "verb": {"type": "string", "enum": ["state", "recent", "search"]},
            "pattern": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["verb"],
        "additionalProperties": False,
    },
    "web": {
        "type": "object",
        "properties": {
            "verb": {"type": "string", "enum": ["text", "fetch"]},
            "url": {"type": "string", "minLength": 1},
            "max_chars": {"type": "integer"},
            "timeout": {"type": "integer"},
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    "web_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "max_results": {"type": "integer"},
            "limit": {"type": "integer"},
            "search_depth": {"type": "string", "enum": ["ultra-fast", "fast", "basic", "advanced"]},
            "topic": {"type": "string", "enum": ["general", "news", "finance"]},
            "time_range": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "country": {"type": "string"},
            "include_domains": {"type": "array", "items": {"type": "string"}},
            "exclude_domains": {"type": "array", "items": {"type": "string"}},
            "include_answer": {"type": ["boolean", "string"]},
            "include_raw_content": {"type": ["boolean", "string"]},
            "include_images": {"type": "boolean"},
            "include_favicon": {"type": "boolean"},
            "include_usage": {"type": "boolean"},
            "safe_search": {"type": "boolean"},
            "auto_parameters": {"type": "boolean"},
            "exact_match": {"type": "boolean"},
            "max_chars": {"type": "integer"},
            "timeout": {"type": "integer"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "recall": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "search_history": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "generate_image": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1},
            "negative_prompt": {"type": "string"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "steps": {"type": "integer"},
            "seed": {"type": "integer"},
            "model": {"type": "string"},
            "guidance_scale": {"type": "number"},
            "batch_size": {"type": "integer"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
    "generate_audio": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1},
            "duration": {"type": "number"},
            "sample_rate": {"type": "integer"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
    "soundtrap": {
        "type": "object",
        "properties": {
            "verb": {"type": "string"},
            "name": {"type": "string"},
            "path": {"type": "string"},
            "project_id": {"type": "string"},
            "clip_id": {"type": "string"},
            "placement_id": {"type": "string"},
            "track": {"type": "string"},
            "start_beat": {"type": "number"},
            "length_beats": {"type": "number"},
            "gain": {"type": "number"},
            "muted": {"type": "boolean"},
            "prompt": {"type": "string"},
            "duration": {"type": "number"},
            "sample_rate": {"type": "integer"},
            "bpm": {"type": "number"},
            "delete_file": {"type": "boolean"},
        },
        # 'op' and 'id' are envelope keys stripped by validate_tool_arguments.
        "required": [],
        "additionalProperties": False,
    },
    "write_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    "edit_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "find": {"type": "string"},
            "replace": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["path", "find", "replace"],
        "additionalProperties": False,
    },
    "run_command": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1},
            "cwd": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "run_tests": {
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "runner": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": [],
        "additionalProperties": False,
    },
    "git": {
        "type": "object",
        "properties": {
            "verb": {"type": "string", "enum": ["status", "diff", "log", "branch"]},
            "cwd": {"type": "string"},
            "limit": {"type": "integer"},
            "full": {"type": "boolean"},
        },
        "required": ["verb"],
        "additionalProperties": False,
    },
    "zip_files": {
        "type": "object",
        "properties": {
            "paths": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "output": {"type": "string", "minLength": 1},
            "base_dir": {"type": "string"},
        },
        "required": ["paths", "output"],
        "additionalProperties": False,
    },
    "unzip_file": {
        "type": "object",
        "properties": {
            "archive": {"type": "string", "minLength": 1},
            "output_dir": {"type": "string", "minLength": 1},
        },
        "required": ["archive", "output_dir"],
        "additionalProperties": False,
    },
    "set_session_meta": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": ["string", "array"]},
        },
        "required": [],
        "additionalProperties": False,
    },
    "llm_call": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "messages": {"type": "array", "items": {"type": "object"}},
            "system": {"type": "string"},
            "max_tokens": {"type": ["integer", "string"]},
            "temp": {"type": "number"},
            "top_p": {"type": "number"},
            "max_chars": {"type": "integer"},
            "thinking": {"type": "boolean"},
        },
        "required": [],
        "additionalProperties": False,
    },
    "reload_skills": {
        "type": "object",
        "properties": {"max_names": {"type": "integer"}},
        "required": [],
        "additionalProperties": False,
    },
    "create_tool": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "description": {"type": "string"},
            "overwrite": {"type": "boolean"},
            "overwrite_executor": {"type": "boolean"},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    "get_budget_score": {
        "type": "object",
        "properties": {
            "evaluate_message": {"type": "boolean"},
            "message": {"type": "string"},
            "message_count": {"type": "integer"},
        },
        "required": [],
        "additionalProperties": False,
    },
    "get_context_summary": {
        "type": "object",
        "properties": {"include_last_prompt": {"type": "boolean"}},
        "required": [],
        "additionalProperties": False,
    },
    "scratchpad": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "category": {"type": "string"},
            "source": {"type": "string"},
            "evidence": {"type": "string"},
            "supersedes": {"type": "integer"},
            "reason": {"type": "string"},
            "include_retired": {"type": "boolean"},
            # propose_amendment fields (Monolith-authored substrate amendments)
            "target": {"type": "string"},
            "section": {"type": "string"},
            "current_text": {"type": "string"},
            "proposed_text": {"type": "string"},
            "rationale": {"type": "string"},
            # record_confidence fields (ANALYSIS loop calibration log)
            "value": {"type": "integer"},
            "claim": {"type": "string"},
            "premise": {"type": "string"},
            # review_loop fields
            "item_id": {"type": "string"},
            "action": {"type": "string"},
            "kind": {"type": "string"},
            "subkind": {"type": "string"},
            "limit": {"type": "integer"},
            "summary": {"type": "string"},
            "note": {"type": "string"},
            "severity": {"type": "integer"},
            "snooze_hours": {"type": "number"},
            "hours": {"type": "number"},
            "snoozed_until": {"type": "string"},
            # pin self-violation check override (pin-10 incident guard)
            "bypass_self_violation_check": {"type": "boolean"},
            # introspect filter: substring match against subsystem entry name
            "name": {"type": "string"},
        },
        # 'op' and 'id' are envelope keys stripped by validate_tool_arguments,
        # so they cannot be enforced here. The executor handles missing/invalid
        # op + id with explicit error returns.
        "required": [],
        "additionalProperties": False,
    },
    "inspect_trace": {
        "type": "object",
        "properties": {
            "verb": {"type": "string"},
            "limit": {"type": "integer"},
            "turn_id": {"type": "string"},
        },
        # 'verb' resolution and 'turn_id' presence checked at executor.
        "required": [],
        "additionalProperties": False,
    },
    "inspect_pipeline": {
        "type": "object",
        "properties": {
            "verb": {"type": "string"},
            "turn_id": {"type": "string"},
            "limit": {"type": "integer"},
            "fault_kind": {"type": "string"},
            "since": {"type": "string"},
        },
        # 'verb' resolution and 'turn_id' presence checked at executor.
        "required": [],
        "additionalProperties": False,
    },
    "monosearch": {
        "type": "object",
        "properties": {
            "verb": {"type": "string"},
            "query": {"type": "string"},
            "meta": {"type": "string"},
            "source": {"type": "string"},
            "id": {"type": "string"},
            "limit": {"type": "integer"},
            "since": {"type": "string"},
        },
        # 'verb' resolution + 'source' alias checked at the executor.
        "required": [],
        "additionalProperties": False,
    },
    "monopulse": {
        "type": "object",
        "properties": {
            "verb": {"type": "string", "enum": ["pulse", "hotspots", "stalled", "drift", "changed"]},
            "limit": {"type": "integer"},
        },
        # Executor defaults to verb=pulse for manual convenience.
        "required": [],
        "additionalProperties": False,
    },
    "ask_user": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "minLength": 1},
            "options": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "minLength": 1},
                        "description": {"type": "string"},
                    },
                    "required": ["label"],
                    "additionalProperties": False,
                },
            },
            "header": {"type": "string"},
            "multi_select": {"type": "boolean"},
        },
        # Option count cap (≤4), label-length caps, etc. enforced at executor
        # for friendlier error messages.
        "required": ["question", "options"],
        "additionalProperties": False,
    },
    "stats": {
        "type": "object",
        "properties": {
            "verb": {"type": "string", "minLength": 1},
            "range": {"type": "string"},
            "plane": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["verb"],
        "additionalProperties": False,
    },
}


def _is_int_like(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number_like(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        if _is_int_like(value):
            return True
        return isinstance(value, str) and _INT_RE.fullmatch(value.strip()) is not None
    if expected == "number":
        if _is_number_like(value):
            return True
        return isinstance(value, str) and _NUM_RE.fullmatch(value.strip()) is not None
    if expected == "boolean":
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            lowered = value.strip().lower()
            return lowered in {"true", "false"}
        return False
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def _validate_value(field: str, value: Any, schema: dict[str, Any], errors: list[str]) -> None:
    expected = schema.get("type")
    expected_types: list[str]
    if isinstance(expected, list):
        expected_types = [str(item) for item in expected]
    elif isinstance(expected, str):
        expected_types = [expected]
    else:
        expected_types = []

    if expected_types and not any(_matches_type(value, kind) for kind in expected_types):
        errors.append(
            f"field '{field}' expected {', '.join(expected_types)} but got {type(value).__name__}"
        )
        return

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value.strip()) < min_length:
            errors.append(f"field '{field}' must be at least {min_length} chars")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"field '{field}' must contain at least {min_items} item(s)")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_value(f"{field}[{index}]", item, item_schema, errors)


def validate_tool_arguments(tool_name: str, cmd: dict[str, Any]) -> list[str]:
    schema = BUILTIN_TOOL_SCHEMAS.get(str(tool_name or "").strip().lower())
    if not isinstance(schema, dict):
        # Dynamic tools may define their own argument semantics at runtime.
        return []

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    additional_allowed = bool(schema.get("additionalProperties", True))

    args = {
        k: v
        for k, v in cmd.items()
        if k not in _ENVELOPE_KEYS and not str(k).startswith("_")
    }
    errors: list[str] = []

    for key in required:
        if key not in args:
            errors.append(f"missing required field '{key}'")

    if not additional_allowed:
        unknown = sorted(k for k in args if k not in properties)
        if unknown:
            errors.append(f"unknown field(s): {', '.join(unknown)}")

    for key, value in args.items():
        field_schema = properties.get(key)
        if isinstance(field_schema, dict):
            _validate_value(key, value, field_schema, errors)

    # Tool-specific cross-field constraints.
    if tool_name == "llm_call":
        has_prompt = isinstance(args.get("prompt"), str) and bool(str(args.get("prompt")).strip())
        has_messages = isinstance(args.get("messages"), list) and len(args.get("messages", [])) > 0
        if not has_prompt and not has_messages:
            errors.append("llm_call requires either non-empty 'prompt' or non-empty 'messages'")

    return errors
