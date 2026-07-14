from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from core.skill_registry import ToolSpec, canonical_tool_name, get_tool

_DISCOVERED_ID_RE = re.compile(r"\b(?:tool|skill):([A-Za-z0-9_-]+)\b")


@dataclass(frozen=True)
class SessionPaletteEntry:
    name: str
    reason: str
    spec: ToolSpec


def _message_text(msg: dict) -> str:
    text = msg.get("text")
    if isinstance(text, str):
        return text
    content = msg.get("content")
    return content if isinstance(content, str) else ""


def _parse_tool_payload(text: str) -> dict | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _add_entry(entries: dict[str, SessionPaletteEntry], name: str, reason: str) -> None:
    spec = get_tool(canonical_tool_name(name))
    if spec is None:
        return
    existing = entries.get(spec.name)
    if existing is not None:
        if existing.reason != "called" and reason == "called":
            entries[spec.name] = SessionPaletteEntry(spec.name, reason, spec)
        return
    entries[spec.name] = SessionPaletteEntry(spec.name, reason, spec)


def extract_session_tool_palette(
    messages: Iterable[dict],
    *,
    max_tools: int = 8,
) -> list[SessionPaletteEntry]:
    entries: dict[str, SessionPaletteEntry] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool_result":
            continue
        text = _message_text(msg).strip()
        if not text:
            continue
        payload = _parse_tool_payload(text)
        result_text = text
        if payload is not None:
            tool_name = str(payload.get("tool", "")).strip()
            if tool_name:
                _add_entry(entries, tool_name, "called")
            result_text = str(payload.get("result", "") or text)
        for match in _DISCOVERED_ID_RE.finditer(result_text):
            _add_entry(entries, match.group(1), "discovered")
        if len(entries) >= max_tools:
            break
    return list(entries.values())[:max_tools]


def _param_signature(spec: ToolSpec) -> str:
    if not spec.params:
        return "no args"
    return ", ".join(f"{p.name}{'?' if not p.required else ''}" for p in spec.params)


def _trim(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _session_priors(names: set[str]) -> list[str]:
    priors: list[str] = []
    if {"edit_file", "write_file", "author_workshop_card"} & names:
        priors.append("After writes, verify with read_file/grep and focused run_tests when relevant.")
    if {"grep", "find_files", "list_files"} & names:
        priors.append("After locating candidates, read the exact file before editing or asserting behavior.")
    if {"run_command", "run_tests"} & names:
        priors.append("After shell/test failure, inspect the error text and rerun the narrowest fixed target.")
    if "run_workshop" in names:
        priors.append("After a workflow run, inspect outputs/faults before changing the card.")
    if "monosearch" in names:
        priors.append("Before unfamiliar tool use, fetch monosearch get id=\"tool:<name>\" for exact args.")
    return priors[:4]


def render_session_tool_palette(
    messages: Iterable[dict],
    *,
    max_tools: int = 8,
) -> str | None:
    entries = extract_session_tool_palette(messages, max_tools=max_tools)
    if not entries:
        return None
    lines = [
        "[SESSION TOOL PALETTE]",
        "Session-local capability cache. These tools/skills were called or discovered earlier in this chat; fresh sessions start empty.",
        "Known this chat:",
    ]
    for entry in entries:
        marker = "called" if entry.reason == "called" else "discovered"
        lines.append(
            f"- {entry.name} ({_param_signature(entry.spec)}) [{marker}]: "
            f"{_trim(entry.spec.description, 150)}"
        )
    priors = _session_priors({entry.name for entry in entries})
    if priors:
        lines.append("High-probability next moves:")
        for prior in priors:
            lines.append(f"- {prior}")
    lines.extend(
        [
            "Discovery fallback:",
            '- find: monosearch verb="find" meta="tools|skills|capabilities" query="<intent>"',
            '- store: monosearch source="<store>" query="<intent>"  (stores: knowledge, warrants, claim_graph, faults, history, turns, identity, curiosity)',
            '- schema: monosearch verb="get" id="tool:<name>"',
            "[/SESSION TOOL PALETTE]",
        ]
    )
    return "\n".join(lines)
