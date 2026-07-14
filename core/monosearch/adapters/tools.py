from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record
from core.skill_registry import ToolSpec, canonical_tool_name, get_tool, list_tools

_ID_PREFIX = "tool:"

_TOOL_USE_HINTS: dict[str, str] = {
    "open_file": (
        "open/read/inspect local files of many formats; first choice for pdf, docx, xlsx, "
        "csv, json, code, markdown, zip/archive members, images, screenshots, scanned pdf, "
        "ocr fallback status; use read_file only for known plain text"
    ),
    "web_search": (
        "current/latest/live online internet search when no exact URL is known; "
        "chain web_search -> web when page content is needed"
    ),
    "web": "read a specific known URL; use web_search first when no URL is known",
}


def _now_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


def _param_signature(spec: ToolSpec) -> str:
    if not spec.params:
        return "no args"
    return ", ".join(f"{p.name}{'?' if not p.required else ''}" for p in spec.params)


def _required_params(spec: ToolSpec) -> list[str]:
    return [p.name for p in spec.params if p.required]


def _optional_params(spec: ToolSpec) -> list[str]:
    return [p.name for p in spec.params if not p.required]


def _tool_use_hint(spec: ToolSpec) -> str:
    return _TOOL_USE_HINTS.get(spec.name, "")


def _example_arguments(spec: ToolSpec) -> str:
    """A VALID-JSON arguments object for the call_hint — never a bare ``{...}``.

    The model fetches this schema and copies the hint verbatim; a literal
    ``{...}`` is unparseable and produced "malformed tool call - could not parse
    JSON". No required params -> ``{}``; otherwise each required param becomes a
    ``"<name>"`` placeholder (still valid JSON the model fills in)."""
    required = _required_params(spec)
    if not required:
        return "{}"
    body = ", ".join(f'"{name}": "<{name}>"' for name in required)
    return "{" + body + "}"


def _search_blob(spec: ToolSpec) -> str:
    parts = [
        spec.name,
        spec.name.replace("_", " "),
        spec.description,
        _tool_use_hint(spec),
        " ".join(p.name for p in spec.params),
        " ".join(p.detail for p in spec.params),
        " ".join(spec.legacy_ops),
    ]
    return " ".join(parts).lower()


def _score(spec: ToolSpec, query: str) -> int:
    q = str(query or "").strip().lower()
    if not q:
        return 1
    blob = _search_blob(spec)
    canonical = canonical_tool_name(q)
    score = 0
    if canonical == spec.name:
        score += 100
    elif canonical in spec.name:
        score += 60
    if q in blob:
        score += 40
    for term in re.findall(r"[a-z0-9_]+", q):
        if term in blob:
            score += 8
    return score


def _tool_metadata(spec: ToolSpec) -> dict:
    return {
        "kind": "tool",
        "name": spec.name,
        "path": str(spec.path),
        "required_params": _required_params(spec),
        "optional_params": _optional_params(spec),
        "example_call": spec.example_call,
        "legacy_ops": list(spec.legacy_ops),
        "json_schema": spec.json_schema,
    }


def _summary_text(spec: ToolSpec) -> str:
    hint = _tool_use_hint(spec)
    text = f"tool:{spec.name} ({_param_signature(spec)}) - {spec.description}"
    if hint:
        text += f" use_when: {hint}."
    text += f" next: monosearch get id=\"tool:{spec.name}\" for exact schema/example."
    return text


def _detail_text(spec: ToolSpec) -> str:
    lines = [
        "[TOOL]",
        f"name: {spec.name}",
        f"description: {spec.description}",
        f"path: {spec.path}",
        f"params: {_param_signature(spec)}",
    ]
    hint = _tool_use_hint(spec)
    if hint:
        lines.append(f"use_when: {hint}")
    if spec.params:
        lines.append("param_details:")
        for param in spec.params:
            required = "required" if param.required else "optional"
            lines.append(f"- {param.name} ({required}): {param.detail}")
    if spec.legacy_ops:
        lines.append(f"aliases: {', '.join(spec.legacy_ops)}")
    if spec.example_call:
        lines.append(f"example_call: {spec.example_call}")
    if spec.json_schema:
        lines.append("json_schema:")
        lines.append(json.dumps(spec.json_schema, ensure_ascii=False, sort_keys=True))
    # The model copies the call_hint VERBATIM (reprobe 2026-06-16: a "<prompt>"
    # placeholder became a literal garbage spawn). So the hint carries the REAL
    # example_call when one exists; only no-example tools fall back to the
    # generated (still valid-JSON) shape.
    if spec.example_call:
        hint_body = spec.example_call
    else:
        hint_body = f'{{"name":"{spec.name}","arguments":{_example_arguments(spec)}}}'
    lines.append(f"call_hint: use <tool_call>{hint_body}</tool_call>")
    return "\n".join(lines)


class ToolsAdapter(SourceAdapter):
    name = "tools"
    evidence_tier = EvidenceTier.TELEMETRY

    def _to_record(self, spec: ToolSpec, *, detail: bool = False) -> Record:
        return Record(
            namespaced_id=f"{_ID_PREFIX}{spec.name}",
            source=self.name,
            provenance=Provenance.SELF,
            recurrence_key=None,
            text=_detail_text(spec) if detail else _summary_text(spec),
            metadata=_tool_metadata(spec),
            ts=_now_epoch(),
            evidence_tier=self.evidence_tier,
        )

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        scored: list[tuple[int, ToolSpec]] = []
        for spec in list_tools():
            score = _score(spec, query)
            if score > 0:
                scored.append((score, spec))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [self._to_record(spec) for _, spec in scored[: max(1, int(limit))]]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        name = namespaced_id[len(_ID_PREFIX):]
        spec = get_tool(name)
        if spec is None:
            return None
        return self._to_record(spec, detail=True)

    def list(self, filters: dict, limit: int) -> list[Record]:
        specs = sorted(list_tools(), key=lambda spec: spec.name)
        return [self._to_record(spec) for spec in specs[: max(1, int(limit))]]
