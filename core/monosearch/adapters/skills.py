from __future__ import annotations

from datetime import datetime, timezone

from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record
from core.skill_registry import ToolSpec, canonical_skill_name, get_skill, list_skills

from .tools import _param_signature, _score, _tool_use_hint

_ID_PREFIX = "skill:"


def _now_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


def _skill_metadata(spec: ToolSpec) -> dict:
    return {
        "kind": "skill",
        "name": spec.name,
        "path": str(spec.path),
        "tool_schema_id": f"tool:{spec.name}",
        "example_call": spec.example_call,
    }


def _summary_text(spec: ToolSpec) -> str:
    hint = _tool_use_hint(spec)
    text = f"skill:{spec.name} - {spec.description}"
    if hint:
        text += f" use_when: {hint}."
    text += f" tool_schema: tool:{spec.name}; next: monosearch get id=\"skill:{spec.name}\"."
    return text


def _detail_text(spec: ToolSpec) -> str:
    lines = [
        "[SKILL]",
        f"name: {spec.name}",
        f"description: {spec.description}",
        f"path: {spec.path}",
        f"executable_tool: {spec.name}",
        f"tool_schema_id: tool:{spec.name}",
        f"params: {_param_signature(spec)}",
    ]
    hint = _tool_use_hint(spec)
    if hint:
        lines.append(f"use_when: {hint}")
    if spec.example_call:
        lines.append(f"example_call: {spec.example_call}")
    lines.append(
        "call_hint: fetch tool:<name> for exact executable arguments before unfamiliar use."
    )
    return "\n".join(lines)


class SkillsAdapter(SourceAdapter):
    name = "skills"
    evidence_tier = EvidenceTier.TELEMETRY

    def _to_record(self, spec: ToolSpec, *, detail: bool = False) -> Record:
        return Record(
            namespaced_id=f"{_ID_PREFIX}{spec.name}",
            source=self.name,
            provenance=Provenance.SELF,
            recurrence_key=None,
            text=_detail_text(spec) if detail else _summary_text(spec),
            metadata=_skill_metadata(spec),
            ts=_now_epoch(),
            evidence_tier=self.evidence_tier,
        )

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        scored: list[tuple[int, ToolSpec]] = []
        for spec in list_skills():
            score = _score(spec, query)
            if score > 0:
                scored.append((score, spec))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [self._to_record(spec) for _, spec in scored[: max(1, int(limit))]]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        name = canonical_skill_name(namespaced_id[len(_ID_PREFIX):])
        spec = get_skill(name)
        if spec is None:
            return None
        return self._to_record(spec, detail=True)

    def list(self, filters: dict, limit: int) -> list[Record]:
        specs = sorted(list_skills(), key=lambda spec: spec.name)
        return [self._to_record(spec) for spec in specs[: max(1, int(limit))]]
