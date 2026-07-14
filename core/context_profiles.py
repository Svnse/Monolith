from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ContextProfile:
    name: str
    max_tool_catalog_entries: int
    monosearch_result_count: int
    runtime_lane_budget_chars: int
    history_refresh_threshold_chars: int
    prompt_budget_chars: int

    def to_dict(self) -> dict:
        return asdict(self)


_PROFILES: dict[str, ContextProfile] = {
    "tiny_local": ContextProfile(
        name="tiny_local",
        max_tool_catalog_entries=12,
        monosearch_result_count=4,
        runtime_lane_budget_chars=1800,
        history_refresh_threshold_chars=12000,
        prompt_budget_chars=24000,
    ),
    "standard_local": ContextProfile(
        name="standard_local",
        max_tool_catalog_entries=512,
        monosearch_result_count=10,
        runtime_lane_budget_chars=5000,
        history_refresh_threshold_chars=36000,
        prompt_budget_chars=96000,
    ),
    "large_context": ContextProfile(
        name="large_context",
        max_tool_catalog_entries=1024,
        monosearch_result_count=18,
        runtime_lane_budget_chars=9000,
        history_refresh_threshold_chars=96000,
        prompt_budget_chars=240000,
    ),
    "cloud_full": ContextProfile(
        name="cloud_full",
        max_tool_catalog_entries=2048,
        monosearch_result_count=24,
        runtime_lane_budget_chars=12000,
        history_refresh_threshold_chars=160000,
        prompt_budget_chars=400000,
    ),
}


def list_context_profiles() -> tuple[ContextProfile, ...]:
    return tuple(_PROFILES.values())


def resolve_context_profile(name: str | None = None) -> ContextProfile:
    key = str(name or "").strip() or "standard_local"
    return _PROFILES.get(key, _PROFILES["standard_local"])


def active_context_profile() -> ContextProfile:
    try:
        from core.config import get_config

        return resolve_context_profile(get_config().llm.context_profile)
    except Exception:
        return resolve_context_profile("standard_local")


def trim_tool_specs(specs: tuple, profile: ContextProfile | None = None) -> tuple:
    prof = profile or active_context_profile()
    max_entries = max(1, int(prof.max_tool_catalog_entries))
    if len(specs) <= max_entries:
        return specs
    required_names = {
        "open_file",
        "read_file",
        "grep",
        "find_files",
        "list_files",
        "run_tests",
        "run_command",
        "inspect_trace",
        "inspect_pipeline",
        "monosearch",
        "monopulse",
        "get_budget_score",
        "get_context_summary",
        "scratchpad",
    }
    required = [spec for spec in specs if getattr(spec, "name", "") in required_names]
    remainder = [spec for spec in specs if getattr(spec, "name", "") not in required_names]
    return tuple((required + remainder)[:max_entries])


def build_profiled_tool_catalog(profile: ContextProfile | None = None) -> str:
    return "\n".join(
        [
            "[TOOL DISCOVERY KERNEL]",
            "Do not rely on a preloaded static tool catalog. Discover capabilities through monosearch, then call the exact tool schema you fetched.",
            "Primary discovery calls:",
            '- tools: <tool_call>{"name":"monosearch","arguments":{"verb":"find","meta":"tools","query":"edit file","limit":5}}</tool_call>',
            '- local file/document reading: <tool_call>{"name":"monosearch","arguments":{"verb":"find","meta":"tools","query":"open read pdf zip docx xlsx image OCR file","limit":5}}</tool_call>',
            '- live/current web search: <tool_call>{"name":"monosearch","arguments":{"verb":"find","meta":"tools","query":"latest online web search","limit":3}}</tool_call>',
            '- skills: <tool_call>{"name":"monosearch","arguments":{"verb":"find","meta":"skills","query":"workflow card","limit":5}}</tool_call>',
            '- broad capabilities: <tool_call>{"name":"monosearch","arguments":{"verb":"find","meta":"capabilities","query":"what I need to do","limit":8}}</tool_call>',
            '- exact schema: <tool_call>{"name":"monosearch","arguments":{"verb":"get","id":"tool:<name>"}}</tool_call>',
            "Search routing rule: use meta for broad intent buckets; use source for an exact store. If a meta search misses, retry the likely exact source before concluding no record exists.",
            'Useful meta values: tools, skills, capabilities, debug, memory, workflows. `meta=debug` searches faults/turns/stages/ratings/health; it does not search canonical_log.',
            'Useful source values: tools, skills, faults, ratings, knowledge, warrants or claim_graph, history or canonical_log, turns, stages, memory, bearing, identity, curiosity, reminders, investigations, lag, health.',
            'Canonical timeline/log queries: use source="history" or source="canonical_log"; empty query lists recent rows; get ids with id="clog:<event_id>".',
            'Delivery/tool-loop debugging: search source="history" for canonical events and source="turns" for turn/frame telemetry; do not rely on meta="debug" alone.',
            "After a tool is discovered or used, it may appear in [SESSION TOOL PALETTE] for this chat only. Fresh sessions start empty.",
            "Monosearch itself is always callable with fields: verb, meta, query, source, id, limit, since. Verb may be omitted for query/source/meta searches and id gets.",
            "[/TOOL DISCOVERY KERNEL]",
        ]
    ).strip()
