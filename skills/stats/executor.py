"""stats — verb-routed read-only telemetry query tool.

Routes each `verb` to a StatsStore method and returns a JSON-serialized
envelope. Used by the stats addon's Wrapped generator (the model invokes
this tool to fetch its own data brief) and available to any future feature
that needs aggregate telemetry.

Entry point name `run` is the contract _load_dynamic_executor looks for.
"""
from __future__ import annotations

import json
from typing import Any

from core.stats_store import StatsStore


def run(cmd: dict, ctx: Any) -> str:
    verb = str(cmd.get("verb") or "").strip().lower()
    if not verb:
        return "[stats: error - 'verb' is required]"
    store = StatsStore()
    range_key = str(cmd.get("range") or "all").strip().lower()

    try:
        if verb == "lifetime":
            payload = store.get_lifetime_summary()
            payload["streak"] = store.get_streak()
        elif verb == "rollups":
            from datetime import date as _date_cls
            today = _date_cls.today().isoformat()
            payload = store.get_daily_rollups("0001-01-01", today) if hasattr(store, "get_daily_rollups") else []
        elif verb == "distribution":
            plane = str(cmd.get("plane") or "").strip().lower()
            if not plane:
                return "[stats: error - 'plane' is required for verb=distribution]"
            payload = store.get_mode_distribution(plane, range_key) if plane != "effort" else store.get_effort_distribution(range_key)
        elif verb == "records":
            payload = store.get_personal_records()
        elif verb == "achievements":
            limit = int(cmd.get("limit") or 8)
            payload = store.get_achievements(limit=limit)
        elif verb == "substrate":
            payload = store.get_substrate_summary()
        elif verb == "time_rhythm":
            rhythm = store.get_time_rhythm(range_key)
            payload = {f"{wk}:{hr:02d}": count for (wk, hr), count in rhythm.items()}
        elif verb == "pipeline_cost":
            payload = store.get_pipeline_cost_breakdown(range_key)
        elif verb == "wrapped_brief":
            payload = {
                "lifetime": store.get_lifetime_summary(),
                "streak": store.get_streak(),
                "rating_histogram": store.get_rating_histogram(range_key),
                "effort_distribution": store.get_effort_distribution(range_key),
                "reasoning_distribution": store.get_mode_distribution("reasoning", range_key),
                "fault_summary": store.get_fault_summary(range_key),
                "tool_usage": store.get_tool_usage(range_key, top_n=5),
                "personal_records": store.get_personal_records(),
                "achievements": store.get_achievements(limit=10),
                "substrate": store.get_substrate_summary(),
            }
        else:
            return f"[stats: error - unknown verb '{verb}']"
    except Exception as exc:
        return f"[stats: error - {type(exc).__name__}: {exc}]"

    return f"[stats: {verb}]\n{json.dumps(payload, indent=2, ensure_ascii=False, default=str)}"
