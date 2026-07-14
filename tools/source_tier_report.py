"""Print the Source-Tier distribution over recent frames (graded straight off
frame_traces.metadata_json). Run after activating MONOLITH_SOURCE_TIER_V1 to
sanity-check the classifier on live turns BEFORE wiring any consumer hard.

Usage:
    python tools/source_tier_report.py [N]   # default N=200 recent turns
"""
from __future__ import annotations

import os
import sys
from collections import Counter

# Make the repo root importable when run as `python tools/source_tier_report.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import turn_trace as tt


def _turn_id(summary) -> str | None:
    # list_recent_turns returns TurnTraceSummary objects (attribute access);
    # stay defensive in case the shape changes.
    if isinstance(summary, dict):
        return summary.get("turn_id")
    return getattr(summary, "turn_id", None)


def main(n: int = 200) -> None:
    rows = tt.list_recent_turns(limit=n)
    dist: Counter = Counter()
    inconsistent = 0
    for r in rows:
        tid = _turn_id(r)
        if not tid:
            continue
        joined = tt.get_turn_trace(tid)
        if joined is None or joined.frame is None:
            continue
        meta = joined.frame.metadata or {}
        tier = meta.get("source_tier", "(unset)")
        dist[tier] += 1
        if tier == "tool" and meta.get("region_tiers", {}).get("tool") != "tool":
            inconsistent += 1
    total = sum(dist.values())
    print(f"frames inspected: {total}")
    for tier, count in dist.most_common():
        pct = (100.0 * count / total) if total else 0.0
        print(f"  {tier:>14}: {count:4d}  ({pct:5.1f}%)")
    print(f"inconsistent 'tool' rows (no tool region): {inconsistent}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
