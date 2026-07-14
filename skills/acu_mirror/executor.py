"""acu_mirror skill wrapper.

Deterministic and read-only. All inspection logic lives in core.acu_mirror.
"""
from __future__ import annotations

from typing import Any

from core import acu_mirror


def run(cmd: dict, ctx: Any) -> str:
    op = str((cmd or {}).get("op") or "snapshot").strip().lower()
    if op not in {"snapshot", "inspect", "detect"}:
        return "[acu_mirror: unknown op {!r} - use 'snapshot']".format(op)

    snapshot = acu_mirror.build_snapshot(
        threshold=cmd.get("threshold", acu_mirror.DEFAULT_THRESHOLD),
        near_band=cmd.get("near_band", acu_mirror.DEFAULT_NEAR_BAND),
        limit=cmd.get("limit", acu_mirror.DEFAULT_LIMIT),
        scan_cap=cmd.get("scan_cap", acu_mirror.SCAN_CAP),
        backend=cmd.get("backend"),
    )
    fmt = str(cmd.get("format") or "text").strip().lower()
    if fmt == "json":
        return "[acu_mirror:json]\n" + acu_mirror.snapshot_json(snapshot)
    return acu_mirror.format_snapshot(snapshot)
