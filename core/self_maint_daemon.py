"""Self-maintenance daemon — SAFETY CORE (the testable invariants).

This module holds the audit-required safety primitives the threaded wake loop will
wrap. The loop itself (adapting engine/expedition_runner.py — thread, wake-event,
single-flight generation_lock, fault-halt, guaranteed record_frame) is the proven-
pattern wrapper, built + fresh-audited separately. The dangerous decisions live
HERE so they can be unit-tested:

  - Gap 4: `build_wake_context` pins level=2. The dispatch allow-list gate in
    skill_runtime only fires when ctx.level > 1; a level=1 context bypasses it and
    every tool (write_file/run_command/spawn_subagent) becomes reachable.
  - The allow-list is the NARROW set: the single-purpose `review_act` actuator +
    read-only inspection tools. NOT `scratchpad` (its 13 ops leak substrate writes),
    NOT any mutator. This is the Gap 1 fix made concrete at the daemon boundary.
  - `try_wake` gates each wake on the persisted, restart-safe daily leash (Gap 5).
"""
from __future__ import annotations

import json
import os
from typing import Callable

from core import self_maint_leash
from core.paths import CONFIG_DIR, LOG_DIR

# Read tools were REMOVED from WAKE_TOOLS in V1 (audit 2026-06-22, findings F12/F13):
# the read executors resolve arbitrary ABSOLUTE paths with no workspace confinement at
# level>1, so a wake could pull host secrets (config.yaml api_key, the TAVILY key literal
# in the launcher) into the model context + durable traces — and that fires even under
# observe-first (apply-off). Dropping the read tools closes that BY CONSTRUCTION. A V1
# maintenance wake triages from the queue item's own summary/reason text alone. To re-add
# richer triage later, gate the read executors on _path_is_within(_WORKSPACE_ROOT, ...)
# for level>1 first (core/skill_runtime.py), THEN add them back here.
_READ_TOOLS = frozenset({"open_file", "read_file", "grep", "find_files", "list_files", "calculate"})

# The daemon's COMPLETE tool allow-list: the narrow review actuator ONLY. Deliberately
# excludes scratchpad (op-leak), every mutator, AND (V1) the read tools (see above).
WAKE_TOOLS = frozenset({"review_act"})

# Dark flag that gates the daemon's autonomous self-start. Default OFF => the daemon
# never wakes the model unattended (byte-identical to no daemon). Separate from the
# APPLY flag (MONOLITH_SELF_MAINT_V1, checked inside review_act): trigger-on +
# apply-off = the daemon wakes and LOGS what it would do but applies nothing.
_TRIGGER_FLAG = "MONOLITH_SELF_MAINT_TRIGGER_V1"
_WAKE_INTERVAL_ENV = "MONOLITH_SELF_MAINT_WAKE_INTERVAL_S"
_DEFAULT_WAKE_INTERVAL_S = 1800  # 30 min — conservative; E recalibrates
_TRUTHY = {"1", "true", "yes", "on"}

# Per-wake observability ledger (every wake, including skips, with reason).
_WAKE_LEDGER = CONFIG_DIR / "self_maint_trigger.ledger.jsonl"


def trigger_enabled() -> bool:
    return str(os.environ.get(_TRIGGER_FLAG, "")).strip().lower() in _TRUTHY


def wake_interval_s() -> int:
    raw = os.environ.get(_WAKE_INTERVAL_ENV)
    if raw is None:
        return _DEFAULT_WAKE_INTERVAL_S
    try:
        iv = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_WAKE_INTERVAL_S
    # <=0 would remove the firing-rate floor (cap-burst at day-start + busy-spin churn);
    # fail closed to the default floor, like max_wakes_per_day. (audit #5)
    return iv if iv > 0 else _DEFAULT_WAKE_INTERVAL_S


def log_wake(row: dict) -> None:
    """Append one JSONL row to the wake ledger. Best-effort; never raises."""
    try:
        _WAKE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(_WAKE_LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass


def read_wake_tail(n: int = 20) -> list[dict]:
    """The last n parsed wake-ledger rows, oldest->newest (for the companion panel).
    Best-effort; never raises; [] if the ledger is absent."""
    try:
        if not _WAKE_LEDGER.exists():
            return []
        lines = _WAKE_LEDGER.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[dict] = []
    for ln in lines[-max(1, int(n)):]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def wake_context_kwargs(should_cancel: Callable[[], bool]) -> dict:
    """The safety-critical kwargs for the wake context — PURE/testable (no Qt).
    `level=2` is load-bearing (Gap 4): below it skill_runtime's allow-list gate is
    inert. `allowed_tools` is the narrow set. The threaded loop must not override
    these."""
    return {
        "archive_dir": LOG_DIR,
        "should_cancel": should_cancel,
        "level": 2,
        "allowed_tools": WAKE_TOOLS,
        "spawn_depth": 1,
    }


def build_wake_context(should_cancel: Callable[[], bool]):
    """Construct the ToolExecutionContext from the safety kwargs. (Imports
    skill_runtime, which pulls Qt — only callable inside the running app.)"""
    from core.skill_runtime import ToolExecutionContext, ToolResultCache
    return ToolExecutionContext(result_cache=ToolResultCache(), **wake_context_kwargs(should_cancel))


def try_wake(now=None) -> dict:
    """Reserve one wake against the persisted daily leash (Gap 5).
    Returns {ok, count, cap[, reason]}; ok=False when the daily cap is reached."""
    return self_maint_leash.try_consume_wake(now=now)
