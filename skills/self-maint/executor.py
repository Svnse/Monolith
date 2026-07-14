"""self_maint — runtime control for the self-maintenance daemon.

Lets Monolith (or E via the chat) start/stop the autonomous review-queue triage daemon
and tune its cadence WITHOUT a launcher flag + restart. Single-purpose, like review_act.

SAFETY: this controls the OBSERVE loop only. Whether a snooze/escalate actually APPLIES
is still gated by MONOLITH_SELF_MAINT_V1 (E's hard gate) — starting the daemon here
cannot grant the power to mutate the review queue. The daemon's own wakes (level 2) do
NOT carry this tool (WAKE_TOOLS = review_act only), so the daemon cannot re-arm or
re-pace itself in a loop; only a normal (level-1) turn can.
"""
from __future__ import annotations

from typing import Any

from core import self_maint_leash as leash
from engine import self_maint_runner as smr

_OPS = ("start", "stop", "set_interval", "set_cap", "status")


def _first(cmd: dict, *keys):
    for k in keys:
        v = cmd.get(k)
        if v is not None:
            return v
    return None


def _status_line(runner) -> str:
    try:
        s = runner.snapshot()
    except Exception:
        return "[self_maint: status unavailable]"
    apply_state = "on (LIVE)" if s.get("apply_on") else "off (observe-first)"
    return (f"[self_maint: status={s.get('status')} interval={s.get('interval_s')}s "
            f"apply={apply_state} wake#{s.get('wake')}]")


def run(cmd: dict, ctx: Any = None) -> str:
    op = str(cmd.get("op") or cmd.get("action") or "").strip().lower()
    if op not in _OPS:
        return f"[self_maint: op must be one of {', '.join(_OPS)}]"
    runner = smr.get_runner()

    if op == "status":
        return _status_line(runner)

    if op == "set_interval":
        try:
            iv = runner.set_interval(_first(cmd, "seconds", "interval_s", "value"))
        except Exception as e:  # noqa: BLE001
            return f"[self_maint: set_interval failed: {e}]"
        return f"[self_maint: interval set to {iv}s] " + _status_line(runner)

    if op == "set_cap":
        n = _first(cmd, "cap", "max_per_day", "value")
        leash.set_cap_override(n)
        return f"[self_maint: daily cap set to {n}] " + _status_line(runner)

    if op == "stop":
        try:
            runner.stop(timeout=2.0)
        except Exception as e:  # noqa: BLE001
            return f"[self_maint: stop failed: {e}]"
        return "[self_maint: stopped] " + _status_line(runner)

    # op == "start" — apply optional cadence args BEFORE starting, then bind the
    # live-turn guard (the audit's hard requirement) and start in observe-first mode.
    iv = _first(cmd, "seconds", "interval_s")
    if iv is not None:
        runner.set_interval(iv)
    cap = _first(cmd, "cap", "max_per_day")
    if cap is not None:
        leash.set_cap_override(cap)
    ws = getattr(ctx, "world_state", None)
    runner.set_is_busy(lambda ws=ws: smr.engine_is_busy(ws))
    started = runner.start(force=True)
    verb = "started" if started else "already running"
    # The appended status line is the single source of truth for the apply posture
    # (observe-first vs LIVE) — don't hardcode a posture that can contradict it.
    return f"[self_maint: {verb}] " + _status_line(runner)
