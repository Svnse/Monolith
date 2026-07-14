"""The active-agents spine — pure projection layer.

Two model-facing projections of the model's own sub-agent activity (so it can
SEE what it's doing — proprioception), injected like bearing/runtime state:
  * ``render_active_agents_block`` ([ACTIVE AGENTS]) — the LIVE, running-only view
    from the in-memory worker set (immediate; main-thread on spawn).
  * ``render_agent_recap`` ([AGENTS THIS TURN]) — the DURABLE recap of this turn's
    spawns + outcomes, read from turn_trace governance events; closes the
    completed-spawn half (a folded agent vanishes from [ACTIVE AGENTS]) and gives
    the otherwise-unread spawn events a reader.

Neither is the result — the ``[SUBAGENT_RESULT]`` fold-back remains the model's
work product, untouched. No source of truth lives here — the source is the
PageChat's live workers + the durable turn_trace; this module only formats.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

RUNNING = "running"
DONE = "done"

# --- shared read-channel ----------------------------------------------------
# The SOURCE of truth is the PageChat's live spawn workers. The PageChat pushes
# a derived snapshot here on spawn-start / done (both on the main thread); the
# model-context contributor reads it during prompt assembly. This is a
# projection channel (same role as world_state), NOT a competing source.
_lock = threading.Lock()
_active: list["AgentRecord"] = []


@dataclass(frozen=True)
class AgentRecord:
    agent_id: str
    frame: str
    level: int
    status: str  # RUNNING | DONE
    child_turn_id: str = ""  # the sub-agent's own turn id — the zoom-in key
    tool_count: int = 0

    @property
    def is_running(self) -> bool:
        return self.status == RUNNING


_TURN_RE = re.compile(r"\bturn=(\S+)")


def parse_child_turn_id(fenced: str) -> str:
    """Pull the sub-agent's ``child_turn_id`` out of its ``[SUBAGENT_RESULT
    ... turn=<id> ...]`` fence — the key the UI uses to zoom into that agent's
    own trace. Empty when absent."""
    m = _TURN_RE.search(str(fenced or ""))
    return m.group(1).strip() if m else ""


def _label(rec: AgentRecord) -> str:
    frame = str(rec.frame or "").strip() or f"L{rec.level}"
    return frame


def render_active_agents_block(agents: list[AgentRecord]) -> str:
    """Terse model-context block listing the sub-agents running RIGHT NOW.

    Empty string when nothing is running (no live agents = nothing to inject —
    keeps the prompt clean, same discipline as the bearing/runtime lanes).
    """
    running = [a for a in agents if a.is_running]
    if not running:
        return ""
    lines = ["[ACTIVE AGENTS]"]
    for rec in running:
        lines.append(f"- {_label(rec)} (L{rec.level}, running)")
    lines.append("[/ACTIVE AGENTS]")
    return "\n".join(lines)


# --- projection channel + model-context contributor -------------------------

def set_active(records) -> None:
    """Push the current active-agent snapshot (called by the PageChat on the
    main thread whenever an agent starts or finishes)."""
    global _active
    with _lock:
        _active = [r for r in records]


def get_active() -> list[AgentRecord]:
    with _lock:
        return list(_active)


def contribute_section(messages: list[dict], config: dict):
    """Ephemeral-coalescer contributor: inject the live [ACTIVE AGENTS] view so
    the model can see which sub-agents are running this turn. Only on a real
    (non-ephemeral) user turn; empty when nothing is running."""
    from core.ephemeral_coalescer import SectionResult

    if not any(m.get("role") == "user" and not m.get("ephemeral") for m in messages):
        return None
    block = render_active_agents_block(get_active())
    if not block:
        return None
    return SectionResult(name="active_agents", text=block)


def render_agent_recap(events: list[dict]) -> str:
    """The durable "what I spawned THIS turn and what it returned" view, built from
    the spawn governance events (subagent_spawned / subagent_folded / spawn_denied).

    Closes the completed-spawn half of proprioception: the live [ACTIVE AGENTS]
    lane is running-only, so a sub-agent vanishes from it the moment it folds —
    yet the model should still see what it did this turn. This reads the DURABLE
    fault_traces events that previously reached the model through NO reader (they
    are written with fault_kind=None, invisible to the fault path).
    """
    spawned: dict[str, dict] = {}
    order: list[str] = []
    denied = 0
    for ev in events or []:
        kind = str((ev or {}).get("event_kind") or "")
        payload = (ev or {}).get("payload") or {}
        if kind == "subagent_spawned":
            cid = str(payload.get("child_turn_id") or f"_{len(order)}")
            if cid not in spawned:
                order.append(cid)
            label = str(payload.get("label") or "").strip() or f"L{payload.get('level', 2)}"
            spawned[cid] = {"label": label, "level": payload.get("level", 2), "folded": False}
        elif kind == "subagent_folded":
            cid = str(payload.get("child_turn_id") or "")
            if cid in spawned:
                spawned[cid]["folded"] = True
        elif kind in ("spawn_denied", "spawn_budget_exhausted"):
            denied += 1
    if not spawned and not denied:
        return ""
    lines = ["[AGENTS THIS TURN]"]
    for cid in order:
        rec = spawned[cid]
        lines.append(f"- {rec['label']} (L{rec['level']}, {'done' if rec['folded'] else 'running'})")
    if denied:
        lines.append(f"- {denied} spawn(s) denied")
    lines.append("[/AGENTS THIS TURN]")
    return "\n".join(lines)


def contribute_recap_section(messages: list[dict], config: dict):
    """Coalescer contributor: the durable recap of THIS turn's spawns + outcomes,
    read from turn_trace.list_governance_events for the current OUTER turn
    (config["_parent_turn_id"] on a followup, else config["_turn_id"]). This is
    the reader that closes the unread durable spawn-event loop."""
    from core.ephemeral_coalescer import SectionResult

    if not any(m.get("role") == "user" and not m.get("ephemeral") for m in messages):
        return None
    cfg = config or {}
    outer = str(cfg.get("_parent_turn_id") or cfg.get("_turn_id") or "").strip()
    if not outer:
        return None
    try:
        from core import turn_trace
        recs = turn_trace.list_governance_events(outer)
    except Exception:
        return None
    events: list[dict] = []
    for rec in recs or []:
        payload = {}
        raw = getattr(rec, "payload_json", None)
        if raw:
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {}
        events.append({"event_kind": getattr(rec, "event_kind", ""), "payload": payload})
    block = render_agent_recap(events)
    if not block:
        return None
    return SectionResult(name="agent_recap", text=block)
