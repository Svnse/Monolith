"""The active-agents spine: ONE source (the live spawn workers) projected two ways
— a terse model-context block and a compact UI one-liner. Pure formatters, tested
here; the source enrichment + interceptor wiring live in chat.py.

Origin: E wants the verbose subagent message replaced by a one-line "Agent"
indicator on the command, AND the model to see which agents are active while
spawn_subagent is working (2026-06-16)."""
from __future__ import annotations

from core.active_agents import (
    AgentRecord,
    render_active_agents_block,
)


def test_block_empty_when_no_running_agents() -> None:
    assert render_active_agents_block([]) == ""
    # done agents alone do not warrant a live block
    done = [AgentRecord(agent_id="a1", frame="probe", level=2, status="done")]
    assert render_active_agents_block(done) == ""


def test_block_lists_running_agents_for_the_model() -> None:
    agents = [
        AgentRecord(agent_id="a1", frame="read-only view", level=2, status="running"),
        AgentRecord(agent_id="a2", frame="api summary", level=3, status="running"),
    ]
    block = render_active_agents_block(agents)
    assert block.startswith("[ACTIVE AGENTS]")
    assert block.endswith("[/ACTIVE AGENTS]")
    assert "read-only view" in block
    assert "api summary" in block
    assert "L2" in block and "L3" in block
    # terse — the advisor's caveat: it's a view, not the result; keep it short
    assert len(block.splitlines()) == 4  # header + 2 agents + footer


def test_block_counts_only_running_among_mixed() -> None:
    agents = [
        AgentRecord(agent_id="a1", frame="live", level=2, status="running"),
        AgentRecord(agent_id="a2", frame="finished", level=2, status="done"),
    ]
    block = render_active_agents_block(agents)
    assert "live" in block
    assert "finished" not in block


# --- durable recap: "what I spawned this turn" (closes the unread spawn-event loop) ---

def test_recap_empty_when_no_events() -> None:
    from core.active_agents import render_agent_recap
    assert render_agent_recap([]) == ""
    assert render_agent_recap(None) == ""


def test_recap_shows_spawned_running_and_folded_done_in_order() -> None:
    from core.active_agents import render_agent_recap
    events = [
        {"event_kind": "subagent_spawned", "payload": {"child_turn_id": "c1", "label": "web-research", "level": 3}},
        {"event_kind": "subagent_folded", "payload": {"child_turn_id": "c1"}},
        {"event_kind": "subagent_spawned", "payload": {"child_turn_id": "c2", "label": "summary", "level": 2}},
    ]
    block = render_agent_recap(events)
    assert block.startswith("[AGENTS THIS TURN]")
    assert block.endswith("[/AGENTS THIS TURN]")
    # c1 folded -> done, c2 still spawned -> running; spawn order preserved
    assert block.index("web-research (L3, done)") < block.index("summary (L2, running)")


def test_recap_counts_denied_spawns() -> None:
    from core.active_agents import render_agent_recap
    events = [
        {"event_kind": "spawn_denied", "payload": {}},
        {"event_kind": "spawn_budget_exhausted", "payload": {}},
    ]
    assert "2 spawn(s) denied" in render_agent_recap(events)


def test_recap_contributor_silent_without_a_turn_id() -> None:
    from core import active_agents as aa
    assert aa.contribute_recap_section([{"role": "user", "text": "hi"}], {}) is None


def test_recap_contributor_reads_governance_events_for_the_outer_turn(monkeypatch) -> None:
    import json as _json

    from core import active_agents as aa
    import core.turn_trace as tt

    class _Rec:
        def __init__(self, kind, payload):
            self.event_kind = kind
            self.payload_json = _json.dumps(payload)

    captured = {}

    def fake_events(outer):
        captured["outer"] = outer
        return [
            _Rec("subagent_spawned", {"child_turn_id": "c1", "label": "probe", "level": 2}),
            _Rec("subagent_folded", {"child_turn_id": "c1"}),
        ]

    monkeypatch.setattr(tt, "list_governance_events", fake_events)
    # _parent_turn_id (the OUTER turn on a followup) wins over _turn_id
    res = aa.contribute_recap_section(
        [{"role": "user", "text": "hi"}], {"_turn_id": "gen2", "_parent_turn_id": "outer1"}
    )
    assert captured["outer"] == "outer1"
    assert res is not None and res.name == "agent_recap"
    assert "probe (L2, done)" in res.text


# --- projection channel + contributor ---------------------------------------

def test_contributor_none_when_nothing_active() -> None:
    from core import active_agents as aa
    aa.set_active([])
    assert aa.contribute_section([{"role": "user", "text": "hi"}], {}) is None


def test_contributor_injects_block_for_the_model_when_active() -> None:
    from core import active_agents as aa
    aa.set_active([AgentRecord(agent_id="a1", frame="probe", level=2, status="running")])
    try:
        res = aa.contribute_section([{"role": "user", "text": "hi"}], {})
        assert res is not None
        assert res.name == "active_agents"
        assert "[ACTIVE AGENTS]" in res.text
        assert "probe" in res.text
    finally:
        aa.set_active([])


def test_parse_child_turn_id_from_fence() -> None:
    from core.active_agents import parse_child_turn_id
    fence = "[SUBAGENT_RESULT level=2 frame=probe turn=ab12cd34 ok=true]\nbody\n[/SUBAGENT_RESULT]"
    assert parse_child_turn_id(fence) == "ab12cd34"


def test_parse_child_turn_id_absent_is_empty() -> None:
    from core.active_agents import parse_child_turn_id
    assert parse_child_turn_id("no turn marker here") == ""
    assert parse_child_turn_id("") == ""


def test_agent_record_carries_zoom_key() -> None:
    rec = AgentRecord(agent_id="a1", frame="probe", level=2, status="done",
                      child_turn_id="ab12cd34", tool_count=3)
    assert rec.child_turn_id == "ab12cd34"
    assert rec.tool_count == 3
    assert not rec.is_running


def test_contributor_silent_on_ephemeral_only_turn() -> None:
    from core import active_agents as aa
    aa.set_active([AgentRecord(agent_id="a1", frame="probe", level=2, status="running")])
    try:
        # a follow-up/ephemeral-only turn must not re-inject (mirrors runtime_state)
        msgs = [{"role": "user", "text": "x", "ephemeral": True}]
        assert aa.contribute_section(msgs, {}) is None
    finally:
        aa.set_active([])
