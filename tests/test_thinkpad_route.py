"""POST /thinkpad — the agent-server entry point for reasoning-branch training.

Tests the non-Qt core (`agent_server._process_thinkpad`, mirroring the
`_process_rating` testability pattern) plus the turn_trace stamp that makes
branch turns trainable:

  * turn_trace.set_turn_monothink_active() — follow-up UPDATE on a recorded
    frame (the record_source_tier pattern). Branches run the monothink scaffold
    in-frame but run_subagent records their frames with monothink_active=False,
    which would gate `will_evolve` off for branch ratings.
  * _process_thinkpad() — validates the body, fans out via an injected
    run_live, stamps each branch trace trainable, serializes branches+advisory.
"""
from __future__ import annotations

import os
import pytest

import core.turn_trace as tt
import engine.agent_server as ags
from core.turn_trace import FrameTraceRecord
from core.thinkpad import Branch, ThinkpadResult
from core.grounded_verdict import Candidate, Ranked


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    tt.set_db_path(tmp_path / "thinkpad_route_turn_trace.sqlite3")
    os.environ["MONOLITH_TURN_TRACE_V1"] = "1"
    yield
    tt.set_db_path(None)


def _frame(turn_id, *, parent=None, monothink=False, captured_at="2026-06-10T00:00:00Z"):
    return FrameTraceRecord(
        turn_id=turn_id,
        captured_at=captured_at,
        backend="subagent",
        engine_key="subagent:L3",
        gen_id=0,
        final_messages=(),
        system_prompt_chars=0,
        user_prompt_chars=0,
        total_chars=0,
        parent_turn_id=parent,
        monothink_active=monothink,
    )


# ── turn_trace.set_turn_monothink_active ─────────────────────────────────


def test_set_turn_monothink_active_stamps_recorded_frame():
    tt.record_frame(_frame("child-1", parent="outer-1"))
    assert tt.get_turn_monothink_active("child-1") is False
    assert tt.set_turn_monothink_active("child-1") is True
    assert tt.get_turn_monothink_active("child-1") is True


def test_set_turn_monothink_active_missing_turn_is_honest_noop():
    assert tt.set_turn_monothink_active("never-recorded") is False
    assert tt.get_turn_monothink_active("never-recorded") is False


# ── agent_server._process_thinkpad ────────────────────────────────────────


def _fake_run_live(branches):
    """run_thinkpad_live stand-in capturing the call and returning canned branches."""
    calls = {}

    def run_live(messages, base_config, *, n, parent_turn_id):
        calls.update(messages=messages, n=n, parent_turn_id=parent_turn_id)
        ranked = tuple(
            Ranked(candidate=Candidate(id=b.id, cites=b.cites),
                   authority=3 if b.cites else 0,
                   grounded=bool(b.cites),
                   winning_cite=(b.cites[0] if b.cites else None))
            for b in branches
        )
        return ThinkpadResult(tuple(branches), ranked), tuple(b.trace_id for b in branches)

    return run_live, calls


def test_process_thinkpad_returns_branches_and_advisory_and_stamps_trainable():
    # run_subagent records each branch frame monothink_active=False (prod behavior)
    tt.record_frame(_frame("child-A", parent="outer-1"))
    tt.record_frame(_frame("child-B", parent="outer-1"))
    branches = [
        Branch(id="c0", raw="r0", answer="A [cite: R1]", think="T0",
               cites=("R1",), trace_id="child-A"),
        Branch(id="c1", raw="r1", answer="B [no-ground]", think="T1",
               cites=(), trace_id="child-B"),
    ]
    run_live, calls = _fake_run_live(branches)
    payload, status = ags._process_thinkpad(
        {"message": "probe question", "n": 2}, run_live=run_live)
    assert status == 200
    assert payload["ok"] is True
    assert calls["n"] == 2
    assert calls["messages"] == [{"role": "user", "content": "probe question"}]
    out = payload["branches"]
    assert [b["id"] for b in out] == ["c0", "c1"]
    assert out[0]["answer"] == "A [cite: R1]"
    assert out[0]["think"] == "T0"
    assert out[0]["cites"] == ["R1"]
    assert out[0]["trace_id"] == "child-A"
    adv = payload["advisory"]
    assert adv[0] == {"id": "c0", "authority": 3, "grounded": True, "winning_cite": "R1"}
    assert adv[1] == {"id": "c1", "authority": 0, "grounded": False, "winning_cite": None}
    # the stamp: branch turns are now trainable (will_evolve gate passes)
    assert tt.get_turn_monothink_active("child-A") is True
    assert tt.get_turn_monothink_active("child-B") is True


def test_process_thinkpad_requires_message():
    run_live, _ = _fake_run_live([])
    payload, status = ags._process_thinkpad({"n": 2}, run_live=run_live)
    assert status == 400
    assert payload["ok"] is False


def test_process_thinkpad_clamps_n():
    tt.record_frame(_frame("child-A"))
    branches = [Branch(id="c0", raw="r", answer="a", think="t", cites=(), trace_id="child-A")]
    run_live, calls = _fake_run_live(branches)
    payload, status = ags._process_thinkpad(
        {"message": "q", "n": 99}, run_live=run_live)
    assert status == 200
    assert calls["n"] == 4  # hard cap: fan-out is N real inferences on E's GPU
    run_live2, calls2 = _fake_run_live(branches)
    ags._process_thinkpad({"message": "q", "n": 0}, run_live=run_live2)
    assert calls2["n"] == 1


def test_process_thinkpad_branch_failure_is_reported_not_raised():
    def boom(messages, base_config, *, n, parent_turn_id):
        raise RuntimeError("engine offline")

    payload, status = ags._process_thinkpad({"message": "q"}, run_live=boom)
    assert status == 500
    assert payload["ok"] is False
    assert "engine offline" in payload["error"]


def test_process_thinkpad_scaffold_override_for_shadow_ablation():
    """Phase 4 (shadow ablation): the caller may supply a scaffold variant
    (lesson present/absent). Branches run on it; nothing is written live.
    Without the override the live scaffold loads as before."""
    tt.record_frame(_frame("child-A"))
    branches = [Branch(id="c0", raw="r", answer="a", think="t", cites=(), trace_id="child-A")]
    seen = {}

    def run_live(messages, base_config, *, n, parent_turn_id, scaffold=None):
        seen["scaffold"] = scaffold
        ranked = ()
        return ThinkpadResult(tuple(branches), ranked), ("child-A",)

    payload, status = ags._process_thinkpad(
        {"message": "q", "n": 1, "scaffold": "## Ablated variant\nbody"},
        run_live=run_live)
    assert status == 200
    assert seen["scaffold"] == "## Ablated variant\nbody"
    payload2, _ = ags._process_thinkpad({"message": "q", "n": 1}, run_live=run_live)
    assert seen["scaffold"] is None   # absent -> live scaffold loads downstream
