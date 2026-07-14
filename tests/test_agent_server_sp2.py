"""SP2 of the monothink training loop — the agent-server surface.

Tests the non-Qt logic that makes the loop reachable by an external rater:
  * turn_trace.latest_outer_turn() — which turn /chat just produced (+ trainable?)
  * monothink.set_monothink_toggle() — enable monothink from outside the UI
  * agent_server._process_rating() — the /rating endpoint core. This is the same
    record_outcome metadata->hook seam the UI /rating uses, now unit-testable
    (the integration gap flagged at the end of SP1).

Spec: docs/superpowers/specs/2026-06-03-monothink-training-loop-sp1-rating-contract-design.md
"""
from __future__ import annotations

import os
import pytest

import core.turn_trace as tt
import core.monothink as mt
import engine.agent_server as ags
from core.turn_trace import FrameTraceRecord


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    tt.set_db_path(tmp_path / "sp2_turn_trace.sqlite3")
    os.environ["MONOLITH_TURN_TRACE_V1"] = "1"
    yield
    tt.set_db_path(None)


def _frame(turn_id, *, parent=None, monothink=False, captured_at="2026-06-03T00:00:00Z"):
    return FrameTraceRecord(
        turn_id=turn_id,
        captured_at=captured_at,
        backend="test",
        engine_key="llm",
        gen_id=0,
        final_messages=(),
        system_prompt_chars=0,
        user_prompt_chars=0,
        total_chars=0,
        parent_turn_id=parent,
        monothink_active=monothink,
    )


# ── turn_trace.latest_outer_turn ──────────────────────────────────────────


def test_latest_outer_turn_returns_most_recent_outer():
    tt.record_frame(_frame("T1", monothink=True, captured_at="2026-06-03T00:00:01Z"))
    tt.record_frame(_frame("T2-child", parent="T1", captured_at="2026-06-03T00:00:02Z"))
    tt.record_frame(_frame("T3", monothink=False, captured_at="2026-06-03T00:00:03Z"))
    latest = tt.latest_outer_turn()
    assert latest is not None
    assert latest["turn_id"] == "T3"  # most recent OUTER (T2-child excluded)
    assert latest["monothink_active"] is False


def test_latest_outer_turn_reports_monothink_active():
    tt.record_frame(_frame("Tm", monothink=True, captured_at="2026-06-03T00:00:05Z"))
    latest = tt.latest_outer_turn()
    assert latest["turn_id"] == "Tm"
    assert latest["monothink_active"] is True


def test_latest_outer_turn_none_when_empty():
    assert tt.latest_outer_turn() is None


# ── monothink.set_monothink_toggle ────────────────────────────────────────


class _FakeWS:
    def __init__(self):
        self.state = {}

    def set_monothink(self, v):
        self.state["monothink_enabled"] = bool(v)

    def set_monothink_once(self, v):
        self.state["monothink_once"] = bool(v)


def test_set_monothink_toggle_on_off_once_status(monkeypatch):
    ws = _FakeWS()
    monkeypatch.setattr(mt, "_monothink_world_state", ws)
    assert mt.set_monothink_toggle("on")["monothink"] == "on"
    assert ws.state["monothink_enabled"] is True
    assert mt.set_monothink_toggle("off")["monothink"] == "off"
    assert ws.state["monothink_enabled"] is False
    r = mt.set_monothink_toggle("once")
    assert r["ok"] is True and ws.state["monothink_once"] is True
    assert mt.set_monothink_toggle("status")["ok"] is True


def test_set_monothink_toggle_unknown_mode(monkeypatch):
    monkeypatch.setattr(mt, "_monothink_world_state", _FakeWS())
    assert mt.set_monothink_toggle("bogus")["ok"] is False


def test_set_monothink_toggle_no_world_state(monkeypatch):
    monkeypatch.setattr(mt, "_monothink_world_state", None)
    assert mt.set_monothink_toggle("on")["ok"] is False


# ── agent_server._process_rating (the /rating endpoint core) ───────────────


def _seed_monothink_turn(turn_id):
    tt.record_frame(_frame(turn_id, monothink=True, captured_at="2026-06-03T00:00:09Z"))


def test_process_rating_valid_records_and_fires_hook(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        "core.monothink.maybe_evolve_after_rating",
        lambda tid, rv, tags, think_block=None, replay_input=None, rater_note=None: calls.update(
            tid=tid, rv=rv, tags=tags, think=think_block, replay=replay_input, note=rater_note,
        ),
    )
    _seed_monothink_turn("T-rate")
    payload, status = ags._process_rating({
        "turn_id": "T-rate",
        "rating": 35,
        "failure_tags": ["missing_branch_pressure", "bogus_tag"],
        "surface_note": "felt long overall",
        "think_block": "trace here",
        "replay_input": "original ask",
    })
    assert status == 200
    assert payload["ok"] is True
    assert payload["failure_tags"] == ["missing_branch_pressure"]
    assert payload["dropped_unknown_tags"] == ["bogus_tag"]
    assert payload["will_evolve"] is True
    # the metadata->hook seam: normalized tags + think_block reached the decider,
    # surface_note did NOT.
    assert calls["tags"] == ["missing_branch_pressure"]
    assert calls["think"] == "trace here"
    assert calls["replay"] == "original ask"
    assert calls["note"] == "felt long overall"
    assert payload["has_replay_input"] is True
    # outcome row persisted
    trace = tt.get_turn_trace("T-rate")
    assert trace is not None


def test_process_rating_requires_turn_id():
    _payload, status = ags._process_rating({"rating": 50, "failure_tags": ["premise_unchecked"]})
    assert status == 400


def test_process_rating_rejects_out_of_range_rating():
    _payload, status = ags._process_rating({"turn_id": "X", "rating": 150})
    assert status == 400


def test_process_rating_no_tags_records_but_does_not_evolve(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(
        "core.monothink.maybe_evolve_after_rating",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )
    _seed_monothink_turn("T-notags")
    payload, status = ags._process_rating({"turn_id": "T-notags", "rating": 80, "failure_tags": []})
    assert status == 200
    assert payload["will_evolve"] is False
    assert called["n"] == 0  # triviality gate held; outcome still recorded
    assert tt.get_turn_trace("T-notags") is not None


# ── live HTTP round-trip (verifies the full route wiring, no Qt needed) ─────


def test_http_monothink_and_rating_roundtrip(monkeypatch):
    import json
    import urllib.request

    ws = _FakeWS()
    monkeypatch.setattr(mt, "_monothink_world_state", ws)
    # A non-monothink outer turn so /rating records but does NOT trigger a real
    # LLM evolution (will_evolve False) — keeps the live test hermetic.
    tt.record_frame(_frame("RT-1", monothink=False, captured_at="2026-06-03T00:00:20Z"))

    server = ags.AgentServer()
    server.start(0)
    try:
        host, port = server._httpd.server_address

        def post(path, body):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))

        st, j = post("/monothink", {"mode": "on"})
        assert st == 200 and j["ok"] is True and j["monothink"] == "on"
        assert ws.state["monothink_enabled"] is True

        st, j = post("/rating", {
            "turn_id": "RT-1", "rating": 40,
            "failure_tags": ["premise_unchecked"], "surface_note": "note",
        })
        assert st == 200 and j["ok"] is True
        assert j["failure_tags"] == ["premise_unchecked"]
        assert j["will_evolve"] is False  # RT-1 was not monothink-active
        assert tt.get_turn_trace("RT-1") is not None
    finally:
        server.stop()


def test_http_chat_returns_the_turn_it_produced_not_a_stale_one():
    """Turn-attribution: /chat must return the turn THIS call produced, not a
    pre-existing one. The engine records the frame at turn-build (before
    streaming), so the new outer turn appears between the pre-dispatch snapshot
    and push_done. Verified against a deliberately-stale seeded prior turn."""
    import json
    import urllib.request

    # A stale prior outer turn that must NOT be returned.
    tt.record_frame(_frame("STALE", monothink=False, captured_at="2026-06-03T00:00:00Z"))

    server = ags.AgentServer()

    def fake_engine(_agent, _msg):
        # Mirror the engine's real order: record THIS turn's frame (as
        # engine/llm.py does before dispatch), then stream + signal done.
        tt.record_frame(_frame("FRESH", monothink=True, captured_at="2026-06-03T00:00:40Z"))
        server.push_token("hi there")
        server.push_done()

    server.on_message = fake_engine
    server.start(0)
    try:
        host, port = server._httpd.server_address
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=json.dumps({"message": "hello", "agent": "A"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            st, j = resp.status, json.loads(resp.read().decode("utf-8"))
        assert st == 200 and j["ok"] is True
        assert j["turn_id"] == "FRESH"  # not "STALE"
        assert j["monothink_active"] is True
    finally:
        server.stop()
