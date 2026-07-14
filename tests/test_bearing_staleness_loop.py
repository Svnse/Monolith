"""Integration tests for the channel-staleness closure actuator wired into
compiler.bearing_interceptor.

V1 contract: flag-OFF is byte-identical to the legacy single-nudge path; flag-ON
runs the actuator (de-spam via escalating-but-advisory framing + streak persisted
+ cleared on resolution). Channel signal is escalates=False → NEVER a fault.
"""
from __future__ import annotations

import pytest

from addons.system.bearing import audit, compiler, store
from addons.system.bearing import schema as bs

_STALE_FRAME = "collaborating with Claude on the carve"   # "claude" trips USER-channel mismatch


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    monkeypatch.delenv("MONOLITH_BEARING_V1", raising=False)  # defaults ON
    yield tmp_path


def _user_msgs():
    return [{"role": "user", "content": "[CHANNEL: USER]\nhello"}]


# ── flag OFF: regression — legacy behavior, no state writes ───────────


def test_flag_off_keeps_legacy_single_nudge_and_writes_no_state(env, monkeypatch) -> None:
    monkeypatch.delenv("MONOLITH_BEARING_STALENESS_V2", raising=False)  # off (default)
    store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
    out = compiler.bearing_interceptor(_user_msgs(), {})
    assert out is not None
    assert "possible staleness" in out[0]["content"].lower()  # today's nudge, verbatim
    assert store.get_pending_staleness() is None              # actuator did not run


# ── flag ON: the closure loop ────────────────────────────────────────


def test_flag_on_turn1_nudges_and_persists_streak(env, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
    store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
    out = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
    assert out is not None
    assert "staleness" in out[0]["content"].lower()
    assert store.get_pending_staleness() == {"signal_id": "channel:user", "streak": 1, "turn_id": "t1"}
    assert any(r.get("kind") == "staleness_nudged" and r.get("streak") == 1
               for r in audit.read_recent())


def test_flag_on_turn2_escalates_framing_and_streak(env, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
    store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
    compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
    out2 = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t2"})
    content = out2[0]["content"].lower()
    assert "turn 2" in content                         # escalating framing (varies the text)
    assert "no update is needed" in content            # but stays advisory (the false-positive out)
    assert store.get_pending_staleness()["streak"] == 2


def test_flag_on_clears_when_mismatch_resolves(env, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
    store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
    compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
    assert store.get_pending_staleness() is not None
    # model resolves: frame no longer carries a peer marker on the USER channel
    store.set_bearing(bs.Bearing(current_frame="carving the system prompt"))
    compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t2"})
    assert store.get_pending_staleness() is None
    assert any(r.get("kind") == "staleness_cleared" for r in audit.read_recent())


def test_flag_on_no_mismatch_is_inert(env, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
    store.set_bearing(bs.Bearing(current_frame="carving the system prompt"))
    out = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
    assert store.get_pending_staleness() is None
    assert "staleness" not in out[0]["content"].lower()


def test_flag_on_never_emits_fault_even_when_persistent(env, monkeypatch) -> None:
    # channel signal is escalates=False — persistence must NOT coerce via a fault (V1 invariant)
    monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
    import core.fault_response as fr
    calls = {"n": 0}
    monkeypatch.setattr(fr, "emit_fault", lambda **k: calls.__setitem__("n", calls["n"] + 1) or 1)
    store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
    for i in range(6):
        compiler.bearing_interceptor(_user_msgs(), {"_turn_id": f"t{i}"})
    assert store.get_pending_staleness()["streak"] == 6
    assert calls["n"] == 0


def test_flag_on_double_fire_defended_no_double_increment(env, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
    store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
    once = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
    twice = compiler.bearing_interceptor(once, {"_turn_id": "t1"})
    assert twice is None
    assert store.get_pending_staleness()["streak"] == 1  # not double-incremented


def test_flag_on_does_not_inflate_streak_within_a_turn(env, monkeypatch) -> None:
    # The interceptor re-fires per generation; tool-loop followups (same OUTER turn,
    # carried via _parent_turn_id) must NOT inflate the streak or re-audit.
    monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
    store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
    compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "tA"})                            # outer
    compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "tF1", "_parent_turn_id": "tA"})  # followup
    compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "tF2", "_parent_turn_id": "tA"})  # followup
    assert store.get_pending_staleness()["streak"] == 1
    nudged = [r for r in audit.read_recent() if r.get("kind") == "staleness_nudged"]
    assert len(nudged) == 1


def test_flag_on_advances_on_next_outer_turn_after_followups(env, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
    store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
    compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "tA"})                            # turn A → streak 1
    compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "tF1", "_parent_turn_id": "tA"})  # followup → 1
    compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "tB"})                            # turn B → streak 2
    assert store.get_pending_staleness()["streak"] == 2


def test_flag_on_coexists_with_pending_rejection(env, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
    store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
    store.set_pending_rejection(["D1"], turn_id="tprev", ts="2026-06-06T00:00:00+00:00")
    out = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
    content = out[0]["content"]
    assert "[BEARING_UPDATE_REJECTED]" in content
    assert "staleness" in content.lower()
    assert store.get_pending_staleness() is not None
    assert store.get_pending_rejection() is not None  # neither clobbers the other
