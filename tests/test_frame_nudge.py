"""Tests for the empty-bearing <frame> nudge (MONOLITH_FRAME_NUDGE_V1).

Covers:
  - kill_switch.frame_nudge_is_enabled() flag behaviour
  - compiler._empty_bearing_signal() shape
  - _apply_staleness / bearing_interceptor: both flags OFF → legacy (byte-identical);
    frame_nudge ON + empty bearing → nudge; frame_nudge ON + non-empty bearing → silent;
    streak escalation uses empty_bearing wording at streak ≥ 2
  - staleness_v2 ON + frame_nudge OFF → channel path unchanged
  - staleness.render_nudge: empty_bearing branch wording at streak 1 vs ≥ 2
"""
from __future__ import annotations

import pytest

from addons.system.bearing import audit, compiler, store
from addons.system.bearing import kill_switch
from addons.system.bearing import schema as bs
from addons.system.bearing import staleness as st
from addons.system.bearing.compiler import _empty_bearing_signal


# ── shared fixtures ───────────────────────────────────────────────────

@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated store + audit, flags cleared, MONOLITH_BEARING_V1 defaults ON."""
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    monkeypatch.delenv("MONOLITH_BEARING_V1", raising=False)          # defaults ON
    monkeypatch.delenv("MONOLITH_BEARING_STALENESS_V2", raising=False) # defaults OFF
    monkeypatch.delenv("MONOLITH_FRAME_NUDGE_V1", raising=False)       # defaults OFF
    yield tmp_path


def _user_msgs():
    return [{"role": "user", "content": "[CHANNEL: USER]\nhello"}]


# ─────────────────────────────────────────────────────────────────────
# 1. kill_switch.frame_nudge_is_enabled()
# ─────────────────────────────────────────────────────────────────────

class TestFrameNudgeFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MONOLITH_FRAME_NUDGE_V1", raising=False)
        assert kill_switch.frame_nudge_is_enabled() is False

    def test_on_when_set_to_1(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "1")
        assert kill_switch.frame_nudge_is_enabled() is True

    def test_on_when_set_to_true(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "true")
        assert kill_switch.frame_nudge_is_enabled() is True

    def test_on_when_set_to_yes(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "yes")
        assert kill_switch.frame_nudge_is_enabled() is True

    def test_on_when_set_to_on(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "on")
        assert kill_switch.frame_nudge_is_enabled() is True

    def test_off_when_set_to_0(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "0")
        assert kill_switch.frame_nudge_is_enabled() is False

    def test_off_when_set_to_false(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "false")
        assert kill_switch.frame_nudge_is_enabled() is False

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "TRUE")
        assert kill_switch.frame_nudge_is_enabled() is True


# ─────────────────────────────────────────────────────────────────────
# 2. _empty_bearing_signal() shape
# ─────────────────────────────────────────────────────────────────────

class TestEmptyBearingSignalShape:
    def test_signal_id(self):
        sig = _empty_bearing_signal()
        assert sig.signal_id == "bearing:empty"

    def test_kind(self):
        sig = _empty_bearing_signal()
        assert sig.kind == "empty_bearing"

    def test_escalates_false(self):
        sig = _empty_bearing_signal()
        assert sig.escalates is False

    def test_text_mentions_frame_tag(self):
        sig = _empty_bearing_signal()
        assert "<frame>" in sig.text

    def test_text_mentions_heartbeat(self):
        sig = _empty_bearing_signal()
        assert "heartbeat" in sig.text.lower()

    def test_detail_mentions_empty(self):
        sig = _empty_bearing_signal()
        assert "empty" in sig.detail.lower()

    def test_threshold_is_3(self):
        sig = _empty_bearing_signal()
        assert sig.threshold == 3


# ─────────────────────────────────────────────────────────────────────
# 3. render_nudge: empty_bearing branch
# ─────────────────────────────────────────────────────────────────────

class TestRenderNudgeEmptyBearing:
    def test_streak_1_returns_verbatim_signal_text(self):
        sig = _empty_bearing_signal()
        out = st.render_nudge(sig, 1)
        assert out == sig.text

    def test_streak_2_uses_empty_bearing_wording(self):
        sig = _empty_bearing_signal()
        out = st.render_nudge(sig, 2)
        assert "still empty" in out.lower()
        assert "2" in out
        # Must include original text
        assert "<frame>" in out

    def test_streak_3_uses_empty_bearing_wording(self):
        sig = _empty_bearing_signal()
        out = st.render_nudge(sig, 3)
        assert "still empty" in out.lower()
        assert "3" in out

    def test_streak_2_does_not_use_channel_wording(self):
        """The empty_bearing branch must NOT bleed into channel-mismatch copy."""
        sig = _empty_bearing_signal()
        out = st.render_nudge(sig, 2)
        # Channel-specific phrasing must be absent
        assert "referencing a peer" not in out
        assert "no update is needed" not in out

    def test_channel_mismatch_streak_2_unaffected(self):
        """Channel-mismatch signals still hit the old non-escalating branch."""
        sig = st.StalenessSignal(
            signal_id="channel:user",
            kind="channel_mismatch",
            text="CHANNEL-NUDGE",
            escalates=False,
            threshold=3,
        )
        out = st.render_nudge(sig, 2)
        assert "no update is needed" in out.lower()
        assert "still empty" not in out.lower()


# ─────────────────────────────────────────────────────────────────────
# 4. _apply_staleness / bearing_interceptor: flag-matrix
# ─────────────────────────────────────────────────────────────────────

class TestApplyStalenessMatrix:

    # ── 4a. Both flags OFF → legacy path, byte-identical ─────────────

    def test_both_off_empty_bearing_no_nudge(self, env, monkeypatch):
        """Legacy path: empty bearing, both flags off → no frame nudge."""
        # flags already off via fixture
        store.set_bearing(bs.Bearing())  # empty
        out = compiler.bearing_interceptor(_user_msgs(), {})
        assert out is not None
        content = out[0]["content"]
        assert "<frame>" not in content
        assert "bearing:empty" not in content
        assert store.get_pending_staleness() is None  # actuator never ran

    def test_both_off_channel_mismatch_legacy_nudge(self, env, monkeypatch):
        """Legacy path: non-empty frame with peer marker → single legacy nudge (no streak)."""
        _STALE_FRAME = "collaborating with Claude on the carve"
        store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
        out = compiler.bearing_interceptor(_user_msgs(), {})
        assert out is not None
        assert "possible staleness" in out[0]["content"].lower()
        assert store.get_pending_staleness() is None  # no state written

    # ── 4b. frame_nudge ON + empty bearing → nudge injected ──────────

    def test_frame_nudge_on_empty_bearing_nudge_appended(self, env, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "1")
        store.set_bearing(bs.Bearing())  # empty
        out = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
        assert out is not None
        content = out[0]["content"]
        assert "<frame>" in content
        assert "BEARING — empty" in content
        assert store.get_pending_staleness() == {
            "signal_id": "bearing:empty", "streak": 1, "turn_id": "t1"
        }

    def test_frame_nudge_on_empty_bearing_writes_audit(self, env, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "1")
        store.set_bearing(bs.Bearing())
        compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
        rows = audit.read_recent()
        nudged = [r for r in rows if r.get("kind") == "staleness_nudged"]
        assert len(nudged) == 1
        assert nudged[0]["signal_id"] == "bearing:empty"
        assert nudged[0]["streak"] == 1

    # ── 4c. frame_nudge ON + NON-empty bearing → no empty nudge ──────

    def test_frame_nudge_on_non_empty_bearing_no_nudge(self, env, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "1")
        store.set_bearing(bs.Bearing(current_frame="working on the system prompt carve"))
        out = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
        assert out is not None
        content = out[0]["content"]
        assert "<frame>" not in content
        assert "bearing:empty" not in content
        assert store.get_pending_staleness() is None

    # ── 4d. streak escalation at streak ≥ 2 ──────────────────────────

    def test_frame_nudge_streak_escalates_to_empty_bearing_wording(self, env, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "1")
        store.set_bearing(bs.Bearing())  # stays empty across both turns
        compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
        out2 = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t2"})
        assert out2 is not None
        content = out2[0]["content"].lower()
        assert "still empty" in content
        assert "turn 2" in content
        assert store.get_pending_staleness()["streak"] == 2

    def test_frame_nudge_streak_escalation_no_channel_wording(self, env, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "1")
        store.set_bearing(bs.Bearing())
        compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
        out2 = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t2"})
        content = out2[0]["content"]
        assert "no update is needed" not in content.lower()
        assert "referencing a peer" not in content.lower()

    # ── 4e. frame_nudge ON, bearing goes non-empty → self-clears ─────

    def test_frame_nudge_clears_when_bearing_populated(self, env, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "1")
        store.set_bearing(bs.Bearing())
        compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
        assert store.get_pending_staleness() is not None
        # Model populates bearing
        store.set_bearing(bs.Bearing(current_frame="now working on spec writing"))
        compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t2"})
        assert store.get_pending_staleness() is None
        rows = audit.read_recent()
        assert any(r.get("kind") == "staleness_cleared" for r in rows)

    # ── 4f. staleness_v2 ON + frame_nudge OFF → channel path unchanged ─

    def test_staleness_v2_only_empty_bearing_does_not_fire(self, env, monkeypatch):
        """Empty bearing must NOT trigger the nudge when only staleness_v2 is on."""
        monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
        # frame_nudge stays OFF (default)
        store.set_bearing(bs.Bearing())  # empty
        out = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
        assert out is not None
        content = out[0]["content"]
        assert "<frame>" not in content
        assert "bearing:empty" not in content
        assert store.get_pending_staleness() is None

    def test_staleness_v2_only_channel_mismatch_still_nudges(self, env, monkeypatch):
        """staleness_v2 ON + frame_nudge OFF: channel mismatch still fires (unchanged)."""
        monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
        _STALE_FRAME = "collaborating with Claude on the carve"
        store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
        out = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
        assert out is not None
        content = out[0]["content"].lower()
        assert "staleness" in content
        assert store.get_pending_staleness() == {
            "signal_id": "channel:user", "streak": 1, "turn_id": "t1"
        }

    def test_staleness_v2_only_streak_escalates_channel_wording(self, env, monkeypatch):
        """staleness_v2 ON + frame_nudge OFF: streak 2 uses channel wording, not empty_bearing."""
        monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
        _STALE_FRAME = "collaborating with Claude on the carve"
        store.set_bearing(bs.Bearing(current_frame=_STALE_FRAME))
        compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
        out2 = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t2"})
        content = out2[0]["content"].lower()
        assert "no update is needed" in content  # channel-mismatch advisory wording
        assert "still empty" not in content       # empty_bearing wording must be absent

    # ── 4g. both ON: empty_bearing takes precedence ───────────────────

    def test_both_on_empty_bearing_wins_over_channel(self, env, monkeypatch):
        """With both flags on, empty bearing → empty signal (not channel signal)."""
        monkeypatch.setenv("MONOLITH_FRAME_NUDGE_V1", "1")
        monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", "1")
        store.set_bearing(bs.Bearing())  # empty
        out = compiler.bearing_interceptor(_user_msgs(), {"_turn_id": "t1"})
        assert out is not None
        content = out[0]["content"]
        assert "<frame>" in content
        state = store.get_pending_staleness()
        assert state is not None
        assert state["signal_id"] == "bearing:empty"
