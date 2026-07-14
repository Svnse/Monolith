"""Tests for the bearing staleness closure spine (addons/system/bearing/staleness.py).

Pure state-machine + nudge-rendering tests. No I/O — the actuator wiring
(store/audit/compiler) is tested in test_bearing_staleness_loop.py.

Cadence: bearing_interceptor runs PER GENERATION (tool-loop followups re-fire it
within one user turn — see engine/llm.py:1429), so `evaluate` takes the OUTER
turn id and advances the streak only when it changes. Same turn → re-render at
the same streak, no advance, no escalation (keeps "persistent across turns" true
and V2-safe).

Escalation is PER-SIGNAL (`escalates`, `threshold`): channel-mismatch is a
precision heuristic with false positives, so V1 ships it `escalates=False`
(nudge-only, never a fault). The escalating path is kept + tested for a future
signal that opts in behind an acknowledgment-clear escape (V2).
"""
from __future__ import annotations

from addons.system.bearing import staleness as st


def _sig(signal_id: str = "channel:user", kind: str = "channel_mismatch",
         text: str = "NUDGE-TEXT", detail: str = "d",
         escalates: bool = True, threshold: int = 3) -> "st.StalenessSignal":
    return st.StalenessSignal(signal_id=signal_id, kind=kind, text=text,
                              detail=detail, escalates=escalates, threshold=threshold)


# ── evaluate: new / persisting episode ───────────────────────────────


def test_new_signal_starts_streak_at_one() -> None:
    d = st.evaluate(_sig(), prior_state=None, turn_id="t1")
    assert d.next_state == {"signal_id": "channel:user", "streak": 1, "turn_id": "t1"}
    assert d.advanced is True
    assert d.text is not None
    assert d.cleared is False
    assert d.should_escalate is False


def test_new_turn_with_same_signal_increments_streak() -> None:
    prior = {"signal_id": "channel:user", "streak": 1, "turn_id": "t1"}
    d = st.evaluate(_sig(), prior_state=prior, turn_id="t2")
    assert d.next_state["streak"] == 2
    assert d.advanced is True
    assert d.should_escalate is False


# ── evaluate: per-generation cadence (the bug the advisor caught) ─────


def test_same_turn_re_render_does_not_advance_streak() -> None:
    # tool-loop followup within the SAME turn: re-render, do NOT inflate the streak
    prior = {"signal_id": "channel:user", "streak": 1, "turn_id": "tA"}
    d = st.evaluate(_sig(), prior_state=prior, turn_id="tA")
    assert d.next_state["streak"] == 1
    assert d.advanced is False
    assert d.text is not None  # still renders the nudge into the (rebuilt) prompt


def test_same_turn_does_not_re_escalate_mid_turn() -> None:
    # at threshold, a within-turn re-fire must NOT emit the fault again (V2 safety)
    prior = {"signal_id": "channel:user", "streak": 3, "turn_id": "tA"}
    d = st.evaluate(_sig(escalates=True, threshold=3), prior_state=prior, turn_id="tA")
    assert d.advanced is False
    assert d.should_escalate is False


# ── evaluate: escalation (per-signal, on a genuine advance only) ──────


def test_escalating_signal_escalates_exactly_at_threshold_on_advance() -> None:
    prior = {"signal_id": "channel:user", "streak": 2, "turn_id": "tA"}
    d = st.evaluate(_sig(escalates=True, threshold=3), prior_state=prior, turn_id="tB")
    assert d.next_state["streak"] == 3
    assert d.advanced is True
    assert d.should_escalate is True


def test_does_not_re_escalate_after_threshold() -> None:
    prior = {"signal_id": "channel:user", "streak": 3, "turn_id": "tA"}
    d = st.evaluate(_sig(escalates=True, threshold=3), prior_state=prior, turn_id="tB")
    assert d.next_state["streak"] == 4
    assert d.should_escalate is False


def test_non_escalating_signal_never_escalates() -> None:
    prior = {"signal_id": "channel:user", "streak": 5, "turn_id": "tA"}
    d = st.evaluate(_sig(escalates=False, threshold=3), prior_state=prior, turn_id="tB")
    assert d.next_state["streak"] == 6
    assert d.should_escalate is False
    assert d.text is not None


def test_different_signal_resets_streak() -> None:
    prior = {"signal_id": "channel:connect", "streak": 5, "turn_id": "tA"}
    d = st.evaluate(_sig(signal_id="channel:user"), prior_state=prior, turn_id="tB")
    assert d.next_state == {"signal_id": "channel:user", "streak": 1, "turn_id": "tB"}
    assert d.advanced is True
    assert d.should_escalate is False


def test_escalate_detail_includes_streak_and_signal_detail() -> None:
    prior = {"signal_id": "channel:user", "streak": 2, "turn_id": "tA"}
    d = st.evaluate(_sig(detail="frame X"), prior_state=prior, turn_id="tB")
    assert "frame X" in d.escalate_detail
    assert "3" in d.escalate_detail


# ── evaluate: resolution ─────────────────────────────────────────────


def test_no_signal_with_prior_state_clears() -> None:
    prior = {"signal_id": "channel:user", "streak": 2, "turn_id": "tA"}
    d = st.evaluate(None, prior_state=prior, turn_id="tB")
    assert d.cleared is True
    assert d.next_state is None
    assert d.text is None


def test_no_signal_no_prior_is_noop() -> None:
    d = st.evaluate(None, prior_state=None, turn_id="t1")
    assert d.cleared is False
    assert d.next_state is None
    assert d.text is None
    assert d.should_escalate is False


# ── render_nudge: escalating framing, original detail preserved ───────


def test_render_streak_one_is_verbatim_signal_text() -> None:
    sig = _sig(text="ORIGINAL NUDGE")
    assert st.render_nudge(sig, 1) == "ORIGINAL NUDGE"


def test_render_escalating_streak_two_says_still_and_keeps_original() -> None:
    sig = _sig(text="ORIGINAL NUDGE", escalates=True, threshold=3)
    out = st.render_nudge(sig, 2)
    assert "still" in out.lower()
    assert "2" in out
    assert "ORIGINAL NUDGE" in out


def test_render_escalating_at_threshold_flags_persistence() -> None:
    sig = _sig(text="ORIGINAL NUDGE", escalates=True, threshold=3)
    out = st.render_nudge(sig, 3)
    assert ("flagged" in out.lower()) or ("persistent" in out.lower())
    assert "3" in out
    assert "ORIGINAL NUDGE" in out


def test_render_non_escalating_stays_advisory_and_names_the_out() -> None:
    sig = _sig(text="ORIGINAL NUDGE", escalates=False, threshold=3)
    out = st.render_nudge(sig, 3)
    assert "no update" in out.lower()
    assert "flagged" not in out.lower()
    assert "ORIGINAL NUDGE" in out
