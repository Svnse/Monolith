from __future__ import annotations

import pytest

from addons.system.bearing import compiler
from addons.system.bearing import schema as bs
from addons.system.bearing import store


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    yield tmp_path


# ── format_bearing_block ─────────────────────────────────────────────


def test_empty_bearing_renders_marker_block() -> None:
    block = compiler.format_bearing_block(bs.Bearing())
    assert "[BEARING]" in block
    assert "[/BEARING]" in block
    assert "no situational state established" in block
    assert "active state, NOT ambient context" in block


def test_active_state_framing_in_full_block() -> None:
    b = bs.Bearing(active_goal="x")
    block = compiler.format_bearing_block(b)
    assert "active state" in block.lower()
    assert "NOT ambient context" in block


def test_full_bearing_renders_all_sections() -> None:
    b = bs.Bearing(
        current_frame="frame text",
        active_goal="goal text",
        trajectory="traj text",
        open_tensions=(bs.Tension(text="t1-text", opened_at_turn="turn-a"),),
        referents=(bs.Referent(name="file.py", kind="file", status="observed", grounded_at_turn="turn-b"),),
        modal_branches=(
            bs.ModalBranch(text="branch-text", status="rejected", reason="r-reason", last_touched_turn="turn-c"),
        ),
        stakes=bs.Stakes(reversibility="hard", urgency="medium", cost_if_wrong="time lost"),
        user_model=bs.UserModel(intent_read="ir-text", register="literal", confidence=0.91),
        next_move="next-text",
        updated_at_turn="turn-d",
    )
    block = compiler.format_bearing_block(b)
    assert "frame text" in block
    assert "goal text" in block
    assert "traj text" in block
    assert "t1-text" in block
    assert "file.py" in block
    assert "branch-text" in block
    assert "ir-text" in block
    assert "time lost" in block
    assert "next-text" in block
    assert "turn-a" in block
    assert "turn-b" in block
    assert "turn-c" in block
    assert "turn-d" in block


def test_empty_slots_are_omitted_from_full_block() -> None:
    b = bs.Bearing(active_goal="only goal", updated_at_turn="t1")
    block = compiler.format_bearing_block(b)
    # active_goal present
    assert "active_goal: only goal" in block
    # current_frame is empty → its line should be absent
    assert "current_frame:" not in block
    # trajectory empty → absent
    assert "trajectory:" not in block


def test_pending_rejection_appended_when_provided() -> None:
    pending = {
        "failed_rules": ["D1", "D2"],
        "turn_id": "t-prev",
        "ts": "2026-05-20T00:00:00+00:00",
    }
    block = compiler.format_bearing_block(bs.Bearing(), pending_rejection=pending)
    assert "[BEARING_UPDATE_REJECTED]" in block
    assert "D1" in block
    assert "D2" in block
    assert "t-prev" in block
    assert "One repair attempt allowed this turn" in block


def test_pending_rejection_omitted_when_none() -> None:
    block = compiler.format_bearing_block(bs.Bearing(), pending_rejection=None)
    assert "[BEARING_UPDATE_REJECTED]" not in block


# ── readable turn-count age render ───────────────────────────────────


def test_renders_age_in_turns_when_current_turn_n_given() -> None:
    b = bs.Bearing(current_frame="f", updated_at_turn="uuid-x", updated_at_turn_n=312)
    block = compiler.format_bearing_block(b, current_turn_n=354)
    assert "updated_at_turn: 312 (42 turns ago)" in block
    assert "uuid-x" not in block  # UUID replaced by the readable form


def test_renders_this_turn_for_zero_age() -> None:
    b = bs.Bearing(current_frame="f", updated_at_turn="u", updated_at_turn_n=10)
    block = compiler.format_bearing_block(b, current_turn_n=10)
    assert "updated_at_turn: 10 (this turn)" in block


def test_renders_singular_for_age_one() -> None:
    b = bs.Bearing(current_frame="f", updated_at_turn="u", updated_at_turn_n=10)
    block = compiler.format_bearing_block(b, current_turn_n=11)
    assert "updated_at_turn: 10 (1 turn ago)" in block


def test_falls_back_to_uuid_when_no_current_turn_n() -> None:
    """Flag-off path: interceptor passes current_turn_n=None → byte-identical
    UUID line, exactly as before the feature existed."""
    b = bs.Bearing(current_frame="f", updated_at_turn="uuid-x", updated_at_turn_n=312)
    block = compiler.format_bearing_block(b)
    assert "updated_at_turn: uuid-x" in block
    assert "turns ago" not in block


def test_falls_back_to_uuid_when_turn_n_zero() -> None:
    """Pre-feature bearing (updated_at_turn_n=0) → UUID line even if current given."""
    b = bs.Bearing(current_frame="f", updated_at_turn="uuid-x", updated_at_turn_n=0)
    block = compiler.format_bearing_block(b, current_turn_n=354)
    assert "updated_at_turn: uuid-x" in block
    assert "turns ago" not in block


def test_no_negative_age_when_current_behind_stamp() -> None:
    """Defensive: stamp ahead of current (shouldn't happen) → UUID line, never
    a negative age."""
    b = bs.Bearing(current_frame="f", updated_at_turn="uuid-x", updated_at_turn_n=400)
    block = compiler.format_bearing_block(b, current_turn_n=354)
    assert "updated_at_turn: uuid-x" in block
    assert "-46" not in block


# ── deterministic key order (KV cache safety) ────────────────────────


def test_block_is_deterministic_across_calls() -> None:
    b = bs.Bearing(
        current_frame="a",
        active_goal="b",
        trajectory="c",
        open_tensions=(bs.Tension(text="t1", opened_at_turn="u1"),),
        referents=(bs.Referent(name="r1", kind="file", status="observed", grounded_at_turn="u2"),),
        modal_branches=(bs.ModalBranch(text="m1", status="active", reason="r1", last_touched_turn="u3"),),
        next_move="nm",
    )
    block1 = compiler.format_bearing_block(b)
    block2 = compiler.format_bearing_block(b)
    assert block1 == block2


# ── bearing_interceptor ──────────────────────────────────────────────


def test_interceptor_returns_none_when_no_user_message(tmp_store) -> None:
    result = compiler.bearing_interceptor([], {})
    assert result is None


def test_interceptor_injects_before_last_user_message(tmp_store) -> None:
    store.set_bearing(bs.Bearing(active_goal="goal"))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    result = compiler.bearing_interceptor(messages, {})
    assert result is not None
    # Bearing message inserted right before the LAST user message (index 3)
    assert result[3]["source"] == "bearing"
    assert result[3]["ephemeral"] is True
    assert result[4]["content"] == "second"  # original last-user shifted
    assert "active_goal: goal" in result[3]["content"]


def test_interceptor_injects_empty_marker_when_bearing_empty(tmp_store) -> None:
    messages = [{"role": "user", "content": "u"}]
    result = compiler.bearing_interceptor(messages, {})
    assert result is not None
    assert "no situational state established" in result[0]["content"]


def test_interceptor_kill_switch_disables(monkeypatch, tmp_store) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_V1", "0")
    store.set_bearing(bs.Bearing(active_goal="x"))
    messages = [{"role": "user", "content": "u"}]
    result = compiler.bearing_interceptor(messages, {})
    assert result is None


def test_interceptor_kill_switch_off_disables_via_env(monkeypatch, tmp_store) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_V1", "off")
    messages = [{"role": "user", "content": "u"}]
    assert compiler.bearing_interceptor(messages, {}) is None


def test_interceptor_double_fire_defended(tmp_store) -> None:
    messages = [
        {"role": "user", "content": "u"},
    ]
    once = compiler.bearing_interceptor(messages, {})
    assert once is not None
    # Second pass on the already-injected list returns None
    twice = compiler.bearing_interceptor(once, {})
    assert twice is None


def test_interceptor_includes_pending_rejection_block(tmp_store) -> None:
    store.set_pending_rejection(["D1"], turn_id="t-prev", ts="2026-05-20T00:00:00+00:00")
    messages = [{"role": "user", "content": "u"}]
    result = compiler.bearing_interceptor(messages, {})
    assert result is not None
    content = result[0]["content"]
    assert "[BEARING_UPDATE_REJECTED]" in content
    assert "D1" in content


def test_interceptor_renders_age_when_flag_on(tmp_store, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_TURN_COUNTER_V1", "1")
    store.set_bearing(bs.Bearing(
        current_frame="f", updated_at_turn="uuid", updated_at_turn_n=300))
    result = compiler.bearing_interceptor([{"role": "user", "content": "hi"}], {"_turn_n": 342})
    assert result is not None
    assert "updated_at_turn: 300 (42 turns ago)" in result[0]["content"]


def test_interceptor_uuid_when_flag_off(tmp_store, monkeypatch) -> None:
    """Flag off → byte-identical UUID render even if config carries _turn_n."""
    monkeypatch.delenv("MONOLITH_TURN_COUNTER_V1", raising=False)
    store.set_bearing(bs.Bearing(
        current_frame="f", updated_at_turn="uuid", updated_at_turn_n=300))
    result = compiler.bearing_interceptor([{"role": "user", "content": "hi"}], {"_turn_n": 342})
    assert result is not None
    assert "updated_at_turn: uuid" in result[0]["content"]
    assert "turns ago" not in result[0]["content"]


# ── _detect_bearing_mismatch: precision + actionable, self-author nudge ──
#
# The nudge amplifies a deterministic cross-channel-context signal. It is only
# safe to make actionable if it is HIGH PRECISION — common words ("user",
# "connect") in ordinary technical frames must NOT fire, or the model learns to
# ignore it (cry-wolf) or spams bearing updates to silence it. And when it does
# fire it must name the actual action (<bearing_update>) and leave an out, since
# bearing is model-authored and the frame may still hold.


def _chan(role_tag: str) -> list[dict]:
    return [{"role": "user", "content": f"[CHANNEL: {role_tag}]\nhello"}]


def test_mismatch_fires_on_peer_name_in_user_channel() -> None:
    b = bs.Bearing(current_frame="collaborating with Claude on the carve")
    out = compiler._detect_bearing_mismatch(b, _chan("USER"))
    assert out is not None
    assert "<bearing_update>" in out  # names the actual self-author action


def test_mismatch_nudge_leaves_an_out() -> None:
    # advisory, not imperative — the model may decline if the frame still holds
    b = bs.Bearing(current_frame="exchange with Codex about kernels")
    out = compiler._detect_bearing_mismatch(b, _chan("USER"))
    assert out is not None
    assert "no update is needed" in out.lower()


def test_mismatch_fires_on_e_reference_in_connect_channel() -> None:
    b = bs.Bearing(current_frame="answering E's question about bearing")
    out = compiler._detect_bearing_mismatch(b, _chan("CONNECT/claude"))
    assert out is not None
    assert "<bearing_update>" in out


def test_mismatch_silent_when_common_word_user_in_connect_channel() -> None:
    # "user" is a common technical word — must not be read as stale peer-context
    b = bs.Bearing(current_frame="fixing the user login flow")
    assert compiler._detect_bearing_mismatch(b, _chan("CONNECT/claude")) is None


def test_mismatch_silent_when_common_word_connect_in_user_channel() -> None:
    b = bs.Bearing(current_frame="connecting the auth service to the database")
    assert compiler._detect_bearing_mismatch(b, _chan("USER")) is None


def test_mismatch_silent_when_frame_matches_channel() -> None:
    b = bs.Bearing(current_frame="carving the system prompt")
    assert compiler._detect_bearing_mismatch(b, _chan("USER")) is None


def test_mismatch_silent_when_frame_empty() -> None:
    assert compiler._detect_bearing_mismatch(bs.Bearing(), _chan("USER")) is None
