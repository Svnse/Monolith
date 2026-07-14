from __future__ import annotations

import pytest

from core import continuity


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    """Redirect the continuity store to a temp file for each test."""
    store_path = tmp_path / "continuity.json"
    monkeypatch.setattr(continuity, "_STORE_PATH", store_path)
    yield store_path


# ── store: pin / retire / read ───────────────────────────────────────


def test_pin_assigns_incrementing_ids(tmp_store) -> None:
    p1 = continuity.pin("first")
    p2 = continuity.pin("second")
    assert p1["id"] == 1
    assert p2["id"] == 2


def test_pin_normalizes_unknown_category_and_source(tmp_store) -> None:
    p = continuity.pin("text", category="bogus", source="alien")
    assert p["category"] == "lesson"
    assert p["source"] == "i_inferred"


def test_pin_text_trimmed_to_category_limit(tmp_store) -> None:
    long = "x" * 600
    lesson = continuity.pin(long, category="lesson")
    anchor = continuity.pin(long, category="anchor")
    assert len(lesson["text"]) <= 200
    assert len(anchor["text"]) <= 500


def test_pin_with_evidence_keeps_field(tmp_store) -> None:
    p = continuity.pin(
        "calibration", category="lesson", evidence="seen at T+5m"
    )
    assert p.get("evidence") == "seen at T+5m"


def test_pin_with_empty_text_raises(tmp_store) -> None:
    with pytest.raises(ValueError):
        continuity.pin("   ", category="lesson")


def test_pin_cap_auto_retires_oldest_lesson(tmp_store) -> None:
    for i in range(8):
        continuity.pin(f"lesson {i}", category="lesson")
    continuity.pin("lesson 9", category="lesson")
    snap = continuity.read(include_retired=True, retired_limit=10)
    assert snap["counts"]["active"] == 8
    assert any(
        r["text"] == "lesson 0" and r["retire_reason"] == "aged_out"
        for r in snap["retired"]
    )


def test_pin_cap_protects_anchors_over_lessons(tmp_store) -> None:
    continuity.pin("anchor 1", category="anchor")
    continuity.pin("anchor 2", category="anchor")
    for i in range(6):
        continuity.pin(f"lesson {i}", category="lesson")
    # 9th — overflow. A lesson should retire, not an anchor.
    continuity.pin("lesson overflow", category="lesson")
    snap = continuity.read()
    cats = [p["category"] for p in snap["active"]]
    assert cats.count("anchor") == 2
    assert cats.count("lesson") == 6


def test_pin_cap_protects_pendings_over_lessons(tmp_store) -> None:
    for i in range(8):
        continuity.pin(f"pending {i}", category="pending")
    # 9th — only pendings present; oldest pending retires (lessons absent)
    continuity.pin("pending 8", category="pending")
    snap = continuity.read(include_retired=True)
    assert snap["counts"]["active"] == 8
    assert snap["retired"][-1]["text"] == "pending 0"


def test_supersedes_auto_retires_predecessor(tmp_store) -> None:
    p1 = continuity.pin("old", category="lesson")
    p2 = continuity.pin("refined", category="lesson", supersedes=p1["id"])
    snap = continuity.read(include_retired=True)
    active_ids = [p["id"] for p in snap["active"]]
    assert p1["id"] not in active_ids
    assert p2["id"] in active_ids
    retired = snap["retired"][-1]
    assert retired["id"] == p1["id"]
    assert retired["retire_reason"] == f"superseded_by:{p2['id']}"


def test_retire_moves_pin_with_known_reason(tmp_store) -> None:
    p = continuity.pin("a lesson")
    retired = continuity.retire(p["id"], "wrong")
    assert retired is not None
    assert retired["retire_reason"] == "wrong"
    snap = continuity.read(include_retired=True)
    assert snap["counts"]["active"] == 0
    assert snap["retired"][-1]["id"] == p["id"]


def test_retire_unknown_id_returns_none(tmp_store) -> None:
    assert continuity.retire(999, "wrong") is None


def test_retire_unknown_reason_falls_to_user_retired(tmp_store) -> None:
    p = continuity.pin("a lesson")
    r = continuity.retire(p["id"], "frivolous")
    assert r is not None
    assert r["retire_reason"] == "user_retired"


def test_read_default_excludes_retired_field(tmp_store) -> None:
    p = continuity.pin("a lesson")
    continuity.retire(p["id"], "wrong")
    snap = continuity.read()
    assert "retired" not in snap


def test_read_with_include_retired_caps_at_5(tmp_store) -> None:
    for i in range(7):
        p = continuity.pin(f"lesson {i}")
        continuity.retire(p["id"], "wrong")
    snap = continuity.read(include_retired=True)
    assert len(snap["retired"]) == 5


def test_retired_tail_capped_at_16(tmp_store) -> None:
    for i in range(20):
        p = continuity.pin(f"lesson {i}")
        continuity.retire(p["id"], "wrong")
    snap = continuity.read(include_retired=True, retired_limit=0)
    assert len(snap["retired"]) == 16


# ── projection ──────────────────────────────────────────────────────


def test_render_returns_none_when_empty(tmp_store) -> None:
    assert continuity.render_continuity_block() is None


def test_render_orders_anchor_pending_lesson(tmp_store) -> None:
    continuity.pin("a lesson", category="lesson")
    continuity.pin("an anchor", category="anchor")
    continuity.pin("a pending", category="pending")
    block = continuity.render_continuity_block()
    assert block is not None
    lines = block.splitlines()
    assert lines[0].startswith("[CONTINUITY]")
    assert "anchor(" in lines[1]
    assert "pending(" in lines[2]
    assert "lesson(" in lines[3]


def test_render_includes_evidence_when_present(tmp_store) -> None:
    continuity.pin("calibration", category="lesson", evidence="seen at T+5m")
    block = continuity.render_continuity_block()
    assert "(evidence: seen at T+5m)" in block


def test_render_header_counts_match_store(tmp_store) -> None:
    continuity.pin("active 1")
    p = continuity.pin("active 2")
    continuity.retire(p["id"], "wrong")
    block = continuity.render_continuity_block()
    # 1 active, 1 retired
    assert "1 pin" in block
    assert "1 retired" in block


# ── interceptor ─────────────────────────────────────────────────────


def test_interceptor_fires_on_first_turn(tmp_store) -> None:
    continuity.pin("a lesson")
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first user message"},
    ]
    result = continuity.continuity_interceptor(messages, {})
    assert result is not None
    assert any(
        "[CONTINUITY]" in str(m.get("content", "")) for m in result
    )
    # Injected message is ephemeral and inserted before the user turn
    injected = [
        m for m in result
        if "[CONTINUITY]" in str(m.get("content", ""))
    ][0]
    assert injected.get("ephemeral") is True
    assert injected.get("source") == "continuity"
    user_idx = next(
        i for i, m in enumerate(result)
        if m.get("content") == "first user message"
    )
    inj_idx = result.index(injected)
    assert inj_idx < user_idx


def test_interceptor_skips_after_first_turn(tmp_store) -> None:
    continuity.pin("a lesson")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    assert continuity.continuity_interceptor(messages, {}) is None


def test_interceptor_skips_when_flag_off(monkeypatch, tmp_store) -> None:
    monkeypatch.setenv("MONOLITH_CONTINUITY_BOOT_V1", "0")
    continuity.pin("a lesson")
    messages = [{"role": "user", "content": "first"}]
    assert continuity.continuity_interceptor(messages, {}) is None


def test_interceptor_skips_when_no_pins(tmp_store) -> None:
    messages = [{"role": "user", "content": "first"}]
    assert continuity.continuity_interceptor(messages, {}) is None


def test_interceptor_skips_when_already_injected(tmp_store) -> None:
    continuity.pin("a lesson")
    messages = [
        {
            "role": "user",
            "content": "[CONTINUITY] — 1 pin\n- lesson(1): existing",
            "ephemeral": True,
        },
        {"role": "user", "content": "actual question"},
    ]
    assert continuity.continuity_interceptor(messages, {}) is None


def test_interceptor_skips_when_no_user_message(tmp_store) -> None:
    continuity.pin("a lesson")
    messages = [{"role": "system", "content": "system"}]
    assert continuity.continuity_interceptor(messages, {}) is None


def test_interceptor_does_not_refire_after_regen_wipes_assistant(tmp_store) -> None:
    """Fix #7 (2026-05-14 audit): /regen on turn 2+ wipes the assistant
    message but keeps user1+user2. The old gate ('any assistant in history')
    would re-fire continuity here. The corrected gate ('non-ephemeral user
    count == 1') won't, since two user messages have been sent.
    """
    continuity.pin("a lesson")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second (regen target — assistant wiped)"},
    ]
    assert continuity.continuity_interceptor(messages, {}) is None


def test_interceptor_fires_on_first_turn_with_ephemeral_assistant_artifact(tmp_store) -> None:
    """Fix #7 (2026-05-14 audit): an ephemeral assistant artifact in turn 1
    history (e.g. a transient addon stub) shouldn't suppress the continuity
    boot. The old gate ('any assistant') would skip; the corrected gate
    ('non-ephemeral user count == 1') still fires, since only one real user
    message has been sent.
    """
    continuity.pin("a lesson")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "transient artifact", "ephemeral": True},
        {"role": "user", "content": "first user message"},
    ]
    result = continuity.continuity_interceptor(messages, {})
    assert result is not None
    injected = [
        m for m in result
        if isinstance(m.get("content"), str) and "[CONTINUITY]" in m["content"]
    ]
    assert len(injected) == 1


# ── executor surface (skills/scratchpad) ────────────────────────────


def test_executor_pin_op_writes_through(tmp_store) -> None:
    from skills.scratchpad.executor import run

    out = run({"op": "pin", "text": "remembered", "category": "lesson"}, ctx=None)
    assert "lesson(1)" in out
    snap = continuity.read()
    assert snap["counts"]["active"] == 1
    assert snap["active"][0]["text"] == "remembered"


def test_executor_pin_with_supersedes_announces_link(tmp_store) -> None:
    from skills.scratchpad.executor import run

    p1 = continuity.pin("old")
    out = run(
        {"op": "pin", "text": "new", "category": "lesson", "supersedes": p1["id"]},
        ctx=None,
    )
    assert f"superseding #{p1['id']}" in out


def test_executor_retire_returns_reason(tmp_store) -> None:
    from skills.scratchpad.executor import run

    p = continuity.pin("a lesson")
    out = run({"op": "retire", "id": p["id"], "reason": "resolved"}, ctx=None)
    assert "retired #" in out
    assert "resolved" in out


def test_executor_read_default_omits_retired(tmp_store) -> None:
    from skills.scratchpad.executor import run

    p = continuity.pin("active")
    p2 = continuity.pin("to retire")
    continuity.retire(p2["id"], "wrong")
    out = run({"op": "read"}, ctx=None)
    assert "Active pins:" in out
    assert "retired" not in out.lower().split("\n")[-1] or "Last" not in out


def test_executor_read_include_retired_lists_retired(tmp_store) -> None:
    from skills.scratchpad.executor import run

    p = continuity.pin("a lesson")
    continuity.retire(p["id"], "stale")
    out = run({"op": "read", "include_retired": True}, ctx=None)
    assert "Last 1 retired:" in out
    assert "[stale]" in out


def test_executor_unknown_op_returns_help(tmp_store) -> None:
    from skills.scratchpad.executor import run

    out = run({"op": "delete"}, ctx=None)
    assert "unknown op" in out
    assert "pin / retire / read" in out


# ── relational-time marker (when-plane) ──────────────────────────────


def test_last_turn_at_none_on_fresh_store(tmp_store) -> None:
    assert continuity.get_last_turn_at() is None


def test_set_last_turn_at_persists(tmp_store) -> None:
    continuity.set_last_turn_at("2026-06-02T12:00:00+00:00")
    assert continuity.get_last_turn_at() == "2026-06-02T12:00:00+00:00"


def test_set_last_turn_at_preserves_pins(tmp_store) -> None:
    """The marker is additive — recording a turn timestamp must not disturb pins."""
    continuity.pin("anchor pin", category="anchor")
    continuity.set_last_turn_at("2026-06-02T12:00:00+00:00")
    snap = continuity.read(include_retired=False)
    assert any(p["text"] == "anchor pin" for p in snap["active"])
    assert continuity.get_last_turn_at() == "2026-06-02T12:00:00+00:00"
