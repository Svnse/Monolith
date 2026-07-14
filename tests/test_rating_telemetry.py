from __future__ import annotations

import pytest

from core import rating_telemetry as rt
from core import turn_trace as tt


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    """Redirect both turn_trace and rating_telemetry to a temp DB per test."""
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    monkeypatch.setenv("MONOLITH_RATING_TELEMETRY_V1", "1")
    yield db
    tt.set_db_path(None)


def _record_rating(turn_id: str, value: int, reason: str | None = None) -> None:
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id=turn_id,
        recorded_at="2026-05-10T00:00:00Z",
        kind="rating",
        rating_value=value,
        reason=reason,
    ))


# ── recent_ratings_summary (turn_trace.py) ─────────────────────────


def test_summary_empty_when_no_ratings(trace_db) -> None:
    snap = tt.recent_ratings_summary()
    assert snap["count"] == 0
    assert snap["rolling_avg"] == 0.0
    assert snap["recent"] == []
    assert snap["worst"] is None
    assert snap["best"] is None


def test_summary_basic_aggregation(trace_db) -> None:
    _record_rating("t1", 80)
    _record_rating("t2", 60, "too vague")
    _record_rating("t3", 95, "exact")
    snap = tt.recent_ratings_summary()
    assert snap["count"] == 3
    assert snap["rolling_avg"] == pytest.approx((80 + 60 + 95) / 3)
    # oldest-first ordering
    assert snap["recent"] == [80, 60, 95]
    assert snap["worst"]["value"] == 60
    assert snap["worst"]["reason"] == "too vague"
    assert snap["best"]["value"] == 95
    assert snap["best"]["reason"] == "exact"


def test_summary_window_keeps_only_most_recent(trace_db) -> None:
    for i in range(15):
        _record_rating(f"t{i}", 50 + i)
    snap = tt.recent_ratings_summary(window=5)
    assert snap["count"] == 5
    # Most recent 5 values, oldest-first
    assert snap["recent"] == [60, 61, 62, 63, 64]


def test_summary_only_counts_kind_rating(trace_db) -> None:
    _record_rating("r1", 75)
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="r1", recorded_at="2026-05-10T00:00:01Z",
        kind="thumbs_up",
    ))
    snap = tt.recent_ratings_summary()
    assert snap["count"] == 1


def test_summary_no_op_when_flag_off(monkeypatch, trace_db) -> None:
    _record_rating("t1", 80)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "0")
    snap = tt.recent_ratings_summary()
    assert snap["count"] == 0


# ── render_telemetry_block ──────────────────────────────────────────


def test_render_returns_none_when_empty(trace_db) -> None:
    assert rt.render_telemetry_block() is None


def test_render_includes_avg_recent_worst_best(trace_db) -> None:
    _record_rating("t1", 90, "exact")
    _record_rating("t2", 50, "missed file path")
    block = rt.render_telemetry_block()
    assert block is not None
    assert block.startswith("[RATING TELEMETRY]")
    assert "rolling avg" in block
    assert "n=2" in block
    assert "missed file path" in block
    assert "exact" in block
    assert "worst recent" in block
    assert "best recent" in block


def test_render_dedupes_when_only_one_rating(trace_db) -> None:
    _record_rating("only", 75, "the only one")
    block = rt.render_telemetry_block()
    assert block is not None
    # Only one sample → worst == best (same turn). Worst line shows; best omitted.
    assert "worst recent" in block
    assert "best recent" not in block


def test_render_truncates_long_reason(trace_db) -> None:
    long = "x" * 200
    _record_rating("t1", 50, long)
    _record_rating("t2", 90, "short")
    block = rt.render_telemetry_block()
    assert block is not None
    assert "x" * 200 not in block
    assert "…" in block


def test_render_caps_recent_list_length(trace_db) -> None:
    for i in range(8):
        _record_rating(f"t{i}", 50 + i * 5)
    block = rt.render_telemetry_block()
    assert block is not None
    # _MAX_RECENT_SHOWN is 5; only last 5 values render in the recent line
    recent_line = next(line for line in block.split("\n") if line.startswith("- recent:"))
    # Count comma-separated values
    values_str = recent_line.split(":", 1)[1].split("(", 1)[0]
    assert values_str.count(",") == 4  # 5 values → 4 separators


# ── interceptor ─────────────────────────────────────────────────────


def test_interceptor_injects_before_user_message(trace_db) -> None:
    _record_rating("t1", 80)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what's up"},
    ]
    out = rt.rating_telemetry_interceptor(msgs, {})
    assert out is not None
    assert any("[RATING TELEMETRY]" in str(m.get("content", "")) for m in out)
    injected = next(m for m in out if "[RATING TELEMETRY]" in str(m.get("content", "")))
    assert injected.get("ephemeral") is True
    assert injected.get("source") == "rating_telemetry"
    # Injected before the user message
    user_idx = next(i for i, m in enumerate(out) if m.get("role") == "user" and not m.get("ephemeral"))
    inject_idx = next(i for i, m in enumerate(out) if m is injected)
    assert inject_idx < user_idx


def test_interceptor_returns_none_when_no_ratings(trace_db) -> None:
    msgs = [{"role": "user", "content": "hi"}]
    assert rt.rating_telemetry_interceptor(msgs, {}) is None


def test_interceptor_skips_when_already_present(trace_db) -> None:
    _record_rating("t1", 80)
    msgs = [
        {"role": "user", "content": "[RATING TELEMETRY]\n- already here", "ephemeral": True},
        {"role": "user", "content": "actual"},
    ]
    assert rt.rating_telemetry_interceptor(msgs, {}) is None


def test_interceptor_skips_with_no_user_message(trace_db) -> None:
    _record_rating("t1", 80)
    msgs = [{"role": "system", "content": "sys"}]
    assert rt.rating_telemetry_interceptor(msgs, {}) is None


def test_interceptor_no_op_when_flag_off(monkeypatch, trace_db) -> None:
    _record_rating("t1", 80)
    monkeypatch.setenv("MONOLITH_RATING_TELEMETRY_V1", "0")
    msgs = [{"role": "user", "content": "hi"}]
    assert rt.rating_telemetry_interceptor(msgs, {}) is None
