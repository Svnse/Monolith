from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core import turn_trace as tt


@pytest.fixture
def populated_db(tmp_path, monkeypatch):
    """A turn_trace DB with synthetic frame_traces across 3 dates."""
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    conn = tt._get_conn()
    # Insert 5 frames across 3 days
    rows = [
        ("t1", "2026-05-15T10:00:00+00:00", 100, 500, 600),
        ("t2", "2026-05-15T11:00:00+00:00", 200, 800, 1000),
        ("t3", "2026-05-16T09:00:00+00:00", 150, 600, 750),
        ("t4", "2026-05-17T14:00:00+00:00", 300, 1200, 1500),
        ("t5", "2026-05-17T15:00:00+00:00", 250, 900, 1150),
    ]
    for turn_id, captured_at, sys_chars, user_chars, total in rows:
        conn.execute(
            "INSERT INTO frame_traces "
            "(turn_id, captured_at, backend, engine_key, gen_id, "
            "system_prompt_chars, user_prompt_chars, total_chars, "
            "final_messages_json, config_snapshot_json, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]', '{}', '{}')",
            (turn_id, captured_at, "gguf_api", "llm", 0, sys_chars, user_chars, total),
        )
    conn.commit()
    yield db
    tt.set_db_path(None)


def test_lifetime_summary_counts_all_frames(populated_db):
    from core.stats_store import StatsStore
    store = StatsStore()
    summary = store.get_lifetime_summary()
    assert summary["turns"] == 5
    assert summary["total_chars"] == 5000
    assert summary["first_turn_date"] == "2026-05-15"
    assert summary["last_turn_date"] == "2026-05-17"
    assert summary["day_count"] == 3


def test_get_streak_counts_consecutive_days(populated_db, monkeypatch):
    """Streak counts consecutive days through 'today' with >= 1 turn.
    Fixture has turns on 2026-05-15, 16, 17. If today is 2026-05-17, streak=3."""
    import core.stats_store as ss_mod
    # Pin "today" to 2026-05-17 so the fixture data forms a 3-day streak
    monkeypatch.setattr(ss_mod, "_today_iso", lambda: "2026-05-17")
    from core.stats_store import StatsStore
    store = StatsStore()
    assert store.get_streak() == 3


def test_get_streak_zero_when_gap_before_today(populated_db, monkeypatch):
    """If yesterday had no turn, streak is 0 (even if today has turns)."""
    import core.stats_store as ss_mod
    monkeypatch.setattr(ss_mod, "_today_iso", lambda: "2026-05-20")
    from core.stats_store import StatsStore
    store = StatsStore()
    assert store.get_streak() == 0


@pytest.fixture
def populated_db_with_ratings(populated_db):
    """Extends populated_db with outcome_traces ratings."""
    conn = tt._get_conn()
    ratings = [
        ("t1", "2026-05-15T10:30:00+00:00", "rating", 85, "good"),
        ("t2", "2026-05-15T11:30:00+00:00", "rating", 60, "ok"),
        ("t3", "2026-05-16T09:30:00+00:00", "rating", 92, "great"),
        ("t4", "2026-05-17T14:30:00+00:00", "rating", 45, "weak"),
    ]
    for turn_id, recorded_at, kind, value, reason in ratings:
        conn.execute(
            "INSERT INTO outcome_traces (turn_id, recorded_at, kind, rating_value, reason, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, '{}')",
            (turn_id, recorded_at, kind, value, reason),
        )
    conn.commit()
    return populated_db


def test_rating_histogram_buckets_by_tens(populated_db_with_ratings):
    """Ratings 85, 60, 92, 45 bucket into 80-89, 60-69, 90-100, 40-49."""
    from core.stats_store import StatsStore
    store = StatsStore()
    hist = store.get_rating_histogram("all")
    assert hist["80-89"] == 1
    assert hist["60-69"] == 1
    assert hist["90-100"] == 1
    assert hist["40-49"] == 1
    assert sum(hist.values()) == 4


def test_rating_trend_per_day_mean(populated_db_with_ratings, monkeypatch):
    """Trend returns (date, mean) pairs for the last N days."""
    import core.stats_store as ss_mod
    monkeypatch.setattr(ss_mod, "_today_iso", lambda: "2026-05-17")
    from core.stats_store import StatsStore
    store = StatsStore()
    trend = dict(store.get_rating_trend(days=3))
    assert trend["2026-05-15"] == pytest.approx((85 + 60) / 2)
    assert trend["2026-05-16"] == pytest.approx(92.0)
    assert trend["2026-05-17"] == pytest.approx(45.0)


@pytest.fixture
def populated_db_with_modes(populated_db):
    """Re-populate frame_traces with effort_tier + reasoning_mode set on each turn."""
    conn = tt._get_conn()
    conn.execute("DELETE FROM frame_traces")
    rows = [
        ("t1", "2026-05-15T10:00:00+00:00", "med",    None),
        ("t2", "2026-05-15T11:00:00+00:00", "high",   None),
        ("t3", "2026-05-16T09:00:00+00:00", "high",   "monothink"),
        ("t4", "2026-05-17T14:00:00+00:00", "ultimate", "monothink"),
        ("t5", "2026-05-17T15:00:00+00:00", "med",    None),
    ]
    for turn_id, captured_at, effort, reasoning in rows:
        conn.execute(
            "INSERT INTO frame_traces "
            "(turn_id, captured_at, backend, engine_key, gen_id, "
            "system_prompt_chars, user_prompt_chars, total_chars, "
            "final_messages_json, config_snapshot_json, metadata_json, "
            "effort_tier, reasoning_mode) "
            "VALUES (?, ?, 'gguf_api', 'llm', 0, 100, 200, 300, '[]', '{}', '{}', ?, ?)",
            (turn_id, captured_at, effort, reasoning),
        )
    conn.commit()
    return populated_db


def test_effort_distribution_counts_per_tier(populated_db_with_modes):
    from core.stats_store import StatsStore
    store = StatsStore()
    dist = store.get_effort_distribution("all")
    assert dist["med"] == 2
    assert dist["high"] == 2
    assert dist["ultimate"] == 1


def test_mode_distribution_for_reasoning_plane(populated_db_with_modes):
    from core.stats_store import StatsStore
    store = StatsStore()
    dist = store.get_mode_distribution("reasoning", "all")
    assert dist.get("monothink") == 2
    assert sum(dist.values()) == 2


@pytest.fixture
def populated_db_with_faults(populated_db):
    """Add fault_traces rows that look like tool failures."""
    conn = tt._get_conn()
    faults = [
        ("t1", "2026-05-15T10:00:01+00:00", "ToolFailedEvent", "grep",        "hard_failure"),
        ("t2", "2026-05-15T11:00:01+00:00", "ToolFailedEvent", "read_file",   "recoverable_with_hint"),
        ("t3", "2026-05-16T09:00:01+00:00", "ToolFailedEvent", "grep",        "informational"),
    ]
    for turn_id, emitted_at, event_kind, source_name, fault_kind in faults:
        conn.execute(
            "INSERT INTO fault_traces "
            "(turn_id, seq, emitted_at, event_kind, source_kind, source_name, fault_kind, payload_json) "
            "VALUES (?, 1, ?, ?, 'producer', ?, ?, '{}')",
            (turn_id, emitted_at, event_kind, source_name, fault_kind),
        )
    conn.commit()
    return populated_db


def test_get_fault_summary_aggregates_by_kind(populated_db_with_faults):
    from core.stats_store import StatsStore
    store = StatsStore()
    summary = store.get_fault_summary("all")
    assert summary["hard_failure"] == 1
    assert summary["recoverable_with_hint"] == 1
    assert summary["informational"] == 1


def test_tool_usage_top_n_orders_by_count(populated_db_with_faults):
    from core.stats_store import StatsStore
    store = StatsStore()
    usage = store.get_tool_usage("all", top_n=3)
    assert usage[0]["tool"] == "grep"
    assert usage[0]["count"] == 2
    assert usage[1]["tool"] == "read_file"


def test_get_time_rhythm_buckets_by_weekday_hour(populated_db_with_modes):
    """Returns {(weekday, hour): count} where weekday is 0=Mon..6=Sun."""
    from core.stats_store import StatsStore
    store = StatsStore()
    rhythm = store.get_time_rhythm("all")
    # 2026-05-15 was a Friday (weekday 4); turns at 10:00 and 11:00
    assert rhythm.get((4, 10)) == 1
    assert rhythm.get((4, 11)) == 1
    # 2026-05-16 Saturday (5); turn at 09
    assert rhythm.get((5, 9)) == 1
    # 2026-05-17 Sunday (6); turns at 14 and 15
    assert rhythm.get((6, 14)) == 1
    assert rhythm.get((6, 15)) == 1


def test_personal_records_returns_six_keys(populated_db_with_modes):
    from core.stats_store import StatsStore
    store = StatsStore()
    records = store.get_personal_records()
    expected_keys = {
        "longest_chat", "highest_rated_turn", "biggest_prompt",
        "best_day", "longest_tool_chain", "oldest_active_pin",
    }
    assert set(records.keys()) >= expected_keys


def test_substrate_summary_reads_continuity_json(tmp_path, monkeypatch):
    """SubstrateBlock data: active pins (by category) + retired count from
    continuity.json."""
    import json
    from core import paths as _paths
    continuity_path = tmp_path / "continuity.json"
    payload = {
        "version": 1,
        "pins": [
            {"id": 1, "category": "anchor", "text": "x", "active": True},
            {"id": 2, "category": "lesson", "text": "y", "active": True},
            {"id": 3, "category": "lesson", "text": "z", "active": False},
            {"id": 4, "category": "pending", "text": "p", "active": True},
        ],
    }
    continuity_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(_paths, "CONFIG_DIR", tmp_path)
    from core.stats_store import StatsStore
    store = StatsStore()
    summary = store.get_substrate_summary()
    assert summary["continuity"]["active"]["anchor"] == 1
    assert summary["continuity"]["active"]["lesson"] == 1
    assert summary["continuity"]["active"]["pending"] == 1
    assert summary["continuity"]["retired"] == 1


def test_get_achievements_returns_newest_first(populated_db):
    """get_achievements pulls from the achievements table, newest first."""
    conn = tt._get_conn()
    rows = [
        ("2026-05-15", "MILESTONE", "100 turns", '{}'),
        ("2026-05-17", "NEW PR", "longest chat: 30", '{}'),
        ("2026-05-16", "STREAK", "7 day streak", '{}'),
    ]
    for unlocked_at, tag, desc, payload in rows:
        conn.execute(
            "INSERT INTO achievements (unlocked_at, tag, description, payload_json) "
            "VALUES (?, ?, ?, ?)",
            (unlocked_at, tag, desc, payload),
        )
    conn.commit()
    from core.stats_store import StatsStore
    store = StatsStore()
    items = store.get_achievements(limit=10)
    assert len(items) == 3
    assert items[0]["unlocked_at"] == "2026-05-17"
    assert items[0]["tag"] == "NEW PR"
    assert items[-1]["unlocked_at"] == "2026-05-15"
