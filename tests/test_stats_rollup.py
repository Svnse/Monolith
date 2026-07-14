from __future__ import annotations

import pytest

from core import turn_trace as tt


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Reset turn_trace's global connection state to a temp DB."""
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    yield db
    tt.set_db_path(None)


def test_daily_rollups_table_exists_on_fresh_install(fresh_db):
    """daily_rollups table is created by _DDL on first connection."""
    conn = tt._get_conn()
    assert conn is not None
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_rollups'"
    )
    assert cur.fetchone() is not None, "daily_rollups table missing"


def test_daily_rollups_columns(fresh_db):
    """daily_rollups has the columns specified in the spec."""
    conn = tt._get_conn()
    cur = conn.execute("PRAGMA table_info(daily_rollups)")
    cols = {row[1] for row in cur.fetchall()}
    expected = {
        "date", "turns", "total_chars",
        "ratings_count", "ratings_sum", "ratings_histogram_json",
        "effort_distribution_json", "conversation_mode_dist_json",
        "reasoning_mode_dist_json", "linguency_mode_dist_json",
        "tool_usage_json", "fault_summary_json",
        "time_rhythm_json", "stage_latency_json",
        "computed_at",
    }
    missing = expected - cols
    assert not missing, f"daily_rollups missing columns: {missing}"


def test_migration_adds_daily_rollups_to_legacy_db(tmp_path, monkeypatch):
    """A pre-existing DB without daily_rollups gets the table on next connect."""
    import sqlite3
    db = tmp_path / "legacy.sqlite3"
    # Simulate a legacy DB by creating only frame_traces (no rollups)
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE frame_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT NOT NULL UNIQUE,
            captured_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    try:
        conn = tt._get_conn()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_rollups'"
        )
        assert cur.fetchone() is not None, "migration did not add daily_rollups"
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='achievements'"
        )
        assert cur.fetchone() is not None, "migration did not add achievements"
    finally:
        tt.set_db_path(None)


def test_materialize_missing_days_writes_rows(fresh_db, monkeypatch):
    """Given frame_traces on 2026-05-15 and 16, materialize creates 2 rows
    (assuming today is 2026-05-17 so today is excluded)."""
    conn = tt._get_conn()
    for turn_id, captured_at in [
        ("t1", "2026-05-15T10:00:00+00:00"),
        ("t2", "2026-05-15T11:00:00+00:00"),
        ("t3", "2026-05-16T09:00:00+00:00"),
    ]:
        conn.execute(
            "INSERT INTO frame_traces "
            "(turn_id, captured_at, backend, engine_key, gen_id, "
            "system_prompt_chars, user_prompt_chars, total_chars, "
            "final_messages_json, config_snapshot_json, metadata_json, effort_tier) "
            "VALUES (?, ?, 'gguf_api', 'llm', 0, 100, 200, 300, '[]', '{}', '{}', 'med')",
            (turn_id, captured_at),
        )
    conn.commit()

    import core.stats_rollup as sr
    monkeypatch.setattr(sr, "_today_iso", lambda: "2026-05-17")
    count = sr.materialize_missing_days(conn)
    assert count == 2

    rows = conn.execute("SELECT date, turns, total_chars FROM daily_rollups ORDER BY date").fetchall()
    assert len(rows) == 2
    assert tuple(rows[0]) == ("2026-05-15", 2, 600)
    assert tuple(rows[1]) == ("2026-05-16", 1, 300)


def test_materialize_is_idempotent(fresh_db, monkeypatch):
    """Re-running materializer doesn't double-insert (uses INSERT OR REPLACE
    keyed on date)."""
    conn = tt._get_conn()
    conn.execute(
        "INSERT INTO frame_traces "
        "(turn_id, captured_at, backend, engine_key, gen_id, "
        "system_prompt_chars, user_prompt_chars, total_chars, "
        "final_messages_json, config_snapshot_json, metadata_json) "
        "VALUES ('t1', '2026-05-15T10:00:00+00:00', 'gguf_api', 'llm', 0, 100, 200, 300, '[]', '{}', '{}')"
    )
    conn.commit()
    import core.stats_rollup as sr
    monkeypatch.setattr(sr, "_today_iso", lambda: "2026-05-17")
    sr.materialize_missing_days(conn)
    sr.materialize_missing_days(conn)  # second run
    rows = conn.execute("SELECT COUNT(*) FROM daily_rollups").fetchone()
    assert rows[0] == 2  # 2 days (15, 16); 17 excluded as 'today'


def test_check_achievements_unlocks_first_milestone(fresh_db, monkeypatch):
    """When lifetime turns crosses 100 for the first time, a MILESTONE
    achievement is inserted."""
    conn = tt._get_conn()
    for i in range(100):
        conn.execute(
            "INSERT INTO frame_traces "
            "(turn_id, captured_at, backend, engine_key, gen_id, "
            "system_prompt_chars, user_prompt_chars, total_chars, "
            "final_messages_json, config_snapshot_json, metadata_json) "
            "VALUES (?, '2026-05-15T10:00:00+00:00', 'gguf_api', 'llm', 0, "
            "100, 200, 300, '[]', '{}', '{}')",
            (f"t{i}",),
        )
    conn.commit()
    import core.stats_rollup as sr
    monkeypatch.setattr(sr, "_today_iso", lambda: "2026-05-17")
    sr.materialize_missing_days(conn)
    achievements = conn.execute(
        "SELECT tag, description FROM achievements"
    ).fetchall()
    tags = {a[0] for a in achievements}
    assert "MILESTONE" in tags
    assert any("100" in a[1] for a in achievements)


def test_check_achievements_does_not_double_unlock(fresh_db, monkeypatch):
    """Re-running materializer doesn't re-insert the same MILESTONE."""
    conn = tt._get_conn()
    for i in range(100):
        conn.execute(
            "INSERT INTO frame_traces "
            "(turn_id, captured_at, backend, engine_key, gen_id, "
            "system_prompt_chars, user_prompt_chars, total_chars, "
            "final_messages_json, config_snapshot_json, metadata_json) "
            "VALUES (?, '2026-05-15T10:00:00+00:00', 'gguf_api', 'llm', 0, "
            "100, 200, 300, '[]', '{}', '{}')",
            (f"t{i}",),
        )
    conn.commit()
    import core.stats_rollup as sr
    monkeypatch.setattr(sr, "_today_iso", lambda: "2026-05-17")
    sr.materialize_missing_days(conn)
    sr.materialize_missing_days(conn)
    n = conn.execute(
        "SELECT COUNT(*) FROM achievements WHERE tag = 'MILESTONE' AND description LIKE '%100%'"
    ).fetchone()[0]
    assert n == 1
