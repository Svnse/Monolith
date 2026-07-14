from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def mirror_env(monkeypatch, tmp_path):
    from core import db_connect as _dbc

    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")

    from core.acatalepsy import canonical_log, intake, schema

    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, attr):
                    delattr(tl, attr)

    from core import identity

    monkeypatch.setattr(
        identity,
        "load_identity",
        lambda: "# Monolith\n\n## What I value\nPrecision over fluency. Naming over narration.",
    )
    yield tmp_path


def _ingest(canonical: str, *, times: int = 1, source: str = "model") -> int:
    from core.acu_store import ACUStore

    store = ACUStore()
    try:
        acu_id = -1
        for _ in range(times):
            acu_id = store.ingest(canonical, source=source)
        return acu_id
    finally:
        store.close()


def _row(acu_id: int) -> dict:
    from core.db_connect import connect_acatalepsy

    conn = connect_acatalepsy(role="reader")
    try:
        row = conn.execute("SELECT * FROM acus WHERE id=?", (int(acu_id),)).fetchone()
        return dict(row)
    finally:
        conn.close()


def _add_pending_candidate(canonical: str, *, contradicts_acu_id: int | None = None) -> int:
    from core.db_connect import authorized_write, connect_acatalepsy

    now = datetime.now(timezone.utc).isoformat()
    conn = connect_acatalepsy(role="memory_writer")
    try:
        with authorized_write("test:acu_mirror_pending"):
            log = conn.execute(
                "INSERT INTO canonical_log(ts, kind, session_id, payload) VALUES(?,?,?,?)",
                (datetime.now(timezone.utc).timestamp(), "test_event", "test", "{}"),
            )
            cur = conn.execute(
                """
                INSERT INTO acu_candidates(
                    canonical_form, evidence_log_id, evidence_char_start,
                    evidence_char_end, evidence_span, source, reason,
                    reinforcement_count, contradicts_acu_id, state, created_at,
                    auditor_run_id
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    canonical,
                    int(log.lastrowid),
                    0,
                    len(canonical),
                    canonical,
                    "model",
                    "test pending candidate",
                    1,
                    contradicts_acu_id,
                    "pending",
                    now,
                    1,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
    finally:
        conn.close()


def test_mirror_surfaces_stable_claims_and_blocks_unstable_without_writes(mirror_env) -> None:
    from core import acu_mirror

    stable = _ingest("monolith | values | precision", times=6)
    fresh = _ingest("monolith | values | naming", times=1)

    assert _row(stable)["confidentity"] == 0.0
    assert _row(fresh)["confidentity"] == 0.0

    snap = acu_mirror.build_snapshot(threshold=0.3, near_band=0.05, backend="lexical")

    surfaceable = {item["id"]: item for item in snap["surfaceable"]}
    blocked = {item["id"]: item for item in snap["blocked"]}
    assert stable in surfaceable
    assert fresh in blocked
    assert any("below stability gate" in r for r in blocked[fresh]["surface_reasons"])

    # The mirror computes confidentity in memory only; it must not persist the score.
    assert _row(stable)["confidentity"] == 0.0
    assert _row(fresh)["confidentity"] == 0.0


def test_mirror_reports_near_threshold_claims(mirror_env) -> None:
    from core import acu_mirror

    acu_id = _ingest("monolith | values | precision", times=6)

    snap = acu_mirror.build_snapshot(threshold=0.35, near_band=0.05, backend="lexical")

    near = {item["id"]: item for item in snap["near_threshold"]}
    assert acu_id in near
    assert near[acu_id]["confidentity"] < 0.35
    assert any("below threshold" in r for r in near[acu_id]["surface_reasons"])


def test_mirror_reports_contradictions_and_pending_candidates(mirror_env) -> None:
    from core import acu_mirror

    first = _ingest("france | capital | paris", source="world")
    second = _ingest("france | capital | lyon", source="world")
    candidate = _add_pending_candidate("france | capital | marseille", contradicts_acu_id=first)

    snap = acu_mirror.build_snapshot(threshold=0.2, limit=10, backend="lexical")

    functional = [c for c in snap["contradictions"] if c["type"] == "functional_conflict"]
    assert any({c["left_id"], c["right_id"]} == {first, second} for c in functional)

    pending_conflicts = [c for c in snap["contradictions"] if c["type"] == "pending_candidate"]
    assert any(c["candidate_id"] == candidate and c["target_id"] == first for c in pending_conflicts)
    assert any(p["id"] == candidate and p["contradicts_acu_id"] == first for p in snap["pending_candidates"])


def test_mirror_decay_preview_is_pure_read(mirror_env) -> None:
    from core import acu_mirror
    from core.db_connect import authorized_write, connect_acatalepsy

    fixed_now = datetime(2026, 6, 17, tzinfo=timezone.utc)
    old_ts = (fixed_now - timedelta(days=90)).isoformat()
    acu_id = _ingest("old claim | values | precision", times=3)

    conn = connect_acatalepsy(role="memory_writer")
    try:
        with authorized_write("test:acu_mirror_decay_timestamp"):
            conn.execute(
                "UPDATE acus SET created_at=?, last_seen=?, last_touched_ts=? WHERE id=?",
                (old_ts, old_ts, old_ts, int(acu_id)),
            )
            conn.commit()
    finally:
        conn.close()

    before = _row(acu_id)
    snap = acu_mirror.build_snapshot(threshold=0.99, backend="lexical", now=fixed_now)
    after = _row(acu_id)

    item = next(i for i in snap["decay"]["items"] if i["id"] == acu_id)
    assert item["effective_reinforcement"] < item["reinforcement"]
    assert item["age_days"] == pytest.approx(90.0)
    assert before["last_touched_ts"] == after["last_touched_ts"]


def test_acu_mirror_skill_is_discoverable_and_read_only(mirror_env, tmp_path) -> None:
    from core.skill_registry import clear_skill_cache
    from core.skill_runtime import (
        ToolExecutionContext,
        clear_dynamic_executor_cache,
        execute_tool_call_enveloped,
    )

    _ingest("monolith | values | precision", times=6)
    clear_skill_cache()
    clear_dynamic_executor_cache()

    result = execute_tool_call_enveloped(
        {"tool": "acu_mirror", "op": "snapshot", "threshold": 0.3, "limit": 5},
        ToolExecutionContext(archive_dir=tmp_path),
    )

    assert result.ok is True
    assert "acu_mirror" in result.text
    assert "read-only snapshot" in result.text
    assert "would_surface_identity_review" in result.text
