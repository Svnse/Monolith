from __future__ import annotations

import pytest


@pytest.fixture
def substrate(monkeypatch, tmp_path):
    """Isolated acatalepsy DB + isolated milestone ledger + injected identity."""
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    monkeypatch.setenv("MONOLITH_IDENTITY_EMERGENCE_V1", "1")
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)
    from core import identity_milestones as m
    monkeypatch.setattr(m, "STORE_PATH", tmp_path / "identity_milestones.json")
    from core import identity
    monkeypatch.setattr(
        identity, "load_identity",
        lambda: "# Monolith\n\n## What I value\nPrecision over fluency. Naming over narration.",
    )
    yield tmp_path


def _ingest(canonical: str, source: str = "model") -> int:
    from core.acu_store import ACUStore
    s = ACUStore()
    try:
        return s.ingest(canonical, source=source)
    finally:
        s.close()


def _ingest_stable(canonical: str, source: str = "model", times: int = 6) -> int:
    """Ingest the same claim repeatedly so it reinforces into the STABLE
    disposition (reinforcement ≥5 → stability ≥ threshold → emergence-eligible)."""
    from core.acu_store import ACUStore
    s = ACUStore()
    try:
        acu_id = -1
        for _ in range(times):
            acu_id = s.ingest(canonical, source=source)
        return acu_id
    finally:
        s.close()


def test_detect_fires_on_aligned_self_derived(substrate) -> None:
    from core import identity_emergence as em
    _ingest_stable("monolith | values | precision")   # reinforced -> stable -> emergence
    _ingest_stable("monolith | values | narration")
    _ingest("weather | is | sunny")                    # fresh + unaligned -> excluded

    rep = em.detect_emergence(threshold_confidentity=0.3, min_new_acus=1, backend="lexical")

    assert rep.fired is True
    canons = [c["canonical"] for c in rep.candidates]
    assert any("precision" in c for c in canons)
    assert all("weather" not in c for c in canons)


def test_detect_persists_confidentity_onto_self_derived_rows(substrate) -> None:
    from core import identity_emergence as em
    from core.acu_store import ACUStore
    acu_id = _ingest("monolith | values | precision")

    em.detect_emergence(threshold_confidentity=0.3, min_new_acus=1, backend="lexical")

    s = ACUStore()
    try:
        row = s.get_by_id(acu_id)
    finally:
        s.close()
    assert row is not None
    assert row["confidentity"] > 0.0  # was 0.0 default before scoring


def test_detect_emits_canonical_log_event_and_advances_watermark(substrate) -> None:
    from core import identity_emergence as em, identity_milestones as m
    from core.acatalepsy import canonical_log
    _ingest_stable("monolith | values | precision")

    em.detect_emergence(threshold_confidentity=0.3, min_new_acus=1, backend="lexical")

    kinds = [e.kind for e in canonical_log.read_since(0, limit=500)]
    assert "identity_emergence_detected" in kinds
    assert m.get_watermark() >= 1


def test_detect_silent_when_nothing_aligned(substrate) -> None:
    from core import identity_emergence as em
    _ingest("weather | is | sunny")
    _ingest("traffic | is | heavy")

    rep = em.detect_emergence(threshold_confidentity=0.3, min_new_acus=1, backend="lexical")

    assert rep.fired is False
    assert rep.candidates == ()


def test_detect_silent_below_min_new_acus(substrate) -> None:
    from core import identity_emergence as em, identity_milestones as m
    _ingest("monolith | values | precision")
    m.set_watermark(0)  # 1 new ACU, but require many

    rep = em.detect_emergence(threshold_confidentity=0.3, min_new_acus=50, backend="lexical")

    assert rep.fired is False


def test_best_effort_retries_database_locked(monkeypatch) -> None:
    import sqlite3
    from core import identity_emergence as em

    calls = []

    def fake_detect(**_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return em.EmergenceReport(True, 1, (), "ok")

    monkeypatch.setattr(em, "detect_emergence", fake_detect)
    monkeypatch.setattr(em.time, "sleep", lambda _seconds: None)

    rep = em.detect_emergence_best_effort(retries=2)

    assert rep.fired is True
    assert len(calls) == 2


def test_best_effort_skips_after_database_lock_retries(monkeypatch) -> None:
    import sqlite3
    from core import identity_emergence as em

    calls = []

    def fake_detect(**_kwargs):
        calls.append(1)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(em, "detect_emergence", fake_detect)
    monkeypatch.setattr(em.time, "sleep", lambda _seconds: None)

    rep = em.detect_emergence_best_effort(retries=1)

    assert rep.fired is False
    assert rep.message == "database locked; skipped"
    assert len(calls) == 2


def test_turn_trace_identity_hook_uses_best_effort_without_lock_trace(monkeypatch, tmp_path) -> None:
    import datetime as dt
    from core import identity_emergence as em
    from core import turn_trace as tt

    called = []

    def fake_best_effort(**kwargs):
        called.append(kwargs)
        return em.EmergenceReport(False, 0, (), "database locked; skipped")

    failures = []
    monkeypatch.setattr(em, "detect_emergence_best_effort", fake_best_effort)
    monkeypatch.setattr(tt, "_trace_failure", failures.append)
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    monkeypatch.setenv("MONOLITH_CURIOSITY_V1", "0")
    tt.set_db_path(tmp_path / "turn_trace.sqlite3")
    try:
        tt.record_outcome(tt.OutcomeTraceRecord(
            turn_id="t-lock",
            recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            kind="rating",
            rating_value=50,
        ))
    finally:
        tt.set_db_path(None)

    assert called == [{}]
    assert not any("identity emergence hook failed" in f for f in failures)
