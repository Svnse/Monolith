"""M3.1 kill-actuator: Monolith may RETIRE a curiosity pull it judges as noise.
The SAFE half of closing the loop — reversible, audited, excluded from future
surfacing. Promotion-into-identity stays human-gated (not here)."""
from __future__ import annotations

import pytest


@pytest.fixture
def substrate(monkeypatch, tmp_path):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    monkeypatch.setenv("MONOLITH_CURIOSITY_V1", "1")
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)
    from core import identity_milestones as m, identity
    monkeypatch.setattr(m, "STORE_PATH", tmp_path / "identity_milestones.json")
    monkeypatch.setattr(
        identity, "load_identity",
        lambda: "# Monolith\n\n## What I value\nPrecision over fluency. Naming over narration.",
    )
    yield tmp_path


def _ingest(canonical, times=1):
    from core.acu_store import ACUStore
    s = ACUStore()
    try:
        for _ in range(times):
            s.ingest(canonical, source="model")
    finally:
        s.close()


def test_ledger_killed_roundtrip(substrate) -> None:
    from core import identity_milestones as m
    assert m.get_curiosity_killed() == {}
    m.kill_curiosity_pull("a | b | c", "noise")
    killed = m.get_curiosity_killed()
    assert "a | b | c" in killed
    assert killed["a | b | c"]["reason"] == "noise"
    m.unkill_curiosity_pull("a | b | c")
    assert "a | b | c" not in m.get_curiosity_killed()


def test_kill_excludes_from_detection_and_logs(substrate) -> None:
    from core import curiosity, identity_milestones as m
    from core.acatalepsy import canonical_log
    _ingest("monolith | values | precision")
    rep1 = curiosity.detect_pulls(align_threshold=0.2, force=True)
    canon = next(p["canonical"] for p in rep1.pulls if "precision" in p["canonical"])

    curiosity.kill_pull(canon, reason="ungroundable rumination")

    assert canon in m.get_curiosity_killed()
    kinds = [e.kind for e in canonical_log.read_since(0, limit=500)]
    assert "curiosity_pull_killed" in kinds

    rep2 = curiosity.detect_pulls(align_threshold=0.2, force=True)
    assert all(p["canonical"] != canon for p in rep2.pulls)  # retired -> not resurfaced


def test_unkill_restores_pull(substrate) -> None:
    from core import curiosity, identity_milestones as m
    _ingest("monolith | values | precision")
    canon = next(p["canonical"] for p in curiosity.detect_pulls(align_threshold=0.2, force=True).pulls)
    curiosity.kill_pull(canon, reason="x")
    assert all(p["canonical"] != canon for p in curiosity.detect_pulls(align_threshold=0.2, force=True).pulls)
    m.unkill_curiosity_pull(canon)
    assert any(p["canonical"] == canon for p in curiosity.detect_pulls(align_threshold=0.2, force=True).pulls)
