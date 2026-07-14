from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_skill():
    p = Path(__file__).parent.parent / "skills" / "curiosity" / "executor.py"
    spec = importlib.util.spec_from_file_location("curiosity_exec_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(monkeypatch, tmp_path):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
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
    yield _load_skill()


def _ingest(canonical, times=1):
    from core.acu_store import ACUStore
    s = ACUStore()
    try:
        for _ in range(times):
            s.ingest(canonical, source="model")
    finally:
        s.close()


def test_detect_op_reports_pulls(env) -> None:
    skill = env
    _ingest("monolith | values | precision")  # fresh aligned -> pull
    out = skill.run({"op": "detect", "threshold": 0.2}, None)
    assert "curiosity" in out.lower()
    assert "precision" in out


def test_detect_op_silent_when_nothing(env) -> None:
    skill = env
    _ingest("weather | is | sunny")  # unaligned
    out = skill.run({"op": "detect", "threshold": 0.2}, None)
    assert "precision" not in out
    assert "0 curiosity" in out.lower() or "nothing" in out.lower()
