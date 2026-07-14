from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_skill():
    p = Path(__file__).parent.parent / "skills" / "identity_review" / "executor.py"
    spec = importlib.util.spec_from_file_location("identity_review_exec_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(monkeypatch, tmp_path):
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
    from core import identity_milestones as m, proposals, identity, llm_config
    monkeypatch.setattr(m, "STORE_PATH", tmp_path / "identity_milestones.json")
    monkeypatch.setattr(proposals, "STORE_PATH", tmp_path / "proposals.json")
    monkeypatch.setattr(identity, "load_identity", lambda: identity._DEFAULT_IDENTITY)
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "test-model")
    skill = _load_skill()
    monkeypatch.setattr(skill, "_proposals", proposals)
    yield skill


def _ingest_aligned():
    # identity_review drafts CONSOLIDATION (emergence) amendments → candidates
    # must be STABLE (reinforced), so ingest each claim repeatedly.
    from core.acu_store import ACUStore
    s = ACUStore()
    try:
        for c in ("monolith | values | precision", "monolith | holds | continuity",
                  "monolith | credits | observation"):
            for _ in range(6):
                s.ingest(c, source="model")
    finally:
        s.close()


def test_detect_op_reports_candidates_without_llm(env) -> None:
    skill = env
    _ingest_aligned()
    out = skill.run({"op": "detect", "threshold": 0.3, "min_new": 1}, None)
    assert "identity_review" in out
    assert "confidentity" in out.lower()
    assert "precision" in out  # a real emerged candidate surfaced, no LLM call


def test_draft_op_files_emergent_proposal(env, monkeypatch) -> None:
    skill = env
    _ingest_aligned()
    monkeypatch.setattr(
        skill, "_call_llm",
        lambda prompt: "EMERGENT_CLAIM: I lean on adversarial verification before declaring done.\n"
                       "RATIONALE: Repeatedly observed across sessions; an earned posture, not a redeclaration.",
    )
    out = skill.run({"op": "draft", "threshold": 0.3, "min_new": 1}, None)
    assert "queued as proposal" in out

    from core import proposals
    items = proposals.list_proposals()
    assert len(items) == 1
    assert items[0]["target"] == "identity.md"
    assert items[0]["section"] == "Emergent"
    assert "adversarial verification" in items[0]["proposed_text"]

    from core.acatalepsy import canonical_log
    kinds = [e.kind for e in canonical_log.read_since(0, limit=500)]
    assert "identity_amendment_proposed" in kinds


def test_draft_op_no_candidates_proposes_nothing(env, monkeypatch) -> None:
    skill = env
    from core.acu_store import ACUStore
    s = ACUStore()
    try:
        s.ingest("weather | is | sunny", source="model")
    finally:
        s.close()
    monkeypatch.setattr(skill, "_call_llm", lambda prompt: "EMERGENT_CLAIM: x\nRATIONALE: y")
    out = skill.run({"op": "draft", "threshold": 0.3, "min_new": 1}, None)
    assert "no emergent" in out.lower() or "nothing" in out.lower()

    from core import proposals
    assert proposals.list_proposals() == []
