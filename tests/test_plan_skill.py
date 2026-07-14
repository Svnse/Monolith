from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_skill():
    p = Path(__file__).parent.parent / "skills" / "plan" / "executor.py"
    spec = importlib.util.spec_from_file_location("plan_exec_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LLM_OK = (
    "PLAN: build the widget\n"
    "STEP: read | spec.md | depends: none\n"
    "STEP: draft | widget.py | depends: 1\n"
    "STEP: test | widget.py | depends: 2\n"
)


@pytest.fixture
def env(monkeypatch, tmp_path):
    from core import plans, planner
    plans.set_db_path(tmp_path / "turn_trace.sqlite3")
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, canonical_log
    schema.migrate()
    tl = getattr(canonical_log, "_tl", None)
    if tl is not None:
        for a in ("writer_conn", "reader_conn"):
            if hasattr(tl, a):
                delattr(tl, a)
    monkeypatch.setattr(planner, "_call_llm", lambda prompt: _LLM_OK)
    monkeypatch.setattr(planner, "_bearing_goal", lambda: "ship V0")
    monkeypatch.setattr(planner, "_curiosity_pulls", lambda: [
        {"canonical": "monolith | values | precision", "pull_strength": 0.33}])
    yield _load_skill()
    plans.set_db_path(None)


def test_decompose_then_show(env) -> None:
    skill = env
    out = skill.run({"op": "decompose", "goal": "build the widget"}, None)
    assert "plan" in out.lower()
    assert "read" in out and "draft" in out and "test" in out

    shown = skill.run({"op": "show"}, None)
    assert "build the widget" in shown
    assert "0/3" in shown or "next" in shown.lower()


def test_candidates_lists_bearing_and_curiosity(env) -> None:
    skill = env
    out = skill.run({"op": "candidates"}, None)
    assert "ship V0" in out
    assert "precision" in out
    assert "bearing" in out.lower() and "curiosity" in out.lower()


def test_mark_advances_progress(env) -> None:
    skill = env
    skill.run({"op": "decompose", "goal": "build the widget"}, None)
    out = skill.run({"op": "mark", "step": 1, "status": "done"}, None)
    assert "1/3" in out or "done" in out.lower()
    shown = skill.run({"op": "show"}, None)
    assert "draft" in shown  # step 2 is now the ready frontier


def test_show_no_active_plan(env) -> None:
    skill = env
    out = skill.run({"op": "show"}, None)
    assert "no active plan" in out.lower()
