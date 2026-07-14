"""MonoExplore V0 — deterministic core tests (no LLM in the asserted paths).

Isolation: the `bearing_path` fixture redirects both the bearing store and the
plan store (turn_trace.sqlite3) to tmp, so coherence reads can't go flaky on the
dev machine's real state.
"""
import importlib.util
import pathlib

import pytest

from addons.system.bearing import store as bstore
from addons.system.bearing.schema import Bearing, Referent


@pytest.fixture
def bearing_path(tmp_path, monkeypatch):
    from core import plans
    monkeypatch.setattr(bstore, "_STORE_PATH", tmp_path / "bearing.json")
    plans.set_db_path(tmp_path / "tt.sqlite3")  # isolate get_active_plan from real turn_trace
    yield
    plans.set_db_path(None)


def _set_bearing(active_goal="", trajectory="", referents=()):
    bstore.set_bearing(
        Bearing(active_goal=active_goal, trajectory=trajectory, referents=tuple(referents))
    )


# ── coherence signal (keystone) ───────────────────────────────────────


def test_coherence_red_when_referents_ungrounded(bearing_path):
    from core import monoexplore
    _set_bearing(
        active_goal="map the engine",
        trajectory="reading engine modules",
        referents=tuple(
            Referent(name=f"claim {i}", kind="claim", status="inferred") for i in range(4)
        ),
    )
    rep = monoexplore.coherence_report()
    assert rep["verdict"] == "RED"
    assert "ground" in rep["reason"].lower()


def test_coherence_green_when_grounded_and_on_trajectory(bearing_path):
    from core import monoexplore
    _set_bearing(
        active_goal="map engine modules",
        trajectory="reading engine turn_pipeline modules",
        referents=(
            Referent(name="engine/turn_pipeline.py", kind="file", status="observed"),
            Referent(name="engine/llm.py", kind="file", status="observed"),
        ),
    )
    rep = monoexplore.coherence_report()
    assert rep["verdict"] == "GREEN"


def test_coherence_yellow_on_trajectory_drift(bearing_path):
    from core import monoexplore
    _set_bearing(
        active_goal="map the engine",
        trajectory="reading engine modules",
        referents=(
            Referent(name="recipes/cake.txt", kind="file", status="observed"),
            Referent(name="photos/holiday.jpg", kind="file", status="observed"),
        ),
    )
    rep = monoexplore.coherence_report()
    assert rep["verdict"] == "YELLOW"
    assert rep["dims"]["grounding_ratio"] == 1.0  # grounded, but off-trajectory


# ── goal-seeding + expedition lifecycle ───────────────────────────────


def test_seed_goal_explicit():
    from core import monoexplore
    assert monoexplore.seed_goal("explore the engine") == {
        "goal": "explore the engine",
        "source": "explicit",
    }


def test_seed_goal_falls_back_to_default_when_no_candidates(monkeypatch):
    from core import monoexplore, planner
    monkeypatch.setattr(planner, "propose_candidates", lambda: [])
    seed = monoexplore.seed_goal("")
    assert seed["source"] == "default"
    assert "filesystem" in seed["goal"].lower()


def test_seed_goal_uses_top_candidate(monkeypatch):
    from core import monoexplore, planner
    monkeypatch.setattr(
        planner,
        "propose_candidates",
        lambda: [{"goal": "understand acatalepsy", "source": "curiosity", "basis": "pull 0.9"}],
    )
    assert monoexplore.seed_goal("") == {"goal": "understand acatalepsy", "source": "curiosity"}


def test_start_expedition_decomposes_and_activates(tmp_path, monkeypatch):
    from core import monoexplore, planner, plans
    plans.set_db_path(tmp_path / "tt.sqlite3")
    try:
        # decompose makes ONE LLM call — monkeypatch it like tests/test_planner.py.
        monkeypatch.setattr(
            planner, "_call_llm", lambda prompt: "PLAN: x\nSTEP: read | engine/ | depends: none\n"
        )
        exp = monoexplore.start_expedition("map the engine", force=True)
        assert exp is not None and exp["source"] == "explicit"
        active = plans.get_active_plan()
        assert active["plan_uid"] == exp["plan_uid"] and active["status"] == "active"
    finally:
        plans.set_db_path(None)


def test_dark_by_default(monkeypatch, tmp_path):
    from core import monoexplore, plans
    monkeypatch.delenv("MONOLITH_MONOEXPLORE_V1", raising=False)
    plans.set_db_path(tmp_path / "tt.sqlite3")
    try:
        assert monoexplore.flag_enabled() is False
        assert monoexplore.start_expedition("x") is None  # no force + flag off -> no-op
    finally:
        plans.set_db_path(None)


# ── skill dispatch surface ────────────────────────────────────────────


def test_skill_status_and_coherence_ops(bearing_path):
    p = (
        pathlib.Path(__file__).resolve().parent.parent
        / "skills" / "monoexplore" / "executor.py"
    )
    spec = importlib.util.spec_from_file_location("monoexplore_exec", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.run({"op": "coherence"}, None)
    assert "monoexplore" in out.lower()
    assert any(v in out for v in ("GREEN", "YELLOW", "RED"))
    assert mod.run({"op": "bogus"}, None).startswith("[monoexplore: unknown op")


def test_skill_registers_via_frontmatter():
    """The wiring the executor-loading tests bypass: SKILL.md frontmatter must
    parse so core.skill_registry actually surfaces monoexplore as a tool."""
    from core import skill_registry
    skill_registry.clear_skill_cache()
    spec = skill_registry.get_tool("monoexplore")
    assert spec is not None and spec.name == "monoexplore"
    assert spec.description  # parser requires a non-empty description


# ── V1: read-only set + leash config ──────────────────────────────────


def test_read_only_set_excludes_writes():
    from core import monoexplore
    assert "run_command" not in monoexplore.READ_ONLY_SET
    assert "write_file" not in monoexplore.READ_ONLY_SET
    assert {"open_file", "read_file", "grep", "find_files", "list_files"} <= monoexplore.READ_ONLY_SET


def test_leash_roundtrip(tmp_path, monkeypatch):
    from core import monoexplore
    monkeypatch.setattr(monoexplore, "_LEASH_PATH", tmp_path / "monoexplore.json")
    monoexplore.save_leash({"tool_policy": "read_only", "max_ticks_per_wake": 3, "tick_interval_s": 15})
    assert monoexplore.load_leash()["max_ticks_per_wake"] == 3


def test_leash_defaults_when_absent(tmp_path, monkeypatch):
    from core import monoexplore
    monkeypatch.setattr(monoexplore, "_LEASH_PATH", tmp_path / "absent.json")
    leash = monoexplore.load_leash()
    assert leash["tool_policy"] == "read_only" and leash["max_ticks_per_wake"] == 6


# ── V1: grounded-finding ingest (the external-evidence gate) ───────────


def test_ingest_only_findings_that_overlap_evidence(monkeypatch):
    from core import monoexplore
    calls = []

    class _FakeStore:
        def ingest(self, claim, source="model"):
            calls.append((claim, source))
            return 1

        def close(self):
            pass

    monkeypatch.setattr(monoexplore, "_acu_store_factory", lambda: _FakeStore())
    # "GeneratorWorker" is in the evidence -> ingested; "saturn|orbits|sun" is not -> dropped.
    n = monoexplore.ingest_grounded_findings(
        ["engine_llm | defines | GeneratorWorker", "saturn | orbits | sun"],
        evidence_text="[read_file engine/llm.py] class GeneratorWorker: ...",
    )
    assert n == 1 and calls == [("engine_llm | defines | GeneratorWorker", "tool")]


def test_ingest_skips_when_no_evidence(monkeypatch):
    from core import monoexplore
    monkeypatch.setattr(
        monoexplore, "_acu_store_factory",
        lambda: (_ for _ in ()).throw(AssertionError("must not ingest with no evidence")),
    )
    assert monoexplore.ingest_grounded_findings(["a | r | b"], evidence_text="") == 0
