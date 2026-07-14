"""ExpeditionRunner (MonoExplore V1) — orchestration tests.

These prove the LOOP orchestration with the three composed primitives mocked.
They do NOT prove the real primitives compose end-to-end: the V1 acceptance
criterion is one UNMOCKED tick against a live OpenAI-compat backend that runs a
real read_file and lands a real world-provenance ACU (apply stage, needs the app).
"""
import threading
import time

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from addons.system.bearing import store as bstore
    from core import monoexplore, plans
    monkeypatch.setattr(bstore, "_STORE_PATH", tmp_path / "bearing.json")
    monkeypatch.setattr(monoexplore, "_LEASH_PATH", tmp_path / "monoexplore.json")
    plans.set_db_path(tmp_path / "tt.sqlite3")
    yield
    plans.set_db_path(None)


def _patch_tick_externals(er, monkeypatch):
    monkeypatch.setattr(er, "load_config", lambda: {})
    monkeypatch.setattr(er, "finalize_assistant_turn", lambda **k: None)
    monkeypatch.setattr(er, "record_frame", lambda rec: None)


def _stub_self_context(runner, monkeypatch):
    # avoid real identity/bearing/curiosity reads + real canonical_log writes
    monkeypatch.setattr(runner, "_build_system_prompt", lambda catalog: "sys")
    monkeypatch.setattr(runner, "_log_tick", lambda *a, **k: None)


# ── one tick (COMPOSE core) ───────────────────────────────────────────


def test_one_tick_grounded_generates_dispatches_ingests(isolated, monkeypatch):
    from engine import expedition_runner as er
    _patch_tick_externals(er, monkeypatch)
    gens = iter([
        ("READ engine/llm.py", "I'm drawn to the engine"),  # hop 1: (output, thinking) — triggers a read
        ("<findings>\nengine_llm | defines | GeneratorWorker\n</findings>\nDone.", ""),  # hop 2: terminal + finding
    ])
    monkeypatch.setattr(er, "generate_sync_parts_from_config", lambda *a, **k: next(gens))
    monkeypatch.setattr(
        er, "extract_commands",
        lambda raw, strict=False: [{"tool": "read_file", "arguments": {"path": "engine/llm.py"}}] if "READ" in raw else [],
    )
    monkeypatch.setattr(er, "expand_calls", lambda cmd: [cmd])

    class _Env:
        ok = True
        call_id = "t1"
        text = "[read_file engine/llm.py] class GeneratorWorker"
        data = {}

    monkeypatch.setattr(er, "execute_tool_call_enveloped", lambda call, ctx: _Env())
    ingested = []
    monkeypatch.setattr(
        er, "_ingest_grounded",
        lambda findings, evidence_text: ingested.append((tuple(findings), evidence_text)) or len(findings),
    )

    runner = er.ExpeditionRunner()
    _stub_self_context(runner, monkeypatch)
    result = runner._run_one_tick(turn_id="exp_test")
    assert result["generated"] is True
    assert result["tools_run"] >= 1 and result["grounded"] is True
    assert ingested and ingested[0][1]  # real tool-output evidence text passed to the gate
    assert "engine_llm | defines | GeneratorWorker" in ingested[0][0]


def test_tick_noop_on_empty_generation(isolated, monkeypatch):
    from engine import expedition_runner as er
    _patch_tick_externals(er, monkeypatch)
    monkeypatch.setattr(er, "generate_sync_parts_from_config", lambda *a, **k: ("", ""))  # G1 degrade
    monkeypatch.setattr(er, "_ingest_grounded", lambda f, evidence_text: 0)
    runner = er.ExpeditionRunner()
    _stub_self_context(runner, monkeypatch)
    result = runner._run_one_tick(turn_id="exp_empty")
    assert result["generated"] is False
    assert result["tools_run"] == 0 and result["grounded"] is False


def test_extract_findings_salvages_unclosed_findings_block():
    from engine import expedition_runner as er

    text = (
        "Done.\n<findings>\n"
        "engine/llm.py | defines | GeneratorWorker\n"
        "not an atomic finding\n"
    )

    assert er._extract_findings(text) == ["engine/llm.py | defines | GeneratorWorker"]


def test_extract_findings_strips_malformed_close_tag():
    from engine import expedition_runner as er

    text = "<findings>\ncore/plans.py | stores | active plan </finding>\n"

    assert er._extract_findings(text) == ["core/plans.py | stores | active plan"]


# ── observed-ledger (A-durable: the memory loop) ──────────────────────


def test_tick_records_observations_to_plan_ledger(isolated, monkeypatch):
    from engine import expedition_runner as er
    from core import plans
    uid = plans.create_plan(goal="explore", source="monoexplore",
                            steps=[{"verb": "read", "target": "x", "depends_on": []}])
    _patch_tick_externals(er, monkeypatch)
    gens = iter([
        ("READ engine/llm.py", ""),
        ("<findings>\nengine_llm | defines | GeneratorWorker\n</findings>\nDone.", ""),
    ])
    monkeypatch.setattr(er, "generate_sync_parts_from_config", lambda *a, **k: next(gens))
    monkeypatch.setattr(
        er, "extract_commands",
        lambda raw, strict=False: [{"tool": "read_file", "arguments": {"path": "engine/llm.py"}}] if "READ" in raw else [],
    )
    monkeypatch.setattr(er, "expand_calls", lambda cmd: [cmd])

    class _Env:
        ok = True
        call_id = "t1"
        text = "[read_file engine/llm.py] class GeneratorWorker"
        data = {}

    monkeypatch.setattr(er, "execute_tool_call_enveloped", lambda call, ctx: _Env())
    monkeypatch.setattr(er, "_ingest_grounded", lambda findings, evidence_text: len(findings))

    runner = er.ExpeditionRunner()
    _stub_self_context(runner, monkeypatch)
    runner._run_one_tick(turn_id="exp_rec")

    obs = plans.get_observations(uid)
    assert "read_file engine/llm.py" in obs["visited"]      # the anti-re-listing signal
    assert any("GeneratorWorker" in f for f in obs["findings"])


def test_expedition_directive_injects_observed_so_far(isolated, monkeypatch):
    from engine import expedition_runner as er
    from core import plans
    uid = plans.create_plan(goal="explore the tree", source="monoexplore",
                            steps=[{"verb": "read", "target": "x", "depends_on": []}])
    plans.record_observations(uid, "exp_1",
                              visited=["list_files /root"], findings=["root | contains | 35 dirs"])
    runner = er.ExpeditionRunner()
    directive = runner._expedition_directive()
    assert "[OBSERVED SO FAR]" in directive
    assert "list_files /root" in directive
    assert "root | contains | 35 dirs" in directive
    assert "Take ONE grounded move now." in directive


def test_expedition_directive_omits_observed_when_empty(isolated, monkeypatch):
    from engine import expedition_runner as er
    from core import plans
    plans.create_plan(goal="fresh", source="monoexplore",
                      steps=[{"verb": "read", "target": "x", "depends_on": []}])
    runner = er.ExpeditionRunner()
    directive = runner._expedition_directive()
    assert "[OBSERVED SO FAR]" not in directive
    assert "Take ONE grounded move now." in directive


# ── STOP (asymmetric) ─────────────────────────────────────────────────


def test_stop_flag_propagates_to_ctx(isolated):
    from engine import expedition_runner as er
    runner = er.ExpeditionRunner()
    assert runner._ctx.should_cancel() is False
    runner.stop()
    assert runner._ctx.should_cancel() is True


def test_stop_halts_tool_loop(isolated, monkeypatch):
    from engine import expedition_runner as er
    _patch_tick_externals(er, monkeypatch)
    monkeypatch.setattr(er, "generate_sync_parts_from_config", lambda *a, **k: ("READ x", ""))
    monkeypatch.setattr(er, "extract_commands", lambda raw, strict=False: [{"tool": "read_file", "arguments": {}}])
    monkeypatch.setattr(er, "expand_calls", lambda cmd: [cmd])
    monkeypatch.setattr(er, "_ingest_grounded", lambda f, evidence_text: 0)
    runner = er.ExpeditionRunner()
    _stub_self_context(runner, monkeypatch)

    def _exec(call, ctx):
        runner._stop = True  # STOP arrives during dispatch
        class _E:
            ok = True
            call_id = "t"
            text = ""
            data = {}
        return _E()

    monkeypatch.setattr(er, "execute_tool_call_enveloped", _exec)
    result = runner._run_one_tick(turn_id="exp_stop")
    assert result["tools_run"] <= 1  # loop did not re-dispatch generation after stop


# ── daemon loop: refuse-goalless / RED-halt / budget ──────────────────


def test_stop_wakes_interval_sleep_and_marks_stopped(isolated, monkeypatch):
    from engine import expedition_runner as er
    from core import monoexplore

    monkeypatch.setattr(monoexplore, "coherence_report", lambda: {"verdict": "GREEN", "reason": "ok", "dims": {}})
    runner = er.ExpeditionRunner()
    runner.set_tick_interval(30)
    ticked = threading.Event()

    def _tick(*, turn_id):
        ticked.set()

    monkeypatch.setattr(runner, "_run_one_tick", _tick)
    thread = threading.Thread(target=runner._run_loop, kwargs={"max_ticks": 2})
    thread.start()
    assert ticked.wait(1.0)

    started = time.monotonic()
    runner.stop(timeout=1.0)
    thread.join(1.0)

    assert not thread.is_alive()
    assert time.monotonic() - started < 1.0
    assert runner.status == "stopped"
    assert runner._ctx.should_cancel() is True


def test_pause_wakes_interval_sleep_and_marks_paused(isolated, monkeypatch):
    from engine import expedition_runner as er
    from core import monoexplore

    monkeypatch.setattr(monoexplore, "coherence_report", lambda: {"verdict": "GREEN", "reason": "ok", "dims": {}})
    runner = er.ExpeditionRunner()
    runner.set_tick_interval(30)
    ticked = threading.Event()
    ticks = []

    def _tick(*, turn_id):
        ticks.append(turn_id)
        ticked.set()

    monkeypatch.setattr(runner, "_run_one_tick", _tick)
    thread = threading.Thread(target=runner._run_loop, kwargs={"max_ticks": 2})
    thread.start()
    assert ticked.wait(1.0)

    started = time.monotonic()
    runner.pause()
    thread.join(1.0)

    assert not thread.is_alive()
    assert time.monotonic() - started < 1.0
    assert len(ticks) == 1
    assert runner.status == "paused"
    assert runner._ctx.should_cancel() is False


def test_start_can_clear_prior_thinking(isolated, monkeypatch):
    from engine import expedition_runner as er

    runner = er.ExpeditionRunner()
    runner._push_thinking("exp_old", "old chain", "old output")
    monkeypatch.setattr(runner, "_bootstrap_and_run", lambda: None)

    assert runner.start("new goal", clear_thinking=True) is True
    runner._thread.join(1.0)

    assert runner.snapshot()["thinking"] == []


def test_bootstrap_no_plan_when_nothing_decomposable(isolated, monkeypatch):
    from engine import expedition_runner as er
    from core import monoexplore
    monkeypatch.setattr(monoexplore, "start_expedition", lambda goal, force=False: None)  # decompose fails
    runner = er.ExpeditionRunner()
    runner._pending_goal = "x"
    runner._bootstrap_and_run()  # synchronous — no UI-thread LLM call
    assert runner.status == "no-plan"


def test_bootstrap_runs_loop_when_plan_ready(isolated, monkeypatch):
    from engine import expedition_runner as er
    from core import monoexplore, plans
    monkeypatch.setattr(monoexplore, "start_expedition",
                        lambda goal, force=False: {"plan_uid": "p1", "goal": goal or "auto", "source": "x"})
    monkeypatch.setattr(plans, "get_active_plan", lambda: {"plan_uid": "p1", "goal": "g", "steps": []})
    runner = er.ExpeditionRunner()
    ran = {}
    monkeypatch.setattr(runner, "_run_loop", lambda *, max_ticks: ran.update(max_ticks=max_ticks))
    runner._pending_goal = "explore engine"
    runner._bootstrap_and_run()
    assert ran.get("max_ticks", 0) >= 1


def test_red_coherence_is_diagnostic_not_halt(isolated, monkeypatch):
    # RED must NOT block generation or halt — that deadlocks (RED -> no explore ->
    # nothing grounds -> stays RED). It keeps exploring; RED is logged only.
    from engine import expedition_runner as er
    from core import monoexplore
    monkeypatch.setattr(monoexplore, "coherence_report", lambda: {"verdict": "RED", "reason": "drift", "dims": {}})
    runner = er.ExpeditionRunner()
    runner.set_tick_interval(0)
    ran = []
    monkeypatch.setattr(runner, "_run_one_tick", lambda *, turn_id: ran.append(turn_id))
    runner._run_loop(max_ticks=3)
    assert len(ran) == 3 and runner.status == "idle"  # kept exploring through RED; no false halt


def test_loop_runs_budget_then_idles(isolated, monkeypatch):
    from engine import expedition_runner as er
    from core import monoexplore
    monkeypatch.setattr(monoexplore, "coherence_report", lambda: {"verdict": "GREEN", "reason": "ok", "dims": {}})
    runner = er.ExpeditionRunner()
    runner.set_tick_interval(0)
    n = []
    monkeypatch.setattr(runner, "_run_one_tick", lambda *, turn_id: n.append(1))
    runner._run_loop(max_ticks=3)
    assert len(n) == 3 and runner.status == "idle"


def test_system_prompt_tells_model_the_parseable_envelope(isolated, monkeypatch):
    # Guards the production-dead-loop bug: the model must be told the exact wire
    # format cmd_parser.extract_commands accepts, or no tick ever grounds.
    from engine import expedition_runner as er
    runner = er.ExpeditionRunner()
    monkeypatch.setattr(runner, "_identity_block", lambda: "")
    monkeypatch.setattr(runner, "_bearing_block", lambda: "")
    monkeypatch.setattr(runner, "_desire_block", lambda: "")
    p = runner._build_system_prompt("- read_file: read a file")
    assert "<tool_call>" in p and '"name":"read_file"' in p   # the format the parser accepts
    assert "<findings>" in p and "entity | relation | entity" in p
    assert "read_file" in p                                    # the read-only catalog


def test_snapshot_exposes_live_telemetry(isolated):
    from engine import expedition_runner as er
    runner = er.ExpeditionRunner()
    runner._tokens = 1234
    runner._push_thinking("exp_1", "I am reasoning about the verifier", "I read verifier.py")
    runner._push_activity("tick 1 · 1 tool(s) · +1 world · 1 finding(s)")
    snap = runner.snapshot()
    assert snap["tokens"] == 1234
    assert snap["thinking"][-1]["thinking"] == "I am reasoning about the verifier"
    assert snap["activity"][-1].startswith("tick 1")
    assert "verdict" in snap["coherence"] and snap["status"] == "idle"
    assert set(snap) >= {"goal", "next_move", "tools_total", "world_acus", "fault_streak", "referents", "last_lesson", "flag_on"}


def test_get_runner_is_singleton(isolated):
    from engine import expedition_runner as er
    er._runner_singleton = None
    assert er.get_runner() is er.get_runner()
