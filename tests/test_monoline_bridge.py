from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

import core.turn_trace as tt
from engine import monoline_bridge as br
from tests._monoline_requirement import requires_monoline


pytestmark = requires_monoline


def _echo_blueprint() -> dict:
    # NO "schema_version" key (PROVEN, do not add one). _looks_like_preset() keys on
    # schema_version+blocks+connections; WITH it, load_workflow routes to the strict
    # PipelinePreset.from_dict, which REQUIRES a non-empty "id" and raises "World file
    # missing id". WITHOUT it, load_workflow routes to the lenient build_preset_from_blueprint,
    # which synthesizes ids and actually runs the world. Matches the working Task-2 blueprint.
    return {
        "name": "Echo",
        "blocks": [
            {"id": "input", "kind": "port", "config": {"direction": "in", "label": "request", "source": "user_input"}},
            {"id": "assistant", "kind": "llm"},
            {"id": "output", "kind": "port", "config": {"direction": "out", "label": "response", "source": "subgraph"}},
        ],
        "connections": [["input.value", "assistant.prompt"], ["assistant.response", "output.value"]],
    }


@pytest.fixture
def echo_world(tmp_path):
    p = Path(tmp_path) / "echo.monoline"
    p.write_text(json.dumps(_echo_blueprint()), encoding="utf-8")
    from core.workflow_registry import Workflow
    return Workflow(id="echo", name="Echo", description="", kind="monoline", source_path=p)


def test_make_engine_func_native_branch_runs(echo_world, monkeypatch):
    # Stub the native engine_call so we never need a real model.
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: "NATIVE_OK")
    fn = br.make_engine_func(parent_turn_id="T1", spawn_budget=None,
                             should_cancel=lambda: False, is_busy=lambda: False,
                             allow_egress=False)
    out = fn([{"role": "user", "content": "hi"}], {"provider": "local"})
    assert out == "NATIVE_OK"


def test_make_engine_func_egress_rejected_without_optin(monkeypatch):
    fn = br.make_engine_func(parent_turn_id="T1", spawn_budget=None,
                             should_cancel=lambda: False, is_busy=lambda: False,
                             allow_egress=False)
    with pytest.raises(Exception):
        fn([{"role": "user", "content": "hi"}], {"provider": "api"})


def test_make_engine_func_monolith_branch_uses_atom(monkeypatch):
    import core.subagent as sub
    seen = {}

    def fake_run(messages, config, **kw):
        seen.update(kw)
        return sub.SubagentResult(ok=True, text="ATOM_TEXT", fenced="...",
                                  child_turn_id="c", level=kw.get("level", 3), tools_run=0)
    monkeypatch.setattr(sub, "run_subagent", fake_run)
    fn = br.make_engine_func(parent_turn_id="ROOT", spawn_budget=None,
                             should_cancel=lambda: False, is_busy=lambda: False,
                             allow_egress=False)
    out = fn([{"role": "user", "content": "hi"}], {"provider": "monolith", "label": "blk"})
    assert out == "ATOM_TEXT"
    assert seen["level"] == 3
    assert seen["parent_turn_id"] == "ROOT"


def test_monolith_branch_generates_with_monolith_runtime_config(monkeypatch):
    # Regression (2026-06-14): the monolith provider must hand run_subagent MONOLITH's runtime
    # config (api_base/api_model), NOT the Monoline block config (which has none) -- else
    # generate_sync_parts_from_config raises 'Missing api_base or api_model' and the block
    # yields EMPTY output (the live 'Untitled World' bug). run_subagent -> generate is API-only.
    import core.subagent as sub
    seen = {}

    def fake_run(messages, config, **kw):
        seen["config"] = config
        return sub.SubagentResult(ok=True, text="ATOM_OK", fenced="[r]",
                                  child_turn_id="c", level=kw.get("level", 3), tools_run=0)
    monkeypatch.setattr(sub, "run_subagent", fake_run)
    monkeypatch.setattr(br, "load_config",
                        lambda: {"api_base": "https://m", "api_model": "deepseek", "api_key": "k"})
    fn = br.make_engine_func(parent_turn_id="ROOT", spawn_budget=None,
                             should_cancel=lambda: False, is_busy=lambda: False)
    block_cfg = {"provider": "monolith", "label": "Assistant",
                 "api_url": "", "model": "", "model_path": "", "temperature": 0.7}
    out = fn([{"role": "user", "content": "hEY"}], block_cfg)
    assert out == "ATOM_OK"
    assert seen["config"].get("api_base") == "https://m"     # used Monolith's runtime config
    assert seen["config"].get("api_model") == "deepseek"


def test_make_engine_func_monolith_busy_retries_then_raises(monkeypatch):
    import core.subagent as sub
    calls = {"n": 0}

    def always_busy(messages, config, **kw):
        calls["n"] += 1
        return sub.SubagentResult(ok=False, text="", fenced="...", child_turn_id="c",
                                  level=3, tools_run=0, halt_reason="busy")
    monkeypatch.setattr(sub, "run_subagent", always_busy)
    fn = br.make_engine_func(parent_turn_id="ROOT", spawn_budget=None,
                             should_cancel=lambda: False, is_busy=lambda: False,
                             allow_egress=False, busy_retries=2, busy_backoff=0.0)
    with pytest.raises(Exception):
        fn([{"role": "user", "content": "hi"}], {"provider": "monolith"})
    assert calls["n"] == 3  # initial + 2 retries, then raise


def _echo_engine(messages, _cfg):
    # Echo the latest user content so output contains the input (a "NATIVE_OK" constant
    # would make the `"hi" in output` assertion below false -- the world really runs).
    for msg in reversed(messages):
        if str(msg.get("role", "")).lower() == "user":
            return f"echo:{msg.get('content', '')}"
    return "echo:"


def test_run_monoline_world_records_root_frame(echo_world, tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(Path(tmp_path) / "turn_trace.sqlite3")
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", _echo_engine)
    try:
        # parent_turn_id="" => dry-run model: the bridge records the run-root with a NULL
        # parent, so it IS a top-level governance root, discoverable via the monoline_block
        # fault-union in latest_governance_root (the echo flow uses the 'local' provider, so
        # there is NO atom child frame -- the run is found purely by its monoline_block faults).
        run = br.run_monoline_world(
            echo_world, user_input="hi", parent_turn_id="",
            spawn_budget=None, should_cancel=lambda: False, is_busy=lambda: False,
            on_step=None, should_stop=None)
        assert "hi" in run.result.output
        root = tt.latest_governance_root()
        assert root is not None
        events = tt.list_governance_events(root)
        # at least one monoline_block fault overlay was written
        assert any(getattr(e, "event_kind", "") == "monoline_block" for e in events)
    finally:
        tt.set_db_path(None)


def test_make_tool_func_passes_only_tool_specific_arg(monkeypatch):
    # Regression (2026-06-14): a strict tool schema (calculate wants 'expr') rejects an extra
    # generic 'input' field -> "[tool: invalid arguments ... unknown field(s): input]". The
    # adapter must pass ONLY the tool-specific arg for mapped tools, not also a generic 'input'.
    import core.skill_runtime as sr
    seen = {}
    monkeypatch.setattr(sr, "execute_tool_call",
                        lambda cmd, ctx: (seen.__setitem__("cmd", cmd) or "OK"))
    tool_func = br.make_tool_func(parent_turn_id="ROOT", should_cancel=lambda: False)
    out = tool_func("calculate", "2+2", {})
    assert out == "OK"
    assert seen["cmd"].get("expr") == "2+2"
    assert "input" not in seen["cmd"]      # the field calculate's strict schema rejects


def test_run_monoline_world_emits_run_event_stream(echo_world, tmp_path, monkeypatch):
    # Spec §3: the bridge emits a normalized RunEvent stream (run_started → block_finished* → run_finished).
    # Redirect the trace DB to tmp so this NEVER writes to the real store even if the flag is on in env.
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", _echo_engine)
    events: list = []
    try:
        br.run_monoline_world(
            echo_world, user_input="hi", parent_turn_id="",
            spawn_budget=None, should_cancel=lambda: False, is_busy=lambda: False,
            on_step=None, should_stop=None, on_event=events.append)
    finally:
        tt.set_db_path(None)
    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "RunStarted"
    assert "BlockFinished" in kinds
    assert kinds[-1] == "RunFinished"
    started = events[0]
    assert started.user_input == "hi"
    assert started.flow_id == "echo"
    assert any("->" in w for w in started.wires)        # wires read from the .monoline
    assert any(b.kind == "llm" for b in started.graph)  # graph carried
    finals = [e for e in events if type(e).__name__ == "BlockFinished"]
    assert any("hi" in " ".join(str(v) for v in (e.outputs or {}).values()) for e in finals)
    assert "hi" in (events[-1].output or "")


def test_block_output_and_timing_persisted_but_invisible_to_monosearch(echo_world, tmp_path, monkeypatch):
    # Spec §4: per-block output + precise timing land in the monoline_block PAYLOAD (so the
    # historical browser matches the live card) -- but stay invisible to the model-facing
    # MonoSearch adapter (FaultAdapter never reads payload; monoline_block is fault_kind=None).
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(Path(tmp_path) / "turn_trace.sqlite3")
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, _c: "SECRET_BLOCK_OUTPUT_XYZ")
    try:
        br.run_monoline_world(
            echo_world, user_input="hi", parent_turn_id="",
            spawn_budget=None, should_cancel=lambda: False, is_busy=lambda: False,
            on_step=None, should_stop=None)
        root = tt.latest_governance_root()
        blocks = [e for e in tt.list_governance_events(root)
                  if getattr(e, "event_kind", "") == "monoline_block"]
        assert blocks
        # persisted: per-port outputs + block_id + precise timing in payload
        assert any("SECRET_BLOCK_OUTPUT_XYZ" in str((e.payload or {}).get("outputs") or {})
                   for e in blocks)
        assert all("started_at" in (e.payload or {}) and "completed_at" in (e.payload or {})
                   and "block_id" in (e.payload or {}) for e in blocks)
        # invisible: the only model-facing fault reader never surfaces the output text
        from core.monosearch.adapters.faults import FaultAdapter
        ad = FaultAdapter()
        recs = ad.search("SECRET_BLOCK_OUTPUT_XYZ", {}, 200) + ad.list({}, 200)
        assert all("SECRET_BLOCK_OUTPUT_XYZ" not in (r.text or "") for r in recs)
    finally:
        tt.set_db_path(None)


def test_load_monoline_restores_monolith_core_and_path():
    # SAFETY NET for the sys.modules swap (the whole pivot rides on this). After a load,
    # Monolith's own core.* must be intact and the Monoline root must be OFF sys.path
    # (else Monoline's top-level ui/ shadows Monolith's on the next uncached import).
    import core.paths as cp_before
    m = br.load_monoline()
    assert hasattr(m["headless"], "run_workflow")
    import core.paths as cp_after
    import core.subagent as cs_after
    repo_root = Path(__file__).resolve().parents[1]
    assert Path(cp_after.__file__).resolve() == (repo_root / "core" / "paths.py").resolve()
    assert Path(cs_after.__file__).resolve() == (repo_root / "core" / "subagent.py").resolve()
    assert cp_after is cp_before                       # same Monolith module object
    assert not any(p.replace("\\", "/").rstrip("/").endswith("Project/Monoline")
                   for p in sys.path)                  # Monoline root cleaned off sys.path


def test_load_monoline_syspath_restored_even_if_root_preexisting(monkeypatch):
    # Regression: the swap inserts/removes the Monoline root UNCONDITIONALLY. Even when the
    # root is ALREADY on sys.path (and not first), a cold load must restore sys.path to EXACTLY
    # its pre-call state -- no lingering insert, no duplicate. (A buggy "only touch if absent"
    # would either skip cleanup or leave a dangling entry that shadows Monolith's top-level ui/.)
    monkeypatch.setattr(br, "_MONOLINE_CACHE", None)   # force a cold swap this call
    root = str(br.ensure_monoline_on_path())
    sys.path.append(root)                              # pre-seed: present but NOT index 0
    before = list(sys.path)
    try:
        m = br.load_monoline()
        assert hasattr(m["headless"], "run_workflow")
        assert sys.path == before                      # exact restoration: no dup, no leak
        assert sys.path.count(root) == 1
    finally:
        try:
            sys.path.remove(root)                      # undo our pre-seed
        except ValueError:
            pass


def test_validate_chat_workflow_rejects_blank_local_llm(tmp_path, monkeypatch):
    monkeypatch.delenv("MONOLINE_DEFAULT_LLM_PROVIDER", raising=False)
    m = br.load_monoline()
    preset = m["model"].build_blank_preset()
    p = Path(tmp_path) / "blank-local.monoline"
    p.write_text(json.dumps(preset.to_dict()), encoding="utf-8")
    from core.workflow_registry import Workflow
    wf = Workflow(id="blank-local", name="Blank Local", description="", kind="monoline", source_path=p)

    problem = br.validate_chat_workflow(wf)

    assert problem is not None
    assert "local provider requires a model_path" in problem


def test_validate_chat_workflow_accepts_monolith_llm(tmp_path):
    m = br.load_monoline()
    preset = m["model"].build_blank_preset()
    for block in preset.blocks:
        if block.kind == "llm":
            block.config["provider"] = "monolith"
    p = Path(tmp_path) / "monolith.monoline"
    p.write_text(json.dumps(preset.to_dict()), encoding="utf-8")
    from core.workflow_registry import Workflow
    wf = Workflow(id="monolith", name="Monolith", description="", kind="monoline", source_path=p)

    assert br.validate_chat_workflow(wf) is None


def test_monolith_launched_canvas_defaults_new_llm_blocks_to_monolith(monkeypatch):
    m = br.load_monoline()
    monkeypatch.setenv("MONOLINE_DEFAULT_LLM_PROVIDER", "monolith")

    block = m["model"].make_block("llm")

    assert block.config["provider"] == "monolith"


def test_open_create_canvas_sets_monolith_provider_env(monkeypatch):
    calls = {}

    class _Proc:
        pass

    def _fake_popen(args, *, cwd, env):
        calls.update(args=args, cwd=cwd, env=env)
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    parent = types.SimpleNamespace()

    br.open_create_canvas(parent)

    assert calls["env"]["MONOLINE_DEFAULT_LLM_PROVIDER"] == "monolith"
    assert getattr(parent, "_monoline_canvas_procs")
