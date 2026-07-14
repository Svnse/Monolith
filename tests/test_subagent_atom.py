from __future__ import annotations

from pathlib import Path

import pytest

from core import subagent as sa
from core.skill_runtime import ToolExecutionContext, L1_PRINCIPAL_TOOLS


def _l1(tmp_path):
    return ToolExecutionContext(archive_dir=Path(tmp_path), level=1,
                               allowed_tools=L1_PRINCIPAL_TOOLS,
                               parent_turn_id="root-turn")


def test_run_subagent_single_inference_returns_fenced_result(tmp_path, monkeypatch):
    calls = {"n": 0}
    def fake_gen(cfg, messages, llm_config=None, *, thinking_enabled=None, should_cancel=None):
        calls["n"] += 1
        return ("child answer", "")
    monkeypatch.setattr(sa, "generate_sync_parts_from_config", fake_gen)
    res = sa.run_subagent(
        [{"role": "user", "content": "do x"}], {"api_base": "b", "api_model": "m"},
        level=2, frame="plane:research", parent_turn_id="root-turn",
        allowed_tools=ToolExecutionContext(archive_dir=Path(tmp_path)).allowed_tools,
        should_cancel=None, max_followups=0, spawn_budget=None)
    assert res.ok is True
    assert calls["n"] == 1                      # ONE inference by default (not a loop)
    assert "child answer" in res.text
    assert res.fenced.startswith("[SUBAGENT_RESULT level=2")
    assert res.fenced.rstrip().endswith("[/SUBAGENT_RESULT]")
    assert res.child_turn_id  # a freshly minted uuid hex


def test_run_subagent_refuses_when_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "generate_sync_parts_from_config",
                        lambda *a, **k: ("x", ""))
    res = sa.run_subagent(
        [{"role": "user", "content": "x"}], {"api_base": "b", "api_model": "m"},
        level=3, frame="leaf", parent_turn_id="root-turn",
        allowed_tools=frozenset(), should_cancel=None, max_followups=0,
        spawn_budget=None, is_busy=lambda: True)
    assert res.ok is False
    assert res.halt_reason == "busy"
    assert "[subagent: generator busy]" in res.fenced


def test_run_subagent_writes_child_frame_and_allow_fault(tmp_path, monkeypatch):
    import core.turn_trace as tt
    frames, faults = [], []
    monkeypatch.setattr(sa, "generate_sync_parts_from_config",
                        lambda *a, **k: ("ans", ""))
    monkeypatch.setattr(tt, "record_frame", lambda rec: frames.append(rec))
    monkeypatch.setattr(tt, "record_fault", lambda rec: faults.append(rec))
    sa.run_subagent(
        [{"role": "user", "content": "x"}], {"api_base": "b", "api_model": "m"},
        level=2, frame="skill:write", parent_turn_id="root-turn",
        allowed_tools=frozenset(), should_cancel=None, max_followups=0, spawn_budget=None)
    assert len(frames) == 1
    assert frames[0].parent_turn_id == "root-turn"
    assert frames[0].backend == "subagent"
    assert frames[0].engine_key == "subagent:L2"
    kinds = {getattr(f, "event_kind", "") for f in faults}
    assert "subagent_spawned" in kinds
    assert "subagent_folded" in kinds


def test_subagent_engine_adapter_returns_str(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "generate_sync_parts_from_config",
                        lambda *a, **k: ("adapter answer", ""))
    out = sa.subagent_engine(
        [{"role": "user", "content": "x"}],
        {"api_base": "b", "api_model": "m",
         "_subagent": {"level": 3, "frame": "leaf", "parent_turn_id": "root-turn",
                       "allowed_tools": [], "max_followups": 0}})
    assert isinstance(out, str)
    assert "adapter answer" in out


def test_worker_loop_runs_allowed_tool_through_the_gate(tmp_path, monkeypatch):
    """An L2 worker with max_followups=1: hop 1 emits a read_file call (allowed),
    the atom runs it through the gate, folds the result, then hop 2 answers."""
    from core.skill_runtime import L2_WORKER_TOOLS
    f = Path(tmp_path) / "doc.txt"; f.write_text("PAYLOAD", encoding="utf-8")
    hops = {"n": 0}
    folded = []
    call_json = '<tool_call>{"tool":"read_file","path":"%s"}</tool_call>' % str(f).replace("\\", "/")
    def fake_gen(cfg, messages, llm_config=None, *, thinking_enabled=None, should_cancel=None):
        hops["n"] += 1
        if hops["n"] == 2 and messages:          # capture what got folded in as the user turn
            folded.append(messages[-1]["content"])
        return (call_json, "") if hops["n"] == 1 else ("final synthesis", "")
    monkeypatch.setattr(sa, "generate_sync_parts_from_config", fake_gen)
    res = sa.run_subagent(
        [{"role": "user", "content": "read the doc"}], {"api_base": "b", "api_model": "m"},
        level=2, frame="skill:read", parent_turn_id="root-turn",
        allowed_tools=L2_WORKER_TOOLS, should_cancel=None, max_followups=1,
        spawn_budget=None)
    assert hops["n"] == 2          # generate -> run tool -> regenerate
    assert res.tools_run == 1
    assert folded and "PAYLOAD" in folded[0]  # the tool output reached the loop (folded turn)
    assert res.text == "final synthesis"


def test_worker_loop_denies_disallowed_tool_inside_the_loop(tmp_path, monkeypatch):
    """An L3 leaf that emits a write_file call: the gate denies it (capability)
    INSIDE the atom's loop -- the ladder is enforced on child traffic, not just L1."""
    import core.skill_runtime as srt
    write_json = '<tool_call>{"tool":"write_file","path":"x","content":"y"}</tool_call>'
    seen = {"n": 0}
    folded = []
    def fake_gen(cfg, messages, llm_config=None, *, thinking_enabled=None, should_cancel=None):
        seen["n"] += 1
        if seen["n"] == 2 and messages:
            folded.append(messages[-1]["content"])
        return (write_json, "") if seen["n"] == 1 else ("done", "")
    monkeypatch.setattr(sa, "generate_sync_parts_from_config", fake_gen)
    res = sa.run_subagent(
        [{"role": "user", "content": "try to write"}], {"api_base": "b", "api_model": "m"},
        level=3, frame="leaf", parent_turn_id="root-turn",
        allowed_tools=srt.L3_LEAF_TOOLS, should_cancel=None, max_followups=1,
        spawn_budget=None)
    assert res.tools_run == 1                     # the call was attempted (and denied)
    assert seen["n"] == 2                         # loop continued after the denial
    assert folded and "denied" in folded[0].lower()
    assert "write_file" in folded[0]


from core.skill_runtime import (
    execute_tool_call_enveloped, execute_spawn_subagent, derive_child_context)


def test_spawn_subagent_headless_inline_returns_fenced(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "generate_sync_parts_from_config",
                        lambda *a, **k: ("leaf answer", ""))
    # L1 ctx with NO on_spawn_subagent => headless inline path.
    env = execute_tool_call_enveloped(
        {"tool": "spawn_subagent", "level": 3, "frame": "leaf",
         "prompt": "summarize", "messages": []},
        _l1(tmp_path))
    assert env.ok is True
    assert "[SUBAGENT_RESULT level=3" in env.text


def test_llm_call_still_single_shots_via_atom(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "generate_sync_parts_from_config",
                        lambda *a, **k: ("summary text", ""))
    env = execute_tool_call_enveloped(
        {"tool": "llm_call", "prompt": "summarize this"}, _l1(tmp_path))
    assert env.ok is True
    assert "summary text" in env.text


def test_llm_call_has_no_direct_sync_bridge_call():
    """Rule-6 deletion: execute_llm_call must NOT call the inference primitive directly."""
    import inspect
    import core.skill_runtime as srt
    src = inspect.getsource(srt.execute_llm_call)
    assert "generate_sync_from_config" not in src


def test_llm_call_busy_when_generation_lock_held(tmp_path, monkeypatch):
    """Advisor-flagged behavior change: llm_call now goes through the atom's
    non-blocking lock. If the lock is held, it returns a graceful 'busy' result --
    never blocks, never crashes."""
    import core.turn_trace as tt
    monkeypatch.setattr(tt, "record_fault", lambda rec: None)
    monkeypatch.setattr(tt, "record_frame", lambda rec: None)
    from core.generation import generation_lock
    got = generation_lock.acquire(blocking=False)
    assert got is True
    try:
        env = execute_tool_call_enveloped(
            {"tool": "llm_call", "prompt": "hi"}, _l1(tmp_path))
    finally:
        generation_lock.release()
    assert "busy" in env.text.lower()


def test_is_engine_busy_reads_world_state_running():
    """INV-C Arm 2 helper + chat.py import-cleanliness check."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from ui.pages.chat import _engine_is_busy
    busy_ws = type("WS", (), {"snapshot": lambda self: {"engines": {"llm": {"status": "RUNNING"}}}})()
    idle_ws = type("WS", (), {"snapshot": lambda self: {"engines": {"llm": {"status": "idle"}}}})()
    assert _engine_is_busy(busy_ws) is True
    assert _engine_is_busy(idle_ws) is False
    assert _engine_is_busy(None) is False


def test_governance_read_apis_reconstruct_tree(tmp_path, monkeypatch):
    """Seed a 3-level tree + 1 deny in an isolated temp DB; reconstruct via the CTE."""
    import core.turn_trace as tt
    import datetime as _dt
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)  # ensure seed writes persist

    def _ts():
        return _dt.datetime.now(_dt.timezone.utc).isoformat()

    tt.set_db_path(Path(tmp_path) / "turn_trace.sqlite3")
    try:
        def frame(turn_id, parent, level):
            tt.record_frame(tt.FrameTraceRecord(
                turn_id=turn_id, captured_at=_ts(),
                backend="subagent" if parent else "llm",
                engine_key=f"subagent:L{level}" if parent else "llm", gen_id=0,
                final_messages=tuple(), system_prompt_chars=0, user_prompt_chars=0,
                total_chars=10, parent_turn_id=parent,
                metadata={"kind": "subagent", "level": f"L{level}"}))
        frame("L1", None, 1)
        frame("L2", "L1", 2)
        frame("L3", "L2", 3)
        # one denied spawn keyed on the caller (L1) -- no child frame exists for it
        tt.record_fault(tt.FaultTraceRecord(
            turn_id="L1", parent_turn_id="L1", seq=0, emitted_at=_ts(),
            event_kind="spawn_denied", source_kind="policy", source_name="subagent_gate",
            authority_tier="dispatch", fault_kind="spawn_denied", severity="hard",
            payload={"level": 2, "requested_level": 2, "deny_reason": "L2 may spawn L3 only"}))

        assert tt.latest_governance_root() == "L1"
        children = tt.list_child_frames("L1")
        assert {c.turn_id for c in children} == {"L2"}  # direct children only
        events = tt.list_governance_events("L1")
        assert "spawn_denied" in {getattr(e, "event_kind", "") for e in events}
    finally:
        tt.set_db_path(None)


def test_run_subagent_exposes_native_thinking(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "generate_sync_parts_from_config",
                        lambda *a, **k: ("the answer", "native chain of thought"))
    res = sa.run_subagent(
        [{"role": "user", "content": "x"}], {"api_base": "b", "api_model": "m"},
        level=3, frame="thinkpad:c0", parent_turn_id="root-turn",
        allowed_tools=frozenset(), should_cancel=None, max_followups=0, spawn_budget=None)
    assert res.ok is True
    assert res.thinking == "native chain of thought"   # not discarded
    assert res.text == "the answer"                    # text unchanged


def test_run_subagent_threads_cancel_into_generation_and_halts_on_cancel(tmp_path, monkeypatch):
    # Enforce-stop: run_subagent must (1) pass its should_cancel down to the streaming sync call
    # so an in-flight generation can break mid-stream, and (2) if a stop is observed around that
    # call, return halt_reason="cancelled" with NO leaked partial text — so the workshop run halts
    # rather than treating the partial as a finished block output.
    state = {"cancelled": False, "saw_should_cancel": False}

    def fake_gen(cfg, messages, llm_config=None, *, thinking_enabled=None, should_cancel=None):
        state["saw_should_cancel"] = should_cancel is not None  # the cancel callback was threaded in
        state["cancelled"] = True                               # simulate: stop pressed mid-stream
        return ("partial answer so far", "")                    # the stream returned a partial

    monkeypatch.setattr(sa, "generate_sync_parts_from_config", fake_gen)
    res = sa.run_subagent(
        [{"role": "user", "content": "x"}], {"api_base": "b", "api_model": "m"},
        level=3, frame="leaf", parent_turn_id="root-turn",
        allowed_tools=frozenset(), should_cancel=lambda: state["cancelled"],
        max_followups=0, spawn_budget=None)

    assert state["saw_should_cancel"] is True      # should_cancel reached the sync streaming call
    assert res.halt_reason == "cancelled"
    assert res.text == ""                          # partial NOT surfaced as the block result
