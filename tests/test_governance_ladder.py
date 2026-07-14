from __future__ import annotations

from core.skill_registry import clear_skill_cache, get_tool
from core import skill_runtime as sr


def test_t11_level_set_names_resolve_via_get_tool():
    """T11: every name in every level set must be a real registered tool."""
    clear_skill_cache()  # pick up skills/spawn-subagent/SKILL.md created this task
    all_names = sr.L1_PRINCIPAL_TOOLS | sr.L2_WORKER_TOOLS | sr.L3_LEAF_TOOLS
    unresolved = sorted(n for n in all_names if get_tool(n) is None)
    assert unresolved == [], f"level sets reference unregistered tools: {unresolved}"


def test_executor_reachability_audit_l1():
    """Rule-4 audit: every registered executor must stay reachable at L1.
    Catches llm_call silently breaking when the live turn gets allowed_tools."""
    clear_skill_cache()
    for name in sr._TOOL_EXECUTORS:
        if get_tool(name) is not None:  # resolvable today
            assert name in sr.L1_PRINCIPAL_TOOLS, f"{name} unreachable at L1"


def test_recall_is_unregistered_and_excluded():
    """recall is NOT a SKILL.md tool (monoexplore.py:157); it must be absent
    from every level set, so its omission is harmless (chokepoint 'unknown tool')."""
    clear_skill_cache()
    assert get_tool("recall") is None
    assert "recall" not in (sr.L1_PRINCIPAL_TOOLS | sr.L2_WORKER_TOOLS | sr.L3_LEAF_TOOLS)


def test_level_floor_monotonic_subset():
    assert sr.L3_LEAF_TOOLS <= sr.L2_WORKER_TOOLS <= sr.L1_PRINCIPAL_TOOLS
    assert sr.MAX_SPAWN_LEVEL == 3
    assert sr._LEVEL_DEFAULT_TOOLS == {
        1: sr.L1_PRINCIPAL_TOOLS, 2: sr.L2_WORKER_TOOLS, 3: sr.L3_LEAF_TOOLS,
    }


def test_spawn_budget_counts_across_tree():
    b = sr.SpawnBudget(max_total_spawns=2)
    assert b.can_spawn() is True
    b.charge(); assert b.can_spawn() is True
    b.charge(); assert b.can_spawn() is False


from pathlib import Path
from core.skill_runtime import (
    ToolExecutionContext, derive_child_context,
    L1_PRINCIPAL_TOOLS, L2_WORKER_TOOLS, L3_LEAF_TOOLS, SpawnBudget,
)


def _l1_ctx(tmp_path, **kw):
    base = dict(archive_dir=Path(tmp_path), level=1,
               allowed_tools=L1_PRINCIPAL_TOOLS, spawn_depth=0)
    base.update(kw)
    return ToolExecutionContext(**base)


def test_t1_l1_to_l2(tmp_path):
    child = derive_child_context(_l1_ctx(tmp_path), 2, label="plane:research")
    assert child.level == 2
    assert child.spawn_depth == 1
    assert child.allowed_tools == L2_WORKER_TOOLS
    assert child.subagent_label == "plane:research"


def test_t2_l2_to_l3_intersection_is_l3(tmp_path):
    l2 = derive_child_context(_l1_ctx(tmp_path), 2)
    l3 = derive_child_context(l2, 3)
    assert l3.level == 3
    assert l3.spawn_depth == 2
    assert l3.allowed_tools == (L2_WORKER_TOOLS & L3_LEAF_TOOLS) == L3_LEAF_TOOLS


def test_t4_l1_to_l3_direct(tmp_path):
    l3 = derive_child_context(_l1_ctx(tmp_path), 3)
    assert l3.level == 3
    assert l3.spawn_depth == 1  # depth 1, level 3 -- direct


def test_t8_narrowed_parent_never_regrants(tmp_path):
    narrow = _l1_ctx(tmp_path, allowed_tools=frozenset({"read_file"}))
    l2 = derive_child_context(narrow, 2)
    l3 = derive_child_context(l2, 3)
    assert l2.allowed_tools == frozenset({"read_file"})
    assert l3.allowed_tools == frozenset({"read_file"})


def test_t10_shared_cancel_and_budget(tmp_path):
    budget = SpawnBudget()
    flag = {"stop": True}
    parent = _l1_ctx(tmp_path, should_cancel=lambda: flag["stop"], spawn_budget=budget)
    child = derive_child_context(parent, 2)
    assert child.should_cancel() is True
    assert child.spawn_budget is budget  # SAME object, shared by reference
    assert child.result_cache is None    # clean cache, no cross-level bleed
    assert child.on_spawn_subagent is None  # children spawn inline


def test_existing_construction_sites_still_compile(tmp_path):
    """Defaults keep all existing ToolExecutionContext(archive_dir=...) sites working."""
    ctx = ToolExecutionContext(archive_dir=Path(tmp_path))
    assert ctx.level == 1
    assert ctx.allowed_tools == L1_PRINCIPAL_TOOLS
    assert ctx.spawn_depth == 0
    assert ctx.spawn_budget is None


import threading


def test_generation_lock_is_single_process_wide_object():
    from core.generation import generation_lock
    import core.generation as g
    assert generation_lock is g.generation_lock  # one shared module-level Lock


def test_generation_lock_try_acquire_refuses_when_held():
    from core.generation import generation_lock
    got = generation_lock.acquire(blocking=False)
    assert got is True
    try:
        assert generation_lock.acquire(blocking=False) is False  # held => refuse
    finally:
        generation_lock.release()
    assert generation_lock.acquire(blocking=False) is True
    generation_lock.release()


def test_expedition_lock_is_the_shared_lock():
    import engine.expedition_runner as er
    from core.generation import generation_lock
    assert er._expedition_lock is generation_lock


from core.skill_runtime import execute_tool_call_enveloped, _LEVEL_DEFAULT_TOOLS


def _ctx_at(level, tmp_path, **kw):
    base = dict(archive_dir=Path(tmp_path), level=level,
               allowed_tools=_LEVEL_DEFAULT_TOOLS[level], spawn_depth=level - 1)
    base.update(kw)
    return ToolExecutionContext(**base)


def test_t3_l3_cannot_spawn(tmp_path):
    env = execute_tool_call_enveloped(
        {"tool": "spawn_subagent", "level": 4, "prompt": "x"}, _ctx_at(3, tmp_path))
    assert env.ok is False
    assert env.data.get("denied") is True
    assert env.data.get("reason") == "capability"  # Guard A: spawn_subagent not in L3


def test_t5_l2_cannot_spawn_l2(tmp_path):
    env = execute_tool_call_enveloped(
        {"tool": "spawn_subagent", "level": 2, "prompt": "x"}, _ctx_at(2, tmp_path))
    assert env.ok is False
    assert env.data.get("reason") == "spawn_cap"
    assert "L3 Leaf only" in env.text


def test_t6_requested_level_4_exceeds_cap(tmp_path):
    env = execute_tool_call_enveloped(
        {"tool": "spawn_subagent", "level": 4, "prompt": "x"}, _ctx_at(1, tmp_path))
    assert env.ok is False
    assert env.data.get("reason") == "spawn_cap"
    assert "exceeds the L3 cap" in env.text


def test_t7_l3_write_file_denied_capability(tmp_path):
    env = execute_tool_call_enveloped(
        {"tool": "write_file", "path": "x", "content": "y"}, _ctx_at(3, tmp_path))
    assert env.ok is False
    assert env.data.get("reason") == "capability"
    assert env.data.get("level") == 3


def test_t9_denied_spawn_never_reaches_executor(tmp_path, monkeypatch):
    import engine.sync_bridge as sb
    def _boom(*a, **k):
        raise AssertionError("inference primitive was reached on a denied path")
    monkeypatch.setattr(sb, "generate_sync_parts_from_config", _boom)
    env = execute_tool_call_enveloped(
        {"tool": "spawn_subagent", "level": 9, "prompt": "x"}, _ctx_at(3, tmp_path))
    assert env.ok is False  # _boom never raised => never called


def test_t12_budget_exhaustion_denies_with_fault(tmp_path, monkeypatch):
    import core.turn_trace as tt
    captured = []
    monkeypatch.setattr(tt, "record_fault", lambda rec: captured.append(rec))
    budget = SpawnBudget(max_total_spawns=0)  # already exhausted
    ctx = _ctx_at(1, tmp_path, spawn_budget=budget, parent_turn_id="turn-abc")
    env = execute_tool_call_enveloped(
        {"tool": "spawn_subagent", "level": 2, "prompt": "x"}, ctx)
    assert env.ok is False
    assert env.data.get("reason") == "spawn_cap"
    assert any(getattr(r, "fault_kind", None) == "spawn_denied" for r in captured)


def test_l1_full_traffic_is_not_denied_by_gate(tmp_path):
    """The gate is a no-op for the principal (L1) -- read_file at L1 runs."""
    f = Path(tmp_path) / "hello.txt"
    f.write_text("hi", encoding="utf-8")
    env = execute_tool_call_enveloped(
        {"tool": "read_file", "path": str(f)}, _ctx_at(1, tmp_path))
    assert env.data.get("denied") is None


def test_budget_charged_at_gate_bounds_fanout(tmp_path):
    """Budget charged synchronously at the gate (not the atom), so N spawn calls
    in one generation cannot all pass before any charge lands (R5)."""
    budget = SpawnBudget(max_total_spawns=2)
    ctx = _ctx_at(1, tmp_path, spawn_budget=budget, parent_turn_id="turn-x",
                  on_spawn_subagent=lambda c: "[spawn_subagent: PENDING]")
    e1 = execute_tool_call_enveloped({"tool": "spawn_subagent", "level": 2, "prompt": "a"}, ctx)
    e2 = execute_tool_call_enveloped({"tool": "spawn_subagent", "level": 2, "prompt": "b"}, ctx)
    e3 = execute_tool_call_enveloped({"tool": "spawn_subagent", "level": 2, "prompt": "c"}, ctx)
    assert e1.data.get("denied") is None      # granted, charged -> used=1
    assert e2.data.get("denied") is None      # granted, charged -> used=2
    assert e3.data.get("reason") == "spawn_cap"  # budget exhausted at the gate
    assert budget._used == 2


def test_cmd_parser_threads_governance_into_ctx(tmp_path, monkeypatch):
    """process_response forwards level/allowed_tools/parent_turn_id/spawn_budget."""
    import core.cmd_parser as cp
    seen = {}
    real = cp.execute_tool_call_enveloped
    def spy(call, ctx):
        seen["level"] = ctx.level
        seen["allowed"] = ctx.allowed_tools
        seen["parent"] = ctx.parent_turn_id
        seen["budget"] = ctx.spawn_budget
        return real(call, ctx)
    monkeypatch.setattr(cp, "execute_tool_call_enveloped", spy)
    f = Path(tmp_path) / "x.txt"; f.write_text("hi", encoding="utf-8")
    text = '<tool_call>{"tool":"read_file","path":"%s"}</tool_call>' % str(f).replace("\\", "/")
    budget = SpawnBudget()
    cp.process_response(
        text, archive_dir=Path(tmp_path),
        level=1, allowed_tools=L1_PRINCIPAL_TOOLS,
        parent_turn_id="live-turn", spawn_budget=budget)
    assert seen["level"] == 1
    assert seen["allowed"] == L1_PRINCIPAL_TOOLS
    assert seen["parent"] == "live-turn"
    assert seen["budget"] is budget


def test_expedition_ctx_is_l2_narrowed():
    from engine.expedition_runner import ExpeditionRunner
    from core.monoexplore import READ_ONLY_SET
    r = ExpeditionRunner()
    assert r._ctx.level == 2
    assert r._ctx.allowed_tools == READ_ONLY_SET
    assert r._ctx.spawn_depth == 1
