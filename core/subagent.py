"""The reusable sub-inference atom (Phase A subagent substrate).

ONE blocking, Qt-free function (run_subagent) plus an engine_func-shaped adapter
(subagent_engine) so Phase B's Monoline LLM-node can bind it verbatim. The atom:
  - acquires the process-wide generation_lock (try-acquire, refuse on busy);
  - honors a mandatory is_busy() engine-RUNNING guard (INV-C Arm 2);
  - runs ONE child inference by default (the outer tool-loop stays the live turn's);
  - writes a child frame + allow faults to the SINGLE turn_trace store;
  - folds its result back as a provenance-fenced [SUBAGENT_RESULT ...] envelope.
The governance gate (skill_runtime) is what authorizes a spawn; this atom assumes
it has already been authorized (it is reached only via execute_spawn_subagent).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from core.generation import generation_lock
from core import turn_trace as _tt
from engine.sync_bridge import generate_sync_parts_from_config


@dataclass
class SubagentResult:
    ok: bool
    text: str            # the child's raw content
    fenced: str          # the [SUBAGENT_RESULT ...] envelope (what folds back)
    child_turn_id: str
    level: int
    tools_run: int = 0
    halt_reason: str | None = None   # "busy" | "cancelled" | "error" | None
    thinking: str = ""               # engine-separated native reasoning (last hop)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fence(level: int, frame: str, child_turn_id: str, ok: bool,
           tools_run: int, body: str) -> str:
    """Provenance fence: content inside is UNTRUSTED DATA, never instructions."""
    head = (f"[SUBAGENT_RESULT level={level} frame={frame} turn={child_turn_id} "
            f"ok={'true' if ok else 'false'} tools={tools_run}]")
    return f"{head}\n{body}\n[/SUBAGENT_RESULT]"


def run_subagent(messages, config, *, level: int, frame: str,
                 parent_turn_id: str | None, allowed_tools, should_cancel,
                 max_followups: int = 0, spawn_budget=None, llm_config=None,
                 is_busy=None) -> SubagentResult:
    """Blocking, Qt-free. Runs ONE child inference under the process-wide lock.
    is_busy=None => assume the generator is free (expedition convention)."""
    child_turn_id = uuid.uuid4().hex

    if should_cancel is not None:
        try:
            if should_cancel():
                return SubagentResult(False, "", _fence(level, frame, child_turn_id,
                                      False, 0, "[subagent: cancelled]"),
                                      child_turn_id, level, halt_reason="cancelled")
        except Exception:
            pass

    # INV-C Arm 2 -- yield/refuse if the main streaming lane is generating.
    if is_busy is not None:
        try:
            if is_busy():
                return SubagentResult(False, "", _fence(level, frame, child_turn_id,
                                      False, 0, "[subagent: generator busy]"),
                                      child_turn_id, level, halt_reason="busy")
        except Exception:
            pass

    # NOTE: the shared budget is charged at the GATE (Guard B grant branch), NOT here --
    # the L1 async path runs this atom later on a worker thread, so charging here would
    # let a single fan-out generation blow the budget (R5). The atom never charges.

    # Allow fault: the spawn was authorized and is about to run.
    try:
        _tt.record_fault(_tt.FaultTraceRecord(
            turn_id=str(parent_turn_id or ""), parent_turn_id=parent_turn_id, seq=0,
            emitted_at=_now(), event_kind="subagent_spawned", source_kind="kernel",
            source_name="subagent_atom", authority_tier="dispatch",
            fault_kind=None, severity=None,
            payload={"level": level, "parent_turn_id": parent_turn_id,
                     "child_turn_id": child_turn_id, "label": frame}))
    except Exception:
        pass

    # The child's OWN execution context: its capability is `allowed_tools` and any tool
    # it fires (incl. a nested spawn_subagent) flows through the gate at THIS level.
    # parent_turn_id is the MINTED child_turn_id (NOT the passed-in parent), so a
    # grandchild records THIS child as its parent. on_spawn_subagent=None => nested spawns
    # run inline (recursion composes for free).
    from core.skill_runtime import ToolExecutionContext, execute_tool_call_enveloped
    from core.cmd_parser import extract_commands, expand_calls
    from core.paths import LOG_DIR
    exec_ctx = ToolExecutionContext(
        archive_dir=LOG_DIR, level=level, allowed_tools=allowed_tools,
        spawn_depth=level - 1, subagent_label=frame, parent_turn_id=child_turn_id,
        spawn_budget=spawn_budget, should_cancel=should_cancel,
        result_cache=None, on_spawn_subagent=None)

    work = list(messages)
    raw, thinking, tools_run = "", "", 0
    # ONE inference by default (max_followups=0 => leaf/llm_call). Workers loop up to
    # max_followups+1 hops: generate -> run allowed tools -> fold fenced results ->
    # regenerate. This is the expedition _run_one_tick shape minus the daemon.
    for _hop in range(int(max_followups) + 1):
        if should_cancel is not None:
            try:
                if should_cancel():
                    break
            except Exception:
                pass
        # INV-C Arm 1 -- serialize sync-path generators; refuse (do NOT block) if held.
        if not generation_lock.acquire(blocking=False):
            return SubagentResult(False, raw, _fence(level, frame, child_turn_id,
                                  False, tools_run, "[subagent: generator busy]"),
                                  child_turn_id, level, tools_run, halt_reason="busy")
        try:
            try:
                # should_cancel is threaded into the streaming sync call so a stop request
                # breaks the in-flight generation mid-stream (enforce-STOP), instead of the
                # block running to completion. Kernel Contract v2: STOP is immediate.
                raw, thinking = generate_sync_parts_from_config(
                    config, work,
                    llm_config=(llm_config or {"max_tokens": 1024, "temp": 0.4}),
                    thinking_enabled=True, should_cancel=should_cancel)
            except Exception as exc:
                raw, thinking = "", f"[subagent: error - {exc}]"
        finally:
            generation_lock.release()
        # If a stop was observed while generating (the stream broke mid-flight), report a clean
        # cancellation rather than letting the partial output flow on as a finished result.
        if should_cancel is not None:
            try:
                _cancelled = bool(should_cancel())
            except Exception:
                _cancelled = False
            if _cancelled:
                return SubagentResult(False, "", _fence(level, frame, child_turn_id,
                                      False, tools_run, "[subagent: cancelled]"),
                                      child_turn_id, level, tools_run, halt_reason="cancelled")
        if not raw:
            break
        calls = [c for cmd in extract_commands(raw, strict=False) for c in expand_calls(cmd)]
        ran = False
        fenced_results: list[str] = []
        for n, call in enumerate(calls):
            call.setdefault("id", f"{child_turn_id}_t{n}")
            env = execute_tool_call_enveloped(call, exec_ctx)  # gated at THIS level
            fenced_results.append(getattr(env, "display_text", "") or "")
            tools_run += 1
            ran = True
        if not ran:
            break                       # terminal: no tool calls this hop
        work = work + [{"role": "user", "content": "\n\n".join(fenced_results)}]

    ok = bool(raw)
    body = raw or thinking or "(no output)"
    fenced = _fence(level, frame, child_turn_id, ok, tools_run, body)

    # Child frame (Layer B): one record_frame per allowed spawn.
    try:
        sys_chars = len(messages[0]["content"]) if messages else 0
        usr_chars = len(messages[1]["content"]) if len(messages) > 1 else 0
        total = sum(len(m.get("content", "")) for m in messages) + len(raw) + len(thinking)
        _tt.record_frame(_tt.FrameTraceRecord(
            turn_id=child_turn_id, captured_at=_now(), backend="subagent",
            engine_key=f"subagent:L{level}", gen_id=0,
            final_messages=tuple(_tt.FrameMessage.from_message(m) for m in messages),
            system_prompt_chars=sys_chars, user_prompt_chars=usr_chars,
            total_chars=total, parent_turn_id=parent_turn_id,
            metadata={"kind": "subagent", "level": f"L{level}", "role": frame,
                      "spawner_turn_id": parent_turn_id, "decision": "allow"}))
    except Exception:
        pass

    # Fold fault: the result is folded back.
    try:
        _tt.record_fault(_tt.FaultTraceRecord(
            turn_id=str(parent_turn_id or ""), parent_turn_id=parent_turn_id, seq=0,
            emitted_at=_now(), event_kind="subagent_folded", source_kind="kernel",
            source_name="subagent_atom", authority_tier="dispatch",
            fault_kind=None, severity=None,
            payload={"level": level, "child_turn_id": child_turn_id,
                     "status": "ok" if ok else "error", "result_chars": len(body),
                     "result_preview": body[:200]}))
    except Exception:
        pass

    return SubagentResult(ok, raw, fenced, child_turn_id, level, tools_run,
                          halt_reason=None if ok else "error", thinking=thinking)


def subagent_engine(messages, config) -> str:
    """engine_func-shaped adapter (Phase B reuse). Params ride in config['_subagent']."""
    params = dict(config.get("_subagent", {}) or {})
    allowed = params.get("allowed_tools", [])
    res = run_subagent(
        messages, config,
        level=int(params.get("level", 3)),
        frame=str(params.get("frame", "leaf")),
        parent_turn_id=params.get("parent_turn_id"),
        allowed_tools=frozenset(allowed),
        should_cancel=params.get("should_cancel"),
        max_followups=int(params.get("max_followups", 0)),
        spawn_budget=params.get("spawn_budget"),
        is_busy=params.get("is_busy"))
    return res.fenced


def subagent_engine_raw(messages, config) -> str:
    """Like subagent_engine but returns the child's RAW content (res.text), not the
    fenced envelope -- an LLM block wants the answer text, not a SUBAGENT_RESULT wrapper.
    NOTE: on a busy/refused atom, res.text is '' -- callers needing busy-vs-empty must
    call run_subagent directly and branch on res.halt_reason (see monoline_bridge)."""
    params = dict(config.get("_subagent", {}) or {})
    res = run_subagent(
        messages, config,
        level=int(params.get("level", 3)),
        frame=str(params.get("frame", "leaf")),
        parent_turn_id=params.get("parent_turn_id"),
        allowed_tools=frozenset(params.get("allowed_tools", [])),
        should_cancel=params.get("should_cancel"),
        max_followups=int(params.get("max_followups", 0)),
        spawn_budget=params.get("spawn_budget"),
        is_busy=params.get("is_busy"))
    return res.text
