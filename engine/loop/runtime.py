"""
LoopRuntime â€” the autonomous agent executor.

One loop.  One Pad.  Hard walls.
The agent picks its own cognitive mode each cycle.
The runtime validates, executes, enforces, and stops.

Usage:
    runtime = LoopRuntime(
        infer_fn=my_llm_call,       # (messages) -> str
        tool_executor=my_executor,  # (name, args) -> str | raises
        policy=RunPolicy(...),
    )
    result = runtime.run("Fix the login bug", tools=[...])
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from engine.loop.contracts import (
    Evidence,
    Pad,
    PreflightResult,
    RunContext,
    RunPolicy,
    RunResult,
    Step,
    ToolSpec,
)
from engine.loop.prompt import build_messages, build_retry
from engine.loop.tool_intelligence import (
    classify_tool_failure,
    extract_install_target_from_command,
    extract_missing_dependency_from_code,
    normalize_action_signature,
    pivot_directive_for_class,
    should_trip_circuit,
    FAILURE_MISSING_DEPENDENCY,
)
from engine.loop.policy import (
    POLICY_ACTION_DENY,
    POLICY_ACTION_NEEDS_APPROVAL,
    PolicyKernel,
)
from engine.loop.walls import WallChecker

# Type aliases for the two required callbacks.
InferFn = Callable[[list[dict[str, str]]], str]
ToolExecFn = Callable[[str, dict[str, Any]], str]
ApprovalFn = Callable[[str, dict[str, Any], ToolSpec], bool]
StructuredInferFn = Callable[[list[dict[str, str]]], Step]

# Optional observer callback:  (event_kind, payload_dict)
EventFn = Callable[[str, dict[str, Any]], None]


class LoopRuntime:
    """
    Goal-seeking loop over a single Pad.

    Parameters
    ----------
    infer_fn
        ``(messages: list[dict]) -> str``  â€” call the LLM, return raw text.
    tool_executor
        ``(tool_name: str, args: dict) -> str`` â€” execute a tool, return
        output string.  Raise on failure.
    policy
        Budget / approval rules.  Immutable for the run.
    on_event
        Optional observer.  Called with ``(kind, payload)`` at each
        interesting point so the UI / trace system can follow along.
        Kinds: ``cycle_start``, ``step_parsed``, ``action_result``,
        ``wall_hit``, ``finish``, ``retry``, ``error``.
    """

    def __init__(
        self,
        infer_fn: InferFn,
        tool_executor: ToolExecFn,
        structured_infer_fn: StructuredInferFn | None = None,
        approval_fn: ApprovalFn | None = None,
        policy: RunPolicy | None = None,
        on_event: EventFn | None = None,
        on_stop_requested: Callable[[], None] | None = None,
    ) -> None:
        self._infer = infer_fn
        self._exec = tool_executor
        self._structured_infer = structured_infer_fn
        self._approve = approval_fn
        self._policy = policy or RunPolicy()
        self._policy_kernel = PolicyKernel()
        self._walls = WallChecker()
        self._emit = on_event or (lambda *_: None)
        self._on_stop_requested = on_stop_requested
        self._stopped = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Request graceful stop (checked at the top of each cycle)."""
        self._stopped = True
        try:
            if callable(self._on_stop_requested):
                self._on_stop_requested()
        except Exception:
            pass

    def run(
        self,
        goal: str,
        tools: list[ToolSpec],
        *,
        run_id: str | None = None,
        preflight: PreflightResult | None = None,
        env_block: str = "",
        pre_run_elapsed_sec: float = 0.0,
    ) -> RunResult:
        """Execute the goal-seeking loop.  Blocks until done."""
        self._stopped = False
        rid = str(run_id or "").strip() or uuid.uuid4().hex[:12]
        pad = _build_pad(goal=goal, preflight=preflight, env_block=env_block)
        ctx = RunContext(
            run_id=rid,
            goal=pad.goal,
            policy=self._policy,
            tools=tools,
            pad=pad,
            start_time=time.time() - max(0.0, float(pre_run_elapsed_sec or 0.0)),
        )

        _empty_action_count = 0
        while not self._stopped:
            ctx.cycle += 1
            self._emit("cycle_start", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "pad": _pad_snapshot(ctx.pad),
            })

            # ---- infer ----
            step = self._infer_step(ctx)
            if step is None:
                if self._stopped:
                    return self._terminate(ctx, wall="stopped")
                elapsed = max(0.0, (time.time() - ctx.start_time) - float(getattr(ctx, "paused_sec", 0.0) or 0.0))
                if elapsed >= float(ctx.policy.max_elapsed_sec):
                    self._emit("wall_hit", {
                        "run_id": ctx.run_id,
                        "wall": "max_elapsed",
                        "cycle": ctx.cycle,
                    })
                    return self._terminate(ctx, wall="max_elapsed")
                # Exhausted retries â€” treat as malformed wall
                return self._terminate(ctx, wall="malformed")

            if not step.actions and not bool(step.task_finished):
                amputated = bool(getattr(step, "_actions_likely_amputated", False))
                if amputated:
                    self._emit("empty_action_from_repair", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "reason": "actions_likely_amputated",
                    })
                _empty_action_count += 1
                if _empty_action_count >= 2:
                    self._emit("wall_hit", {
                        "run_id": ctx.run_id,
                        "wall": "empty_action_stall",
                        "cycle": ctx.cycle,
                    })
                    return self._terminate(ctx, wall="empty_action_stall")
                forced_messages = build_messages(ctx)
                forced_messages.append({
                    "role": "user",
                    "content": (
                        "Your last output produced no executable actions. "
                        + (
                            "It also appears truncated/repaired. Respond with a SHORTER JSON and include one concrete tool action. "
                            if amputated else
                            "You MUST either call a tool or set task_finished=true. "
                        )
                        + "If complete, set task_finished=true with a one-sentence finish_summary."
                    ),
                })
                step = self._infer_step_from_messages(ctx, forced_messages)
                if step is None:
                    return self._terminate(ctx, wall="malformed")
                if step.actions or bool(step.task_finished):
                    _empty_action_count = 0
            else:
                _empty_action_count = 0

            self._emit("step_parsed", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "step": _step_dict(step),
                "reasoning": step.reasoning,
                "response": getattr(step, "response", "") or "",
                "step_ok": step.step_ok,
                "self_check": getattr(step, "self_check", "") or "",
            })

            # ---- wall check (before execution) ----
            self._emit("wall_check_detail", _wall_check_detail(ctx, step))
            wall = self._walls.check(ctx, step)
            if wall:
                self._emit("wall_hit", {"run_id": ctx.run_id, "wall": wall, "cycle": ctx.cycle})
                return self._terminate(ctx, wall=wall)

            # ---- execute actions ----
            step, evidence = self._execute_with_same_cycle_retry(ctx, step)
            if step is None:
                return self._terminate(ctx, wall="malformed")

            # ---- update pad ----
            pad_diff = self._update_pad(ctx, step, evidence)
            if pad_diff:
                self._emit("pad_diff", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "diff": pad_diff,
                })

            # ---- verify boundary (soft-replan trigger) ----
            if _is_verify_intent(step.intent) and step.step_ok is False:
                ctx.pad.open_questions = [
                    "Verification failed. Try a different verification approach."
                ][:ctx.pad.MAX_OPEN_QUESTIONS]
                self._emit("verify_boundary", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "action": "replan",
                    "reason": "verify_incomplete",
                    "current_outcome": str(step.self_check or "")[:200],
                })

            # ---- finish? ----
            if bool(step.task_finished):
                # Runtime-owned boundary: failed verification cannot finish.
                if _is_verify_intent(step.intent) and step.step_ok is False:
                    ctx.pad.open_questions = [
                        "Verification failed. Try a different verification approach."
                    ][:ctx.pad.MAX_OPEN_QUESTIONS]
                    self._emit("verify_boundary", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "action": "continue",
                        "reason": "finish_blocked_verify_failed",
                        "current_outcome": str(step.self_check or "")[:200],
                    })
                    continue

                uncrystallized = [
                    t for t in ctx.pad.todo_state
                    if not t.get("crystallized") and t.get("blocking", False)
                ]
                if uncrystallized and ctx.pad.preflight is not None:
                    remaining = [str(t.get("directive") or "") for t in uncrystallized[:3] if str(t.get("directive") or "").strip()]
                    ctx.pad.open_questions = [f"Incomplete: {d}" for d in remaining][:ctx.pad.MAX_OPEN_QUESTIONS]
                    self._emit("finish_blocked", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "reason": "uncrystallized_blocking_todos",
                        "remaining": remaining,
                    })
                    continue

                if not _has_successful_verification(ctx.pad):
                    ctx.pad.open_questions = [
                        "No successful verification after implementation. Use read_file or run_cmd to verify before finishing."
                    ][:ctx.pad.MAX_OPEN_QUESTIONS]
                    self._emit("finish_blocked", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "reason": "no_verification_evidence",
                    })
                    continue
                result = RunResult(
                    run_id=ctx.run_id,
                    success=True,
                    summary=self._compose_finish_summary(step),
                    pad=ctx.pad,
                    cycles_used=ctx.cycle,
                    tool_calls_used=ctx.total_tool_calls,
                )
                self._emit("finish", {"run_id": ctx.run_id, "result": _result_dict(result)})
                return result

            # ---- deterministic convergence guard ----
            # If the plan is complete and the model keeps issuing redundant
            # verification actions, end the run successfully instead of looping.
            if _should_auto_finish_after_redundant_verify(ctx, step, evidence):
                summary = _compose_auto_finish_summary(step, ctx.pad)
                result = RunResult(
                    run_id=ctx.run_id,
                    success=True,
                    summary=summary,
                    pad=ctx.pad,
                    cycles_used=ctx.cycle,
                    tool_calls_used=ctx.total_tool_calls,
                )
                self._emit("auto_finish", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "reason": "redundant_verification_after_completed_plan",
                    "summary": summary,
                })
                self._emit("finish", {"run_id": ctx.run_id, "result": _result_dict(result)})
                return result

        # stopped externally
        return self._terminate(ctx, wall="stopped")

    # ------------------------------------------------------------------
    # Inference + parsing
    # ------------------------------------------------------------------

    def _infer_step(self, ctx: RunContext) -> Step | None:
        """Call the LLM, parse the response into a Step.  Retries on failure."""
        return self._infer_step_from_messages(ctx, build_messages(ctx))

    def _start_infer_watchdog(self, ctx: RunContext) -> tuple[threading.Event, threading.Thread | None]:
        """
        Watch elapsed wall clock during a single infer call.
        If max_elapsed is reached mid-inference, request stop so the adapter
        can terminate streaming generation promptly.
        """
        done = threading.Event()

        def _watch() -> None:
            while not done.wait(0.2):
                elapsed = max(0.0, (time.time() - ctx.start_time) - float(getattr(ctx, "paused_sec", 0.0) or 0.0))
                if elapsed >= float(ctx.policy.max_elapsed_sec):
                    self._emit("wall_watchdog_stop", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "wall": "max_elapsed",
                        "elapsed_sec": round(elapsed, 3),
                        "max_elapsed_sec": float(ctx.policy.max_elapsed_sec),
                    })
                    self.stop()
                    return

        try:
            th = threading.Thread(target=_watch, daemon=True, name=f"loop-watchdog-{ctx.run_id}-{ctx.cycle}")
            th.start()
            return done, th
        except Exception:
            return done, None

    @staticmethod
    def _stop_infer_watchdog(done: threading.Event, thread_obj: threading.Thread | None) -> None:
        done.set()
        if thread_obj is not None and thread_obj.is_alive():
            thread_obj.join(timeout=0.3)

    def _infer_step_from_messages(self, ctx: RunContext, messages: list[dict[str, str]]) -> Step | None:
        """Call the LLM on prebuilt messages, parse into a Step, retry on malformed output."""
        messages = list(messages)
        retries = ctx.policy.max_retries

        if self._structured_infer is not None:
            for attempt in range(1 + retries):
                if self._stopped:
                    return None
                ctx.llm_call_count += 1
                call_index = int(ctx.llm_call_count)
                self._emit("llm_input", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "call_index": call_index,
                    "attempt": attempt + 1,
                    "backend": "structured",
                    "messages": messages,
                    "prompt_digest": _messages_digest(messages),
                    "prompt_messages": len(messages),
                    "prompt_chars": _messages_chars(messages),
                })
                try:
                    wd_done, wd_thread = self._start_infer_watchdog(ctx)
                    try:
                        step = self._structured_infer(messages)
                    finally:
                        self._stop_infer_watchdog(wd_done, wd_thread)
                except Exception as exc:
                    self._emit("llm_call", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "call_index": call_index,
                        "attempt": attempt + 1,
                        "backend": "structured",
                        "ok": False,
                        "error": f"structured infer failed: {exc}",
                        "output": "",
                        "response_chars": 0,
                        "token_stats_available": False,
                    })
                    self._emit("retry", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "attempt": attempt + 1,
                        "error": f"structured infer failed: {exc}",
                        "raw_preview": "",
                    })
                    messages.append(build_retry("", f"structured infer failed: {exc}"))
                    continue
                if isinstance(step, Step):
                    self._emit("llm_call", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "call_index": call_index,
                        "attempt": attempt + 1,
                        "backend": "structured",
                        "ok": True,
                        "error": "",
                        "output": json.dumps(_step_dict(step), ensure_ascii=False),
                        "response_chars": len(json.dumps(_step_dict(step), ensure_ascii=False)),
                        "token_stats_available": False,
                    })
                    return step
                self._emit("llm_call", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "call_index": call_index,
                    "attempt": attempt + 1,
                    "backend": "structured",
                    "ok": False,
                    "error": "structured infer returned non-Step",
                    "output": "",
                    "response_chars": 0,
                    "token_stats_available": False,
                })
                self._emit("retry", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "attempt": attempt + 1,
                    "error": "structured infer returned non-Step",
                    "raw_preview": "",
                })
                messages.append(build_retry("", "structured infer returned non-Step"))
            return None

        for attempt in range(1 + retries):
            if self._stopped:
                return None

            ctx.llm_call_count += 1
            call_index = int(ctx.llm_call_count)
            self._emit("llm_input", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "call_index": call_index,
                "attempt": attempt + 1,
                "backend": "json",
                "messages": messages,
                "prompt_digest": _messages_digest(messages),
                "prompt_messages": len(messages),
                "prompt_chars": _messages_chars(messages),
            })
            wd_done, wd_thread = self._start_infer_watchdog(ctx)
            try:
                raw = self._infer(messages)
            finally:
                self._stop_infer_watchdog(wd_done, wd_thread)
            step, err = _parse_step(raw)
            repair_flags = _parse_repair_flags(raw, step)
            if step is not None and repair_flags.get("actions_likely_amputated", False):
                try:
                    setattr(step, "_actions_likely_amputated", True)
                except Exception:
                    pass
            self._emit("llm_call", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "call_index": call_index,
                "attempt": attempt + 1,
                "backend": "json",
                "ok": step is not None,
                "error": err if step is None else "",
                "output": str(raw or ""),
                "response_chars": len(str(raw or "")),
                "token_stats_available": False,
                "repair_flags": repair_flags,
            })
            if step is not None and repair_flags.get("actions_likely_amputated", False):
                self._emit("parse_repair", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "call_index": call_index,
                    "reason": "actions_likely_amputated",
                    "raw_chars": len(str(raw or "")),
                    "raw_preview": str(raw or "")[:280],
                })
            if step is not None:
                return step

            self._emit("retry", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "attempt": attempt + 1,
                "error": err,
                "raw_preview": raw[:200],
            })
            messages.append(build_retry(raw, err))

        return None

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute(self, ctx: RunContext, step: Step) -> list[Evidence]:
        """Execute each action in the step, collect evidence."""
        results: list[Evidence] = []
        tool_index = {t.name: t for t in ctx.tools}

        for action in step.actions:
            if self._stopped:
                break

            name = action.get("tool", "")
            args = action.get("args", {})
            decision = self._policy_kernel.decide(
                ctx=ctx,
                tool_name=name,
                args=args if isinstance(args, dict) else {},
                tool_index=tool_index,
            )
            self._emit("policy_decision", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "tool": name,
                "args": args,
                "scope": decision.scope,
                "action": decision.action,
                "reason_code": decision.reason_code,
                "reason": decision.reason_code,
                "path_scope": _classify_path_scope(args),
            })

            # Policy deny -> record failed evidence and continue safely.
            if decision.action == POLICY_ACTION_DENY:
                denied_tool = str(name or "").strip() or "<empty>"
                reason = str(decision.reason_code or "policy_denied")
                denied_output = f"policy denied tool '{denied_tool}': {reason}"
                results.append(Evidence(
                    tool=name,
                    args=args,
                    output=denied_output,
                    ok=False,
                    cycle=ctx.cycle,
                ))
                ctx.total_tool_calls += 1
                self._emit("action_result", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "tool": name,
                    "args": args,
                    "ok": False,
                    "status": "failed",
                    "error": reason,
                    "output": denied_output,
                    "output_preview": denied_output[:200],
                    "truncated": False,
                })
                continue

            spec = tool_index[name]
            args_dict = args if isinstance(args, dict) else {}
            signature = normalize_action_signature(name, args_dict)
            if (
                str(name or "").strip().lower() == "run_cmd"
                and str(ctx.action_failure_class.get(signature) or "") == FAILURE_MISSING_DEPENDENCY
                and int(ctx.action_failure_counts.get(signature, 0)) >= 1
            ):
                dep = extract_missing_dependency_from_code(str(ctx.action_failure_code.get(signature) or ""))
                if dep and dep not in ctx.attempted_installs:
                    directive = "Run is dependency-blocked. Use run_cmd to install dependency, or use read_file for static verification."
                    msg = f"routing blocked action pending dependency install: {signature}"
                    results.append(Evidence(tool=name, args=args_dict, output=msg, ok=False, cycle=ctx.cycle))
                    existing_qs = [str(q) for q in (ctx.pad.open_questions or [])]
                    if directive not in existing_qs:
                        ctx.pad.open_questions = (existing_qs + [directive])[-ctx.pad.MAX_OPEN_QUESTIONS:]
                    self._emit("routing_blocked", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "tool": name,
                        "signature": signature,
                        "failure_class": FAILURE_MISSING_DEPENDENCY,
                        "dependency": dep,
                        "directive": directive,
                        "failure_count": int(ctx.action_failure_counts.get(signature, 0)),
                    })
                    self._emit("action_result", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "tool": name,
                        "args": args_dict,
                        "ok": False,
                        "status": "failed",
                        "error": "routing_blocked",
                        "output": msg,
                        "output_preview": msg[:200],
                        "truncated": False,
                    })
                    continue
            if (
                str(getattr(spec, "scope", "") or "") in {"read", "list", "search", "grep"}
                and int(ctx.action_success_cycle.get(signature, -99999)) == int(ctx.cycle - 1)
            ):
                msg = f"noop blocked repeated discovery action: {signature}"
                results.append(Evidence(tool=name, args=args_dict, output=msg, ok=False, cycle=ctx.cycle))
                # Keep the guard active across cycles so identical discovery
                # actions cannot pass again on cycle+2 without a change.
                ctx.action_success_cycle[signature] = int(ctx.cycle)
                self._emit("noop_blocked", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "tool": name,
                    "signature": signature,
                    "reason": "repeated_discovery_action",
                    "last_success_cycle": int(ctx.action_success_cycle.get(signature, -1)),
                })
                self._emit("action_result", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "tool": name,
                    "args": args_dict,
                    "ok": False,
                    "status": "failed",
                    "error": "noop_blocked",
                    "output": msg,
                    "output_preview": msg[:200],
                    "truncated": False,
                })
                continue
            if should_trip_circuit(signature=signature, failure_counts=ctx.action_failure_counts, threshold=2):
                failure_class = str(ctx.action_failure_class.get(signature) or "unknown")
                directive = pivot_directive_for_class(failure_class)
                question = f"Circuit breaker ({name}): {directive}"
                existing_qs = [str(q) for q in (ctx.pad.open_questions or [])]
                if question not in existing_qs:
                    ctx.pad.open_questions = (existing_qs + [question])[-ctx.pad.MAX_OPEN_QUESTIONS:]
                blocked_msg = f"circuit breaker blocked repeated action: {signature}"
                results.append(Evidence(tool=name, args=args_dict, output=blocked_msg, ok=False, cycle=ctx.cycle))
                self._emit("circuit_breaker", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "tool": name,
                    "signature": signature,
                    "failure_class": failure_class,
                    "directive": directive,
                    "failure_count": int(ctx.action_failure_counts.get(signature, 0)),
                })
                self._emit("action_result", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "tool": name,
                    "args": args_dict,
                    "ok": False,
                    "status": "failed",
                    "error": "circuit_breaker",
                    "output": blocked_msg,
                    "output_preview": blocked_msg[:200],
                    "truncated": False,
                })
                continue

            # Scope check (approval gating)
            if decision.action == POLICY_ACTION_NEEDS_APPROVAL:
                self._emit("approval_required", {
                    "run_id": ctx.run_id, "cycle": ctx.cycle, "tool": name, "scope": spec.scope, "args": args,
                    "reason_code": decision.reason_code,
                })
                if self._approve is None:
                    denied_output = f"approval required for tool '{name}', but no approval handler is configured"
                    results.append(Evidence(
                        tool=name,
                        args=args,
                        output=denied_output,
                        ok=False,
                        cycle=ctx.cycle,
                    ))
                    ctx.total_tool_calls += 1
                    self._emit("action_result", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "tool": name,
                        "args": args,
                        "ok": False,
                        "status": "failed",
                        "error": "approval_unavailable",
                        "output": denied_output,
                        "output_preview": denied_output[:200],
                        "truncated": False,
                    })
                    continue
                approval_started = time.time()
                try:
                    approved = bool(self._approve(name, args, spec))
                except Exception as exc:
                    approved = False
                    self._emit("error", {"run_id": ctx.run_id, "cycle": ctx.cycle, "error": f"approval callback failed: {exc}"})
                finally:
                    waited = max(0.0, time.time() - approval_started)
                    ctx.paused_sec += waited
                if not approved:
                    results.append(Evidence(
                        tool=name,
                        args=args,
                        output=f"approval denied for tool '{name}'",
                        ok=False,
                        cycle=ctx.cycle,
                    ))
                    ctx.total_tool_calls += 1
                    denied_output = f"approval denied for tool '{name}'"
                    self._emit("action_result", {
                        "run_id": ctx.run_id,
                        "cycle": ctx.cycle,
                        "tool": name,
                        "args": args,
                        "ok": False,
                        "status": "failed",
                        "error": "approval_denied",
                        "output": denied_output,
                        "output_preview": denied_output[:200],
                        "truncated": False,
                    })
                    continue

            self._emit("tool_started", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "tool": name,
                "args": args,
                "scope": str(getattr(spec, "scope", "") or ""),
                "path_scope": _classify_path_scope(args),
            })
            tool_started_at = time.perf_counter()
            try:
                output = self._exec(name, args)
                ok = True
            except Exception as exc:
                output = str(exc)
                ok = False
            tool_duration_ms = int(max(0.0, (time.perf_counter() - tool_started_at) * 1000.0))

            if str(name or "").strip().lower() == "run_cmd":
                dep_attempt = extract_install_target_from_command(str(args_dict.get("command") or ""))
                if dep_attempt:
                    ctx.attempted_installs.add(dep_attempt)

            ev = Evidence(tool=name, args=args, output=output, ok=ok, cycle=ctx.cycle)
            results.append(ev)
            ctx.total_tool_calls += 1
            if not ok:
                self._record_tool_failure_guidance(
                    ctx,
                    spec,
                    args_dict,
                    output,
                    signature=signature,
                )
            else:
                ctx.action_failure_counts.pop(signature, None)
                ctx.action_failure_class.pop(signature, None)
                ctx.action_failure_code.pop(signature, None)
                ctx.action_success_cycle[signature] = int(ctx.cycle)
            self._emit("action_result", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "tool": name,
                "args": args,
                "ok": ok,
                "status": "ok" if ok else "failed",
                "output": output,
                "output_preview": output[:200] if output else "",
                "truncated": bool(output and len(output) > 200),
                "duration_ms": tool_duration_ms,
                "path_scope": _classify_path_scope(args),
            })
            self._emit("tool_timing", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "tool": name,
                "duration_ms": tool_duration_ms,
                "ok": ok,
            })

        return results

    def _execute_with_same_cycle_retry(self, ctx: RunContext, step: Step) -> tuple[Step | None, list[Evidence]]:
        """
        Execute with same-cycle reflection retries.

        Max 3 attempts per cycle total (initial + 2 retries). If still failing,
        force task_finished=False and continue to next cycle.
        """
        max_attempts = 3
        attempt = 1
        combined: list[Evidence] = []
        current = step

        while True:
            ev = self._execute(ctx, current)
            combined.extend(ev)

            if bool(current.task_finished):
                return current, combined

            if attempt >= max_attempts:
                note = "Same-cycle retry limit reached; carrying forward with task_finished=false."
                existing = str(getattr(current, "self_check", "") or "").strip()
                current.self_check = f"{existing} {note}".strip()
                current.task_finished = False
                self._emit("same_cycle_retry_limit", {
                    "run_id": ctx.run_id,
                    "cycle": ctx.cycle,
                    "attempts": attempt,
                    "reason": "max_attempts_reached",
                })
                return current, combined

            if not _should_same_cycle_retry(current, ev):
                return current, combined

            reflect_user = _build_same_cycle_reflect_message(
                prev_step=current,
                evidence=ev,
                attempt=attempt + 1,
                max_attempts=max_attempts,
            )
            msgs = build_messages(ctx)
            msgs.append({"role": "user", "content": reflect_user})
            next_step = self._infer_step_from_messages(ctx, msgs)
            if next_step is None:
                return None, combined

            self._emit("same_cycle_retry", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
                "previous_intent": str(current.intent or ""),
                "next_intent": str(next_step.intent or ""),
            })
            self._emit("step_parsed", {
                "run_id": ctx.run_id,
                "cycle": ctx.cycle,
                "step": _step_dict(next_step),
                "reasoning": next_step.reasoning,
                "response": getattr(next_step, "response", "") or "",
                "step_ok": next_step.step_ok,
                "self_check": getattr(next_step, "self_check", "") or "",
                "same_cycle_retry": True,
                "retry_attempt": attempt + 1,
            })
            current = next_step
            attempt += 1

    def _record_tool_failure_guidance(
        self,
        ctx: RunContext,
        spec: ToolSpec,
        args: dict[str, Any],
        output: str,
        *,
        signature: str,
    ) -> None:
        pad = ctx.pad
        tool_name = str(getattr(spec, "name", "") or "")
        classification = classify_tool_failure(tool_name, args, output)
        failure_class = classification.failure_class
        error_code = classification.error_code
        recovery = str(classification.directive or "").strip() or str(getattr(spec, "failure_recovery", "") or "").strip()
        note = f"{tool_name} failed: {str(output or '').strip()[:160]}"
        question = f"Recover {tool_name}: {recovery}"
        existing_qs = [str(q) for q in (pad.open_questions or [])]
        if question not in existing_qs:
            pad.open_questions = (existing_qs + [question])[-pad.MAX_OPEN_QUESTIONS:]
        prev = int(ctx.action_failure_counts.get(signature, 0))
        failure_count = prev + 1
        ctx.action_failure_counts[signature] = failure_count
        ctx.action_failure_class[signature] = failure_class
        ctx.action_failure_code[signature] = error_code
        pad.record_tool_failure(
            cycle=ctx.cycle,
            tool=tool_name,
            signature=signature,
            failure_class=failure_class,
            error_code=error_code,
            note=classification.detail or note,
        )
        if failure_count >= 2:
            follow_up = pivot_directive_for_class(failure_class)
            if follow_up not in pad.open_questions:
                pad.open_questions = (list(pad.open_questions) + [follow_up])[-pad.MAX_OPEN_QUESTIONS:]
        else:
            follow_up = (
                f"If {tool_name} fails again, use list_dir/read_file to gather evidence, "
                f"then choose a different tool call. Guidance: {recovery}"
            )
            if follow_up not in pad.open_questions:
                pad.open_questions = (list(pad.open_questions) + [follow_up])[-pad.MAX_OPEN_QUESTIONS:]

        self._emit("tool_failure_guidance", {
            "run_id": ctx.run_id,
            "cycle": ctx.cycle,
            "tool": tool_name,
            "signature": signature,
            "failure_class": failure_class,
            "error_code": error_code,
            "recovery": recovery,
            "failure_count_hint": failure_count,
            "note": note,
        })

    # ------------------------------------------------------------------
    # Pad update
    # ------------------------------------------------------------------

    @staticmethod
    def _update_pad(ctx: RunContext, step: Step, evidence: list[Evidence]) -> dict[str, Any]:
        pad = ctx.pad
        before = _pad_state_brief(pad)

        # Record intent
        pad.steps.append(step.intent)
        ctx.recent_intents.append(step.intent)
        # Keep recent_intents bounded
        if len(ctx.recent_intents) > ctx.policy.repetition_window * 2:
            ctx.recent_intents = ctx.recent_intents[-ctx.policy.repetition_window * 2:]

        # Append evidence + trim
        pad.evidence.extend(evidence)
        pad.trim_evidence()
        pad.record_check(
            cycle=ctx.cycle,
            ok=step.step_ok,
            note=step.self_check,
        )

        # Crystallize todo items based on successful evidence and todo_update.
        if pad.todo_state and evidence:
            for ev in evidence:
                if not ev.ok:
                    continue
                for item in pad.todo_state:
                    if item.get("crystallized"):
                        continue
                    hint = str(item.get("tool_hint") or "").strip().lower()
                    tool_name = str(ev.tool or "").strip().lower()
                    if hint and not _tool_hint_is_unmapped(hint) and hint == tool_name:
                        item["crystallized"] = True
                        item["cycle_crystallized"] = int(ctx.cycle)
                        item["evidence_ref"] = str(ev.output or "")[:80]
                        break
                    if _tool_hint_is_unmapped(hint) and _evidence_semantically_matches_todo(ev, item):
                        item["crystallized"] = True
                        item["cycle_crystallized"] = int(ctx.cycle)
                        item["evidence_ref"] = str(ev.output or "")[:80]
                        break

        if isinstance(step.todo_update, str) and step.todo_update.strip():
            update_lower = step.todo_update.strip().lower()
            for item in pad.todo_state:
                if item.get("crystallized"):
                    continue
                directive = str(item.get("directive") or "").strip().lower()
                if update_lower in directive or directive in update_lower:
                    item["crystallized"] = True
                    item["cycle_crystallized"] = int(ctx.cycle)
                    break

        # Record artifact-like outputs from successful write operations.
        for ev in evidence:
            if not ev.ok or ev.tool not in {"write_file", "mkdir", "copy_path", "move_path", "apply_patch"}:
                continue
            args = ev.args if isinstance(ev.args, dict) else {}
            path = str(args.get("path") or args.get("dst") or "").strip()
            if path:
                pad.artifacts[path] = f"cycle {ctx.cycle}"

        # Update plan/progress from runtime state only.
        pad.plan = pad.render_todo_text()
        pad.progress = pad.compute_progress()

        # Verification-aware continuity hints: preserve failure signals inside
        # the Pad so the agent can change strategy instead of repeating.
        step_ok = getattr(step, "step_ok", None)
        check_note = str(getattr(step, "self_check", "") or "").strip()
        if step_ok is False and check_note:
            question = f"Resolve failed check: {check_note[:180]}"
            existing_qs = [str(q) for q in pad.open_questions]
            if question not in existing_qs:
                pad.open_questions = (existing_qs + [question])[-pad.MAX_OPEN_QUESTIONS:]

        # Snapshot progress for stall detection
        pad.record_progress()
        after = _pad_state_brief(pad)
        return _pad_state_diff(before, after)

    # ------------------------------------------------------------------
    # Termination helper
    # ------------------------------------------------------------------

    @staticmethod
    def _compose_finish_summary(step: Step) -> str:
        summary = _one_sentence(str(step.finish_summary or "").strip())
        if summary:
            return summary
        self_check = _one_sentence(str(getattr(step, "self_check", "") or "").strip())
        return self_check or "(agent declared finish)"

    def _terminate(self, ctx: RunContext, wall: str) -> RunResult:
        result = RunResult(
            run_id=ctx.run_id,
            success=False,
            summary=f"Run terminated: {wall}",
            pad=ctx.pad,
            cycles_used=ctx.cycle,
            tool_calls_used=ctx.total_tool_calls,
            wall_hit=wall,
        )
        self._emit("finish", {"run_id": ctx.run_id, "result": _result_dict(result)})
        return result


# ======================================================================
# Step parser
# ======================================================================

def _parse_step(raw: str) -> tuple[Step | None, str]:
    """
    Extract a StepPacket JSON from raw LLM output.

    Returns ``(Step, "")`` on success or ``(None, error_message)`` on failure.
    """
    text = raw.strip()

    # Try ```json ... ``` fenced block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        extracted, _repaired = _extract_json_object(text)
        if extracted is None:
            return None, "no JSON object found in response"
        text = extracted

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        repaired_text = _strip_trailing_commas(text)
        if repaired_text != text:
            try:
                data = json.loads(repaired_text)
            except json.JSONDecodeError:
                trimmed = _repair_from_top_level_commas(text)
                if trimmed is not None:
                    try:
                        data = json.loads(trimmed)
                    except json.JSONDecodeError:
                        return None, f"JSON parse error: {exc}"
                else:
                    return None, f"JSON parse error: {exc}"
        else:
            trimmed = _repair_from_top_level_commas(text)
            if trimmed is not None:
                try:
                    data = json.loads(trimmed)
                except json.JSONDecodeError:
                    return None, f"JSON parse error: {exc}"
            else:
                return None, f"JSON parse error: {exc}"

    if not isinstance(data, dict):
        return None, "top-level value is not an object"
    if "intent" not in data:
        return None, "missing required field: intent"

    actions = data.get("actions", [])
    if not isinstance(actions, list):
        return None, "'actions' must be a list"
    # Validate each action has at least a tool name
    for i, a in enumerate(actions):
        if not isinstance(a, dict) or "tool" not in a:
            return None, f"action[{i}] must be {{'tool': '...', 'args': {{...}}}}"

    return Step(
        intent=str(data.get("intent", "")),
        response=str(data.get("response", "")),
        reasoning=str(data.get("reasoning", "")),
        actions=actions,
        self_check=str(data.get("self_check", "")),
        step_ok=(data.get("step_ok") if isinstance(data.get("step_ok"), bool) or data.get("step_ok") is None else None),
        todo_update=(str(data.get("todo_update")).strip() if isinstance(data.get("todo_update"), str) else None),
        task_finished=(
            data.get("task_finished")
            if ("task_finished" in data and (isinstance(data.get("task_finished"), bool) or data.get("task_finished") is None))
            else bool(data.get("finish", False))
        ),
        finish_summary=str(data.get("finish_summary", "")),
    ), ""


def _extract_json_object(text: str) -> tuple[str | None, bool]:
    """
    Extract the first balanced JSON object from text.

    Returns (object_text, repaired) where repaired=True means closing braces
    were appended to recover a truncated object.
    """
    start = text.find("{")
    if start < 0:
        return None, False

    depth = 0
    in_str = False
    escaped = False
    end = None

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end is not None:
        return text[start : end + 1], False

    if depth > 0:
        return text[start:] + ("}" * depth), True

    return None, False


def _strip_trailing_commas(text: str) -> str:
    """
    Remove trailing commas before '}' or ']' to salvage near-JSON outputs.
    """
    out = text
    while True:
        cleaned = re.sub(r",\s*([}\]])", r"\1", out)
        if cleaned == out:
            return out
        out = cleaned


def _top_level_commas(text: str) -> list[int]:
    commas: list[int] = []
    depth = 0
    in_str = False
    escaped = False
    for idx, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            depth += 1
            continue
        if ch in "}]":
            depth = max(0, depth - 1)
            continue
        if ch == "," and depth == 1:
            commas.append(idx)
    return commas


def _repair_from_top_level_commas(text: str) -> str | None:
    """
    Salvage truncated JSON by keeping only complete top-level members.
    """
    commas = _top_level_commas(text)
    if not commas:
        return None
    for cut in reversed(commas):
        candidate = text[:cut].rstrip()
        if not candidate.startswith("{"):
            continue
        candidate = _strip_trailing_commas(candidate).rstrip()
        if candidate.endswith("{"):
            continue
        candidate = candidate + "}"
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return candidate
    return None


def _parse_repair_flags(raw: str, step: Step | None) -> dict[str, Any]:
    text = str(raw or "")
    lowered = text.lower()
    has_actions_key = '"actions"' in lowered
    actions_count = len(getattr(step, "actions", []) or []) if step is not None else -1
    likely_amputated = bool(step is not None and has_actions_key and actions_count == 0 and not bool(getattr(step, "task_finished", False)))
    return {
        "raw_chars": len(text),
        "has_actions_key": has_actions_key,
        "actions_count": actions_count,
        "actions_likely_amputated": likely_amputated,
    }


def _is_verify_intent(intent: str) -> bool:
    lowered = str(intent or "").strip().lower()
    if not lowered:
        return False
    return any(k in lowered for k in ("verify", "verification", "validate", "check", "self-check", "test"))


def _tool_hint_is_unmapped(hint: str) -> bool:
    value = str(hint or "").strip().lower()
    if not value:
        return True
    return value in {"n/a", "na", "none", "unknown", "not applicable", "not_applicable", "any", "*"}


def _todo_directive_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_.\\/:-]+", str(text or "").lower())
    stop = {"create", "update", "make", "new", "file", "directory", "the", "and", "with", "into", "in", "to"}
    out: list[str] = []
    for tok in tokens:
        if len(tok) < 3:
            continue
        if tok in stop:
            continue
        out.append(tok)
    return out


def _evidence_semantically_matches_todo(ev: Evidence, item: dict[str, Any]) -> bool:
    directive = str(item.get("directive") or "").strip().lower()
    if not directive:
        return False
    args = ev.args if isinstance(ev.args, dict) else {}
    hay_parts = [
        str(ev.tool or ""),
        str(ev.output or "")[:400],
        str(args.get("path") or ""),
        str(args.get("src") or ""),
        str(args.get("dst") or ""),
        str(args.get("file") or ""),
        str(args.get("command") or ""),
    ]
    hay = " ".join(hay_parts).lower()
    # Strong signal: full directive or filename/path fragment appears in evidence.
    if directive in hay:
        return True
    for quoted in re.findall(r"'([^']+)'|\"([^\"]+)\"", directive):
        candidate = (quoted[0] or quoted[1] or "").strip().lower()
        if candidate and candidate in hay:
            return True
    # Fallback: token overlap from directive.
    for tok in _todo_directive_tokens(directive):
        if tok in hay:
            return True
    return False


def _should_same_cycle_retry(step: Step, evidence: list[Evidence]) -> bool:
    if bool(getattr(step, "task_finished", False)):
        return False
    if not isinstance(getattr(step, "actions", None), list) or not step.actions:
        return False
    if not evidence:
        return True
    failed = [e for e in evidence if not bool(getattr(e, "ok", False))]
    return len(failed) > 0


def _build_same_cycle_reflect_message(
    *,
    prev_step: Step,
    evidence: list[Evidence],
    attempt: int,
    max_attempts: int,
) -> str:
    lines: list[str] = [
        "Same-cycle retry requested.",
        f"Attempt {attempt}/{max_attempts}.",
        "Your previous action results:",
    ]
    if not evidence:
        lines.append("- No tool output captured.")
    else:
        for ev in evidence[-6:]:
            status = "OK" if ev.ok else "FAIL"
            out = str(ev.output or "").replace("\n", " ").strip()
            if len(out) > 180:
                out = out[:180] + "..."
            lines.append(f"- {ev.tool}: {status} - {out or '<empty>'}")
    lines.extend([
        "",
        "Now respond with a new Step JSON for this same cycle.",
        "Rules:",
        "- If previous action failed, pick a different concrete action/tool or materially different args.",
        "- Do not repeat an identical failing action.",
        "- Set task_finished=true only if done with evidence.",
        "- If still not done, set task_finished=false and provide the next best action.",
    ])
    prev_intent = str(getattr(prev_step, "intent", "") or "").strip()
    if prev_intent:
        lines.append(f"Previous intent: {prev_intent}")
    return "\n".join(lines)


def _one_sentence(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    # Keep the first sentence-like segment for concise finish summaries.
    parts = re.split(r"(?<=[.!?])\s+", value)
    first = str(parts[0] or "").strip()
    return first or value


def _has_successful_verification(pad: Pad) -> bool:
    last_write_cycle = -1
    for ev in pad.evidence:
        if ev.tool in {"write_file", "apply_patch", "mkdir", "copy_path", "move_path"}:
            last_write_cycle = max(last_write_cycle, int(ev.cycle))
    if last_write_cycle < 0:
        return True
    for ev in pad.evidence:
        if int(ev.cycle) <= last_write_cycle:
            continue
        if not ev.ok:
            continue
        if ev.tool in {"read_file", "list_dir", "glob_files", "grep_search", "run_cmd", "run_tests"}:
            return True
    return False


def _has_mutation_evidence(pad: Pad) -> bool:
    for ev in pad.evidence:
        if ev.ok and ev.tool in {"write_file", "apply_patch", "mkdir", "copy_path", "move_path"}:
            return True
    return False


def _all_blocking_todos_done(pad: Pad) -> bool:
    if not pad.todo_state:
        return False
    for item in pad.todo_state:
        if bool(item.get("blocking", False)) and not bool(item.get("crystallized", False)):
            return False
    return True


def _is_discovery_only_step(step: Step, evidence: list[Evidence]) -> bool:
    allowed = {"read_file", "list_dir", "glob_files", "grep_search"}
    actions = list(getattr(step, "actions", []) or [])
    if not actions:
        return False
    for action in actions:
        if not isinstance(action, dict):
            return False
        tool = str(action.get("tool") or "").strip()
        if tool not in allowed:
            return False
    if not evidence:
        return False
    return all(ev.ok and ev.tool in allowed for ev in evidence)


def _intent_norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _should_auto_finish_after_redundant_verify(ctx: RunContext, step: Step, evidence: list[Evidence]) -> bool:
    if bool(getattr(step, "task_finished", False)):
        return False
    if not _all_blocking_todos_done(ctx.pad):
        return False
    if not _has_mutation_evidence(ctx.pad):
        return False
    if not _has_successful_verification(ctx.pad):
        return False
    if not _is_discovery_only_step(step, evidence):
        return False
    recent_intent = str(ctx.recent_intents[-1] if ctx.recent_intents else "").strip()
    if not recent_intent:
        return False
    # Require intent repetition so first verification pass still gets a chance.
    if _intent_norm(step.intent) != _intent_norm(recent_intent):
        return False
    return True


def _compose_auto_finish_summary(step: Step, pad: Pad) -> str:
    check = _one_sentence(str(getattr(step, "self_check", "") or "").strip())
    if check:
        return f"Task completed with verification evidence. {check}"
    goal = _one_sentence(str(getattr(pad, "goal", "") or "").strip())
    if goal:
        return f"Task completed with verification evidence: {goal}"
    return "Task completed with verification evidence."


def _build_pad(goal: str, preflight: PreflightResult | None, env_block: str) -> Pad:
    pad = Pad(
        goal=(preflight.vision_primary if preflight and preflight.vision_primary else goal),
        preflight=preflight,
        env_block=str(env_block or ""),
    )
    if preflight is not None:
        for item in preflight.todo:
            directive = str(item.get("directive") or "").strip()
            tool_hint = str(item.get("tool_hint") or "").strip()
            if not directive:
                continue
            pad.todo_state.append({
                "directive": directive,
                "tool_hint": tool_hint,
                "crystallized": False,
                "cycle_crystallized": None,
                "evidence_ref": None,
                "blocking": bool(item.get("blocking", False)),
            })
        for inv in preflight.invariants:
            pad.add_invariant(inv, origin="env")
        pad.plan = pad.render_todo_text()
    pad.progress = pad.compute_progress()
    return pad


# ======================================================================
# Serialization helpers  (for on_event payloads â€” keep them JSON-safe)
# ======================================================================

def _pad_snapshot(pad: Pad) -> dict[str, Any]:
    return {
        "goal": pad.goal,
        "plan": pad.plan[:200],
        "todo_state": list(pad.todo_state),
        "steps_count": len(pad.steps),
        "evidence_count": len(pad.evidence),
        "last_check": dict(pad.last_check) if isinstance(pad.last_check, dict) else None,
        "tool_failures_count": len(getattr(pad, "tool_failures", []) or []),
        "invariants_count": len(getattr(pad, "invariants", []) or []),
        "todo_open": sum(1 for t in pad.todo_state if not t.get("crystallized")),
        "progress": pad.progress,
    }


def _step_dict(step: Step) -> dict[str, Any]:
    return {
        "response": step.response[:300] if getattr(step, "response", "") else "",
        "intent": step.intent,
        "actions_count": len(step.actions),
        "step_ok": step.step_ok,
        "todo_update": str(step.todo_update or ""),
        "self_check": (step.self_check[:200] if getattr(step, "self_check", "") else ""),
        "task_finished": step.task_finished,
    }


def _result_dict(result: RunResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "success": result.success,
        "summary": result.summary[:300],
        "cycles_used": result.cycles_used,
        "tool_calls_used": result.tool_calls_used,
        "wall_hit": result.wall_hit,
    }


def _messages_digest(messages: list[dict[str, Any]]) -> str:
    try:
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    except Exception:
        payload = str(messages)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _messages_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        if isinstance(m, dict):
            total += len(str(m.get("content") or ""))
    return total


def _resolve_workspace_root_for_debug() -> Path | None:
    try:
        from engine.tools import get_boundary_config
        cfg = get_boundary_config()
        ws = str(cfg.get("workspace_root") or "").strip()
        if not ws:
            return None
        return Path(ws).expanduser().resolve()
    except Exception:
        return None


def _is_within_root(path: Path, root: Path) -> bool:
    p = str(path.resolve())
    r = str(root.resolve())
    return p == r or p.startswith(r + os.sep)


def _classify_path_scope(args: Any) -> str:
    if not isinstance(args, dict):
        return "none"
    ws = _resolve_workspace_root_for_debug()
    if ws is None:
        return "unknown"
    any_path = False
    outside = False
    for key in ("path", "src", "dst", "file"):
        v = args.get(key)
        if not isinstance(v, str) or not v.strip():
            continue
        any_path = True
        raw = v.strip()
        try:
            p = Path(raw).expanduser()
            target = p.resolve() if p.is_absolute() else (ws / p).resolve()
            if not _is_within_root(target, ws):
                outside = True
        except Exception:
            continue
    if not any_path:
        return "none"
    return "outside_workspace" if outside else "inside_workspace"


def _wall_check_detail(ctx: RunContext, step: Step) -> dict[str, Any]:
    elapsed = max(0.0, (time.time() - ctx.start_time) - float(getattr(ctx, "paused_sec", 0.0) or 0.0))
    hist = list(getattr(ctx.pad, "_progress_history", []) or [])
    recent_tail = hist[-int(max(1, getattr(ctx.policy, "stall_window", 4))):] if hist else []
    recent_intents = list(ctx.recent_intents[-max(1, int(getattr(ctx.policy, "repetition_window", 3) - 1)):])
    return {
        "run_id": ctx.run_id,
        "cycle": ctx.cycle,
        "pending_actions": len(getattr(step, "actions", []) or []),
        "tool_calls_used": int(ctx.total_tool_calls),
        "max_tool_calls": int(ctx.policy.max_tool_calls),
        "elapsed_sec": round(float(elapsed), 3),
        "max_elapsed_sec": float(ctx.policy.max_elapsed_sec),
        "progress_tail": recent_tail,
        "recent_intents_tail": recent_intents,
        "stall_window": int(ctx.policy.stall_window),
        "repetition_window": int(ctx.policy.repetition_window),
    }


def _pad_state_brief(pad: Pad) -> dict[str, Any]:
    return {
        "plan": str(getattr(pad, "plan", "") or ""),
        "progress": float(getattr(pad, "progress", 0.0) or 0.0),
        "todo_open": sum(1 for t in (getattr(pad, "todo_state", []) or []) if not t.get("crystallized")),
        "steps_count": len(getattr(pad, "steps", []) or []),
        "evidence_count": len(getattr(pad, "evidence", []) or []),
        "open_questions_count": len(getattr(pad, "open_questions", []) or []),
    }


def _pad_state_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    keys = sorted(set(before.keys()) | set(after.keys()))
    for k in keys:
        if before.get(k) != after.get(k):
            diff[k] = {"before": before.get(k), "after": after.get(k)}
    return diff

