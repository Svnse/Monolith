from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from core.paths import CONFIG_DIR
from core.state import AppState, SystemStatus
from engine.llm import LLMEngine
from engine.loop import LoopRuntime, RunPolicy
from engine.loop.awareness import (
    HardwareProfile,
    RunProfile,
    RuntimeAwareness,
    load_hardware_profile,
    save_hardware_profile,
)
from engine.loop.commands import (
    CMD_APPROVAL_RESPONSE,
    CMD_GET_BOUNDARY_CONFIG,
    CMD_GET_LAST_RESULT,
    CMD_GET_RUN_JOURNAL,
    CMD_SET_BOUNDARY_CONFIG,
    CMD_SET_INFER_BACKEND,
    CMD_SET_INFER_CONFIG,
    CMD_SET_WORKSPACE_ROOT,
    CMD_SET_TOOLS,
    CMD_STOP,
)
from engine.loop.contracts import PreflightResult, RunResult, ToolSpec
from engine.loop.events import (
    CONTROL_EVENT_SPEC_VERSION,
    EVENT_CONTROL,
    CTRL_REDIRECT_STARTED,
    CTRL_APPROVAL_RESPONSE,
    CTRL_TURN_COMPLETED,
    CTRL_TURN_FAILED,
    CTRL_TURN_STARTED,
    CTRL_TURN_STOP_REQUESTED,
    REASON_COMPLETED,
    REASON_ERROR,
    REASON_APPROVAL_DENIED,
    REASON_APPROVAL_GRANTED,
    REASON_STOP_REQUESTED,
    REASON_TURN_STARTED,
    REASON_USER_REDIRECT,
    REASON_WALL_HIT,
    RUN_STATE_COMPLETED,
    RUN_STATE_FAILED,
    RUN_STATE_REDIRECTED,
    RUN_STATE_RUNNING,
    RUN_STATE_STOPPED,
    RUN_STATE_STOPPING,
)
from engine.loop.monolith_adapter import (
    LLMEngineInferAdapter,
    build_monolith_tool_specs,
    execute_monolith_tool,
    make_loop_trace_emitter,
)
from engine.loop.preflight import run_preflight
from engine.loop.env_resolve import resolve_environment, render_env_block
from engine.tools import get_boundary_config, set_boundary_config, set_workspace_root


class _LoopWorker(QThread):
    token = Signal(str)
    trace = Signal(str)
    event = Signal(dict)
    done = Signal(object)   # RunResult
    failed = Signal(str)

    def __init__(
        self,
        *,
        goal: str,
        user_prompt: str,
        run_id: str,
        preflight: PreflightResult | None,
        preflight_infer_fn,
        on_runtime_stop_requested,
        env_block: str,
        tools: list[ToolSpec],
        policy: RunPolicy,
        infer_fn,
        structured_infer_fn,
        target_lines_per_step: int | None = None,
        tool_executor,
    ) -> None:
        super().__init__()
        self.goal = goal
        self._user_prompt = str(user_prompt or "")
        self._preflight = preflight
        self._preflight_infer_fn = preflight_infer_fn
        self._on_runtime_stop_requested = on_runtime_stop_requested
        self._env_block = str(env_block or "")
        self._run_id = str(run_id or "").strip()
        self.tools = tools
        self.policy = policy
        self._target_lines_per_step = (
            int(target_lines_per_step) if target_lines_per_step is not None else None
        )
        self._runtime = LoopRuntime(
            infer_fn=infer_fn,
            structured_infer_fn=structured_infer_fn,
            tool_executor=tool_executor,
            approval_fn=self._request_approval,
            policy=policy,
            on_event=self._on_loop_event,
            on_stop_requested=self._on_runtime_stop_requested,
        )
        self._approval_lock = threading.Lock()
        self._approval_pending: dict[str, dict[str, Any]] = {}
        if not self._run_id:
            self._run_id = ""

    def request_stop(self) -> None:
        try:
            self._runtime.stop()
        except Exception:
            pass
        with self._approval_lock:
            for item in self._approval_pending.values():
                item["decision"] = False
                item["event"].set()
        self.requestInterruption()

    def resolve_approval(self, request_id: str, allow: bool) -> bool:
        rid = str(request_id or "").strip()
        with self._approval_lock:
            item = self._approval_pending.get(rid)
            if item is None:
                return False
            item["decision"] = bool(allow)
            item["event"].set()
            return True

    def _on_loop_event(self, kind: str, data: dict[str, Any]) -> None:
        if isinstance(data, dict):
            rid = str(data.get("run_id") or "").strip()
            if rid:
                self._run_id = rid
        payload = {
            "event": "loop",
            "kind": kind,
            "data": data,
        }
        self.event.emit(payload)

    def _request_approval(self, tool_name: str, args: dict[str, Any], spec: ToolSpec) -> bool:
        request_id = f"apr_{uuid.uuid4().hex[:10]}"
        gate = threading.Event()
        item = {
            "event": gate,
            "decision": None,
            "tool": tool_name,
            "args": dict(args or {}),
            "scope": str(getattr(spec, "scope", "")),
        }
        with self._approval_lock:
            self._approval_pending[request_id] = item

        self.event.emit({
            "event": "loop",
            "kind": "approval_prompt",
            "data": {
                "request_id": request_id,
                "run_id": self._run_id,
                "tool": tool_name,
                "scope": str(getattr(spec, "scope", "")),
                "args": dict(args or {}),
            },
        })

        # Block worker thread until approved/denied/stop. Timeout fails closed.
        gate.wait(timeout=300.0)
        with self._approval_lock:
            pending = self._approval_pending.pop(request_id, None)
        if pending is None:
            return False
        return bool(pending.get("decision"))

    def run(self) -> None:
        try:
            preflight = self._preflight
            preflight_elapsed_sec = 0.0
            if preflight is None and callable(self._preflight_infer_fn):
                self.event.emit({
                    "event": "loop",
                    "kind": "preflight_started",
                    "data": {
                        "run_id": self._run_id,
                        "user_prompt_preview": self._user_prompt[:180],
                    },
                })
                pf_started = datetime.now(timezone.utc).timestamp()
                preflight = run_preflight(
                    self._user_prompt,
                    self._env_block,
                    self._preflight_infer_fn,
                    tools=self.tools,
                    target_lines_per_step=self._target_lines_per_step,
                )
                preflight_elapsed_sec = max(0.0, datetime.now(timezone.utc).timestamp() - pf_started)
                self.event.emit({
                    "event": "loop",
                    "kind": "preflight_result",
                    "data": {
                        "run_id": self._run_id,
                        "ok": bool(preflight is not None),
                        "todo_items": len(list(preflight.todo or [])) if preflight is not None else 0,
                        "elapsed_sec": round(preflight_elapsed_sec, 3),
                    },
                })
                if preflight is None:
                    self.trace.emit("[LOOP] preflight unavailable; continuing with raw user prompt")
                else:
                    self.trace.emit(f"[LOOP] preflight ready: todo_items={len(list(preflight.todo or []))}")

            goal = self.goal
            if preflight is not None:
                vision = str(preflight.vision_primary or "").strip()
                if vision:
                    goal = vision
            result = self._runtime.run(
                goal,
                self.tools,
                run_id=(self._run_id or None),
                preflight=preflight,
                env_block=self._env_block,
                pre_run_elapsed_sec=preflight_elapsed_sec,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.done.emit(result)


class LoopEngine(QObject):
    """
    EnginePort-compatible wrapper for engine.loop.

    Owns a dedicated LLMEngine (model lifecycle) and runs LoopRuntime inside a
    separate QThread so MonoGuard can stop it like any other engine.
    """

    sig_token = Signal(str)
    sig_trace = Signal(str)
    sig_status = Signal(SystemStatus)
    sig_finished = Signal()
    sig_usage = Signal(int)
    sig_agent_event = Signal(dict)
    sig_model_capabilities = Signal(dict)
    _RUN_JOURNAL_LIMIT = 500

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._llm = LLMEngine(state)
        self._worker: _LoopWorker | None = None
        self._status: SystemStatus = SystemStatus.READY
        self._last_result: RunResult | None = None
        self._tool_allowlist: list[str] | None = None
        self._infer_defaults: dict[str, Any] = {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 1024,
        }
        self._infer_backend: str = "json"
        self._event_seq: int = 0
        self._active_turn_meta: dict[str, Any] = {}
        self._stop_requested: bool = False
        self._stop_request_run_id: str = ""
        self._run_effect_journal: dict[str, list[dict[str, Any]]] = {}
        self._awareness_enabled: bool = True
        self._awareness = RuntimeAwareness()
        self._config_dir = CONFIG_DIR
        self._hardware_profile: HardwareProfile = load_hardware_profile(self._config_dir)
        self._awareness_run_profiles: dict[str, RunProfile] = {}

        self._llm.sig_trace.connect(self._on_llm_trace)
        self._llm.sig_status.connect(self._on_llm_status)
        self._llm.sig_model_capabilities.connect(self.sig_model_capabilities)
        self._llm.sig_usage.connect(self.sig_usage)

    # ---------------- EnginePort methods ----------------
    def set_model_path(self, payload: dict) -> None:
        self._llm.set_model_path(payload)

    def set_ctx_limit(self, payload: dict) -> None:
        self._llm.set_ctx_limit(payload)

    def set_history(self, payload: dict) -> None:
        # Loop runtime is pad-based; history is intentionally ignored.
        self.sig_trace.emit("[LOOP] set_history ignored (loop runtime is pad-based)")

    def load_model(self) -> None:
        self._llm.load_model()

    def unload_model(self) -> None:
        if self._is_loop_running():
            self.sig_trace.emit("[LOOP] unload blocked while run is active")
            return
        self._llm.unload_model()

    def generate(self, payload: dict) -> None:
        if self._is_loop_running():
            self.sig_trace.emit("ERROR: Loop busy. Wait for completion.")
            self._set_status(SystemStatus.ERROR)
            return
        if not bool(getattr(self._llm, "model_loaded", False)):
            self.sig_trace.emit("ERROR: Model offline.")
            self._set_status(SystemStatus.ERROR)
            return

        goal = str((payload or {}).get("goal") or (payload or {}).get("prompt") or "").strip()
        if not goal:
            self.sig_trace.emit("ERROR: Loop goal/prompt is empty.")
            self._set_status(SystemStatus.ERROR)
            return

        turn_meta = dict((payload or {}).get("_turn_meta") or {}) if isinstance((payload or {}).get("_turn_meta"), dict) else {}
        self._active_turn_meta = {
            "session_id": str(turn_meta.get("session_id") or ""),
            "turn_id": str(turn_meta.get("turn_id") or ""),
            "parent_run_id": str(turn_meta.get("parent_run_id") or ""),
            "redirected": bool(turn_meta.get("redirected", False)),
            "user_prompt": str(turn_meta.get("user_prompt") or ""),
            "run_id": uuid.uuid4().hex[:12],
        }

        policy = self._policy_from_payload(payload or {})
        tool_names = self._tool_names_from_payload(payload or {})
        tools = build_monolith_tool_specs(tool_names=tool_names)
        infer_cfg = self._infer_config_from_payload(payload or {})
        infer_backend = self._infer_backend_from_payload(payload or {})
        ENVELOPE_OVERHEAD_TOKENS = 350
        CHARS_PER_TOKEN = 3.5
        CHARS_PER_LINE = 72
        SAFETY_FACTOR = 0.6
        usable_content_tokens = int(infer_cfg["max_tokens"]) - ENVELOPE_OVERHEAD_TOKENS
        usable_lines = (usable_content_tokens * CHARS_PER_TOKEN) / CHARS_PER_LINE
        target_lines = max(15, int(usable_lines * SAFETY_FACTOR))
        user_prompt = str(self._active_turn_meta.get("user_prompt") or goal).strip()
        run_id = str(self._active_turn_meta.get("run_id") or "")
        self._refresh_hardware_profile_for_current_model()
        if self._awareness_enabled:
            verdict = self._awareness.pre_run_gate(
                policy=policy,
                infer_config=infer_cfg,
                hardware=self._hardware_profile,
                preflight=None,
                user_prompt=user_prompt,
            )
            for warning in verdict.warnings:
                self.sig_trace.emit(f"[AWARENESS] {warning}")
            if verdict.adjusted_policy is not None:
                policy = verdict.adjusted_policy
            if verdict.adjusted_infer is not None:
                infer_cfg = {**infer_cfg, **verdict.adjusted_infer}
        if self._awareness_enabled and run_id:
            self._awareness_run_profiles[run_id] = self._awareness.start_run(
                run_id=run_id,
                max_elapsed_sec=float(policy.max_elapsed_sec),
                max_tokens=int(infer_cfg.get("max_tokens", 0) or 0),
                wall_start=time.time(),
            )
        boundary = get_boundary_config()
        workspace_root = str(boundary.get("workspace_root") or "")
        env = resolve_environment(workspace_root=workspace_root)
        env_block = render_env_block(env)

        preflight_infer = LLMEngineInferAdapter(
            self._llm,
            temperature=float(infer_cfg["temperature"]),
            top_p=float(infer_cfg["top_p"]),
            max_tokens=int(infer_cfg["max_tokens"]),
            stream_tokens=False,
            on_token=None,
            should_stop=lambda: bool(self._stop_requested),
            enforce_json_object=True,
        )
        self.sig_trace.emit("[LOOP] preflight queued (worker thread)")
        preflight: PreflightResult | None = None
        goal = user_prompt
        wall_deadline = time.time() + float(policy.max_elapsed_sec) + 30.0
        wall_guard_emitted = False

        def _should_stop_with_wall() -> bool:
            nonlocal wall_guard_emitted
            if self._stop_requested:
                return True
            if time.time() >= (wall_deadline - 5.0):
                if not wall_guard_emitted:
                    self.sig_trace.emit("[AWARENESS] wall_guard: streaming stopped near max_elapsed deadline.")
                    wall_guard_emitted = True
                return True
            return False

        infer_fn = LLMEngineInferAdapter(
            self._llm,
            temperature=float(infer_cfg["temperature"]),
            top_p=float(infer_cfg["top_p"]),
            max_tokens=int(infer_cfg["max_tokens"]),
            # Use streaming internally so STOP can break out mid-inference.
            # Token UI streaming stays off for CODE timeline clarity.
            stream_tokens=True,
            on_token=None,
            should_stop=_should_stop_with_wall,
            enforce_json_object=(infer_backend == "json"),
        )
        structured_infer_fn = self._structured_infer_from_backend(infer_backend, payload or {})
        tool_exec = execute_monolith_tool
        def _runtime_stop_requested() -> None:
            self._stop_requested = True
            try:
                self._llm.stop_generation()
            except Exception:
                pass

        self._worker = _LoopWorker(
            goal=goal,
            user_prompt=user_prompt,
            run_id=run_id,
            preflight=preflight,
            preflight_infer_fn=preflight_infer,
            on_runtime_stop_requested=_runtime_stop_requested,
            env_block=env_block,
            tools=tools,
            policy=policy,
            infer_fn=infer_fn,
            structured_infer_fn=structured_infer_fn,
            target_lines_per_step=target_lines,
            tool_executor=tool_exec,
        )
        self._worker.token.connect(self.sig_token)
        self._worker.trace.connect(self.sig_trace)
        self._worker.event.connect(self._on_worker_event)
        self._worker.done.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._on_worker_thread_finished)
        self._stop_requested = False
        self._stop_request_run_id = ""
        self._set_status(SystemStatus.RUNNING)
        self._emit_control_event(
            CTRL_TURN_STARTED,
            {
                "goal_preview": goal[:240],
                "run_id": str(self._active_turn_meta.get("run_id") or ""),
                "redirected": bool(self._active_turn_meta.get("redirected")),
                "parent_run_id": str(self._active_turn_meta.get("parent_run_id") or ""),
                "infer_backend": infer_backend,
                "tool_names": [t.name for t in tools],
                "policy": {
                    "max_cycles": int(policy.max_cycles),
                    "max_tool_calls": int(policy.max_tool_calls),
                    "auto_approve": sorted(str(s) for s in policy.auto_approve),
                    "require_approval": sorted(str(s) for s in policy.require_approval),
                },
                "preflight_ready": bool(preflight is not None),
                "risk": self._risk_summary_for_tools(tools),
                "state": RUN_STATE_RUNNING,
                "reason_code": REASON_TURN_STARTED,
                "baml_call": str((payload or {}).get("baml_call") or ""),
            },
        )
        if bool(self._active_turn_meta.get("redirected")):
            self._emit_control_event(
                CTRL_REDIRECT_STARTED,
                {
                    "parent_run_id": str(self._active_turn_meta.get("parent_run_id") or ""),
                    "run_id": str(self._active_turn_meta.get("run_id") or ""),
                    "goal_preview": goal[:240],
                    "state": RUN_STATE_REDIRECTED,
                    "reason_code": REASON_USER_REDIRECT,
                },
            )

        # Human-readable loop trace lines derived from the same event stream.
        trace_bridge = make_loop_trace_emitter(self.sig_trace.emit)
        self._worker.event.connect(lambda ev, tr=trace_bridge: tr(str(ev.get("kind") or ""), ev.get("data") or {}))

        self._worker.start()

    def runtime_command(self, command: str, payload: dict | None = None) -> dict:
        cmd = str(command or "").strip().lower()
        data = payload if isinstance(payload, dict) else {}
        if cmd == CMD_STOP:
            self.stop_generation()
            return {"ok": True}
        if cmd == CMD_SET_TOOLS:
            names = data.get("tool_names")
            if isinstance(names, list):
                self._tool_allowlist = [str(n) for n in names if isinstance(n, str) and n]
                return {"ok": True, "tool_names": list(self._tool_allowlist)}
            return {"ok": False, "error": "tool_names must be a list[str]"}
        if cmd == CMD_SET_INFER_CONFIG:
            updated = dict(self._infer_defaults)
            for key in ("temperature", "top_p", "max_tokens"):
                if key in data:
                    updated[key] = data[key]
            try:
                self._infer_defaults = self._normalize_infer_config(updated)
            except Exception as exc:
                return {"ok": False, "error": f"invalid infer config: {exc}"}
            return {"ok": True, "infer": dict(self._infer_defaults)}
        if cmd == CMD_SET_INFER_BACKEND:
            backend = str(data.get("backend") or "").strip().lower()
            if backend not in {"json", "baml"}:
                return {"ok": False, "error": "backend must be 'json' or 'baml'"}
            self._infer_backend = backend
            return {"ok": True, "backend": self._infer_backend}
        if cmd == CMD_GET_LAST_RESULT:
            if self._last_result is None:
                return {"ok": True, "result": None}
            return {
                "ok": True,
                "result": {
                    "run_id": self._last_result.run_id,
                    "success": self._last_result.success,
                    "summary": self._last_result.summary,
                    "cycles_used": self._last_result.cycles_used,
                    "tool_calls_used": self._last_result.tool_calls_used,
                    "wall_hit": self._last_result.wall_hit,
                },
            }
        if cmd == CMD_GET_RUN_JOURNAL:
            run_id = str(data.get("run_id") or "").strip()
            if not run_id:
                run_id = str(getattr(self._last_result, "run_id", "") or "")
            if not run_id and self._worker is not None:
                run_id = str(getattr(self._worker, "_run_id", "") or "")
            entries = list(self._run_effect_journal.get(run_id, [])) if run_id else []
            kind_filter = data.get("kind")
            if isinstance(kind_filter, str) and kind_filter.strip():
                kf = kind_filter.strip()
                entries = [e for e in entries if str(e.get("kind") or "") == kf]
            elif isinstance(kind_filter, list):
                kinds = {str(k).strip() for k in kind_filter if str(k).strip()}
                if kinds:
                    entries = [e for e in entries if str(e.get("kind") or "") in kinds]
            if bool(data.get("reverse", False)):
                entries = list(reversed(entries))
            limit_raw = data.get("limit")
            try:
                limit = int(limit_raw) if limit_raw is not None else 0
            except Exception:
                limit = 0
            if limit > 0:
                entries = entries[:limit]
            resp: dict[str, Any] = {"ok": True, "run_id": run_id, "entries": entries}
            if bool(data.get("include_summary", False)):
                resp["summary"] = self._summarize_run_journal(run_id)
            return resp
        if cmd == CMD_GET_BOUNDARY_CONFIG:
            return {"ok": True, "boundary": get_boundary_config()}
        if cmd == CMD_SET_BOUNDARY_CONFIG:
            try:
                boundary = set_boundary_config(
                    mode=data.get("mode"),
                    allowed_roots=data.get("allowed_roots") if isinstance(data.get("allowed_roots"), list) else None,
                )
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "boundary": boundary}
        if cmd == CMD_SET_WORKSPACE_ROOT:
            raw_path = data.get("path")
            path = str(raw_path).strip() if raw_path is not None else ""
            if not path:
                return {"ok": False, "error": "path is required"}
            try:
                set_workspace_root(path)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "boundary": get_boundary_config()}
        if cmd == CMD_APPROVAL_RESPONSE:
            request_id = str(data.get("request_id") or "")
            allow = bool(data.get("allow", False))
            worker = self._worker
            if worker is None:
                return {"ok": False, "error": "no active loop worker"}
            ok = worker.resolve_approval(request_id, allow)
            run_id = str(getattr(worker, "_run_id", "") or self._stop_request_run_id or "")
            if run_id and request_id:
                self._emit_control_event(
                    CTRL_APPROVAL_RESPONSE,
                    {
                        "run_id": run_id,
                        "request_id": request_id,
                        "allow": allow,
                        "state": RUN_STATE_RUNNING if allow else RUN_STATE_STOPPING,
                        "reason_code": REASON_APPROVAL_GRANTED if allow else REASON_APPROVAL_DENIED,
                    },
                )
            return {"ok": ok, "request_id": request_id, "allow": allow}
        return {"ok": False, "error": f"unsupported runtime_command: {command}"}

    def stop_generation(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.sig_trace.emit("[LOOP] stop requested")
            self._stop_requested = True
            self._stop_request_run_id = str(getattr(self._worker, "_run_id", "") or "")
            try:
                self._llm.stop_generation()
            except Exception as exc:
                self.sig_trace.emit(f"[LOOP] stop propagation warning: {exc}")
            self._emit_control_event(
                CTRL_TURN_STOP_REQUESTED,
                {
                    "state": RUN_STATE_STOPPING,
                    "reason_code": REASON_STOP_REQUESTED,
                    "run_id": str(getattr(self._worker, "_run_id", "") or ""),
                },
            )
            self._worker.request_stop()
        else:
            self._set_status(SystemStatus.READY)

    def shutdown(self) -> None:
        self.stop_generation()
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.wait(3000)
            self._worker = None
        self._llm.shutdown()

    # ---------------- internals ----------------
    def _set_status(self, status: SystemStatus) -> None:
        self._status = status
        self.sig_status.emit(status)

    def _is_loop_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    @staticmethod
    def _risk_level_for_scope(scope: str) -> str:
        s = str(scope or "").strip().lower()
        if s in {"write", "delete"}:
            return "red"
        if s in {"execute", "shell"}:
            return "yellow"
        return "green"

    def _risk_summary_for_tools(self, tools: list[ToolSpec]) -> dict[str, Any]:
        counts = {"green": 0, "yellow": 0, "red": 0}
        by_tool: list[dict[str, str]] = []
        max_level = "green"
        order = {"green": 0, "yellow": 1, "red": 2}
        for t in tools:
            scope = str(getattr(t, "scope", "") or "")
            level = self._risk_level_for_scope(scope)
            counts[level] = counts.get(level, 0) + 1
            by_tool.append({"tool": str(t.name), "scope": scope, "risk": level})
            if order[level] > order[max_level]:
                max_level = level
        return {"overall": max_level, "counts": counts, "tools": by_tool}

    def _refresh_hardware_profile_for_current_model(self) -> None:
        model_hash = str(getattr(self._llm, "_model_fingerprint", "") or "")
        n_ctx = int(getattr(self._llm, "ctx_limit", 0) or 0)
        needs_reload = (
            (model_hash and model_hash != str(self._hardware_profile.model_path_hash or ""))
            or (n_ctx > 0 and int(self._hardware_profile.n_ctx or 0) != n_ctx)
        )
        if not needs_reload:
            return
        self._hardware_profile = load_hardware_profile(
            self._config_dir,
            model_path_hash=model_hash,
            n_ctx=n_ctx,
        )

    def _policy_from_payload(self, payload: dict) -> RunPolicy:
        raw = payload.get("policy")
        if not isinstance(raw, dict):
            return RunPolicy()
        base = RunPolicy()
        kwargs: dict[str, Any] = {}
        for key in (
            "max_cycles",
            "max_tool_calls",
            "max_elapsed_sec",
            "max_retries",
            "stall_window",
            "repetition_window",
        ):
            if key in raw:
                kwargs[key] = raw[key]
        for key in ("auto_approve", "require_approval"):
            if key in raw and isinstance(raw[key], (list, tuple, set, frozenset)):
                kwargs[key] = frozenset(str(v) for v in raw[key])
        try:
            return RunPolicy(**kwargs)
        except Exception as exc:
            self.sig_trace.emit(f"[LOOP] invalid policy override ignored: {exc}")
            return base

    def _tool_names_from_payload(self, payload: dict) -> list[str] | None:
        names = payload.get("tool_names")
        if isinstance(names, list):
            return [str(n) for n in names if isinstance(n, str) and n.strip()]
        if self._tool_allowlist:
            return list(self._tool_allowlist)
        return None

    def _normalize_infer_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        cfg = {
            "temperature": float(raw.get("temperature", self._infer_defaults["temperature"])),
            "top_p": float(raw.get("top_p", self._infer_defaults["top_p"])),
            "max_tokens": int(raw.get("max_tokens", self._infer_defaults["max_tokens"])),
        }
        cfg["temperature"] = max(0.0, min(2.0, cfg["temperature"]))
        cfg["top_p"] = max(0.0, min(1.0, cfg["top_p"]))
        # Loop Step JSON often embeds substantive file content; keep a sane floor
        # to reduce truncation-driven malformed/empty-action retries.
        cfg["max_tokens"] = max(4096, min(16384, cfg["max_tokens"]))
        return cfg

    def _infer_config_from_payload(self, payload: dict) -> dict[str, Any]:
        raw = payload.get("infer")
        if not isinstance(raw, dict):
            return dict(self._infer_defaults)
        try:
            return self._normalize_infer_config(raw)
        except Exception as exc:
            self.sig_trace.emit(f"[LOOP] invalid infer config override ignored: {exc}")
            return dict(self._infer_defaults)

    def _infer_backend_from_payload(self, payload: dict) -> str:
        backend = str(payload.get("infer_backend") or self._infer_backend or "json").strip().lower()
        if backend not in {"json", "baml"}:
            self.sig_trace.emit(f"[LOOP] invalid infer_backend '{backend}' ignored; using json")
            return "json"
        return backend

    def _structured_infer_from_backend(self, backend: str, payload: dict[str, Any] | None = None):
        if backend != "baml":
            return None
        data = payload if isinstance(payload, dict) else {}
        baml_call = str(data.get("baml_call") or "").strip()
        baml_kwargs = data.get("baml_kwargs")
        if not isinstance(baml_kwargs, dict):
            baml_kwargs = {}
        try:
            from engine.loop.baml_step_infer import BamlStepInferAdapter
            adapter = BamlStepInferAdapter.from_config(
                call_path=baml_call,
                call_kwargs=baml_kwargs,
            )
            if baml_call:
                self.sig_trace.emit(f"[LOOP] BAML structured inference enabled ({baml_call})")
            else:
                self.sig_trace.emit("[LOOP] BAML structured inference enabled (env)")
            return adapter
        except Exception as exc:
            if baml_call:
                self.sig_trace.emit(f"[LOOP] BAML unavailable ({exc}) for '{baml_call}'; falling back to JSON parser")
            else:
                self.sig_trace.emit(f"[LOOP] BAML unavailable ({exc}); set baml_call or MONOLITH_LOOP_BAML_CALL; falling back to JSON parser")
            return None

    def _on_llm_trace(self, msg: str) -> None:
        self.sig_trace.emit(msg)

    def _on_llm_status(self, status: SystemStatus) -> None:
        # During a loop run, the loop engine owns RUNNING/READY state. We still
        # forward load/unload/error transitions from the underlying LLM engine.
        if self._is_loop_running() and status == SystemStatus.READY:
            return
        if self._is_loop_running() and status == SystemStatus.RUNNING:
            return
        self._set_status(status)

    def _on_worker_event(self, payload: dict) -> None:
        if self._awareness_enabled and isinstance(payload, dict):
            kind = str(payload.get("kind") or "")
            data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
            run_id = str(data.get("run_id") or self._active_turn_meta.get("run_id") or "")
            profile = self._awareness_run_profiles.get(run_id)
            if profile is not None:
                advisories = self._awareness.observe_event(
                    profile=profile,
                    hardware=self._hardware_profile,
                    kind=kind,
                    data=data,
                )
                for advisory in advisories:
                    self.sig_trace.emit(f"[AWARENESS] {advisory}")
        enriched = self._decorate_agent_event(payload)
        self._journal_agent_event(enriched)
        self.sig_agent_event.emit(enriched)

    def _on_worker_done(self, result: object) -> None:
        completion_data: dict[str, Any] = {}
        if isinstance(result, RunResult):
            self._last_result = result
            wall_hit = str(result.wall_hit or "").strip()
            success = bool(result.success)
            reason_code = REASON_COMPLETED
            if not success:
                if self._stop_requested and (wall_hit in {"", "stopped"}):
                    reason_code = REASON_STOP_REQUESTED
                elif wall_hit:
                    reason_code = REASON_WALL_HIT
            completion_data = {
                "run_id": result.run_id,
                "success": success,
                "wall_hit": result.wall_hit,
                "cycles_used": int(result.cycles_used),
                "tool_calls_used": int(result.tool_calls_used),
                "reason_code": reason_code,
            }
            summary = result.summary or ""
            if summary:
                self.sig_token.emit(summary)
            self.sig_usage.emit(int(result.cycles_used))
            if result.success:
                self.sig_trace.emit(
                    f"[LOOP] completed cycles={result.cycles_used} tools={result.tool_calls_used}"
                )
            else:
                self.sig_trace.emit(
                    f"[LOOP] terminated wall={result.wall_hit} cycles={result.cycles_used} tools={result.tool_calls_used}"
                )
        if completion_data:
            success = bool(completion_data.get("success", False))
            completion_data["state"] = RUN_STATE_COMPLETED if success else RUN_STATE_STOPPED
            completion_data.setdefault("reason_code", REASON_COMPLETED if success else REASON_WALL_HIT)
        if self._awareness_enabled and isinstance(result, RunResult):
            run_id = str(result.run_id or "")
            profile = self._awareness_run_profiles.pop(run_id, None)
            if profile is not None:
                self._hardware_profile = self._awareness.record(
                    profile=profile,
                    hardware=self._hardware_profile,
                )
                try:
                    save_hardware_profile(self._config_dir, self._hardware_profile)
                    trunc_rate = 0.0
                    if int(self._hardware_profile.truncation_total or 0) > 0:
                        trunc_rate = float(self._hardware_profile.truncation_hits) / float(self._hardware_profile.truncation_total)
                    self.sig_trace.emit(
                        "[AWARENESS] recorded: "
                        f"latency_ema={self._hardware_profile.latency_ema:.1f}s, "
                        f"truncation_rate={trunc_rate:.0%}, "
                        f"samples={int(self._hardware_profile.latency_samples)}"
                    )
                except Exception as exc:
                    self.sig_trace.emit(f"[AWARENESS] record warning: failed to save hardware profile ({exc})")
        self._emit_control_event(CTRL_TURN_COMPLETED, completion_data)
        self.sig_finished.emit()
        self._set_status(SystemStatus.READY)

    def _on_worker_failed(self, error: str) -> None:
        if self._awareness_enabled:
            run_id = str(self._active_turn_meta.get("run_id") or self._stop_request_run_id or "")
            profile = self._awareness_run_profiles.pop(run_id, None)
            if profile is not None:
                self._hardware_profile = self._awareness.record(
                    profile=profile,
                    hardware=self._hardware_profile,
                )
                try:
                    save_hardware_profile(self._config_dir, self._hardware_profile)
                except Exception:
                    pass
        self._emit_control_event(
            CTRL_TURN_FAILED,
            {
                "error": str(error or ""),
                "state": RUN_STATE_FAILED,
                "reason_code": REASON_ERROR,
            },
        )
        self.sig_trace.emit(f"[LOOP] ERROR: {error}")
        self.sig_finished.emit()
        self._set_status(SystemStatus.ERROR)
        self._set_status(SystemStatus.READY)
        self._stop_requested = False
        self._stop_request_run_id = ""

    def _on_worker_thread_finished(self) -> None:
        # Keep reference until thread fully exits, then drop it.
        worker = self._worker
        if worker is not None and not worker.isRunning():
            rid = str(getattr(worker, "_run_id", "") or self._active_turn_meta.get("run_id") or "")
            if rid:
                self._awareness_run_profiles.pop(rid, None)
            self._worker = None
            self._active_turn_meta = {}
            self._stop_requested = False
            self._stop_request_run_id = ""

    def _next_event_meta(self) -> dict[str, Any]:
        self._event_seq += 1
        return {
            "spec_version": CONTROL_EVENT_SPEC_VERSION,
            "seq": self._event_seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": str(self._active_turn_meta.get("session_id") or ""),
            "turn_id": str(self._active_turn_meta.get("turn_id") or ""),
            "parent_run_id": str(self._active_turn_meta.get("parent_run_id") or ""),
            "run_id": str(self._active_turn_meta.get("run_id") or ""),
        }

    def _decorate_agent_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {"event": "unknown", "kind": "unknown", "data": {}}
        enriched = dict(payload)
        enriched["_meta"] = self._next_event_meta()
        return enriched

    def _emit_control_event(self, kind: str, data: dict[str, Any] | None = None) -> None:
        payload_data = dict(data or {})
        payload_data.setdefault("reason_code", "")
        payload_data.setdefault("run_id", str(self._active_turn_meta.get("run_id") or ""))
        payload = {
            "event": EVENT_CONTROL,
            "kind": str(kind or "unknown"),
            "data": payload_data,
            "_meta": self._next_event_meta(),
        }
        self._journal_agent_event(payload)
        self.sig_agent_event.emit(payload)

    def _journal_agent_event(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        event_type = str(payload.get("event") or "").strip()
        kind = str(payload.get("kind") or "").strip()
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        meta = payload.get("_meta") if isinstance(payload.get("_meta"), dict) else {}

        run_id = str(data.get("run_id") or "").strip()
        if not run_id and event_type == "control":
            run_id = str(self._stop_request_run_id or "")
        if not run_id:
            return

        entry: dict[str, Any] = {
            "event": event_type,
            "kind": kind,
            "entry_type": f"{event_type}.{kind}" if kind else event_type,
            "run_id": run_id,
            "seq": meta.get("seq"),
            "ts": meta.get("ts"),
            "turn_id": meta.get("turn_id"),
            "session_id": meta.get("session_id"),
        }

        if event_type == "control":
            for key in ("state", "reason_code", "error", "success", "wall_hit", "cycles_used", "tool_calls_used", "parent_run_id"):
                if key in data:
                    entry[key] = data.get(key)
            if "request_id" in data:
                entry["request_id"] = data.get("request_id")
            if "allow" in data:
                entry["allow"] = bool(data.get("allow"))
        elif event_type == "loop":
            if kind in {"policy_decision", "tool_started", "action_result", "approval_prompt", "approval_required", "step_parsed", "finish"}:
                entry["cycle"] = data.get("cycle")
                for key in ("tool", "scope", "action", "reason_code", "status", "ok", "error", "truncated"):
                    if key in data:
                        entry[key] = data.get(key)
                if "args" in data and isinstance(data.get("args"), dict):
                    entry["args"] = dict(data.get("args") or {})
                if kind == "action_result":
                    entry["output_preview"] = str(data.get("output_preview") or "")
                elif kind == "step_parsed":
                    entry["step_ok"] = data.get("step_ok")
                    entry["self_check"] = str(data.get("self_check") or "")
                    step = data.get("step") if isinstance(data.get("step"), dict) else {}
                    if step:
                        task_finished = step.get("task_finished")
                        if not isinstance(task_finished, bool):
                            task_finished = bool(step.get("finish", False))
                        entry["step"] = {
                            "intent": str(step.get("intent") or ""),
                            "task_finished": bool(task_finished),
                            "actions_count": int(step.get("actions_count") or 0) if str(step.get("actions_count") or "").isdigit() else step.get("actions_count"),
                        }
                elif kind == "finish":
                    result = data.get("result") if isinstance(data.get("result"), dict) else {}
                    if result:
                        entry["result"] = {
                            "success": bool(result.get("success", False)),
                            "wall_hit": result.get("wall_hit"),
                            "cycles_used": result.get("cycles_used"),
                            "tool_calls_used": result.get("tool_calls_used"),
                        }
            else:
                return
        else:
            return

        # Keep a normalized payload snapshot for forward-compatible consumers,
        # while preserving top-level convenience fields used today.
        entry["payload"] = {
            "event": event_type,
            "kind": kind,
            "data": dict(data),
        }
        self._append_run_journal_entry(run_id, entry)

    def _append_run_journal_entry(self, run_id: str, entry: dict[str, Any]) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            return
        bucket = self._run_effect_journal.setdefault(rid, [])
        bucket.append(dict(entry or {}))
        if len(bucket) > self._RUN_JOURNAL_LIMIT:
            self._run_effect_journal[rid] = bucket[-self._RUN_JOURNAL_LIMIT:]

    def _summarize_run_journal(self, run_id: str) -> dict[str, Any]:
        rid = str(run_id or "").strip()
        entries = list(self._run_effect_journal.get(rid, [])) if rid else []
        by_kind: dict[str, int] = {}
        by_event: dict[str, int] = {}
        first_seq = None
        last_seq = None
        for e in entries:
            kind = str(e.get("kind") or "unknown")
            ev = str(e.get("event") or "unknown")
            by_kind[kind] = by_kind.get(kind, 0) + 1
            by_event[ev] = by_event.get(ev, 0) + 1
            seq = e.get("seq")
            if isinstance(seq, int):
                first_seq = seq if first_seq is None else min(first_seq, seq)
                last_seq = seq if last_seq is None else max(last_seq, seq)
        return {
            "run_id": rid,
            "entry_count": len(entries),
            "by_kind": by_kind,
            "by_event": by_event,
            "first_seq": first_seq,
            "last_seq": last_seq,
        }
