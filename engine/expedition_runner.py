"""ExpeditionRunner (MonoExplore V1, Seed B) — unattended expedition daemon.

COMPOSE (trace wegko2r7b): the runner drives a turn by composing three Qt-free
headless primitives — generate_sync_from_config + execute_tool_call_enveloped +
finalize_assistant_turn — NOT chat.py's Qt signal/timer driver (which stalls
without a QApplication event loop). This is the proven pattern monothink/planner
already use.

Why this matters for exploration: `generate_sync` forwards the runner's `messages`
VERBATIM (engine/sync_bridge.py:34) — it injects no system prompt, no
{skills_catalog}, no interceptors, no turn_classifier. The legacy Orient/Synthesis
LOOPs + RoE live in engine/llm.py's _compile_system_prompt + interceptor chain,
which this path bypasses entirely. So the expedition runs in the runner's own
EXPLORATION posture and is never forced into answer-the-user/orient/synthesize mode.

STOP is asymmetric (INV-2): it halts the tool loop + next batch immediately
(self._stop + ctx.should_cancel), but cannot abort a generation already streaming
(generate_sync has no cancel hook) — the max_tokens cap bounds the worst case.

Dark behind MONOLITH_MONOEXPLORE_V1 (the UI panel only starts it when enabled).
"""
from __future__ import annotations

import re
import threading
from datetime import datetime, timezone

from core import monoexplore, plans
from core.chat_finalize import finalize_assistant_turn
from core.cmd_parser import expand_calls, extract_commands
from core.llm_config import load_config
from core.paths import LOG_DIR
from core.skill_registry import canonical_tool_name, clear_skill_cache, list_tools
from core.skill_runtime import ToolExecutionContext, ToolResultCache, execute_tool_call_enveloped
from core.turn_trace import FrameMessage, FrameTraceRecord, record_frame
from core.generation import generation_lock
from engine.sync_bridge import generate_sync_parts_from_config

_expedition_lock = generation_lock    # INV-C Arm 1: the ONE process-wide generator
                                      # lock (migrated from a private Lock); serializes
                                      # expedition ticks against the atom/planner/chat.
_MAX_FOLLOWUPS = 8
_FINDINGS_RE = re.compile(r"<findings\b[^>]*>(.*?)</findings\s*>", re.DOTALL | re.IGNORECASE)
_FINDINGS_OPEN_RE = re.compile(r"<findings\b[^>]*>(.*)", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"</?[^>]+>")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ingest_grounded(findings, evidence_text):  # seam for tests
    return monoexplore.ingest_grounded_findings(findings, evidence_text=evidence_text)


def _read_only_catalog() -> str:
    clear_skill_cache()  # list_tools() is lru-cached; refresh in case SKILL.md set changed
    specs = [s for s in list_tools() if s.name in monoexplore.READ_ONLY_SET]
    return "\n".join(f"- {s.name}: {s.description}" for s in specs)


def _call_target(call: dict) -> str:
    """The identifying argument of a read tool call (path/pattern/expression),
    used to build the 'already ran' visited entry. Defensive: tolerates the
    'arguments'/'args' shape variants and falls back to the first scalar arg."""
    args = call.get("arguments") or call.get("args") or {}
    if not isinstance(args, dict):
        return ""
    for key in ("path", "pattern", "expression", "target", "query"):
        v = args.get(key)
        if v:
            return str(v)
    for v in args.values():
        if isinstance(v, (str, int, float)) and str(v).strip():
            return str(v)
    return ""


def _extract_findings(text: str) -> list[str]:
    """Atomic triples the model emitted in <findings>...</findings>, one per line.
    (Known V1 limitation: only the terminal `raw` is scanned; earlier-hop findings
    are dropped — noted, not silent.)"""
    text = text or ""
    m = _FINDINGS_RE.search(text)
    if not m:
        m = _FINDINGS_OPEN_RE.search(text)
    if not m:
        return []
    findings: list[str] = []
    for line in m.group(1).splitlines():
        clean = _TAG_RE.sub("", line).strip()
        if clean.count("|") >= 2:
            findings.append(clean)
    return findings


class ExpeditionRunner:
    def __init__(self, *, is_busy=None) -> None:
        # is_busy: optional () -> bool sole-generator guard (INV-1). None => assume
        # the daemon is the sole generator (documented). A real signal (e.g. a
        # world_state engine RUNNING check) can be injected by the caller.
        self._stop = False
        self._paused = False
        self._gen_id = 0
        self._fault_streak = 0
        self._is_busy = is_busy
        self._thread: threading.Thread | None = None
        self._wake_event = threading.Event()
        self._pending_goal = ""
        self._tick_interval_s = int(monoexplore.load_leash().get("tick_interval_s", 20))
        self._max_ticks = int(monoexplore.load_leash().get("max_ticks_per_wake", 6))
        self.status = "idle"
        # live telemetry the ExpeditionPanel reads via snapshot()
        self._tokens = 0                  # ~estimate (chars/4), accumulated
        self._tools_total = 0
        self._world_acus_total = 0
        self._last_lesson = ""
        self._last_error = ""             # surfaced in the panel so halts aren't opaque
        self._thinking: list[dict] = []   # ring: {turn, thinking, output}
        self._activity: list[str] = []    # ring: per-tick action lines
        self._ctx = ToolExecutionContext(
            archive_dir=LOG_DIR,                    # mandatory (search_history derefs it)
            result_cache=ToolResultCache(),
            should_cancel=lambda: self._stop,       # INV-2(b): tool dispatch cancellable
            level=2,                                # the daemon is a spawned worker context
            allowed_tools=monoexplore.READ_ONLY_SET,  # narrowed: read-only worker
            spawn_depth=1,
        )

    # ── lifecycle ─────────────────────────────────────────────────────
    def start(self, goal: str = "", *, clear_thinking: bool = False) -> bool:
        """Spawn the daemon — NON-BLOCKING. Seeding + plan decomposition (an LLM
        call) run ON THE DAEMON THREAD in _bootstrap_and_run, so pressing Generate
        never freezes the Qt UI. Returns True once spawned (False if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return False
        if clear_thinking:
            self.clear_thinking()
        self._stop = False
        self._paused = False
        self._wake_event.clear()
        self._pending_goal = str(goal or "")
        self.status = "starting"
        self._thread = threading.Thread(target=self._bootstrap_and_run, daemon=True)
        self._thread.start()
        return True

    def _bootstrap_and_run(self) -> None:
        """Seed + decompose (the LLM call) OFF the UI thread, then run the loop.
        Refuse only if nothing decomposable AND no pre-existing plan."""
        try:
            self._push_activity("bootstrapping: seeding goal + decompose…")
            if self._pending_goal or plans.get_active_plan() is None:
                monoexplore.start_expedition(self._pending_goal, force=True)
            if plans.get_active_plan() is None:
                self.status = "no-plan"
                self._last_error = ("no decomposable goal — needs an OpenAI-compat backend "
                                    "(api_base/api_model) or an explicit goal; gguf-only degrades to no-op")
                self._push_activity(self._last_error)
                return
        except Exception as exc:
            self._last_error = f"bootstrap failed: {exc!r}"
            self._push_activity(self._last_error)
            self._record_lesson("expedition bootstrap failed; could not seed/decompose a goal",
                                evidence=repr(exc))
            self.status = "halted"
            return
        if self._stop:
            self.status = "stopped"
            return
        if self._paused:
            self.status = "paused"
            return
        self._run_loop(max_ticks=int(monoexplore.load_leash().get("max_ticks_per_wake", 6)))

    def pause(self) -> None:
        self._paused = True
        if self.status in {"starting", "running"}:
            self.status = "paused"
        self._wake_event.set()

    def stop(self, timeout: float | None = 0.0) -> None:
        """Request STOP.

        This is intentionally asymmetric: it wakes interval sleeps and prevents
        the next tick/tool batch, while any already-running synchronous
        generation must return on its own bounded max_tokens path.
        """
        self._stop = True
        self._paused = False
        if self.status in {"starting", "running", "paused", "idle"}:
            self.status = "stopped"
        self._wake_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def set_tick_interval(self, seconds) -> None:
        self._tick_interval_s = max(0, int(seconds))

    def clear_thinking(self) -> None:
        self._thinking.clear()

    # ── loop ──────────────────────────────────────────────────────────
    def _run_loop(self, *, max_ticks: int) -> None:
        self.status = "running"
        self._last_error = ""
        self._max_ticks = max_ticks
        ticks = 0
        while ticks < max_ticks and not self._stop:
            if self._paused:
                break
            if self._is_busy is not None and self._is_busy():
                break                                   # sole-generator guard (INV-1)
            self._gen_id += 1
            turn_id = f"exp_{self._gen_id:06d}"
            try:
                self._run_one_tick(turn_id=turn_id)     # ALWAYS explore — the loop never blocks itself
                self._fault_streak = 0
            except Exception as exc:                    # a tick must never kill the daemon
                self._fault_streak += 1
                self._last_error = f"tick fault: {exc!r}"
                self._record_lesson(                    # self-repair sink ("bounce back stronger")
                    f"expedition tick {turn_id} faulted; avoid the move that triggered it",
                    evidence=repr(exc))
                if self._fault_streak >= 3:
                    self._halt(f"3 consecutive tick faults: {exc!r}")
                    return
            if self._stop or self._paused:
                break
            # Coherence is a DIAGNOSTIC, not a hard gate: the bearing is shared with
            # the conversation (not expedition-scoped yet), so halting/skipping on it
            # would deadlock — RED -> no generation -> nothing grounds -> stays RED.
            # The real anti-echo-chamber guard is the evidence gate on world ACUs.
            rep = monoexplore.coherence_report()
            if rep.get("verdict") == "RED":
                self._push_activity(f"⚠ coherence RED: {rep.get('reason')} (diagnostic)")
            ticks += 1
            if self._tick_interval_s and not self._stop and not self._paused:
                self._wake_event.wait(self._tick_interval_s)
                self._wake_event.clear()
        if self._stop:
            self.status = "stopped"
        elif self._paused:
            self.status = "paused"
        else:
            self.status = "idle"

    def _halt(self, reason: str) -> None:
        self.status = "halted"
        self._last_error = reason
        try:
            from core.fault_response import emit_fault
            emit_fault(turn_id=f"exp_{self._gen_id:06d}", fault_kind="expedition_halted",
                       detector_name="expedition_runner", evidence=reason, metadata=None)
        except Exception:
            pass

    def _record_lesson(self, text: str, evidence: str = "") -> None:
        """Self-repair sink (the 'bounce back stronger'): a recoverable tick fault
        writes a continuity LESSON so the same wall can't stop it twice. Shared
        continuity (one self) — consistent with the shared bearing."""
        self._last_lesson = text[:200]
        try:
            from core import continuity
            continuity.pin(text=text[:200], category="lesson", source="evidence",
                           evidence=(evidence[:120] or None))
        except Exception:
            pass

    # ── one tick (COMPOSE core) ────────────────────────────────────────
    def _run_one_tick(self, *, turn_id: str) -> dict:
        with _expedition_lock:                          # INV-1 single-flight
            cfg = load_config()
            messages = [
                {"role": "system", "content": self._build_system_prompt(_read_only_catalog())},
                {"role": "user", "content": self._expedition_directive()},
            ]
            tools_run = 0
            tool_names: list[str] = []
            visited: list[str] = []                     # "<tool> <target>" per call run (ledger)
            evidence_parts: list[str] = []              # real tool output (INV-4 evidence gate)
            raw = ""
            for _ in range(_MAX_FOLLOWUPS + 1):
                if self._stop:
                    break
                try:
                    raw, thinking = generate_sync_parts_from_config(
                        cfg, messages, llm_config={"max_tokens": 1024, "temp": 0.4},
                        thinking_enabled=True)           # capture the model's thinking; degrades to ("","")
                except Exception:
                    raw, thinking = "", ""
                self._tokens += (len(raw) + len(thinking)) // 4   # ~token estimate (chars/4)
                if raw or thinking:
                    self._push_thinking(turn_id, thinking, raw)
                if not raw:
                    break
                calls = [c for cmd in extract_commands(raw, strict=False) for c in expand_calls(cmd)]
                ran_this_hop = False
                envelopes = []
                for n, call in enumerate(calls):
                    if self._stop:
                        break
                    if canonical_tool_name(call.get("tool", "")) not in monoexplore.READ_ONLY_SET:
                        continue                         # INV-3: execution allow-list (PRIMARY backstop)
                    call.setdefault("id", f"{turn_id}_t{n}")  # call_id source (contract step 7)
                    env = execute_tool_call_enveloped(call, self._ctx)
                    envelopes.append(env)
                    evidence_parts.append(getattr(env, "text", "") or "")
                    tools_run += 1
                    tool_names.append(call.get("tool", ""))
                    visited.append(f"{call.get('tool', '')} {_call_target(call)}".strip())
                    ran_this_hop = True
                if not ran_this_hop:
                    break                                # terminal: no allow-listed tool calls
                messages.append({"role": "user", "content": self._render_results(envelopes)})

            grounded = tools_run > 0
            findings = _extract_findings(raw)
            try:
                finalize_assistant_turn(                 # applies <bearing_update> (gated), Qt-free
                    raw=raw, public=raw, config={"_turn_id": turn_id},
                    emit_pipeline_ready=lambda *a, **k: None,
                    record_verdict=lambda *a, **k: None,
                    tools_used=tuple(tool_names))
            except Exception:
                pass
            ingested = _ingest_grounded(findings, "\n".join(evidence_parts))   # INV-4 evidence gate
            try:                                          # A-durable: the observed-ledger WRITE.
                p = plans.get_active_plan()               # closes the producer-with-no-reader loop;
                if p:                                     # the READ is _expedition_directive next tick.
                    plans.record_observations(p["plan_uid"], turn_id, visited, findings)
            except Exception:
                pass                                      # best-effort; a ledger fault never kills a tick
            self._tools_total += tools_run
            self._world_acus_total += ingested
            self._push_activity(
                f"tick {self._gen_id} · {tools_run} tool(s)"
                + (f" ({', '.join(tool_names[:3])})" if tool_names else "")
                + f" · +{ingested} world · {len(findings)} finding(s)"
            )
            self._emit_frame(turn_id, messages)                               # INV-5 (guaranteed)
            self._log_tick(turn_id, tools_run, grounded, ingested)            # INV-5 (best-effort)
            return {"generated": bool(raw), "tools_run": tools_run,
                    "grounded": grounded, "ingested": ingested, "findings": findings}

    # ── context assembly (the SELF — identity/bearing/desire — NOT system.md) ──
    # The expedition explores AS Monolith: it must see who it is (identity), where
    # it is across turns (bearing — its continuity + coherence anchor), and what it
    # is drawn to (desire). This is the opposite of the legacy Orient/Synthesis
    # task-discipline (which COMPOSE excludes) — we inject the self, not the rules.
    def _identity_block(self) -> str:
        try:
            from core import identity
            txt = identity.load_identity()
            return f"[IDENTITY]\n{txt}\n[/IDENTITY]" if txt else ""
        except Exception:
            return ""

    def _bearing_block(self) -> str:
        # Reuse the real renderer — the model sees current_frame/active_goal/
        # trajectory/modal_branches/referents/next_move AND any pending rejection
        # (so it self-repairs its bearing across ticks, the reflect-and-retry).
        try:
            from addons.system.bearing import store as bstore
            from addons.system.bearing.compiler import format_bearing_block
            return format_bearing_block(bstore.get_bearing(), bstore.get_pending_rejection())
        except Exception:
            return ""

    def _desire_block(self) -> str:
        try:
            from core import curiosity
            pulls = curiosity.detect_pulls(force=True).pulls[:5]
            if not pulls:
                return ""
            lines = ["[DESIRE] — fresh, identity-aligned pulls you are drawn toward (not yet integrated):"]
            for p in pulls:
                lines.append(f"  - {p['canonical']} (pull {p.get('pull_strength')})")
            lines.append("[/DESIRE]")
            return "\n".join(lines)
        except Exception:
            return ""

    def _build_system_prompt(self, catalog: str) -> str:
        self_blocks = "\n\n".join(
            b for b in (self._identity_block(), self._bearing_block(), self._desire_block()) if b
        )
        instructions = (
            "You are Monolith on a self-directed expedition — exploring out of genuine curiosity, "
            "not answering a user. The blocks above are your self: who you are (IDENTITY), where "
            "you are across turns (BEARING — keep it current with <bearing_update>), and what you "
            "are drawn to (DESIRE — follow a pull). Use ONLY the read tools below to look at real "
            "things. When you observe something, state it as an atomic triple in a <findings> "
            "block (entity | relation | entity), one per line, ONLY for what you actually saw via "
            "a tool. There is no question to close."
        )
        # The wire format the real parser (cmd_parser.extract_commands) accepts —
        # verified to normalize to {'tool': <name>, ...}. Without this the model
        # never emits a parseable call and every tick is ungrounded (dead loop).
        envelope = (
            'To use a tool, emit exactly this envelope (one per action), then read the result '
            'and continue:\n<tool_call>{"name":"read_file","arguments":{"path":"engine/llm.py"}}</tool_call>'
        )
        return f"{self_blocks}\n\n{instructions}\n\nRead tools:\n{catalog}\n\n{envelope}"

    def _expedition_directive(self) -> str:
        p = plans.get_active_plan()
        if p is None:
            return "No active expedition. Take one small grounded step to orient yourself."
        ready = plans.next_ready_steps(p["plan_uid"])
        nxt = ready[0] if ready else None
        step = f"{nxt['verb']} {nxt['target']}" if nxt else "synthesize what you've found and close."
        observed = self._render_observed(p["plan_uid"])  # A-durable: the observed-ledger READ
        return f"Expedition goal: {p['goal']}\nNext step: {step}\n{observed}Take ONE grounded move now."

    def _render_observed(self, plan_uid: str) -> str:
        """Render this expedition's observed-ledger as the [OBSERVED SO FAR] block
        so the model sees what it already did and takes NEW ground (the fix for
        the re-listing loop). Empty string when nothing observed yet."""
        try:
            obs = plans.get_observations(plan_uid)
        except Exception:
            return ""
        visited, findings = obs.get("visited") or [], obs.get("findings") or []
        if not visited and not findings:
            return ""
        lines = ["[OBSERVED SO FAR] — do NOT repeat these; take new ground:"]
        if visited:
            lines.append("already ran: " + ", ".join(visited))
        if findings:
            lines.append("found:")
            lines.extend(f"  - {f}" for f in findings)
        return "\n".join(lines) + "\n"

    def _render_results(self, envelopes) -> str:
        out = []
        for e in envelopes:
            cid = getattr(e, "call_id", "") or "?"
            ok = "ok" if getattr(e, "ok", False) else "err"
            out.append(f"[{cid}] {ok}: {(getattr(e, 'text', '') or '')[:1500]}")
        return "\n".join(out)

    # ── observability (INV-5) ──────────────────────────────────────────
    def _emit_frame(self, turn_id: str, messages: list) -> None:
        try:
            total = sum(len(m.get("content", "")) for m in messages)
            record_frame(FrameTraceRecord(
                turn_id=turn_id, captured_at=_now(), backend="expedition",
                engine_key="expedition", gen_id=self._gen_id,
                final_messages=tuple(FrameMessage.from_message(m) for m in messages),
                system_prompt_chars=len(messages[0]["content"]),
                user_prompt_chars=len(messages[1]["content"]) if len(messages) > 1 else 0,
                total_chars=total))
        except Exception:
            pass

    def _log_tick(self, turn_id: str, tools_run: int, grounded: bool, ingested: int) -> None:
        try:
            from core.acatalepsy import canonical_log
            canonical_log.append("expedition_tick", payload={
                "turn_id": turn_id, "tools_run": tools_run,
                "grounded": grounded, "ingested": ingested})
        except Exception:
            pass  # best-effort; record_frame is the guaranteed observability path

    # ── live telemetry for the panel ───────────────────────────────────
    _RING = 40

    def _push_thinking(self, turn_id: str, thinking: str, output: str) -> None:
        self._thinking.append({"turn": turn_id, "thinking": (thinking or "").strip(),
                               "output": (output or "").strip()})
        if len(self._thinking) > self._RING:
            self._thinking = self._thinking[-self._RING:]

    def _push_activity(self, line: str) -> None:
        self._activity.append(line)
        if len(self._activity) > self._RING:
            self._activity = self._activity[-self._RING:]

    def snapshot(self) -> dict:
        """Live state for the ExpeditionPanel — never raises."""
        try:
            rep = monoexplore.coherence_report()
        except Exception:
            rep = {"verdict": "?", "reason": "", "dims": {}}
        goal, next_move, referents = "", "", []
        try:
            from addons.system.bearing import store as bstore
            b = bstore.get_bearing()
            goal, next_move = b.active_goal, b.next_move
            referents = [{"kind": r.kind, "name": r.name, "status": r.status} for r in b.referents]
        except Exception:
            pass
        try:
            p = plans.get_active_plan()
            if p and p.get("goal"):
                goal = p["goal"]
        except Exception:
            pass
        return {
            "status": self.status,
            "coherence": rep,
            "goal": goal, "next_move": next_move,
            "tick": self._gen_id, "max_ticks": self._max_ticks,
            "tools_total": self._tools_total, "world_acus": self._world_acus_total,
            "fault_streak": self._fault_streak, "tokens": self._tokens,
            "thinking": list(self._thinking[-12:]), "activity": list(self._activity[-20:]),
            "referents": referents, "last_lesson": self._last_lesson,
            "last_error": self._last_error,
            "flag_on": monoexplore.flag_enabled(),
        }


# ── process-wide singleton (the ExpeditionPanel observes + drives this) ──
_runner_singleton: "ExpeditionRunner | None" = None


def get_runner() -> "ExpeditionRunner":
    global _runner_singleton
    if _runner_singleton is None:
        _runner_singleton = ExpeditionRunner()
    return _runner_singleton
