"""SelfMaintRunner — the leashed, unattended self-maintenance daemon.

The project's most dangerous subsystem: an autonomous loop that wakes the model with
NO human pressing Generate, to triage Monolith's own [REVIEW QUEUE]. It adapts the
proven Qt-free skeleton of engine/expedition_runner.py (thread + wake-event, single-
flight via the one process-wide generation_lock, fault-streak halt, asymmetric STOP,
guaranteed record_frame, COMPOSE generate->dispatch->finalize) and confines it to the
audited safety core in core/self_maint_daemon.py.

Safety invariants (audit-mandated — see docs/superpowers/plans/2026-06-19-self-maint-
trigger.md), each backed by a test in tests/test_self_maint_runner.py:
  - The daemon self-starts ONLY when MONOLITH_SELF_MAINT_TRIGGER_V1 is set; off => the
    thread is never spawned (byte-identical to no daemon).
  - Two-flag observe-first: trigger-on + apply-off (MONOLITH_SELF_MAINT_V1) => the
    model is woken and its intended action LOGGED, but review_act->safe_review_act
    refuses to apply. Flip the apply flag only after the ledger reads sane.
  - Tick order spends a daily wake-credit ONLY on a real wake: external-busy guard ->
    non-blocking generation_lock -> try_wake() -> run. Capped/busy => skip, logged.
  - The context is built ONLY via daemon.build_wake_context (level=2 + the narrow
    WAKE_TOOLS); a second execution allow-list filter drops any non-WAKE tool before
    dispatch. Host hooks stay None. At most ONE review_act actuation per wake.
  - record_frame + a self_maint_trigger.ledger.jsonl row (incl. the raw model output)
    are emitted every wake.

CONCURRENCY — HARD WIRING REQUIREMENT: the non-blocking generation_lock serializes the
daemon ONLY against OTHER background generators (expedition / subagent / monoline). The
LIVE streaming user turn does NOT acquire generation_lock, so the lock alone does NOT
prevent a wake from running concurrently with the user. The construction/start site MUST
bind is_busy to a live-turn-in-progress signal (world_state engine-busy), mirroring the
subagent atom. Constructing this with is_busy=None in production re-opens that race.

The Qt/model seams (_load_cfg/_generate/_parse_calls/_dispatch/_ensure_ctx) are thin
and overridable so the whole loop is testable in the venv without PySide6 or a model.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

from core import review_loop
from core import self_maint
from core import self_maint_daemon as daemon
from core import turn_trace
from core.generation import generation_lock
from core.skill_registry import canonical_tool_name


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Live-turn statuses (mirrors ui/pages/chat.py::_engine_is_busy, kept Qt-free here so the
# engine-layer daemon does not import the UI layer).
_LIVE_BUSY_STATUSES = frozenset({"running", "generating", "streaming"})


def engine_is_busy(world_state) -> bool:
    """Arm-2 guard: True if any engine reports a live-turn-in-progress status. None =>
    free (headless/test convention). The non-blocking generation_lock does NOT serialize
    the live streaming user turn, so binding this to is_busy is what actually prevents a
    wake from racing the user (the docstring's HARD WIRING REQUIREMENT). Never raises.

    RESIDUAL (audits 2026-06-22, accepted): engine status is READY until the live turn
    calls set_status(RUNNING) after prompt-compile, so a sub-second window exists where a
    just-started user turn isn't yet seen as busy. is_busy is the SOLE serializer against
    the live turn — the daemon's own generation goes through sync_bridge (a fresh
    OpenAICompatLLM), NOT LLMEngine.generate, so there is no llm.py RUNNING-guard backstop
    for it. Harm in that window is cost/contention only — no state corruption (_finalize
    writes zero substrate). More acute at short intervals; the daily cap bounds exposure."""
    if world_state is None:
        return False
    try:
        engines = (world_state.snapshot() or {}).get("engines", {}) or {}
    except Exception:
        return False
    return any(str(e.get("status", "")).strip().lower() in _LIVE_BUSY_STATUSES
               for e in engines.values() if isinstance(e, dict))


def _short(s, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


class SelfMaintRunner:
    _MAX_FOLLOWUPS = 6
    _MAX_ITEMS = 12
    _RING = 40
    _MIN_INTERVAL_S = 15  # runtime floor: prevents a 0/negative spin; the daily cap bounds total wakes

    def __init__(self, *, is_busy=None) -> None:
        # is_busy: optional () -> bool sole-generator guard. None => the daemon relies
        # on the non-blocking generation_lock alone (documented).
        self._stop = False
        self._thread: threading.Thread | None = None
        self._wake_event = threading.Event()
        self._fault_streak = 0
        self._gen_id = 0
        self._is_busy = is_busy
        self._ctx = None
        self._lifecycle_lock = threading.Lock()  # serialize start/stop (no double-spawn)
        self._wake_interval_s = daemon.wake_interval_s()
        self._activity: list[str] = []
        self._last_error = ""
        self.status = "idle"

    # ── lifecycle ─────────────────────────────────────────────────────
    def set_interval(self, seconds) -> int:
        """Adjust the wake cadence at runtime (the control skill). Clamped to
        _MIN_INTERVAL_S (never 0/spin); wakes the loop so the new cadence applies on the
        next tick. Returns the effective interval."""
        try:
            iv = int(seconds)
        except (TypeError, ValueError):
            iv = self._MIN_INTERVAL_S
        self._wake_interval_s = max(self._MIN_INTERVAL_S, iv)
        self._wake_event.set()  # re-evaluate the current sleep with the new interval
        return self._wake_interval_s

    def start(self, *, force: bool = False) -> bool:
        """Spawn the daemon thread. The BOOT path (force=False) refuses unless the
        trigger flag is set — so flag-off is byte-identical to no daemon. The control
        skill passes force=True to start it at runtime (the skill call is the
        authorization). Either way the APPLY flag still gates whether actions apply."""
        if not force and not daemon.trigger_enabled():
            self.status = "disabled"
            return False
        with self._lifecycle_lock:  # check-then-spawn is atomic (no concurrent double-start)
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop = False
            self._fault_streak = 0
            self._wake_event.clear()
            self.status = "starting"
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return True

    def set_is_busy(self, is_busy) -> None:
        """Bind the Arm-2 live-turn guard (the wiring must call this — see docstring)."""
        self._is_busy = is_busy

    def stop(self, timeout: float | None = 0.0) -> None:
        """Asymmetric STOP: halts the next wake immediately (and wakes an interval
        sleep), but cannot abort a generation already streaming — its bounded
        max_tokens path returns on its own."""
        self._stop = True
        self.status = "stopped"
        self._wake_event.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=timeout)
            if not t.is_alive():
                self._thread = None

    # ── loop ──────────────────────────────────────────────────────────
    def _run_loop(self) -> None:
        self.status = "running"
        while not self._stop:
            iv = self._wake_interval_s
            if iv and not self._stop:
                self._wake_event.wait(iv)   # the autonomous timer (interruptible by stop)
                self._wake_event.clear()
            if self._stop:
                break
            try:
                self._tick()
                self._fault_streak = 0
            except Exception as exc:                      # a tick must never kill the daemon
                self._fault_streak += 1
                self._last_error = f"wake fault: {exc!r}"
                daemon.log_wake({"turn_id": f"maint_{self._gen_id:06d}", "fault": repr(exc),
                                 "fault_streak": self._fault_streak, "ts": _now()})  # durable (audit #4)
                if self._fault_streak >= 3:
                    self._halt(f"3 consecutive wake faults: {exc!r}")
                    return
        self.status = "stopped" if self._stop else "idle"

    def _tick(self) -> None:
        """One wake attempt. Order matters: a daily wake-credit is spent ONLY when we
        actually proceed to generate (after the busy + lock guards)."""
        if self._stop:
            return
        if self._is_busy is not None and self._is_busy():
            self._skip("busy_external")
            return
        # NOTE: this serializes the daemon only against OTHER background generators
        # (expedition/subagent/monoline) — the LIVE user turn does NOT take this lock.
        # The live-turn race is prevented by the is_busy guard above, which the wiring
        # MUST bind (see the module docstring's HARD WIRING REQUIREMENT).
        if not generation_lock.acquire(blocking=False):
            self._skip("busy_lock")
            return
        try:
            gate = daemon.try_wake()                        # the persisted daily leash
            if not gate.get("ok"):
                self._skip(gate.get("reason", "capped"))
                return
            self._gen_id += 1
            self._run_one_wake(f"maint_{self._gen_id:06d}")
        finally:
            generation_lock.release()

    def _halt(self, reason: str) -> None:
        self.status = "halted"
        self._last_error = reason
        self._push_activity(f"HALTED: {reason}")
        # Durable in the daemon's OWN ungated ledger — emit_fault below silently no-ops
        # when MONOLITH_TURN_TRACE_V1=0, so it cannot be the only halt record. (audit #4)
        daemon.log_wake({"turn_id": f"maint_{self._gen_id:06d}", "halted": reason,
                         "fault_streak": self._fault_streak, "ts": _now()})
        try:
            from core.fault_response import emit_fault
            emit_fault(turn_id=f"maint_{self._gen_id:06d}", fault_kind="self_maint_halted",
                       detector_name="self_maint_runner", evidence=reason, metadata=None)
        except Exception:
            pass

    # ── one wake (COMPOSE core) ────────────────────────────────────────
    def _run_one_wake(self, turn_id: str) -> dict:
        self._ensure_ctx()
        cfg = self._load_cfg()
        items = review_loop.list_review_items(limit=self._MAX_ITEMS)
        try:
            summary = review_loop.review_summary()
        except Exception:
            summary = {"unresolved_count": len(items)}
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._wake_directive(items, summary)},
        ]
        tool_names: list[str] = []
        raw = ""
        thinking = ""
        for _ in range(self._MAX_FOLLOWUPS + 1):
            if self._stop:
                break
            raw, thinking = self._generate(cfg, messages)
            if not raw:
                break
            calls = self._parse_calls(raw)
            envelopes, ran = self._dispatch_calls(calls, turn_id)
            tool_names.extend(ran)
            if not envelopes:
                break                                       # no allow-listed tool calls => terminal
            if "review_act" in ran:
                break                                       # one safe actuation per wake (hard bound)
            messages.append({"role": "user", "content": self._render_results(envelopes)})

        self._finalize(raw, turn_id, tool_names)
        self._emit_frame(turn_id, messages)                 # guaranteed observability
        # The raw model output + thinking are the dark-system raw-I/O artifact (E's
        # observability contract): the only durable record of WHAT the model said in an
        # unattended wake — including any <bearing_update> it ATTEMPTED (now applied
        # nowhere). Truncated to bound the ledger.
        daemon.log_wake({"turn_id": turn_id, "items_seen": len(items),
                         "tool_calls": tool_names,
                         "raw": (raw or "")[:2000], "thinking": (thinking or "")[:2000],
                         "ts": _now()})
        self._push_activity(
            f"wake {self._gen_id} · {len(items)} item(s) · "
            + (", ".join(tool_names) if tool_names else "no action"))
        return {"generated": bool(raw), "tool_calls": tool_names, "items_seen": len(items)}

    def _dispatch_calls(self, calls, turn_id: str):
        """Run only WAKE_TOOLS calls; drop everything else BEFORE dispatch. Defense-in-depth
        backstop behind the level>1 capability gate. A call is allowed ONLY if EVERY naming
        key the dispatcher might read (tool/skill/op/name) canonicalizes into WAKE_TOOLS —
        so a key-split envelope (filter sees 'review_act' but the resolver would pick
        'self_maint') is dropped, not just caught by the capability gate (audit 2026-06-22)."""
        envelopes, ran = [], []
        for n, call in enumerate(calls):
            if self._stop:
                break
            candidates = [canonical_tool_name(call.get(k) or "")
                          for k in ("tool", "skill", "op", "name") if call.get(k)]
            if not candidates or any(c not in daemon.WAKE_TOOLS for c in candidates):
                continue
            call.setdefault("id", f"{turn_id}_t{n}")
            envelopes.append(self._dispatch(call))
            ran.append(candidates[0])
        return envelopes, ran

    # ── prompt + directive ─────────────────────────────────────────────
    def _wake_directive(self, items, summary) -> str:
        if not items:
            return (
                "[REVIEW QUEUE] empty — nothing needs attention right now.\n\n"
                "This is an autonomous self-maintenance wake; no human is asking you "
                "anything. The queue is empty, so take NO action and end the turn.")
        lines = ["[REVIEW QUEUE] attention-needed substrate items; NOT a user request:"]
        for it in items:
            lines.append(
                "- {id} {kind}.{subkind} sev={sev}: {summary} (reason: {reason})".format(
                    id=it.get("id"), kind=it.get("kind"), subkind=it.get("subkind"),
                    sev=it.get("effective_severity"),
                    summary=_short(it.get("summary"), 120),
                    reason=_short(it.get("reason"), 90)))
        lines.append("[/REVIEW QUEUE]")
        block = "\n".join(lines)
        return (
            f"{block}\n\n"
            "This is an autonomous self-maintenance wake — no human is asking you anything. "
            "Look at the [REVIEW QUEUE] above and take AT MOST ONE safe action on ONE item:\n"
            "  - snooze <id> — defer an item not actionable yet (it resurfaces later)\n"
            "  - escalate <id> — flag an item that genuinely needs E's judgment\n"
            "You CANNOT resolve or dismiss anything — that is E's call alone. If nothing "
            "clearly needs snoozing or escalating, do nothing and end the turn.\n"
            'Act via exactly: <tool_call>{"name":"review_act","arguments":'
            '{"item_id":"acu:25","action":"snooze"}}</tool_call>')

    def _build_system_prompt(self) -> str:
        catalog = (
            "- review_act: act on a [REVIEW QUEUE] item — snooze (defer) or escalate "
            "(flag for E). args: item_id (e.g. acu:25), action (snooze|escalate).")
        instr = (
            "You are Monolith performing autonomous self-maintenance — quietly tending your "
            "own review queue, not answering a user. Triage each item from its summary and "
            "reason; act only via review_act, and only to snooze or escalate. You cannot "
            "resolve or dismiss anything — that authority is E's alone.")
        parts = [b for b in (self._self_blocks(), instr, "Tools:\n" + catalog) if b]
        return "\n\n".join(parts)

    def _self_blocks(self) -> str:
        """Best-effort: the expedition explores AS Monolith, so does maintenance — show
        it who it is + where it is. Every import is wrapped (degrades to '')."""
        blocks = []
        try:
            from core import identity
            txt = identity.load_identity()
            if txt:
                blocks.append(f"[IDENTITY]\n{txt}\n[/IDENTITY]")
        except Exception:
            pass
        try:
            from addons.system.bearing import store as bstore
            from addons.system.bearing.compiler import format_bearing_block
            b = format_bearing_block(bstore.get_bearing(), bstore.get_pending_rejection())
            if b:
                blocks.append(b)
        except Exception:
            pass
        return "\n\n".join(blocks)

    def _render_results(self, envelopes) -> str:
        out = []
        for e in envelopes:
            cid = getattr(e, "call_id", "") or "?"
            ok = "ok" if getattr(e, "ok", False) else "err"
            out.append(f"[{cid}] {ok}: {(getattr(e, 'text', '') or '')[:1200]}")
        return "\n".join(out)

    # ── observability ───────────────────────────────────────────────────
    def _emit_frame(self, turn_id: str, messages: list) -> None:
        try:
            total = sum(len(m.get("content", "")) for m in messages)
            turn_trace.record_frame(turn_trace.FrameTraceRecord(
                turn_id=turn_id, captured_at=_now(), backend="self_maint",
                engine_key="self_maint", gen_id=self._gen_id,
                final_messages=tuple(turn_trace.FrameMessage.from_message(m) for m in messages),
                system_prompt_chars=len(messages[0]["content"]),
                user_prompt_chars=len(messages[1]["content"]) if len(messages) > 1 else 0,
                total_chars=total))
        except Exception:
            pass

    def _skip(self, reason: str) -> None:
        self._push_activity(f"wake skipped: {reason}")
        daemon.log_wake({"turn_id": f"maint_{self._gen_id:06d}", "skipped": reason, "ts": _now()})

    def _push_activity(self, line: str) -> None:
        self._activity.append(line)
        if len(self._activity) > self._RING:
            self._activity = self._activity[-self._RING:]

    # ── Qt/model seams (overridable for tests) ──────────────────────────
    def _ensure_ctx(self) -> None:
        if self._ctx is None:
            self._ctx = daemon.build_wake_context(lambda: self._stop)

    def _load_cfg(self) -> dict:
        from core.llm_config import load_config
        return load_config()

    def _generate(self, cfg, messages):
        from engine.sync_bridge import generate_sync_parts_from_config
        try:
            return generate_sync_parts_from_config(
                cfg, messages, llm_config={"max_tokens": 1024, "temp": 0.3},
                thinking_enabled=True)
        except Exception:
            return ("", "")

    def _parse_calls(self, raw):
        from core.cmd_parser import expand_calls, extract_commands
        return [c for cmd in extract_commands(raw, strict=False) for c in expand_calls(cmd)]

    def _dispatch(self, call):
        from core.skill_runtime import execute_tool_call_enveloped
        return execute_tool_call_enveloped(call, self._ctx)

    def _finalize(self, raw: str, turn_id: str, tool_names) -> None:
        # A maintenance wake is EPHEMERAL triage, NOT a conversation turn — it must write
        # ZERO substrate. The full turn finalizer (chat_finalize.finalize_assistant_turn)
        # would commit a model-emitted <bearing_update>/<frame> envelope to the bearing
        # store + source-tier records, gated by BEARING_V1/FRAME_COMMIT_V1 (ON in the
        # launcher) and NOT by the apply flag — a confinement breach that makes "nothing
        # applied" false under observe-first (audit F5/F6/F11/F14). So: do nothing. (The
        # model's raw output — incl. any <bearing_update> it ATTEMPTED — is preserved as
        # the dark-system artifact in the log_wake row, not applied here.) A wake never
        # reshapes navigational/conversational posture, even when the apply flag is ON.
        return

    # ── live telemetry (for a future panel) ─────────────────────────────
    def snapshot(self) -> dict:
        """Live state for the companion panel — never raises. `recent` is the tail of the
        wake ledger (each row's raw model output = what Monolith WOULD do); `apply_on`
        distinguishes observe-first (logs only) from applying."""
        return {
            "status": self.status,
            "wake": self._gen_id,
            "interval_s": self._wake_interval_s,
            "fault_streak": self._fault_streak,
            "last_error": self._last_error,
            "activity": list(self._activity[-20:]),
            "trigger_on": daemon.trigger_enabled(),
            "apply_on": self_maint.enabled(),
            "recent": daemon.read_wake_tail(20),
        }


# ── process-wide singleton ──────────────────────────────────────────────
_runner_singleton: "SelfMaintRunner | None" = None


def get_runner() -> "SelfMaintRunner":
    global _runner_singleton
    if _runner_singleton is None:
        _runner_singleton = SelfMaintRunner()
    return _runner_singleton


def maybe_start_self_maint(world_state) -> bool:
    """Bootstrap entry point. Starts the self-maintenance daemon IFF its trigger flag
    (MONOLITH_SELF_MAINT_TRIGGER_V1) is set — so when off, this is a pure no-op (the
    daemon is byte-identical to absent). Binds is_busy to the live-turn signal so a wake
    never races the user (the audit's HARD WIRING REQUIREMENT). Returns True if started.
    Never raises — a daemon failure must not break app launch."""
    try:
        if not daemon.trigger_enabled():
            return False
        runner = get_runner()
        runner.set_is_busy(lambda: engine_is_busy(world_state))
        return runner.start()
    except Exception:
        return False
