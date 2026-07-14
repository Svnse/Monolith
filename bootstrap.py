import os
import sys

from PySide6.QtWidgets import QApplication

from core.event_ledger import EventLedger
from core.version import APP_VERSION
from core.config import ConfigWatcher
from core.theme_config import load_theme_config
from core.theme_engine import ThemeEngine
from core.themes import apply_theme
from core.style import refresh_styles
from core.state import AppState
from core.world_state import WorldStateStore, WORLD_STATE_FLUSH_INTERVAL_MS
from core.world_actions import check_policy, PolicyDecision, validate_action
from core.resource_policy import ResourcePolicy
from core.prompt_library import (
    set_world_state as set_prompt_world_state,
    prompt_interceptor,
    tool_interceptor,
)
from core.monothink import (
    set_monothink_world_state,
    monothink_interceptor,
)
from core.ephemeral_coalescer import ephemeral_coalescer_interceptor
from core.adaptive_budget import (
    adaptive_budget_interceptor,
    set_ledger as set_budget_ledger,
    set_world_state as set_budget_world_state,
)
from core.message_interceptors import register_interceptor
from core import log_mirror as _log_mirror
from engine.vision import VisionProcess
from monokernel.bridge import MonoBridge
from monokernel.dock import MonoDock
from monokernel.guard import MonoGuard
from ui.addons.builtin import build_builtin_registry
from ui.addons.context import AddonContext
from ui.addons.host import AddonHost
from ui.addons.loader import load_manifest_addons
from ui.bridge import UIBridge
from ui.main_window import MonolithUI
from ui.overseer import OverseerWindow
from PySide6.QtCore import QTimer


def init_kernel(state: AppState) -> tuple[MonoGuard, MonoDock, MonoBridge, VisionProcess]:
    vision_engine = VisionProcess()
    guard = MonoGuard(state, {"vision": vision_engine})
    dock = MonoDock(guard)
    bridge = MonoBridge(dock)
    # Turn Pipeline — the live-turn event bus / fault-intervention substrate.
    # Boots after Bridge/Dock/Guard so the pipeline can stand alongside them.
    # Validation raises if a policy file under core/pipeline_policies/ is not
    # declared in core/pipeline_registry.py; that's intentional dev-time fail-
    # loud behavior. Soft-handled so a regression in pipeline boot doesn't
    # take down the whole app.
    try:
        from monokernel.turn_pipeline import bootstrap_pipeline as _tp_boot
        _tp_boot()
    except Exception as _tp_exc:  # noqa: BLE001
        guard.sig_trace.emit(
            "system",
            f"[turn_pipeline] bootstrap failed: {type(_tp_exc).__name__}: {_tp_exc}",
        )
    return guard, dock, bridge, vision_engine


def init_ui(
    app: QApplication,
    state: AppState,
    guard: MonoGuard,
    ui_bridge: UIBridge,
) -> tuple[MonolithUI, OverseerWindow, EventLedger]:
    ui = MonolithUI(state, ui_bridge)
    overseer = OverseerWindow(guard, ui_bridge)
    ledger = EventLedger(overseer.db, app_version=APP_VERSION)
    return ui, overseer, ledger


def init_addons(
    state: AppState,
    guard: MonoGuard,
    bridge: MonoBridge,
    ui: MonolithUI,
    ui_bridge: UIBridge,
) -> AddonHost:
    registry = build_builtin_registry()
    load_manifest_addons(registry, on_error=lambda msg: guard.sig_trace.emit("system", msg))

    # VisionArtifactBridge subscribes to the vision engine's sig_image so the
    # generate_image skill can surface arrivals back into the chat (saved PNG
    # + thumb strip in the matching tool-result bubble). Constructed here and
    # injected via the addon services dict; PageChat consumes it through
    # ctx.services["vision_artifact_bridge"]. Failure to construct is soft —
    # the executor falls back to text-only output if the bridge is missing.
    vision_artifact_bridge = None
    try:
        from core.vision_artifact_bridge import VisionArtifactBridge
        vision_engine = guard.engines.get("vision") if hasattr(guard, "engines") else None
        if vision_engine is not None:
            vision_artifact_bridge = VisionArtifactBridge(vision_engine)
    except Exception as exc:
        guard.sig_trace.emit("system", f"[vision_artifact_bridge] init failed: {type(exc).__name__}: {exc}")

    ctx = AddonContext(state=state, guard=guard, bridge=bridge, ui=ui, host=None, ui_bridge=ui_bridge)
    if vision_artifact_bridge is not None:
        ctx.services["vision_artifact_bridge"] = vision_artifact_bridge

    host = AddonHost(registry, ctx)
    ui.attach_host(host)
    return host


def wire_signals(
    app: QApplication,
    state: AppState,
    guard: MonoGuard,
    bridge: MonoBridge,
    ui: MonolithUI,
    ui_bridge: UIBridge,
    overseer: OverseerWindow,
    ledger: EventLedger,
    host: AddonHost,
    vision_engine: VisionProcess,
    config_watcher: ConfigWatcher,
    theme_engine: ThemeEngine,
) -> None:
    ui_bridge.sig_open_overseer.connect(overseer.show)
    ui_bridge.sig_overseer_viz_toggle.connect(guard.enable_viztracer)
    ui_bridge.sig_launch_addon.connect(host.launch_module)
    ui_bridge.sig_reload_modules.connect(host.request_reload_modules)
    ui_bridge.sig_reveal_attachment.connect(
        lambda att: host.ctx.ui.reveal_attachment(att) if getattr(host.ctx, "ui", None) else None
    )

    def _apply_theme(theme_name: str) -> None:
        apply_theme(theme_name)
        refresh_styles()
        theme_engine.apply(app)
        # Force a deep widget-tree refresh so per-widget stylesheets that
        # f-string theme tokens at construction (and would otherwise stay
        # frozen at their construction-time values) actually pick up the
        # new theme. Catches widgets whose sig_theme_changed subscription
        # is broken or was constructed after a prior emission.
        try:
            from core.theme_engine import deep_refresh_theme
            for window in app.topLevelWidgets():
                if window is not None and window.isVisible():
                    deep_refresh_theme(window, theme_name=theme_name)
        except Exception as exc:
            guard.sig_trace.emit("system", f"[THEME] deep refresh failed: {exc}")

    ui_bridge.sig_theme_changed.connect(_apply_theme)

    # ---- World Action dispatch ----
    def _dispatch_world_action_impl(action: dict, *, bypass_policy: bool = False) -> None:
        result = validate_action(action)
        if not result.ok or result.action is None:
            guard.sig_trace.emit("system", f"[WORLD] action rejected: {result.error}")
            return
        act = result.action
        if not bypass_policy:
            decision = check_policy(act)
            if decision == PolicyDecision.BLOCKED:
                guard.sig_trace.emit("system", "[WORLD] action blocked by policy")
                return
            if decision == PolicyDecision.REQUIRE_APPROVAL:
                if getattr(state, "world_state", None) is not None:
                    state.world_state.set_pending_action(act)
                if hasattr(ui_bridge, "sig_world_action_pending"):
                    ui_bridge.sig_world_action_pending.emit(act)
                guard.sig_trace.emit("system", f"[WORLD] action pending approval: {act.get('command', act.get('type'))}")
                return
        if act["type"] == "engine_stop":
            task = bridge.wrap(
                "world",
                "stop",
                act["engine"],
                priority=1,
            )
            bridge.submit(task)
            guard.sig_trace.emit("system", f"[WORLD] stop engine={act['engine']} task={task.id}")
            return
        task = bridge.wrap(
            "world",
            act["command"],
            act["engine"],
            payload=act.get("payload") or {},
            priority=int(act.get("priority", 2)),
        )
        bridge.submit(task)
        guard.sig_trace.emit(
            "system",
            f"[WORLD] submit engine={act['engine']} cmd={act['command']} task={task.id}",
        )

    def _dispatch_world_action(action: dict) -> None:
        _dispatch_world_action_impl(action, bypass_policy=False)

    def _dispatch_world_action_approved(action: dict) -> None:
        if getattr(state, "world_state", None) is not None:
            state.world_state.set_pending_action(None)
            try:
                action_label = (
                    action.get("command")
                    or action.get("type")
                    or "unknown_action"
                )
                state.world_state.set_last_approval_event(action_label, granted_by="user")
            except Exception:
                pass
        _dispatch_world_action_impl(action, bypass_policy=True)

    def _reject_world_action(_action: dict | None = None) -> None:
        if getattr(state, "world_state", None) is not None:
            state.world_state.set_pending_action(None)

    ui_bridge.sig_world_action.connect(_dispatch_world_action)
    if hasattr(ui_bridge, "sig_world_action_approved"):
        ui_bridge.sig_world_action_approved.connect(_dispatch_world_action_approved)
    if hasattr(ui_bridge, "sig_world_action_rejected"):
        ui_bridge.sig_world_action_rejected.connect(_reject_world_action)

    # global chrome-only wiring stays here
    guard.sig_status.connect(ui.update_status)
    guard.sig_usage.connect(lambda _ek, used: ui.update_ctx(used))

    # ---- Event Ledger wiretap (Phase 1) ----
    guard.sig_status.connect(
        lambda ek, s: ledger.record(
            "guard", "state", "status_changed",
            engine_key=ek,
            payload={"engine_key": ek, "status": s})
    )
    guard.sig_finished.connect(
        lambda ek, tid: ledger.record(
            "guard", "lifecycle", "task_finished",
            engine_key=ek,
            payload={"engine_key": ek, "task_id": tid},
            correlation_id=tid)
    )
    guard.sig_engine_ready.connect(
        lambda ek: ledger.record(
            "guard", "lifecycle", "engine_ready",
            engine_key=ek,
            payload={"engine_key": ek})
    )
    guard.sig_trace.connect(
        lambda ek, msg: (
            ledger.record(
                "guard", "error", "trace_warning",
                engine_key=ek,
                payload={"engine_key": ek, "message": msg},
                severity=2)
            if any(k in msg.upper() for k in ("ERROR", "WARN", "REJECTED"))
            else None
        )
    )
    guard.sig_image.connect(
        lambda ek, img: ledger.record(
            "guard", "artifact", "image_generated",
            engine_key=ek,
            payload={"type": "image", "engine_key": ek})
    )
    ui_bridge.sig_theme_changed.connect(
        lambda t: ledger.record(
            "ui.bridge", "intent", "theme_changed",
            payload={"theme": t})
    )
    ui_bridge.sig_apply_operator.connect(
        lambda d: ledger.record(
            "ui.bridge", "intent", "operator_applied",
            payload=d)
    )
    ui_bridge.sig_open_overseer.connect(
        lambda: ledger.record(
            "ui.bridge", "intent", "open_overseer")
    )
    ui_bridge.sig_terminal_header.connect(
        lambda a, b, c: ledger.record(
            "ui.bridge", "state", "terminal_header_updated",
            payload={"args": [a, b, c]})
    )

    app.aboutToQuit.connect(ledger.shutdown)
    app.aboutToQuit.connect(guard.stop)
    app.aboutToQuit.connect(overseer.db.close)
    app.aboutToQuit.connect(lambda: guard.enable_viztracer(False) if guard._viztracer is not None else None)
    app.aboutToQuit.connect(vision_engine.shutdown)
    app.aboutToQuit.connect(config_watcher.stop)
    if getattr(state, "world_state", None) is not None:
        app.aboutToQuit.connect(state.world_state.flush)


def init_policies(state: AppState, guard: MonoGuard) -> None:
    resource_policy = ResourcePolicy.load()

    llm_engine = guard.engines.get("llm")
    if llm_engine is not None and hasattr(llm_engine, "set_timeout"):
        llm_engine.set_timeout(resource_policy.generation_timeout_sec)

    guard.configure_circuit_breaker(
        resource_policy.circuit_breaker_threshold,
        resource_policy.circuit_breaker_cooldown_sec,
    )

    for _ek, _min_free in resource_policy.vram_min_free_mb.items():
        guard.set_vram_quota(_ek, _min_free)


def init_world_state_timers(app: QApplication, state: AppState) -> None:
    world_state = getattr(state, "world_state", None)
    if world_state is None:
        return

    def _sample_resources():
        try:
            import psutil  # optional
        except Exception:
            return
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            ram_used_mb = int(vm.used / (1024 * 1024))
            ram_total_mb = int(vm.total / (1024 * 1024))
        except Exception:
            return
        world_state.set_resources(
            cpu_pct=cpu_pct,
            ram_used_mb=ram_used_mb,
            ram_total_mb=ram_total_mb,
        )

    res_timer = QTimer(app)
    res_timer.setInterval(2000)
    res_timer.timeout.connect(_sample_resources)
    res_timer.start()

    flush_timer = QTimer(app)
    flush_timer.setInterval(int(WORLD_STATE_FLUSH_INTERVAL_MS))
    flush_timer.timeout.connect(world_state.flush)
    flush_timer.start()


def main():
    app = QApplication(sys.argv)
    theme_cfg = load_theme_config()
    apply_theme(theme_cfg.get("theme", "midnight"))
    refresh_styles()
    theme_engine = ThemeEngine()
    theme_engine.apply(app)

    state = AppState()
    state.world_state = WorldStateStore()
    try:
        from core.workshop_seed import seed_workshop_flows
        seed_workshop_flows()  # first-run: populate the Workshop with a starter flow if empty
    except Exception:
        pass  # best-effort; never block boot
    guard, _dock, bridge, vision_engine = init_kernel(state)
    ui_bridge = UIBridge()
    ui, overseer, ledger = init_ui(app, state, guard, ui_bridge)
    host = init_addons(state, guard, bridge, ui, ui_bridge)

    config_watcher = ConfigWatcher()

    def _on_config_change(new_cfg, old_cfg) -> None:
        if old_cfg is None:
            return
        if new_cfg.theme.current != old_cfg.theme.current:
            ui_bridge.sig_theme_changed.emit(new_cfg.theme.current)
        if new_cfg.llm.model_dump() != old_cfg.llm.model_dump():
            ui_bridge.sig_config_changed.emit({"llm": new_cfg.llm.model_dump()})

    config_watcher.on_change(_on_config_change)
    config_watcher.start()

    init_policies(state, guard)
    # Turn-trace retention daemon: clears records older than the TTL on a
    # 24h cadence. Best-effort; failures log to stderr but never break.
    try:
        from core.turn_trace import cleanup_old_records as _trace_cleanup
        _trace_timer = QTimer(app)
        _trace_timer.setInterval(24 * 60 * 60 * 1000)  # 24h
        _trace_timer.timeout.connect(lambda: _trace_cleanup())
        _trace_timer.start()
        # Also run once 5s after boot so a long-idle install doesn't accumulate.
        QTimer.singleShot(5_000, lambda: _trace_cleanup())
    except Exception:
        pass

    # Acatalepsy v1 substrate. Always migrates the schema (idempotent).
    # Auditor worker starts only when MONOLITH_ACATALEPSY_AUDITOR_V1 is set
    # — opt-in for the A1 validation period. See docs/specs/acatalepsy_v1_spec.md
    # and core/acatalepsy/bootstrap.py.
    try:
        from core.acatalepsy.bootstrap import (
            bootstrap_acatalepsy as _acatalepsy_bootstrap,
            shutdown_acatalepsy as _acatalepsy_shutdown,
        )
        _acatalepsy_result = _acatalepsy_bootstrap()
        guard.sig_trace.emit(
            "system",
            f"[acatalepsy] schema_ok={_acatalepsy_result.schema_ok} "
            f"worker_started={_acatalepsy_result.worker_started}"
            + (f" skip={_acatalepsy_result.worker_skip_reason}" if _acatalepsy_result.worker_skip_reason else "")
            + (f" error={_acatalepsy_result.error}" if _acatalepsy_result.error else ""),
        )
        try:
            from core.identity_acus import ensure_origin0_acus_loaded as _load_origin0_acus
            _origin0_ids = _load_origin0_acus()
            guard.sig_trace.emit(
                "system",
                f"[identity_acus] origin0_locked={len(_origin0_ids)}",
            )
        except Exception as _id_acu_exc:  # noqa: BLE001
            guard.sig_trace.emit(
                "system",
                f"[identity_acus] load failed: {type(_id_acu_exc).__name__}: {_id_acu_exc}",
            )
        # M2 identity-emergence: bootstrap heartbeat (second beat alongside the
        # record_outcome feedback hook, so the loop breathes even when E rates
        # rarely). Ships dark (MONOLITH_IDENTITY_EMERGENCE_V1 default OFF) →
        # no-op until enabled. Deterministic, no LLM, propose-only, isolated.
        try:
            from core.identity_emergence import detect_emergence as _detect_emergence
            _er = _detect_emergence()
            guard.sig_trace.emit(
                "system",
                f"[identity_emergence] bootstrap fired={_er.fired} candidates={len(_er.candidates)}",
            )
        except Exception as _em_exc:  # noqa: BLE001
            guard.sig_trace.emit(
                "system",
                f"[identity_emergence] bootstrap check failed: {type(_em_exc).__name__}",
            )
        # M3 curiosity: bootstrap heartbeat (fresh disposition). Ships dark
        # (MONOLITH_CURIOSITY_V1 default OFF). Deterministic, propose-only, isolated.
        try:
            from core.curiosity import detect_pulls as _detect_pulls
            _cr = _detect_pulls()
            guard.sig_trace.emit(
                "system",
                f"[curiosity] bootstrap fired={_cr.fired} pulls={len(_cr.pulls)}",
            )
        except Exception as _cur_exc:  # noqa: BLE001
            guard.sig_trace.emit(
                "system",
                f"[curiosity] bootstrap check failed: {type(_cur_exc).__name__}",
            )
        # Graceful shutdown on app quit
        app.aboutToQuit.connect(lambda: _acatalepsy_shutdown(timeout=3.0))
    except Exception as _ac_exc:
        guard.sig_trace.emit("system", f"[acatalepsy] bootstrap failed: {_ac_exc!r}")
    try:
        from core.monosearch.bootstrap import init_monosearch
        init_monosearch()
    except Exception:
        pass  # MonoSearch is read-only and non-critical to boot
    # Plane scaffolds (effort, conversation, reasoning, linguency) fire BEFORE
    # the coalescer so they inject unconditionally — not subject to the 3000-
    # char budget gate. linguency in particular ships the 37k-char monolith
    # scaffold; the coalescer budget would silently drop it. Each plane has
    # its own loader with its own validation per the plane-separation refactor
    # (S11). Their `contribute_section` variants exist but are NOT routed
    # through the coalescer for the same budget reason.
    #
    # ── Interceptor proximity-to-user ordering (DO NOT INVERT lightly) ──
    # Each interceptor inserts its block at the last-non-ephemeral-user index,
    # so registration order produces this final layout (closer to user = later
    # in this list):
    #
    #     [system, EFFORT, CONVERSATION, REASONING, LINGUENCY, user]
    #
    # The chosen rule: **the most comprehensive scaffold binds tightest to the
    # user message.** Effort tier is the foundational depth frame (broadest);
    # conversation shapes teleology on top; reasoning adds internal audit
    # discipline; linguency layers the output-composition contract closest to
    # the user message. Reading bottom-up (LLM attention), the user message is
    # dominated by linguency first (when active — monolith is 37k+ chars),
    # then reasoning, then conversation, then effort depth, then system.
    #
    # Inverting this (effort-closest-to-user) would put the broadest, least
    # turn-specific frame in the strongest attention position — pushing the
    # specific, high-stakes scaffolds (audit-scorecard, monolith) farther
    # from the user message. That tends to weaken their grip on output. The
    # current order matches the principle that scaffold specificity should
    # scale with proximity. If a future change needs to invert, surface why
    # in this comment — don't silently re-order.
    #
    # Remaining ephemeral contributors coalesce into ONE inserted block via
    # the ephemeral_coalescer. runtime_state owns the former continuity,
    # observed_state/current-execution, recall, and temporal lanes; review_loop,
    # observer, last_turn, confidence_trajectory, rating_telemetry, and
    # context_refresh remain peer sections in the same inserted message. Fixes
    # KV-cache prefix break from many moving inserts (audit defect #6) and adds
    # a total-char budget (MONOLITH_EPHEMERAL_BUDGET_CHARS, default 4000) with
    # priority-based drop-on-overflow (audit defect #8).
    # Unified prompt system: /prompt <name1> <name2> ...
    set_prompt_world_state(state.world_state)
    set_monothink_world_state(state.world_state)
    register_interceptor(prompt_interceptor)                  # 0d. unified prompt scaffolds (direct inject, bypasses budget)
    register_interceptor(monothink_interceptor)               # 0e. monothink scaffold (direct inject, self-evolving)
    register_interceptor(tool_interceptor)                    # 0f. one-shot tool prompts (/skill-creator etc.)
    # Bearing — first-party system addon for situational coherence across
    # turns/sessions. Registered BEFORE the ephemeral_coalescer so the
    # [BEARING] block lands before the coalesced [RUNTIME STATE] block in the
    # rendered prompt. Soft-handled: addon construction failure does not
    # break boot, matching the Turn Pipeline boot pattern above.
    #
    # DIRECT-INJECT, NOT COALESCER CONTRIBUTOR. Bearing's block is typed-slot
    # content (current_frame / active_goal / tensions / branches / referents
    # / etc.) — not free-prose ambient state shaped for the coalescer's char-
    # budget drop-on-overflow heuristic. Going through ephemeral_coalescer's
    # contributor path would subject Bearing to compression rules designed
    # for prose contributors (continuity / context_refresh / etc.) and could
    # silently drop typed slots under budget pressure. Future-CC: do NOT
    # rewire Bearing as a coalescer contributor without first updating the
    # coalescer to understand typed-slot priority. See Bearing V0 plan §13.2.
    try:
        from addons.system.bearing import build_addon as _build_bearing_addon
        _bearing_addon = _build_bearing_addon()
        if _bearing_addon is not None:
            from core.turn_classifier import set_bearing_provider as _set_bearing_provider
            _set_bearing_provider(_bearing_addon.provider)
            register_interceptor(_bearing_addon.interceptor)  # 0d. bearing block
    except Exception as _bearing_exc:  # noqa: BLE001
        guard.sig_trace.emit(
            "system",
            f"[bearing] bootstrap failed: {type(_bearing_exc).__name__}: {_bearing_exc}",
        )
    # Command feedback — outside-in [COMMAND_FAILED] block. Direct-inject (NOT
    # a coalescer contributor) for the same reason as Bearing: a repair
    # instruction telling the model WHY its last command failed in the runtime
    # is load-bearing and must never be silently dropped under budget pressure.
    # Capture is per-surface (e.g. cmd_parser's unrecognized-shape drop); this
    # is the delivery half. See docs/audits/MODEL_OUTPUT_BLINDSPOT_MAP.md.
    try:
        from core.command_feedback import command_feedback_interceptor
        register_interceptor(command_feedback_interceptor)    # 0d. command-failure repair block
    except Exception as _cf_exc:  # noqa: BLE001
        guard.sig_trace.emit(
            "system",
            f"[command_feedback] bootstrap failed: {type(_cf_exc).__name__}: {_cf_exc}",
        )
    register_interceptor(ephemeral_coalescer_interceptor)     # 1. coalesced ephemeral block
    # lag_watch: read-only per-turn JSONL evidence that the unwiring is holding.
    # Writes LOG_DIR/lag_watch.jsonl; gives regression detection if the model
    # ever starts emitting axes again (would_fire populates).
    try:
        from core.lag_watch import lag_watch_interceptor, set_world_state as set_lag_world_state
        set_lag_world_state(state.world_state)
        register_interceptor(lag_watch_interceptor)           # 6. read-only watchdog
    except Exception:
        pass
    # Stage telemetry: apply_interceptors wraps each interceptor with a
    # Layer A StageTraceRecord write to LOG_DIR/turn_trace.sqlite3 when
    # config["_turn_id"] is set. axis_interceptor's bespoke EventLedger
    # event was retired in favor of the unified trace store (Phase 1
    # of the turn-trace spec).
    # adaptive_budget_interceptor parked — see docs (smart-spec re-make 2026-05-09).
    # The interceptor injected [BUDGET GUIDANCE] AND silently capped
    # config["max_tokens"] mid-pipeline; re-enable only after a deterministic
    # /effort surface lands as the canonical authority over the heuristic.
    # set_budget_ledger / set_budget_world_state remain available for the
    # scoring helpers (compute_complexity_score, evaluate_budget_for_message)
    # which can still be called diagnostically without injecting/capping.
    set_budget_ledger(ledger)
    set_budget_world_state(state.world_state)
    # Snapshot the subsystem map after all interceptors / policies / planes /
    # skills have registered. Junior queries this via scratchpad op=introspect
    # — closes the "can't enumerate own subsystems without grep" gap
    # (2026-05-20 audit). Regenerated at every bootstrap; failure is non-fatal.
    try:
        from core.subsystem_map import dump_subsystem_map as _dump_subsystem_map
        _dump_subsystem_map()
    except Exception as _smap_exc:
        guard.sig_trace.emit("system", f"[subsystem_map] dump failed: {_smap_exc!r}")
    init_world_state_timers(app, state)
    wire_signals(
        app,
        state,
        guard,
        bridge,
        ui,
        ui_bridge,
        overseer,
        ledger,
        host,
        vision_engine,
        config_watcher,
        theme_engine,
    )

    # Log mirror — bounded in-memory ring buffer fed by guard's trace signal
    # so external readers (engine/agent_server /log/tail) can see what the
    # cockpit is doing. Best-effort wiring; failures don't break boot.
    try:
        _log_mirror.connect_signals(
            (guard, "sig_trace", "guard", "trace"),
        )
    except Exception:
        pass

    # Self-maintenance daemon — autonomous [REVIEW QUEUE] triage (snooze/escalate only).
    # Gated by MONOLITH_SELF_MAINT_TRIGGER_V1 (default OFF => pure no-op, byte-identical
    # to absent). is_busy is bound to the live-turn signal so a wake never races the user
    # (audit HARD WIRING REQUIREMENT). Failure must never break boot.
    try:
        from engine.self_maint_runner import (
            maybe_start_self_maint as _sm_maybe_start,
            get_runner as _sm_get_runner,
        )
        if _sm_maybe_start(state.world_state):
            guard.sig_trace.emit("system", "[self_maint] daemon started (MONOLITH_SELF_MAINT_TRIGGER_V1 on)")
            app.aboutToQuit.connect(lambda: _sm_get_runner().stop(timeout=2.0))
    except Exception as _sm_exc:
        guard.sig_trace.emit("system", f"[self_maint] bootstrap skipped: {_sm_exc!r}")

    ui.show()
    # ui_v2 cockpit (MONOLITH_UI_V2): a parallel window that reads the same live
    # world_state. MonolithUI stays alive above, so wire_signals / engine / chat
    # are unaffected and one kernel/store is shared. Soft-fail so a v2 regression
    # never breaks boot. Flag off -> branch never taken -> byte-identical. The
    # reference is held on `state` so the window isn't garbage-collected.
    if os.environ.get("MONOLITH_UI_V2"):
        try:
            from ui_v2.app import build_v2_shell
            state._v2_shell = build_v2_shell(state, ui_bridge, guard=guard, bridge=bridge)
            state._v2_shell.show()
        except Exception as _v2_exc:  # noqa: BLE001
            guard.sig_trace.emit("system", f"[ui_v2] launch failed: {_v2_exc!r}")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
