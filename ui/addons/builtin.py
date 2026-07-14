import uuid
import os
from ui.addons.context import AddonContext
from ui.addons.descriptors import CapabilityDescriptor
from ui.addons.registry import AddonRegistry
from ui.addons.spec import AddonSpec
from ui.modules.injector import InjectorWidget
from ui.modules.sd import SDModule
from ui.modules.audiogen import AudioGenModule
from ui.tools.theme import ThemeModule
from ui.pages.chat import PageChat
from ui.pages.connections import ConnectionsPage
from ui.pages.databank import PageFiles
from engine.bridge import EngineBridge
from engine.llm import LLMEngine


def terminal_factory(ctx: AddonContext):
    instance_id = str(uuid.uuid4())
    engine_key = f"llm_{instance_id}"

    short_id = instance_id[:8]

    def _trace(msg):
        ctx.guard.sig_trace.emit("system", msg)

    llm_engine = LLMEngine(ctx.state)
    engine_bridge = EngineBridge(llm_engine)
    ctx.guard.register_engine(engine_key, engine_bridge)
    # Inbound model output must stay on the kernel path:
    # engine -> bridge -> guard -> host.on_engine_event -> PageChat.
    # New text backends/loaders should emit EnginePort signals and let the host
    # deliver them to the UI instead of wiring tokens/status directly here.

    def _on_generate_audio(params: dict) -> None:
        """Lazy resolve: find an AudioGenModule through the host and trigger generation."""
        host = getattr(ctx, "host", None)
        if host is None or not hasattr(host, "ctx") or not host.ctx.ui:
            return
        for i in range(host.ctx.ui.stack.count()):
            widget = host.ctx.ui.stack.widget(i)
            if isinstance(widget, AudioGenModule) and hasattr(widget, "trigger_generation"):
                widget.trigger_generation(params)
                return
        # No audio module open yet — launch one, then trigger
        host.launch_module("audiogen")
        for i in range(host.ctx.ui.stack.count()):
            widget = host.ctx.ui.stack.widget(i)
            if isinstance(widget, AudioGenModule) and hasattr(widget, "trigger_generation"):
                widget.trigger_generation(params)
                return

    def _on_soundtrap(params: dict) -> str:
        """Headless soundtrap: dispatch straight to the core executor (no UI workspace)."""
        from core.soundtrap import execute_soundtrap_command
        return execute_soundtrap_command(dict(params))

    w = PageChat(ctx.state, ctx.ui_bridge, bridge=ctx.bridge, guard=ctx.guard)
    w._on_generate_audio = _on_generate_audio
    w._on_soundtrap = _on_soundtrap
    w._mod_id = instance_id
    w._engine_key = engine_key
    # VisionArtifactBridge — constructed in bootstrap.init_addons, injected via
    # ctx.services. set_vision_artifact_bridge() both stores the bridge and
    # subscribes PageChat to sig_artifact_ready so async image arrivals can
    # rerender the matching tool-result bubble. Optional: if missing, the
    # generate_image executor falls back to text-only output.
    w.set_vision_artifact_bridge(ctx.services.get("vision_artifact_bridge"))
    ctx.ui_bridge.sig_apply_operator.connect(w.apply_operator)

    w.sig_set_model_path.connect(
        lambda payload: ctx.bridge.submit(
            ctx.bridge.wrap("terminal", "set_path", engine_key, payload=payload)
        )
    )
    w.sig_set_ctx_limit.connect(
        lambda limit: None if limit is None else ctx.bridge.submit(
            ctx.bridge.wrap("terminal", "set_ctx_limit", engine_key, payload={"ctx_limit": int(limit)})
        )
    )

    if w.config.get("gguf_path") or w.config.get("api_base"):
        w.sig_set_model_path.emit(w.build_model_payload())
    # Only push a saved ctx_limit through if the user actually set one.
    # An unset (0) value would override the engine's resolved window with a
    # stale persisted default; let the engine emit ground truth via
    # sig_model_capabilities instead.
    _saved_ctx = int(w.config.get("ctx_limit", 0) or 0)
    if _saved_ctx > 0:
        w.sig_set_ctx_limit.emit(_saved_ctx)

    def _submit_direct_generation(request: object) -> None:
        prompt = str(request.get("prompt", "")) if isinstance(request, dict) else str(request)
        # Only enable thinking for models that explicitly support it.
        # For unknown models default to True (the TypeError catch in the engine
        # handles backends that don't accept enable_thinking).
        model_supports_thinking = True
        ws = getattr(ctx.state, "world_state", None)
        if ws is not None:
            engine_meta = (ws.state.get("engines") or {}).get(engine_key) or {}
            preset = engine_meta.get("model_preset") or {}
            caps = preset.get("capabilities") or {}
            if "supports_thinking" in caps:
                model_supports_thinking = bool(caps["supports_thinking"])
        # Generation ctx_limit priority: explicit per-request value > saved
        # config > resolved engine state. 0 means "trust whatever the engine
        # has already negotiated" -- never a default-disguised 8192.
        _state_ctx = int(getattr(ctx.state, "ctx_limit", 0) or 0)
        _saved_ctx = int(w.config.get("ctx_limit", 0) or 0)
        _gen_ctx = _saved_ctx or _state_ctx or 0
        payload = {
            "prompt": prompt,
            "config": w.config,
            "ctx_limit": _gen_ctx,
            "thinking": w.thinking_enabled() and model_supports_thinking,
        }
        if isinstance(request, dict):
            if "config" in request and isinstance(request.get("config"), dict):
                payload["config"] = request["config"]
            if "ctx_limit" in request:
                payload["ctx_limit"] = int(request.get("ctx_limit", _gen_ctx))
            if "thinking" in request:
                payload["thinking"] = bool(request.get("thinking"))
            if "ephemeral" in request:
                payload["ephemeral"] = bool(request.get("ephemeral"))
        task = ctx.bridge.wrap(
            "terminal",
            "generate",
            engine_key,
            payload=payload,
        )
        w.set_task_id(str(task.id))
        ctx.bridge.submit(task)

    # outgoing (addon -> bridge)
    def _on_generate(prompt):
        try:
            w.sig_debug.emit("[CHAT] _on_generate slot hit")
            model_name = w.describe_active_model()
            prompt_text = str(prompt.get("prompt", "")) if isinstance(prompt, dict) else str(prompt)
            prompt_preview = repr(prompt_text[:120])
            _trace(f"[LLM:{short_id}] generating - model={model_name}, prompt={repr(prompt_text[:50])}")
            w.sig_debug.emit(f"[CHAT] _on_generate bridge handoff: prompt={prompt_preview}")
            _submit_direct_generation(prompt)
        except Exception as e:
            _trace(f"[LLM:{short_id}] EXCEPTION in generate: {e}")
            import traceback
            traceback.print_exc()

    w.sig_generate.connect(_on_generate)
    def _on_load():
        _trace(f"[LLM:{short_id}] load requested")
        ctx.bridge.submit(ctx.bridge.wrap("terminal", "load", engine_key))

    def _on_unload():
        _trace(f"[LLM:{short_id}] unload requested")
        ctx.bridge.submit(ctx.bridge.wrap("terminal", "unload", engine_key))

    w.sig_load.connect(_on_load)
    w.sig_unload.connect(_on_unload)
    def _on_stop():
        try:
            _trace(f"[LLM:{short_id}] stopped - generation halted")
            ctx.bridge.stop(engine_key)
        except Exception as e:
            _trace(f"[LLM:{short_id}] EXCEPTION in stop: {e}")
            import traceback
            traceback.print_exc()

    w.sig_stop.connect(_on_stop)

    def _on_sync_history(history):
        try:
            _trace(f"[LLM:{short_id}] syncing history - {len(history)} messages")
            ctx.bridge.submit(
                ctx.bridge.wrap(
                    "terminal",
                    "set_history",
                    engine_key,
                    payload={"history": history},
                )
            )
        except Exception as e:
            _trace(f"[LLM:{short_id}] EXCEPTION in sync_history: {e}")
            import traceback
            traceback.print_exc()

    # Keep Python callbacks alive for the lifetime of the widget. These slots
    # are the outbound bridge from PageChat into the kernel.
    w._kernel_slots = {
        "generate": _on_generate,
        "load": _on_load,
        "unload": _on_unload,
        "stop": _on_stop,
        "sync_history": _on_sync_history,
    }
    w.sig_sync_history.connect(_on_sync_history)
    w.sig_debug.connect(lambda msg: ctx.guard.sig_trace.emit(engine_key, msg))
    # incoming engine->UI events are routed centrally via AddonHost.on_engine_event()

    def _cleanup_terminal(*_args):
        ctx.guard.unregister_engine(engine_key)
        engine_bridge.shutdown()

    w.destroyed.connect(_cleanup_terminal)
    return w

def databank_factory(ctx: AddonContext):
    return PageFiles(ctx.state, ctx.ui_bridge)


def injector_factory(ctx: AddonContext):
    assert ctx.ui is not None, "InjectorWidget requires UI parent"
    return InjectorWidget(ctx.ui, ctx.ui_bridge)


def sd_factory(ctx: AddonContext):
    # Pass the vision_artifact_bridge so SDModule can skip its auto-save
    # when a generation was skill-triggered (the bridge already saved it).
    # Without this dedupe, every LLM-triggered image lands on disk twice.
    return SDModule(
        ctx.bridge,
        ctx.guard,
        ctx.ui_bridge,
        vision_artifact_bridge=ctx.services.get("vision_artifact_bridge"),
    )


def audiogen_factory(ctx: AddonContext):
    return AudioGenModule(ctx.state, ctx.ui_bridge)


def theme_factory(ctx: AddonContext):
    return ThemeModule(ctx.ui_bridge)


def connections_factory(ctx: AddonContext):
    return ConnectionsPage(
        ctx.state,
        ctx.ui_bridge,
        guard=ctx.guard,
        ctx=ctx,
    )


def build_builtin_registry() -> AddonRegistry:
    registry = AddonRegistry()

    registry.register(
        AddonSpec(
            id="terminal",
            kind="module",
            title="CHAT",
            icon="⌖",
            factory=terminal_factory,
            descriptor=CapabilityDescriptor(
                verbs=("generate_text", "chat", "load_model", "unload_model", "stream_tokens"),
                appetites=("text_prompt", "gguf_file", "conversation_history", "system_prompt"),
                emissions=("text_stream", "token_usage", "model_status"),
            ),
        )
    )
    registry.register(
        AddonSpec(
            id="databank",
            kind="module",
            title="FILES",
            icon="▤",
            factory=databank_factory,
            descriptor=CapabilityDescriptor(
                verbs=("browse_files", "select_file"),
                appetites=("file_path", "directory"),
                emissions=("file_selected",),
            ),
        )
    )
    registry.register(
        AddonSpec(
            id="injector",
            kind="module",
            title="RUNTIME",
            icon="💉",
            factory=injector_factory,
            descriptor=CapabilityDescriptor(
                verbs=("inject_context",),
                appetites=("text_content", "file_path", "code_snippet"),
                emissions=("context_injected",),
            ),
        )
    )
    registry.register(
        AddonSpec(
            id="sd",
            kind="module",
            title="VISION",
            icon="⟡",
            factory=sd_factory,
            descriptor=CapabilityDescriptor(
                verbs=("generate_image",),
                appetites=("text_prompt", "image_prompt", "generation_params"),
                emissions=("image_generated",),
            ),
        )
    )
    registry.register(
        AddonSpec(
            id="audiogen",
            kind="module",
            title="AUDIO",
            icon="♫",
            factory=audiogen_factory,
            descriptor=CapabilityDescriptor(
                verbs=("generate_audio",),
                appetites=("text_prompt", "audio_params"),
                emissions=("audio_generated",),
            ),
        )
    )
    registry.register(
        AddonSpec(
            id="theme",
            kind="module",
            title="THEME",
            icon="◌",
            factory=theme_factory,
            descriptor=CapabilityDescriptor(
                verbs=("edit_theme", "apply_theme", "save_theme"),
                appetites=("theme_palette",),
                emissions=("theme_applied", "theme_saved"),
            ),
        )
    )
    registry.register(
        AddonSpec(
            id="connections",
            kind="module",
            title="CONNECT",
            icon="⊕",
            factory=connections_factory,
            descriptor=CapabilityDescriptor(
                verbs=("start_server", "stop_server", "receive_agent_message"),
                appetites=("agent_message",),
                emissions=("chat_injected",),
            ),
        )
    )

    # MonoBase dev panel — Acatalepsy v1 A1 validator. Registered
    # unconditionally; if the auditor worker isn't running (flag off),
    # the panel opens in read-only mode (no "Audit now" button) and the
    # user can still browse / decide on existing candidates.
    def _monobase_dev_factory(ctx: AddonContext):
        from ui.addons.monobase_dev import MonoBaseDevPanel
        return MonoBaseDevPanel(decider_id="user_e")

    registry.register(
        AddonSpec(
            id="monobase_dev",
            kind="module",
            title="MONOBASE",
            icon="◇",
            factory=_monobase_dev_factory,
            descriptor=CapabilityDescriptor(
                verbs=("triage_candidates", "decide_candidate", "trigger_audit"),
                appetites=("acu_candidate",),
                emissions=("candidate_decided",),
            ),
        )
    )

    # MonoSearch companion panel — the human's read-only window into the
    # MonoSearch self-knowledge tool. Dashboard modes (Failing / Recurring /
    # Pulling / Unresolved) + free-text Search over the in-process
    # core.monosearch.service. Read-only: owns no data, writes nothing.
    def _monosearch_factory(ctx: AddonContext):
        from ui.addons.monosearch import MonoSearchPanel
        return MonoSearchPanel()

    registry.register(
        AddonSpec(
            id="monosearch",
            kind="module",
            title="MONOSEARCH",
            icon="⌕",
            factory=_monosearch_factory,
            descriptor=CapabilityDescriptor(
                verbs=("monosearch_query",),
                appetites=(),
                emissions=(),
            ),
        )
    )

    # Monothink Ledger — git-like undo surface for the self-evolving reasoning
    # scaffold. Lists every evolution (tag · rating · diff · state) over
    # core.monothink.list_ledger() and gives each applied version a Revert button
    # (core.monothink.revert_to_version). Append-only: a revert is itself recorded.
    def _monothink_ledger_factory(ctx: AddonContext):
        from ui.addons.monothink_ledger import MonothinkLedgerPanel
        return MonothinkLedgerPanel()

    registry.register(
        AddonSpec(
            id="monothink_ledger",
            kind="module",
            title="MONOTHINK LEDGER",
            icon="📜",
            factory=_monothink_ledger_factory,
            descriptor=CapabilityDescriptor(
                verbs=("revert_to_version",),
                appetites=(),
                emissions=(),
            ),
        )
    )

    # TODO(stats): full integration pending — PageStats is mounted directly in
    # _conversation_stack by MonolithUI.__init__ and routed via _handle_rail_request.
    # A proper addon factory would need to either reuse that singleton or coordinate
    # with the main window so that launching "stats" from the registry switches to
    # the already-mounted page rather than constructing a second instance.
    registry.register(
        AddonSpec(
            id="stats",
            kind="module",
            title="STATS",
            icon="▦",
            factory=lambda ctx: (_ for _ in ()).throw(  # type: ignore[return-value]
                NotImplementedError("stats addon is routed via IconRail, not addon host")
            ),
            descriptor=CapabilityDescriptor(
                verbs=("view_stats",),
                appetites=(),
                emissions=(),
            ),
        )
    )

    return registry
