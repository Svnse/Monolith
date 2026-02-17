import uuid

from ui.addons.context import AddonContext
from ui.addons.descriptors import CapabilityDescriptor
from ui.addons.registry import AddonRegistry
from ui.addons.spec import AddonSpec
from ui.modules.injector import InjectorWidget
from ui.modules.sd import SDModule
from ui.modules.audiogen import AudioGenModule
from ui.modules.manager import PageAddons
from ui.pages.chat import PageChat
from ui.pages.code import PageCode
from ui.pages.databank import PageFiles
from ui.pages.hub import PageHub
from core.operators import OperatorManager
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

    w = PageChat(ctx.state, ctx.ui_bridge)
    w._mod_id = instance_id
    w._engine_key = engine_key
    ctx.ui_bridge.sig_apply_operator.connect(w.apply_operator)
    llm_engine.sig_model_capabilities.connect(w._on_model_capabilities)

    w.sig_set_model_path.connect(
        lambda path: ctx.bridge.submit(
            ctx.bridge.wrap("terminal", "set_path", engine_key, payload={"path": path})
        )
    )
    w.sig_set_ctx_limit.connect(
        lambda limit: None if limit is None else ctx.bridge.submit(
            ctx.bridge.wrap("terminal", "set_ctx_limit", engine_key, payload={"ctx_limit": int(limit)})
        )
    )

    if w.config.get("gguf_path"):
        w.sig_set_model_path.emit(str(w.config.get("gguf_path")))
    w.sig_set_ctx_limit.emit(int(w.config.get("ctx_limit", 8192)))

    # outgoing (addon -> bridge)
    def _on_generate(prompt, thinking_mode):
        try:
            model = w.config.get("gguf_path", "unknown")
            model_name = str(model).rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if model else "none"
            think_label = "think=ON" if thinking_mode else "think=OFF"
            _trace(f"[LLM:{short_id}] generating — {think_label}, model={model_name}, prompt={repr(prompt[:50])}")
            task = ctx.bridge.wrap(
                "terminal",
                "generate",
                engine_key,
                payload={
                    "prompt": prompt,
                    "config": w.config,
                    "thinking_mode": thinking_mode,
                    "ctx_limit": int(w.config.get("ctx_limit", 8192)),
                },
            )
            ctx.bridge.submit(task)
        except Exception as e:
            _trace(f"[LLM:{short_id}] EXCEPTION in generate: {e}")
            import traceback
            traceback.print_exc()

    w.sig_generate.connect(_on_generate)
    w.sig_load.connect(
        lambda: ctx.bridge.submit(ctx.bridge.wrap("terminal", "load", engine_key))
    )
    w.sig_unload.connect(
        lambda: ctx.bridge.submit(ctx.bridge.wrap("terminal", "unload", engine_key))
    )
    def _on_stop():
        try:
            _trace(f"[LLM:{short_id}] stopped — generation halted")
            ctx.bridge.stop(engine_key)
        except Exception as e:
            _trace(f"[LLM:{short_id}] EXCEPTION in stop: {e}")
            import traceback
            traceback.print_exc()

    w.sig_stop.connect(_on_stop)

    def _on_sync_history(history):
        try:
            _trace(f"[LLM:{short_id}] syncing history — {len(history)} messages")
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

    w.sig_sync_history.connect(_on_sync_history)
    ctx.guard.sig_status.connect(w.update_status)
    w.sig_debug.connect(lambda msg: ctx.guard.sig_trace.emit(engine_key, msg))
    # incoming (guard -> addon)
    ctx.guard.sig_token.connect(
        lambda ek, t: w.append_token(t) if ek == engine_key else None
    )
    ctx.guard.sig_trace.connect(
        lambda ek, m: w.append_trace(m) if ek == engine_key else None
    )
    ctx.guard.sig_finished.connect(w.on_guard_finished)

    def _cleanup_terminal(*_args):
        ctx.guard.unregister_engine(engine_key)
        engine_bridge.shutdown()

    w.destroyed.connect(_cleanup_terminal)
    return w


def code_factory(ctx: AddonContext):
    instance_id = str(uuid.uuid4())
    engine_key = f"llm_{instance_id}"

    short_id = instance_id[:8]

    def _trace(msg):
        ctx.guard.sig_trace.emit("system", msg)

    llm_engine = LLMEngine(ctx.state)
    engine_bridge = EngineBridge(llm_engine)
    ctx.guard.register_engine(engine_key, engine_bridge)

    w = PageCode(ctx.state, ctx.ui_bridge)
    w._mod_id = instance_id
    w._engine_key = engine_key
    ctx.ui_bridge.sig_apply_operator.connect(w.apply_operator)

    w.sig_set_model_path.connect(
        lambda path: ctx.bridge.submit(
            ctx.bridge.wrap("code", "set_path", engine_key, payload={"path": path})
        )
    )
    w.sig_set_ctx_limit.connect(
        lambda limit: None if limit is None else ctx.bridge.submit(
            ctx.bridge.wrap("code", "set_ctx_limit", engine_key, payload={"ctx_limit": int(limit)})
        )
    )

    if w.config.get("gguf_path"):
        w.sig_set_model_path.emit(str(w.config.get("gguf_path")))
    w.sig_set_ctx_limit.emit(int(w.config.get("ctx_limit", 8192)))

    def _on_generate(prompt):
        try:
            model = w.config.get("gguf_path", "unknown")
            model_name = str(model).rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if model else "none"
            _trace(f"[LLM:{short_id}] generating — agent=ON, model={model_name}, prompt={repr(prompt[:50])}")
            task = ctx.bridge.wrap(
                "code",
                "generate",
                engine_key,
                payload={
                    "prompt": prompt,
                    "config": w.config,
                    "agent_mode": True,
                    "workspace_root": w._workspace_root,
                    "ctx_limit": int(w.config.get("ctx_limit", 8192)),
                },
            )
            ctx.bridge.submit(task)
        except Exception as e:
            _trace(f"[LLM:{short_id}] EXCEPTION in generate: {e}")
            import traceback
            traceback.print_exc()

    w.sig_generate.connect(_on_generate)
    w.sig_load.connect(
        lambda: ctx.bridge.submit(ctx.bridge.wrap("code", "load", engine_key))
    )
    w.sig_unload.connect(
        lambda: ctx.bridge.submit(ctx.bridge.wrap("code", "unload", engine_key))
    )

    def _on_stop():
        try:
            _trace(f"[LLM:{short_id}] stopped — generation halted")
            ctx.bridge.stop(engine_key)
        except Exception as e:
            _trace(f"[LLM:{short_id}] EXCEPTION in stop: {e}")
            import traceback
            traceback.print_exc()

    w.sig_stop.connect(_on_stop)

    def _on_sync_history(history):
        try:
            _trace(f"[LLM:{short_id}] syncing history — {len(history)} messages")
            ctx.bridge.submit(
                ctx.bridge.wrap(
                    "code",
                    "set_history",
                    engine_key,
                    payload={"history": history},
                )
            )
        except Exception as e:
            _trace(f"[LLM:{short_id}] EXCEPTION in sync_history: {e}")
            import traceback
            traceback.print_exc()

    w.sig_sync_history.connect(_on_sync_history)
    ctx.guard.sig_status.connect(w.update_status)
    w.sig_debug.connect(lambda msg: ctx.guard.sig_trace.emit(engine_key, msg))
    ctx.guard.sig_token.connect(
        lambda ek, t: w.append_token(t) if ek == engine_key else None
    )
    ctx.guard.sig_trace.connect(
        lambda ek, m: w.append_trace(f"[{ek}] {m}" if ek != engine_key else m)
    )
    ctx.guard.sig_agent_event.connect(
        lambda ek, event: w.append_trace(f"[agent-event] {event}") if ek == engine_key else None
    )
    ctx.guard.sig_finished.connect(w.on_guard_finished)

    def _cleanup_terminal(*_args):
        ctx.guard.unregister_engine(engine_key)
        engine_bridge.shutdown()

    w.destroyed.connect(_cleanup_terminal)
    return w


def addons_page_factory(ctx: AddonContext):
    w = PageAddons(ctx.state)
    # route launcher directly to host (host must exist)
    assert ctx.host is not None, "AddonHost must exist before addons page wiring"
    w.sig_launch_addon.connect(lambda addon_id: ctx.host.launch_module(addon_id))
    w.sig_open_vitals.connect(lambda: ctx.ui.toggle_vitals() if ctx.ui else None)
    w.sig_open_overseer.connect(ctx.ui_bridge.sig_open_overseer.emit)
    return w



def hub_factory(ctx: AddonContext):
    manager = OperatorManager()

    def _snapshot_workspace():
        """Capture full workspace state: all open modules + terminal config/messages."""
        if not ctx.ui:
            return {}

        modules = []
        module_order = []
        for mod_id in ctx.ui.module_strip.get_order():
            # Find widget by mod_id
            widget = None
            for i in range(ctx.ui.stack.count()):
                w = ctx.ui.stack.widget(i)
                if getattr(w, '_mod_id', None) == mod_id:
                    widget = w
                    break
            if not widget:
                continue

            addon_id = getattr(widget, '_addon_id', None)
            if not addon_id:
                continue

            module_order.append(addon_id)
            entry = {"addon_id": addon_id}

            # For terminals, capture config + chat messages
            if isinstance(widget, PageChat):
                entry["config"] = dict(widget.config)
                session = getattr(widget, '_current_session', None)
                if session:
                    entry["messages"] = list(session.get("messages", []))
                    entry["session_title"] = session.get("title")
                    entry["assistant_tokens"] = session.get("assistant_tokens", 0)

            modules.append(entry)

        ctx.guard.sig_trace.emit("system", f"[OPERATOR] snapshot: {len(modules)} modules")
        return {"modules": modules, "module_order": module_order}

    w = PageHub(config_provider=_snapshot_workspace, operator_manager=manager, ui_bridge=ctx.ui_bridge)

    def _load_operator(name: str):
        ctx.guard.sig_trace.emit("system", f"[OPERATOR] loading '{name}'")
        try:
            operator_data = manager.load_operator(name)
        except Exception as e:
            ctx.guard.sig_trace.emit("system", f"[OPERATOR] failed to load: {e}")
            return

        if not ctx.ui or not ctx.host:
            return

        # --- New format: has "modules" list ---
        if "modules" in operator_data:
            modules = operator_data["modules"]
            ctx.guard.sig_trace.emit("system", f"[OPERATOR] restoring {len(modules)} modules")

            # Close all existing modules
            for mod_id in list(ctx.ui.module_strip.get_order()):
                ctx.ui.close_module(mod_id)

            # Launch each module from snapshot
            first_terminal_mod_id = None
            for entry in modules:
                addon_id = entry.get("addon_id")
                if not addon_id:
                    continue
                new_mod_id = ctx.host.launch_module(addon_id)
                if not new_mod_id:
                    ctx.guard.sig_trace.emit("system", f"[OPERATOR] failed to launch {addon_id}")
                    continue

                # For terminals with saved state, apply config + messages
                if addon_id == "terminal" and "config" in entry:
                    for i in range(ctx.ui.stack.count()):
                        widget = ctx.ui.stack.widget(i)
                        if getattr(widget, '_mod_id', None) == new_mod_id and isinstance(widget, PageChat):
                            widget.apply_operator(entry)
                            break
                    if not first_terminal_mod_id:
                        first_terminal_mod_id = new_mod_id

            # Switch to first terminal
            if first_terminal_mod_id:
                ctx.ui.switch_to_module(first_terminal_mod_id)

        # --- Legacy format: top-level "config" only ---
        else:
            ctx.guard.sig_trace.emit("system", f"[OPERATOR] legacy format for '{name}'")
            target_widget = None
            for i in range(ctx.ui.stack.count()):
                widget = ctx.ui.stack.widget(i)
                if isinstance(widget, PageChat):
                    target_widget = widget
                    break

            if not target_widget:
                ctx.host.launch_module("terminal")
                for i in range(ctx.ui.stack.count()):
                    widget = ctx.ui.stack.widget(i)
                    if isinstance(widget, PageChat):
                        target_widget = widget
                        break

            ctx.ui_bridge.sig_apply_operator.emit(operator_data)

            if target_widget:
                mod_id = getattr(target_widget, "_mod_id", None)
                if mod_id:
                    ctx.ui.switch_to_module(mod_id)

    w.sig_load_operator.connect(_load_operator)
    w.sig_save_operator.connect(lambda name, data: manager.save_operator(name, data))
    w.sig_presence_drift.connect(
        lambda name, drift_score, threshold: ctx.guard.sig_trace.emit(
            "system",
            f"[PRESENCE] WARN drift exceeded for '{name}': score={drift_score:.2f}, threshold={threshold:.2f}",
        )
    )
    return w

def databank_factory(ctx: AddonContext):
    return PageFiles(ctx.state)


def injector_factory(ctx: AddonContext):
    assert ctx.ui is not None, "InjectorWidget requires UI parent"
    return InjectorWidget(ctx.ui)


def sd_factory(ctx: AddonContext):
    return SDModule(ctx.bridge, ctx.guard)


def audiogen_factory(ctx: AddonContext):
    return AudioGenModule()


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
            id="code",
            kind="module",
            title="CODE",
            icon="▶",
            factory=code_factory,
            descriptor=CapabilityDescriptor(
                verbs=("generate_text", "agent", "tool_use", "code"),
                appetites=("text_prompt", "gguf_file", "workspace_path"),
                emissions=("text_stream", "tool_result", "code_output"),
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
            id="hub",
            kind="page",
            title="HOME",
            icon=None,
            factory=hub_factory,
            descriptor=CapabilityDescriptor(
                verbs=("load_operator", "save_operator", "list_operators"),
                appetites=("operator_name",),
                emissions=("operator_loaded", "operator_saved"),
            ),
        )
    )
    registry.register(
        AddonSpec(
            id="addons",
            kind="page",
            title="MODULES",
            icon=None,
            factory=addons_page_factory,
            descriptor=CapabilityDescriptor(
                verbs=("launch_module", "list_modules"),
                appetites=(),
                emissions=("module_launched",),
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

    return registry
