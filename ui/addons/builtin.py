import uuid

from ui.addons.context import AddonContext
from ui.addons.registry import AddonRegistry
from ui.addons.spec import AddonSpec
from ui.modules.injector import InjectorWidget
from ui.modules.sd import SDModule
from ui.modules.audiogen import AudioGenModule
from ui.modules.manager import PageAddons
from ui.pages.chat import PageChat
from ui.pages.databank import PageFiles
from ui.pages.hub import PageHub
from core.operators import OperatorManager
from engine.bridge import EngineBridge
from engine.llm import LLMEngine


def terminal_factory(ctx: AddonContext):
    instance_id = str(uuid.uuid4())
    engine_key = f"llm_{instance_id}"

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
    w.sig_generate.connect(
        lambda prompt, thinking_mode: ctx.bridge.submit(
            ctx.bridge.wrap(
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
        )
    )
    w.sig_load.connect(
        lambda: ctx.bridge.submit(ctx.bridge.wrap("terminal", "load", engine_key))
    )
    w.sig_unload.connect(
        lambda: ctx.bridge.submit(ctx.bridge.wrap("terminal", "unload", engine_key))
    )
    w.sig_stop.connect(lambda: ctx.bridge.stop(engine_key))
    w.sig_sync_history.connect(
        lambda history: ctx.bridge.submit(
            ctx.bridge.wrap(
                "terminal",
                "set_history",
                engine_key,
                payload={"history": history},
            )
        )
    )
    ctx.guard.sig_status.connect(w.update_status)
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

    def _current_terminal_config():
        if not ctx.ui:
            return {}
        for i in range(ctx.ui.stack.count()):
            widget = ctx.ui.stack.widget(i)
            if isinstance(widget, PageChat):
                return dict(widget.config)
        return {}

    w = PageHub(config_provider=_current_terminal_config, operator_manager=manager)

    def _load_operator(name: str):
        try:
            operator_data = manager.load_operator(name)
        except Exception:
            return
        ctx.ui_bridge.sig_apply_operator.emit(operator_data)

    w.sig_load_operator.connect(_load_operator)
    w.sig_save_operator.connect(lambda name, data: manager.save_operator(name, data))
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
            title="TERMINAL",
            icon="⌖",
            factory=terminal_factory,
        )
    )
    registry.register(
        AddonSpec(
            id="databank",
            kind="module",
            title="DATABANK",
            icon="▤",
            factory=databank_factory,
        )
    )
    registry.register(
        AddonSpec(
            id="hub",
            kind="page",
            title="HUB",
            icon=None,
            factory=hub_factory,
        )
    )
    registry.register(
        AddonSpec(
            id="addons",
            kind="page",
            title="ADDONS",
            icon=None,
            factory=addons_page_factory,
        )
    )
    registry.register(
        AddonSpec(
            id="injector",
            kind="module",
            title="RUNTIME",
            icon="💉",
            factory=injector_factory,
        )
    )
    registry.register(
        AddonSpec(
            id="sd",
            kind="module",
            title="VISION",
            icon="⟡",
            factory=sd_factory,
        )
    )
    registry.register(
        AddonSpec(
            id="audiogen",
            kind="module",
            title="AUDIO",
            icon="♫",
            factory=audiogen_factory,
        )
    )

    return registry
