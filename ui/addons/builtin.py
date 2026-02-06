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


def terminal_factory(ctx: AddonContext):
    w = PageChat(ctx.state, ctx.ui_bridge)
    ctx.ui_bridge.sig_apply_operator.connect(w.apply_operator)
    # outgoing (addon -> bridge)
    w.sig_generate.connect(
        lambda prompt, thinking_mode: ctx.bridge.submit(
            ctx.bridge.wrap(
                "terminal",
                "generate",
                "llm",
                payload={"prompt": prompt, "config": w.config, "thinking_mode": thinking_mode},
            )
        )
    )
    w.sig_load.connect(
        lambda: ctx.bridge.submit(ctx.bridge.wrap("terminal", "load", "llm"))
    )
    w.sig_unload.connect(
        lambda: ctx.bridge.submit(ctx.bridge.wrap("terminal", "unload", "llm"))
    )
    w.sig_stop.connect(lambda: ctx.bridge.stop("llm"))
    w.sig_sync_history.connect(
        lambda history: ctx.bridge.submit(
            ctx.bridge.wrap(
                "terminal",
                "set_history",
                "llm",
                payload={"history": history},
            )
        )
    )
    ctx.guard.sig_status.connect(w.update_status)
    # incoming (guard -> addon)
    ctx.guard.sig_token.connect(w.append_token)
    ctx.guard.sig_trace.connect(w.append_trace)
    ctx.guard.sig_finished.connect(w.on_guard_finished)
    return w


def addons_page_factory(ctx: AddonContext):
    w = PageAddons(ctx.state)
    # route launcher directly to host (host must exist)
    assert ctx.host is not None, "AddonHost must exist before addons page wiring"
    w.sig_launch_addon.connect(lambda addon_id: ctx.host.launch_module(addon_id))
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
