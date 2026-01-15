from ui.addons.context import AddonContext
from ui.addons.registry import AddonRegistry
from ui.addons.spec import AddonSpec
from ui.modules.injector import InjectorWidget
from ui.modules.sd import SDModule
from ui.modules.audiogen import AudioGenModule
from ui.modules.manager import PageAddons
from ui.pages.chat import PageChat
from ui.pages.databank import PageFiles
from ui.pages.settings import PageSettings


def terminal_factory(ctx: AddonContext):
    w = PageChat(ctx.state)
    # outgoing (addon -> guard)
    w.sig_generate.connect(ctx.guard.slot_generate)
    # incoming (guard -> addon)
    ctx.guard.sig_token.connect(w.append_token)
    ctx.guard.sig_trace.connect(w.append_trace)
    return w


def settings_factory(ctx: AddonContext):
    w = PageSettings(ctx.state)
    w.sig_load.connect(ctx.guard.slot_load_model)
    w.sig_unload.connect(ctx.guard.slot_unload_model)
    return w


def addons_page_factory(ctx: AddonContext):
    w = PageAddons(ctx.state)
    # route launcher directly to host (host must exist)
    assert ctx.host is not None, "AddonHost must exist before addons page wiring"
    w.sig_launch_addon.connect(lambda addon_id: ctx.host.launch_module(addon_id))
    return w


def databank_factory(ctx: AddonContext):
    return PageFiles(ctx.state)


def injector_factory(ctx: AddonContext):
    assert ctx.ui is not None, "InjectorWidget requires UI parent"
    return InjectorWidget(ctx.ui)


def sd_factory(ctx: AddonContext):
    return SDModule()


def audiogen_factory(ctx: AddonContext):
    return AudioGenModule()


def build_builtin_registry() -> AddonRegistry:
    registry = AddonRegistry()

    registry.register(
        AddonSpec(
            id="terminal",
            kind="module",
            title="Terminal",
            icon="⌖",
            factory=terminal_factory,
        )
    )
    registry.register(
        AddonSpec(
            id="databank",
            kind="module",
            title="Databank",
            icon="▤",
            factory=databank_factory,
        )
    )
    registry.register(
        AddonSpec(
            id="addons",
            kind="page",
            title="Addons",
            icon=None,
            factory=addons_page_factory,
        )
    )
    registry.register(
        AddonSpec(
            id="settings",
            kind="page",
            title="Settings",
            icon=None,
            factory=settings_factory,
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
            title="Stable Diffusion",
            icon="⟡",
            factory=sd_factory,
        )
    )
    registry.register(
        AddonSpec(
            id="audiogen",
            kind="module",
            title="Audio Gen",
            icon="♫",
            factory=audiogen_factory,
        )
    )

    return registry
