from ui.addons.context import AddonContext
from ui.addons.registry import AddonRegistry
from ui.addons.spec import AddonSpec
from ui.modules.injector import InjectorWidget
from ui.modules.manager import PageAddons
from ui.pages.chat import PageChat
from ui.pages.databank import PageFiles
from ui.pages.settings import PageSettings


def build_builtin_registry() -> AddonRegistry:
    registry = AddonRegistry()

    registry.register(
        AddonSpec(
            id="terminal",
            kind="page",
            title="Terminal",
            icon=None,
            factory=lambda ctx: PageChat(ctx.state),
        )
    )
    registry.register(
        AddonSpec(
            id="databank",
            kind="page",
            title="Databank",
            icon=None,
            factory=lambda ctx: PageFiles(ctx.state),
        )
    )
    registry.register(
        AddonSpec(
            id="addons",
            kind="page",
            title="Addons",
            icon=None,
            factory=lambda ctx: PageAddons(ctx.state),
        )
    )
    registry.register(
        AddonSpec(
            id="settings",
            kind="page",
            title="Settings",
            icon=None,
            factory=lambda ctx: PageSettings(ctx.state),
        )
    )

    registry.register(
        AddonSpec(
            id="injector",
            kind="module",
            title="RUNTIME",
            icon="💉",
            factory=_build_injector,
        )
    )

    return registry


def _build_injector(ctx: AddonContext):
    assert ctx.ui is not None, "InjectorWidget requires UI parent"
    return InjectorWidget(ctx.ui)
