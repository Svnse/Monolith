import uuid
from typing import Dict, Optional

from PySide6.QtWidgets import QWidget

from ui.addons.context import AddonContext
from ui.addons.registry import AddonRegistry


class AddonHost:
    def __init__(self, registry: AddonRegistry, ctx: AddonContext):
        self.registry = registry
        self.ctx = ctx
        self.ctx.host = self
        self._pages: Dict[str, QWidget] = {}

    def mount_page(self, addon_id: str) -> QWidget:
        if addon_id in self._pages:
            return self._pages[addon_id]

        spec = self.registry.get(addon_id)
        if spec.kind != "page":
            raise ValueError(f"Addon '{addon_id}' is not a page")
        widget = spec.factory(self.ctx)
        self._pages[addon_id] = widget
        return widget

    def get_page_widget(self, addon_id: str) -> Optional[QWidget]:
        return self._pages.get(addon_id)

    def launch_module(self, addon_id: str) -> str:
        if not self.ctx.ui:
            raise RuntimeError("AddonHost requires UI for launching modules")

        spec = self.registry.get(addon_id)
        if spec.kind != "module":
            raise ValueError(f"Addon '{addon_id}' is not a module")
        instance_id = str(uuid.uuid4())
        widget = spec.factory(self.ctx)
        widget._mod_id = instance_id

        self.ctx.ui.stack.addWidget(widget)
        self.ctx.ui.module_strip.add_module(instance_id, spec.icon or "?", spec.title)

        if hasattr(widget, "sig_closed"):
            widget.sig_closed.connect(lambda: self.ctx.ui.close_module(instance_id))
        if hasattr(widget, "sig_finished"):
            widget.sig_finished.connect(lambda: self.ctx.ui.module_strip.flash_module(instance_id))

        self.ctx.ui.switch_to_module(instance_id)
        return instance_id
