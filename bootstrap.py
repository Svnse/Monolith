import sys

from PySide6.QtWidgets import QApplication

from core.state import AppState
from engine.bridge import EngineBridge
from engine.vision import VisionEngine
from monokernel.bridge import MonoBridge
from monokernel.dock import MonoDock
from monokernel.guard import MonoGuard
from ui.addons.builtin import build_builtin_registry
from ui.addons.context import AddonContext
from ui.addons.host import AddonHost
from ui.bridge import UIBridge
from ui.main_window import MonolithUI
from ui.overseer import OverseerWindow


def main():
    app = QApplication(sys.argv)

    # Load saved theme before any UI is created
    from core.theme_config import load_theme_config
    from core.themes import apply_theme
    from core.style import refresh_styles
    theme_cfg = load_theme_config()
    apply_theme(theme_cfg.get("theme", "midnight"))
    refresh_styles()

    state = AppState()
    vision_engine_impl = VisionEngine(state)
    vision_engine = EngineBridge(vision_engine_impl)
    guard = MonoGuard(state, {"vision": vision_engine})
    dock = MonoDock(guard)
    bridge = MonoBridge(dock)

    ui_bridge = UIBridge()
    ui = MonolithUI(state, ui_bridge)
    overseer = OverseerWindow(guard, ui_bridge)

    registry = build_builtin_registry()
    ctx = AddonContext(state=state, guard=guard, bridge=bridge, ui=ui, host=None, ui_bridge=ui_bridge)
    host = AddonHost(registry, ctx)
    ui.attach_host(host)

    ui_bridge.sig_open_overseer.connect(overseer.show)
    ui_bridge.sig_overseer_viz_toggle.connect(guard.enable_viztracer)

    # Theme change: refresh style constants and rebuild UI stylesheets
    def _on_theme_changed(theme_name):
        apply_theme(theme_name)
        refresh_styles()
        ui.apply_theme_refresh()
        # Refresh all mounted pages (hub, addons, etc.)
        for page in ui.pages.values():
            if hasattr(page, "apply_theme_refresh"):
                page.apply_theme_refresh()
        # Refresh any open module widgets in the stack
        for i in range(ui.stack.count()):
            w = ui.stack.widget(i)
            if hasattr(w, "apply_theme_refresh"):
                w.apply_theme_refresh()

    ui_bridge.sig_theme_changed.connect(_on_theme_changed)

    # global chrome-only wiring stays here
    guard.sig_status.connect(ui.update_status)
    guard.sig_usage.connect(lambda _ek, used: ui.update_ctx(used))
    app.aboutToQuit.connect(guard.stop)
    app.aboutToQuit.connect(overseer.db.close)
    app.aboutToQuit.connect(lambda: guard.enable_viztracer(False) if guard._viztracer is not None else None)
    app.aboutToQuit.connect(vision_engine.shutdown)

    ui.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
