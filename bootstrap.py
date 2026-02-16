import sys

from PySide6.QtWidgets import QApplication

from core.theme_config import load_theme_config
from core.theme_engine import ThemeEngine
from core.themes import apply_theme
from core.style import refresh_styles
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
    theme_cfg = load_theme_config()
    apply_theme(theme_cfg.get("theme", "midnight"))
    refresh_styles()
    theme_engine = ThemeEngine()
    theme_engine.apply(app)

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

    def _apply_theme(theme_name: str) -> None:
        apply_theme(theme_name)
        refresh_styles()
        theme_engine.apply(app)

        # Backward compatibility while widgets still carry local style hooks.
        for w in app.allWidgets():
            if hasattr(w, "apply_theme_refresh"):
                w.apply_theme_refresh()
            elif hasattr(w, "refresh_style"):
                w.refresh_style()

    ui_bridge.sig_theme_changed.connect(_apply_theme)

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
