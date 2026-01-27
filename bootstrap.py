import sys

from PySide6.QtWidgets import QApplication

from core.state import AppState
from engine.bridge import EngineBridge
from engine.llm import LLMEngine
from engine.vision import VisionEngine
from monokernel.guard import MonoGuard
from ui.addons.builtin import build_builtin_registry
from ui.addons.context import AddonContext
from ui.addons.host import AddonHost
from ui.main_window import MonolithUI


def main():
    app = QApplication(sys.argv)
    state = AppState()
    engine_impl = LLMEngine(state)
    engine = EngineBridge(engine_impl)
    guard = MonoGuard(state, engine)
    vision_engine_impl = VisionEngine(state)
    vision_engine = EngineBridge(vision_engine_impl)
    vision_guard = MonoGuard(state, vision_engine)

    ui = MonolithUI(state)

    registry = build_builtin_registry()
    ctx = AddonContext(state=state, guard=guard, vision_guard=vision_guard, ui=ui, host=None)
    host = AddonHost(registry, ctx)
    ui.attach_host(host)

    # global chrome-only wiring stays here
    guard.sig_status.connect(ui.update_status)
    guard.sig_usage.connect(ui.update_ctx)
    app.aboutToQuit.connect(guard.slot_stop)
    app.aboutToQuit.connect(engine.shutdown)
    app.aboutToQuit.connect(vision_guard.slot_stop)
    app.aboutToQuit.connect(vision_engine.shutdown)

    ui.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
