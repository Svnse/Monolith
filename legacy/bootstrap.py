import sys

from PySide6.QtWidgets import QApplication

from core.state import AppState
from engine.llm import LLMEngine
from monokernel.guard import MonoGuard
from ui.addons.builtin import build_builtin_registry
from ui.addons.context import AddonContext
from ui.addons.host import AddonHost
from ui.main_window import MonolithUI


def main():
    app = QApplication(sys.argv)
    state = AppState()
    engine = LLMEngine(state)
    guard = MonoGuard(state, engine)

    ui = MonolithUI(state)

    registry = build_builtin_registry()
    ctx = AddonContext(state=state, guard=guard, ui=ui, host=None)
    host = AddonHost(registry, ctx)
    ui.attach_host(host)

    # global chrome-only wiring stays here
    guard.sig_status.connect(ui.update_status)
    guard.sig_usage.connect(ui.update_ctx)
    app.aboutToQuit.connect(guard.slot_stop)
    app.aboutToQuit.connect(engine.shutdown)

    ui.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
