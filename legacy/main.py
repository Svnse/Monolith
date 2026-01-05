import sys
from PySide6.QtWidgets import QApplication

from core.state import AppState
from engine.llm import LLMEngine
from monokernel.guard import MonoGuard
from ui.main_window import MonolithUI

def main():
    app = QApplication(sys.argv)
    
    # 1. Initialize State & Core
    state = AppState()
    engine = LLMEngine(state)
    guard = MonoGuard(state, engine)
    
    # 2. Initialize UI
    ui = MonolithUI(state)

    # 3. Connect UI Commands -> Engine
    ui.page_settings.sig_load.connect(guard.slot_load_model)
    ui.page_settings.sig_unload.connect(guard.slot_unload_model)
    ui.page_chat.sig_generate.connect(guard.slot_generate)

    # 4. Connect Engine Updates -> UI
    guard.sig_trace.connect(ui.page_chat.append_trace)
    guard.sig_token.connect(ui.page_chat.append_token)
    guard.sig_status.connect(ui.update_status)
    guard.sig_usage.connect(ui.update_ctx)

    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
