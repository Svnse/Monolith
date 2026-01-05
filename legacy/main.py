import sys
from PySide6.QtWidgets import QApplication

from core.state import AppState
from engine.llm import LLMEngine
from ui.main_window import MonolithUI

def main():
    app = QApplication(sys.argv)
    
    # 1. Initialize State & Core
    state = AppState()
    engine = LLMEngine(state)
    
    # 2. Initialize UI
    ui = MonolithUI(state)

    # 3. Connect UI Commands -> Engine
    ui.page_settings.sig_load.connect(engine.load_model)
    ui.page_settings.sig_unload.connect(engine.unload_model)
    ui.page_chat.sig_generate.connect(engine.generate)

    # 4. Connect Engine Updates -> UI
    engine.sig_trace.connect(ui.page_chat.append_trace)
    engine.sig_token.connect(ui.page_chat.append_token)
    engine.sig_status.connect(ui.update_status)
    engine.sig_usage.connect(ui.update_ctx)

    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
