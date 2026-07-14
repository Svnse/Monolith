from PySide6.QtWidgets import QApplication, QWidget

from core.state import AppState
from ui.bridge import UIBridge
from ui.companion_pane import CompanionState
from ui.main_window import MonolithUI


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _DummyModule(QWidget):
    def __init__(self, addon_id: str, title: str):
        super().__init__()
        self._addon_id = addon_id
        self._title = title

    def windowTitle(self) -> str:  # noqa: N802
        return self._title

    def _refresh_audit_list(self) -> None:
        return

    def _refresh_archive_list(self) -> None:
        return


class _DummyWorldState:
    def __init__(self, snapshot: dict):
        self._snapshot = snapshot

    def snapshot(self) -> dict:
        return self._snapshot


def test_addon_rail_click_toggles_regular_addon_closed():
    _app()
    ui = MonolithUI(AppState(), UIBridge())
    widget = _DummyModule("theme", "THEME")
    widget._mod_id = "mod-theme"
    ui.stack.addWidget(widget)
    ui.register_module("mod-theme", "theme", "*", "THEME")

    ui.switch_to_module("mod-theme")
    assert ui.companion._current_state == CompanionState.ADDON

    ui._handle_rail_request("addon:mod-theme")
    assert ui.companion._current_state == CompanionState.COLLAPSED


def test_addon_rail_click_toggles_media_addon_closed():
    _app()
    ui = MonolithUI(AppState(), UIBridge())
    widget = _DummyModule("sd", "VISION")
    widget._mod_id = "mod-vision"
    ui.stack.addWidget(widget)
    ui.register_module("mod-vision", "sd", "*", "VISION")

    ui.switch_to_module("mod-vision")
    assert ui.companion._current_state == CompanionState.VISION

    ui._handle_rail_request("addon:mod-vision")
    assert ui.companion._current_state == CompanionState.COLLAPSED


def test_switching_to_other_module_does_not_get_forced_back_to_generating():
    _app()
    snapshot = {
        "session": {"pending_action": None},
        "engines": {
            "llm_current": {"status": "RUNNING"},
        },
        "tasks": {
            "llm_current": {
                "active": {"id": "task-1", "command": "generate", "status": "RUNNING"},
                "queue_len": 0,
            },
        },
    }
    state = AppState()
    state.world_state = _DummyWorldState(snapshot)
    ui = MonolithUI(state, UIBridge())

    terminal = _DummyModule("terminal", "CHAT")
    terminal._mod_id = "mod-chat"
    terminal._engine_key = "llm_current"
    addon = _DummyModule("theme", "THEME")
    addon._mod_id = "mod-theme"

    ui.stack.addWidget(terminal)
    ui.stack.addWidget(addon)
    ui.register_module("mod-chat", "terminal", "", "CHAT")
    ui.register_module("mod-theme", "theme", "*", "THEME")

    ui.switch_to_module("mod-chat")
    ui.companion.refresh_from_world()
    assert ui.companion._current_state == CompanionState.GENERATING

    ui.switch_to_module("mod-theme")
    assert ui.companion._current_state == CompanionState.ADDON


def test_generating_button_can_reopen_and_close_trace_panel():
    _app()
    snapshot = {
        "session": {"pending_action": None},
        "engines": {
            "llm_current": {"status": "RUNNING"},
        },
        "tasks": {
            "llm_current": {
                "active": {"id": "task-1", "command": "generate", "status": "RUNNING"},
                "queue_len": 0,
            },
        },
    }
    state = AppState()
    state.world_state = _DummyWorldState(snapshot)
    ui = MonolithUI(state, UIBridge())

    terminal = _DummyModule("terminal", "CHAT")
    terminal._mod_id = "mod-chat"
    terminal._engine_key = "llm_current"
    ui.stack.addWidget(terminal)
    ui.register_module("mod-chat", "terminal", "", "CHAT")

    ui.switch_to_module("mod-chat")
    ui.companion.collapse()
    assert ui.companion._current_state == CompanionState.COLLAPSED

    ui._handle_rail_request("GENERATING")
    assert ui.companion._current_state == CompanionState.GENERATING

    ui._handle_rail_request("GENERATING")
    assert ui.companion._current_state == CompanionState.COLLAPSED


def test_overseer_rail_request_emits_open_signal():
    _app()
    bridge = UIBridge()
    ui = MonolithUI(AppState(), bridge)
    hits: list[str] = []
    bridge.sig_open_overseer.connect(lambda: hits.append("open"))

    ui._handle_rail_request("OVERSEER")

    assert hits == ["open"]


def test_omni_focus_moves_glow_off_top_bar():
    _app()
    ui = MonolithUI(AppState(), UIBridge())

    assert ui.gradient_line._glow_active is True

    ui.omni_bar.sig_focus_glow_changed.emit(True)
    assert ui.gradient_line._glow_active is False

    ui.omni_bar.sig_focus_glow_changed.emit(False)
    assert ui.gradient_line._glow_active is True


def test_accept_external_text_to_chat_routes_to_active_terminal():
    _app()
    ui = MonolithUI(AppState(), UIBridge())

    class _Chat(_DummyModule):
        def __init__(self):
            super().__init__("terminal", "CHAT")
            self.calls: list[tuple[str, str, bool]] = []

        def accept_external_text(self, text: str, *, label: str, force_attachment: bool) -> str:
            self.calls.append((text, label, force_attachment))
            return "attached"

    chat = _Chat()
    chat._mod_id = "mod-chat"
    ui.stack.addWidget(chat)
    ui.register_module("mod-chat", "terminal", "", "CHAT")
    ui.switch_to_module("mod-chat")

    result = ui.accept_external_text_to_chat("payload", label="external", force_attachment=True)

    assert result == "attached"
    assert chat.calls == [("payload", "external", True)]
    assert ui._conversation_stack.currentWidget() is chat
