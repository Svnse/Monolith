from PySide6.QtWidgets import QApplication, QWidget

from ui.bridge import UIBridge
from ui.companion_pane import CompanionPane, CompanionState


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _DummyState:
    world_state = None


class _DummyWorldState:
    def __init__(self, snapshot: dict):
        self._snapshot = snapshot

    def snapshot(self) -> dict:
        return self._snapshot


class _DummyConversation:
    def __init__(self, engine_key: str):
        self._engine_key = engine_key

    def _refresh_audit_list(self) -> None:
        return

    def _refresh_archive_list(self) -> None:
        return


class _ConversationWithPanels(_DummyConversation):
    def __init__(self, engine_key: str):
        super().__init__(engine_key)
        self._config = QWidget()

    def companion_panels(self) -> dict[str, QWidget]:
        return {"CONFIG": self._config}


def test_companion_ignores_stale_other_llm_engine_states():
    _app()
    pane = CompanionPane(_DummyState(), UIBridge())
    pane.set_conversation(_DummyConversation("llm_current"))

    snapshot = {
        "session": {"pending_action": None},
        "engines": {
            "llm_current": {"status": "READY"},
            "llm_stale": {"status": "LOADING"},
        },
        "tasks": {
            "llm_current": {"active": None},
            "llm_stale": {
                "active": {"command": "load", "status": "CANCELLED"},
                "queue_len": 0,
            },
        },
    }

    assert pane.evaluate_state(snapshot) == CompanionState.COLLAPSED


def test_companion_prefers_manual_panel_over_active_conversation_generating_state():
    _app()
    pane = CompanionPane(_DummyState(), UIBridge())
    pane.set_conversation(_DummyConversation("llm_current"))
    pane.show_state(CompanionState.ARCHIVE)

    snapshot = {
        "session": {"pending_action": None},
        "engines": {
            "llm_current": {"status": "RUNNING"},
            "llm_other": {"status": "READY"},
        },
        "tasks": {
            "llm_current": {
                "active": {"command": "generate", "status": "RUNNING"},
                "queue_len": 0,
            },
            "llm_other": {"active": None, "queue_len": 0},
        },
    }

    assert pane.evaluate_state(snapshot) == CompanionState.ARCHIVE


def test_companion_uses_generating_when_no_manual_panel_is_selected():
    _app()
    pane = CompanionPane(_DummyState(), UIBridge())
    pane.set_conversation(_DummyConversation("llm_current"))

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

    assert pane.evaluate_state(snapshot) == CompanionState.GENERATING


def test_companion_prefers_generating_over_vision_when_both_are_active():
    _app()
    pane = CompanionPane(_DummyState(), UIBridge())
    pane.set_conversation(_DummyConversation("llm_current"))

    snapshot = {
        "session": {"pending_action": None},
        "engines": {
            "llm_current": {"status": "RUNNING"},
            "vision": {"status": "RUNNING"},
        },
        "tasks": {
            "llm_current": {
                "active": {"command": "generate", "status": "RUNNING"},
                "queue_len": 0,
            },
            "vision": {
                "active": {"command": "generate", "status": "RUNNING"},
                "queue_len": 0,
            },
        },
    }

    assert pane.evaluate_state(snapshot) == CompanionState.GENERATING


def test_companion_uses_audio_state_when_audio_engine_is_busy():
    _app()
    pane = CompanionPane(_DummyState(), UIBridge())
    pane.set_conversation(_DummyConversation("llm_current"))

    snapshot = {
        "session": {"pending_action": None},
        "engines": {
            "llm_current": {"status": "READY"},
            "audio": {"status": "RUNNING"},
        },
        "tasks": {
            "llm_current": {"active": None, "queue_len": 0},
            "audio": {
                "active": {"command": "generate", "status": "RUNNING"},
                "queue_len": 0,
            },
        },
    }

    assert pane.evaluate_state(snapshot) == CompanionState.AUDIO


def test_companion_collapse_dismisses_generating_until_the_task_changes():
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
    state = _DummyState()
    state.world_state = _DummyWorldState(snapshot)
    pane = CompanionPane(state, UIBridge())
    pane.set_conversation(_DummyConversation("llm_current"))

    pane.refresh_from_world()
    assert pane._current_state == CompanionState.GENERATING

    pane.collapse()
    assert pane._current_state == CompanionState.COLLAPSED
    assert pane.evaluate_state(snapshot) == CompanionState.COLLAPSED

    snapshot["tasks"]["llm_current"]["active"]["id"] = "task-2"
    assert pane.evaluate_state(snapshot) == CompanionState.GENERATING


def test_companion_prefers_conversation_owned_panel_widget():
    _app()
    pane = CompanionPane(_DummyState(), UIBridge())
    controller = _ConversationWithPanels("llm_current")
    pane.set_conversation(controller)

    assert pane.get_panel(CompanionState.CONFIG) is controller._config
