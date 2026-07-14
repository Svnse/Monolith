from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _FakeRunner:
    def __init__(self, snapshot: dict) -> None:
        self._snapshot = snapshot
        self.start_calls: list[tuple[str, bool]] = []
        self.pause_calls = 0
        self.stop_calls = 0

    def snapshot(self) -> dict:
        return dict(self._snapshot)

    def start(self, goal: str = "", *, clear_thinking: bool = False) -> bool:
        self.start_calls.append((goal, clear_thinking))
        if clear_thinking:
            self._snapshot["thinking"] = []
        self._snapshot["status"] = "starting"
        return True

    def pause(self) -> None:
        self.pause_calls += 1
        self._snapshot["status"] = "paused"

    def stop(self, *args, **kwargs) -> None:
        self.stop_calls += 1
        self._snapshot["status"] = "stopped"


def _snapshot(**overrides) -> dict:
    snap = {
        "status": "idle",
        "coherence": {
            "verdict": "GREEN",
            "dims": {"grounding_ratio": 1.0, "progress": 0.5, "drift_overlap": 0.0},
        },
        "goal": "map the engine",
        "next_move": "read engine/llm.py",
        "tick": 1,
        "max_ticks": 6,
        "tools_total": 1,
        "world_acus": 1,
        "fault_streak": 0,
        "tokens": 128,
        "thinking": [],
        "activity": [],
        "referents": [],
        "last_lesson": "",
        "last_error": "",
        "flag_on": True,
    }
    snap.update(overrides)
    return snap


def test_expedition_thinking_scroll_survives_auto_refresh(monkeypatch) -> None:
    app = _app()
    from ui.panels.expedition import ExpeditionPanel

    long_thinking = "\n".join(f"reasoning line {i:03d}" for i in range(220))
    runner = _FakeRunner(
        _snapshot(thinking=[{"turn": "exp_000001", "thinking": long_thinking, "output": "done"}])
    )
    monkeypatch.setattr(ExpeditionPanel, "_runner", staticmethod(lambda: runner))

    panel = ExpeditionPanel()
    try:
        panel.resize(420, 320)
        panel.show()
        panel._timer.stop()
        panel.refresh()
        app.processEvents()

        scrollbar = panel._thinking.verticalScrollBar()
        assert scrollbar.maximum() > 0, "thinking pane must be scrollable for this test"
        scrollbar.setValue(scrollbar.maximum())
        app.processEvents()
        scrolled_to = scrollbar.value()
        assert scrolled_to > 0

        panel.refresh()
        app.processEvents()

        assert panel._thinking.verticalScrollBar().value() == scrolled_to
    finally:
        panel._timer.stop()
        panel.close()


def test_expedition_activity_scroll_survives_auto_refresh(monkeypatch) -> None:
    app = _app()
    from ui.panels.expedition import ExpeditionPanel

    activity = [f"tick {i:03d} - 1 tool - +1 world - 1 finding" for i in range(80)]
    runner = _FakeRunner(_snapshot(activity=activity))
    monkeypatch.setattr(ExpeditionPanel, "_runner", staticmethod(lambda: runner))

    panel = ExpeditionPanel()
    try:
        panel._tabs.setCurrentWidget(panel._activity)
        panel.resize(420, 260)
        panel.show()
        panel._timer.stop()
        panel.refresh()
        app.processEvents()

        scrollbar = panel._activity.verticalScrollBar()
        assert scrollbar.maximum() > 0, "activity pane must be scrollable for this test"
        scrollbar.setValue(scrollbar.maximum())
        app.processEvents()
        scrolled_to = scrollbar.value()
        assert scrolled_to > 0

        panel.refresh()
        app.processEvents()

        assert panel._activity.verticalScrollBar().value() == scrolled_to
    finally:
        panel._timer.stop()
        panel.close()


def test_expedition_non_thinking_tabs_render_runner_snapshot(monkeypatch) -> None:
    _app()
    from ui.panels.expedition import ExpeditionPanel

    runner = _FakeRunner(
        _snapshot(
            activity=["tick 1 - 1 tool - +1 world - 1 finding"],
            referents=[{"kind": "file", "name": "engine/llm.py", "status": "observed"}],
            last_lesson="Observed the sync bridge path.",
        )
    )
    monkeypatch.setattr(ExpeditionPanel, "_runner", staticmethod(lambda: runner))

    panel = ExpeditionPanel()
    try:
        panel._timer.stop()
        panel.refresh()

        assert panel._activity.count() == 1
        assert "tick 1" in panel._activity.item(0).text()
        assert panel._findings.count() == 1
        assert "file:engine/llm.py" in panel._findings.item(0).text()
        assert "Observed the sync bridge path." in panel._self.toPlainText()
    finally:
        panel._timer.stop()
        panel.close()


def test_expedition_pause_and_stop_buttons_drive_runner(monkeypatch) -> None:
    _app()
    from ui.panels.expedition import ExpeditionPanel

    runner = _FakeRunner(_snapshot(status="running"))
    monkeypatch.setattr(ExpeditionPanel, "_runner", staticmethod(lambda: runner))

    panel = ExpeditionPanel()
    try:
        panel._timer.stop()

        panel._on_pause()
        assert runner.pause_calls == 1
        assert "paused" in panel._status.text()

        panel._on_stop()
        assert runner.stop_calls == 1
        assert "stopped" in panel._status.text()
    finally:
        panel._timer.stop()
        panel.close()


def test_expedition_generate_clears_thinking_when_toggle_enabled(monkeypatch) -> None:
    _app()
    from ui.panels.expedition import ExpeditionPanel

    runner = _FakeRunner(
        _snapshot(thinking=[{"turn": "exp_old", "thinking": "old chain", "output": ""}])
    )
    monkeypatch.setattr(ExpeditionPanel, "_runner", staticmethod(lambda: runner))

    panel = ExpeditionPanel()
    try:
        panel._timer.stop()
        panel._goal_in.setText("map engine")
        assert panel._clear_thinking_on_run.isChecked() is True

        panel._on_generate()

        assert runner.start_calls == [("map engine", True)]
        assert "old chain" not in panel._thinking.toPlainText()
    finally:
        panel._timer.stop()
        panel.close()
