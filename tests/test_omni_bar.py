from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
from core.state import AppState
from ui.bridge import UIBridge
from ui.omni_bar import OmniBar


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_omni_bar_returns_monitor_as_search_result():
    _app()
    bar = OmniBar(AppState(), UIBridge())

    results = bar._query("monitor")

    assert any(result.action == "cmd:monitor" for result in results)


def test_omni_bar_returns_generating_panel_as_search_result():
    _app()
    bar = OmniBar(AppState(), UIBridge())

    results = bar._query("generat")

    assert any(result.action == "panel:GENERATING" for result in results)


def test_omni_bar_returns_lineage_results():
    _app()
    bar = OmniBar(AppState(), UIBridge())
    bar._operator_manager.list_operators = lambda: [{"name": "Atlas"}]

    results = bar._query(">lineage at")

    assert any(result.action == "lineage:Atlas" for result in results)


def test_omni_bar_returns_monitor_for_logs_alias():
    _app()
    bar = OmniBar(AppState(), UIBridge())

    results = bar._query("logs")

    assert any(result.action == "cmd:monitor" for result in results)


def test_omni_bar_overlay_results_do_not_resize_bar_frame():
    app = _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    bar = OmniBar(AppState(), UIBridge(), host)
    layout.addWidget(bar)
    host.resize(900, 240)

    before = bar._frame.height()
    bar._input.setText("monitor")
    app.processEvents()

    assert bar._results_popup is not None
    assert bar._results_popup.parentWidget() is host
    assert bar._frame.height() == before

    bar.clear()
    host.deleteLater()
    app.processEvents()


def test_omni_bar_default_results_require_explicit_open():
    app = _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    bar = OmniBar(AppState(), UIBridge(), host)
    layout.addWidget(bar)
    bar._input_has_focus = lambda: True

    assert bar._results_popup is None or bar._results_popup.isVisible() is False

    bar._rebuild_results()
    app.processEvents()

    assert bar._results_popup is None or bar._results_popup.isVisible() is False

    bar.focus_input()
    app.processEvents()

    assert bar._results_popup is not None
    assert len(bar._result_widgets) > 0

    bar.clear()
    host.deleteLater()
    app.processEvents()


def test_omni_bar_click_shows_default_results_immediately():
    app = _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    bar = OmniBar(AppState(), UIBridge(), host)
    layout.addWidget(bar)
    bar._input_has_focus = lambda: True
    focus_events: list[bool] = []
    bar.sig_focus_glow_changed.connect(focus_events.append)

    event = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(4, 4),
        QPointF(4, 4),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    bar.eventFilter(bar._input, event)
    app.processEvents()

    assert bar._default_results_armed is True
    assert bar._frame.property("activeGlow") is True
    assert focus_events[-1] is True
    assert bar._results_popup is not None
    assert len(bar._result_widgets) > 0

    bar.clear()
    assert bar._frame.property("activeGlow") is False
    assert focus_events[-1] is False
    host.deleteLater()
    app.processEvents()


def test_omni_bar_hover_updates_selected_result():
    app = _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    bar = OmniBar(AppState(), UIBridge(), host)
    layout.addWidget(bar)

    bar._default_results_armed = True
    bar._input_has_focus = lambda: True
    bar._rebuild_results()
    app.processEvents()

    assert len(bar._result_widgets) >= 2
    assert bar._selected_index == 0

    second = bar._result_widgets[1]
    second.hovered.emit(second._action)

    assert bar._selected_index == 1

    bar.clear()
    host.deleteLater()
    app.processEvents()
