"""The agent strip widget — live chips fed by the spine, click-to-zoom."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from core.active_agents import AgentRecord


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def test_strip_empty_has_no_chips(app) -> None:
    from ui.components.agents_strip import AgentsStrip
    s = AgentsStrip()
    s.update_agents([])
    assert s._chips == []


def test_strip_builds_one_chip_per_agent(app) -> None:
    from ui.components.agents_strip import AgentsStrip
    s = AgentsStrip()
    s.update_agents([
        AgentRecord("a1", "probe", 2, "running"),
        AgentRecord("a2", "summary", 3, "done", child_turn_id="cid9"),
    ])
    assert len(s._chips) == 2
    assert "probe" in s._chips[0].text()  # running sorts first


def test_done_chip_zooms_with_its_child_turn_id(app) -> None:
    from ui.components.agents_strip import AgentsStrip
    s = AgentsStrip()
    s.update_agents([AgentRecord("a2", "summary", 3, "done", child_turn_id="cid9")])
    got: list = []
    s.sig_zoom_agent.connect(lambda cid, f: got.append((cid, f)))
    s._chips[0].click()
    assert got == [("cid9", "summary")]


def test_running_chip_is_not_zoomable_yet(app) -> None:
    from ui.components.agents_strip import AgentsStrip
    s = AgentsStrip()
    s.update_agents([AgentRecord("a1", "probe", 2, "running")])  # no child_turn_id
    assert not s._chips[0].isEnabled()


def test_update_replaces_not_appends(app) -> None:
    from ui.components.agents_strip import AgentsStrip
    s = AgentsStrip()
    s.update_agents([AgentRecord("a1", "one", 2, "running")])
    s.update_agents([AgentRecord("a2", "two", 2, "running")])
    assert len(s._chips) == 1
    assert "two" in s._chips[0].text()
