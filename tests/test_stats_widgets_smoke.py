from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_all_widgets_importable_and_instantiable(qapp):
    from ui.components.stats_widgets import (
        HeadlineStrip, ActivityCalendar, AchievementFeed, QualityBlock,
        DistributionBlock, RecordsBlock, TimeRhythmMap, PipelineCostBlock,
        SubstrateBlock, WrappedSection,
    )
    widgets = [
        HeadlineStrip(), ActivityCalendar(), AchievementFeed(), QualityBlock(),
        DistributionBlock(), RecordsBlock(), TimeRhythmMap(), PipelineCostBlock(),
        SubstrateBlock(), WrappedSection(),
    ]
    for w in widgets:
        assert hasattr(w, "set_data")
        assert hasattr(w, "_apply_theme")
        w.set_data({})  # exercise the empty path
        w._apply_theme()
