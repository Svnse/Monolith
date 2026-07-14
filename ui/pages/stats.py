"""PageStats — top-level page for the stats addon.

Owns the layout, the range selector, theme refresh, and the Wrapped dispatch.
Section widgets live in ui/components/stats_widgets and are composed here.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

import core.style as _s
from core.paths import NOTES_DIR
from core.stats_store import StatsStore
from ui.components.atoms import MonoButton
from ui.components.complex import GradientLine
from ui.components.stats_widgets import (
    AchievementFeed, ActivityCalendar, DistributionBlock, HeadlineStrip,
    PipelineCostBlock, QualityBlock, RecordsBlock, SubstrateBlock,
    TimeRhythmMap, WrappedSection,
)


class PageStats(QWidget):
    """Top-level stats page. Mounted in IconRail, registered as 'stats' addon."""

    sig_drill_chat = Signal(str)  # archive_path (when records / calendar drill in)

    def __init__(self, state, ui_bridge, bridge=None, guard=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.ui_bridge = ui_bridge
        self._bridge = bridge
        self._guard = guard
        self._store = StatsStore()
        self._loaded = False
        self._build_layout()
        self._apply_theme()
        if self.ui_bridge is not None and hasattr(self.ui_bridge, "sig_theme_changed"):
            self.ui_bridge.sig_theme_changed.connect(self._on_theme_changed)

    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        self._range_key = "month"
        self._range_labels = {"all": "ALL", "year": "YEAR", "month": "MONTH", "week": "WEEK"}

        header = QHBoxLayout()
        header.setSpacing(8)
        self._title = QLabel("[STATS]")
        header.addWidget(self._title)
        self._range_label = QLabel(f"[{self._range_labels[self._range_key]}]")
        header.addWidget(self._range_label)
        header.addStretch()
        self._btn_menu = MonoButton("≡")
        self._btn_menu.setToolTip("Range, refresh, export")
        self._btn_menu.clicked.connect(self._open_menu)
        header.addWidget(self._btn_menu)
        outer.addLayout(header)

        self._menu = QMenu(self)
        range_group = QActionGroup(self._menu)
        range_group.setExclusive(True)
        self._range_actions: dict[str, QAction] = {}
        for key in ("all", "year", "month", "week"):
            act = QAction(self._range_labels[key], self._menu)
            act.setCheckable(True)
            act.setChecked(key == self._range_key)
            act.triggered.connect(lambda _checked=False, k=key: self._set_range(k))
            range_group.addAction(act)
            self._menu.addAction(act)
            self._range_actions[key] = act
        self._menu.addSeparator()
        act_refresh = QAction("Refresh", self._menu)
        act_refresh.triggered.connect(self._on_refresh_clicked)
        self._menu.addAction(act_refresh)
        act_export = QAction("Export JSON", self._menu)
        act_export.triggered.connect(self._on_export_clicked)
        self._menu.addAction(act_export)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        self._headline = HeadlineStrip()
        self._calendar = ActivityCalendar()
        self._achievements = AchievementFeed()
        self._quality = QualityBlock()
        self._distribution = DistributionBlock()
        self._records = RecordsBlock()
        self._rhythm = TimeRhythmMap()
        self._pipeline = PipelineCostBlock()
        self._substrate = SubstrateBlock()
        self._wrapped = WrappedSection()
        self._wrapped.sig_wrapped_requested.connect(self._on_wrapped_requested)

        self._sections = [
            self._headline, GradientLine(),
            self._calendar, self._achievements, GradientLine(),
            self._quality, self._distribution, GradientLine(),
            self._records,
            self._rhythm,
            self._pipeline,
            self._substrate, GradientLine(),
            self._wrapped,
        ]
        for s in self._sections:
            v.addWidget(s)
        scroll.setWidget(host)
        outer.addWidget(scroll, 1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._loaded:
            self._ensure_loaded()
            self._loaded = True

    def _ensure_loaded(self) -> None:
        self._store.refresh()
        self._render_all_sections()

    def _open_menu(self) -> None:
        self._menu.exec(self._btn_menu.mapToGlobal(self._btn_menu.rect().bottomLeft()))

    def _set_range(self, key: str) -> None:
        if key not in self._range_labels:
            return
        self._range_key = key
        self._range_label.setText(f"[{self._range_labels[key]}]")
        act = self._range_actions.get(key)
        if act is not None:
            act.setChecked(True)
        self._render_all_sections()

    def _on_refresh_clicked(self) -> None:
        self._store.refresh()
        self._render_all_sections()

    def _on_export_clicked(self) -> None:
        payload = self._collect_export_payload()
        path = NOTES_DIR / f"stats-export-{int(time.time())}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def _on_wrapped_requested(self, timeframe: str) -> None:
        """Dispatch a Wrapped generation through the existing LLM bridge."""
        if self._bridge is None:
            return
        prompt_template = (Path(__file__).resolve().parent.parent.parent / "skills" / "stats" / "wrapped_prompt.md").read_text(encoding="utf-8")
        prompt = prompt_template.replace("{timeframe}", timeframe)
        task = self._bridge.wrap("llm", "generate", "llm", payload={"prompt": prompt})
        self._bridge.submit(task)

    def _current_range_key(self) -> str:
        return self._range_key

    def _render_all_sections(self) -> None:
        rk = self._current_range_key()
        lifetime = self._store.get_lifetime_summary()
        lifetime["streak"] = self._store.get_streak()
        self._headline.set_data(lifetime)
        self._calendar.set_data({"range": rk})
        self._achievements.set_data(self._store.get_achievements(limit=8))
        self._quality.set_data({
            "histogram": self._store.get_rating_histogram(rk),
            "trend": self._store.get_rating_trend(days=30),
            "fault_summary": self._store.get_fault_summary(rk),
        })
        self._distribution.set_data({
            "effort": self._store.get_effort_distribution(rk),
            "reasoning": self._store.get_mode_distribution("reasoning", rk),
            "tools": self._store.get_tool_usage(rk, top_n=5),
        })
        self._records.set_data(self._store.get_personal_records())
        self._rhythm.set_data(self._store.get_time_rhythm(rk))
        self._pipeline.set_data(self._store.get_pipeline_cost_breakdown(rk))
        self._substrate.set_data(self._store.get_substrate_summary())
        self._wrapped.set_data({"notes_dir": str(NOTES_DIR)})

    def _collect_export_payload(self) -> dict[str, Any]:
        rk = self._current_range_key()
        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "range": rk,
            "lifetime": self._store.get_lifetime_summary(),
            "streak": self._store.get_streak(),
            "rating_histogram": self._store.get_rating_histogram(rk),
            "rating_trend": self._store.get_rating_trend(days=30),
            "effort_distribution": self._store.get_effort_distribution(rk),
            "reasoning_distribution": self._store.get_mode_distribution("reasoning", rk),
            "fault_summary": self._store.get_fault_summary(rk),
            "tool_usage": self._store.get_tool_usage(rk, top_n=10),
            "personal_records": self._store.get_personal_records(),
            "achievements": self._store.get_achievements(limit=50),
            "time_rhythm": {f"{wk}:{hr:02d}": c for (wk, hr), c in self._store.get_time_rhythm(rk).items()},
            "pipeline_cost": self._store.get_pipeline_cost_breakdown(rk),
            "substrate": self._store.get_substrate_summary(),
        }

    def _on_theme_changed(self, *_args) -> None:
        self._apply_theme()
        for s in self._sections:
            if hasattr(s, "_apply_theme"):
                try:
                    s._apply_theme()
                except Exception:
                    pass

    def _apply_theme(self) -> None:
        self._title.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 10px; font-family: Consolas; font-weight: bold; background: transparent;"
        )
        self._range_label.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; background: transparent;"
        )
        self._menu.setStyleSheet(
            f"QMenu {{ background: {_s.BG_PANEL}; color: {_s.FG_TEXT}; "
            f"border: 1px solid {_s.BORDER_LIGHT}; font-family: Consolas; font-size: 11px; padding: 4px 0; }}"
            f"QMenu::item {{ padding: 6px 18px; }}"
            f"QMenu::item:selected {{ background: {_s.ACCENT_PRIMARY}; color: {_s.BG_MAIN}; }}"
            f"QMenu::item:checked {{ color: {_s.ACCENT_PRIMARY}; }}"
            f"QMenu::separator {{ height: 1px; background: {_s.BORDER_SUBTLE}; margin: 4px 8px; }}"
        )
