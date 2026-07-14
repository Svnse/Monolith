"""Section widgets for PageStats.

Each widget exposes:
  set_data(payload: dict) — populate from a StatsStore getter
  _apply_theme()          — re-evaluate stylesheets after theme change

PageStats wires sig_theme_changed → _apply_theme on each section and calls
set_data per section after a StatsStore.refresh().
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from PySide6.QtCore import (
    Property, QEasingCurve, QPoint, QPropertyAnimation, QRect, Qt, Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

import core.style as _s


# ── shared helpers ──────────────────────────────────────────────────


def _label(text: str, *, color: str, size_px: int, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {color}; font-size: {size_px}px; font-family: Consolas;"
        + (" font-weight: bold;" if bold else "")
        + " background: transparent;"
    )
    return lbl


def _fmt_int_compact(n: int) -> str:
    """1.2M / 47k / 342."""
    n = int(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


# ── HeadlineStrip ───────────────────────────────────────────────────


class HeadlineStrip(QWidget):
    """Four large numbers with deltas. Animates from 0 → value on first
    render (per page open)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload: dict[str, Any] = {}
        self._animated_once = False
        self._build_layout()
        self._apply_theme()

    def _build_layout(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(24)
        self._cells: list[dict[str, QLabel]] = []
        for _ in range(4):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            value = _label("0", color=_s.ACCENT_PRIMARY, size_px=22, bold=True)
            label = _label("", color=_s.FG_DIM, size_px=9, bold=True)
            delta = _label("", color=_s.FG_DIM, size_px=9)
            cell_layout.addWidget(value)
            cell_layout.addWidget(label)
            cell_layout.addWidget(delta)
            layout.addWidget(cell, 1)
            self._cells.append({"value": value, "label": label, "delta": delta})

    def set_data(self, payload: dict) -> None:
        self._payload = payload or {}
        turns = int(self._payload.get("turns") or 0)
        total_chars = int(self._payload.get("total_chars") or 0)
        streak = int(self._payload.get("streak") or 0)
        days = int(self._payload.get("day_count") or 0)
        first = str(self._payload.get("first_turn_date") or "")
        wk_turns = int(self._payload.get("delta_turns_this_week") or 0)
        wk_tokens = int(self._payload.get("delta_tokens_this_week") or 0)

        spec = [
            (str(turns), "TURNS", self._delta_str(wk_turns, "wk")),
            (_fmt_int_compact(total_chars // 4), "TOKENS", self._delta_str(wk_tokens // 4, "wk")),
            (self._streak_label(streak), "STREAK", f"longest: {self._payload.get('longest_streak', streak)}"),
            (f"Day {days}", "FIRST TURN", first or "—"),
        ]
        for cell, (value, label, delta) in zip(self._cells, spec):
            cell["value"].setText(value)
            cell["label"].setText(label)
            cell["delta"].setText(delta)
        # Color the streak cell warm when streak is long
        streak_cell_value = self._cells[2]["value"]
        if streak >= 7:
            streak_cell_value.setStyleSheet(
                f"color: {_s.ACCENT_WARM}; font-size: 22px; font-family: Consolas; font-weight: bold; background: transparent;"
            )

    def _streak_label(self, streak: int) -> str:
        """A row of unicode bars representing the streak; capped at 14 visual
        bars so wide streaks don't blow up the cell width."""
        if streak <= 0:
            return "—"
        bars = min(streak, 14)
        return "▮" * bars + f" {streak}d"

    def _delta_str(self, delta: int, suffix: str) -> str:
        if delta == 0:
            return f"±0 this {suffix}"
        sign = "+" if delta > 0 else ""
        return f"{sign}{_fmt_int_compact(delta)} this {suffix}"

    def _apply_theme(self) -> None:
        for cell in self._cells:
            cell["label"].setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; font-weight: bold; background: transparent;"
            )
            cell["delta"].setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; background: transparent;"
            )
            cell["value"].setStyleSheet(
                f"color: {_s.ACCENT_PRIMARY}; font-size: 22px; font-family: Consolas; font-weight: bold; background: transparent;"
            )
        if self._payload:
            self.set_data(self._payload)


# ── _SectionStub base + 9 section subclasses ──────────────────────


class _SectionStub(QFrame):
    """Base stub: title label + status label. Subclasses override SECTION_NAME."""

    SECTION_NAME = "SECTION"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload: Any = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(2)
        self._title = _label(self.SECTION_NAME, color=_s.ACCENT_PRIMARY, size_px=10, bold=True)
        self._status = _label("(no data)", color=_s.FG_DIM, size_px=10)
        layout.addWidget(self._title)
        layout.addWidget(self._status)
        self._apply_theme()

    def set_data(self, payload: Any) -> None:
        self._payload = payload
        if payload is None:
            self._status.setText("(no data)")
        elif isinstance(payload, (list, dict)):
            self._status.setText(f"({len(payload)} items)")
        else:
            self._status.setText(str(payload))

    def _apply_theme(self) -> None:
        self._title.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 10px; font-family: Consolas; font-weight: bold; background: transparent;"
        )
        self._status.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; background: transparent;"
        )


class ActivityCalendar(_SectionStub):
    SECTION_NAME = "ACTIVITY"


class AchievementFeed(_SectionStub):
    SECTION_NAME = "RECENT UNLOCKS"


class QualityBlock(_SectionStub):
    SECTION_NAME = "QUALITY"


class DistributionBlock(_SectionStub):
    SECTION_NAME = "DISTRIBUTIONS"


class RecordsBlock(_SectionStub):
    SECTION_NAME = "PERSONAL RECORDS"


class TimeRhythmMap(_SectionStub):
    SECTION_NAME = "RHYTHM"


class PipelineCostBlock(_SectionStub):
    SECTION_NAME = "PIPELINE COST"


class SubstrateBlock(_SectionStub):
    SECTION_NAME = "SUBSTRATE HEALTH"


class WrappedSection(_SectionStub):
    SECTION_NAME = "WRAPPED"
    sig_wrapped_requested = Signal(str)  # timeframe: "week" | "month" | "year"
