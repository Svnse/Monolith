"""SelfMaintPanel — companion view + control for the self-maintenance daemon.

Lives in CompanionState.SELF_MAINT. Observes the singleton SelfMaintRunner via
get_runner().snapshot() and controls it (Start / Stop / interval / daily cap) — the
daemon is now runtime-controllable (the self_maint skill / this panel), not flag-only.
Shows the daemon STATUS, the two-flag MODE (observe-first vs applying), live vitals,
and the RECENT WAKES feed — each wake's items-seen + what Monolith WOULD do (the model's
raw action) + a snippet of its reasoning. Polls only while visible.

Starting from here is E-initiated and observe-first; whether actions APPLY is still
gated by MONOLITH_SELF_MAINT_V1.
"""
from __future__ import annotations

import re

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

import core.style as _s
from ui.components.atoms import MonoButton

_MONO = "font-family: Consolas; font-size: 9px;"
_STATUS_COLOR = {"running": _s.FG_OK, "idle": _s.FG_DIM, "disabled": _s.FG_DIM,
                 "starting": _s.FG_WARN, "stopped": _s.FG_DIM, "halted": _s.FG_ERROR}
_ACTION_RE = re.compile(r'"item_id"\s*:\s*"([^"]+)".*?"action"\s*:\s*"(\w+)"', re.DOTALL)


def _hhmm(ts: str) -> str:
    s = str(ts or "")
    if "T" in s and len(s) >= 16:
        return s[11:16]
    return s[:5]


def _summarize_action(raw: str) -> str:
    m = _ACTION_RE.search(raw or "")
    if m:
        return f"would {m.group(2)} {m.group(1)}"
    raw = (raw or "").strip().replace("\n", " ")
    return (raw[:80] + "…") if len(raw) > 80 else (raw or "(no output)")


def _row_line(row: dict) -> str:
    # Markers limited to glyphs Consolas actually ships (● ○ ⚠ ·).
    t = _hhmm(row.get("ts", ""))
    if "halted" in row:
        return f"⚠ {t} · HALTED — {str(row.get('halted'))[:80]}"
    if "fault" in row:
        return f"⚠ {t} · fault — {str(row.get('fault'))[:80]} (streak {row.get('fault_streak', '?')})"
    if "skipped" in row:
        return f"· {t} · skipped ({row.get('skipped')})"
    seen = row.get("items_seen", 0)
    calls = row.get("tool_calls") or []
    if calls:
        return f"● {t} · saw {seen} · {_summarize_action(row.get('raw', ''))}"
    return f"○ {t} · saw {seen} · no action"


class SelfMaintPanel(QWidget):
    def __init__(self, world_state=None, parent=None):
        super().__init__(parent)
        self._ws = world_state
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # ── status row (identity lives in the pane header — UI_CONTRACT §2) ──
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addStretch()
        self._status = QLabel("● idle")
        head.addWidget(self._status)
        root.addLayout(head)

        # ── mode (the two flags) + vitals ──
        self._mode = QLabel("—")
        self._mode.setWordWrap(True)
        self._mode.setStyleSheet(f"color:{_s.FG_TEXT};font-family:Consolas;font-size:10px;")
        self._vitals = QLabel("—")
        self._vitals.setStyleSheet(f"color:{_s.FG_SECONDARY};{_MONO}")
        root.addWidget(self._mode)
        root.addWidget(self._vitals)

        # ── controls: Start / Stop · interval · cap (E-initiated, observe-first) ──
        ctl = QHBoxLayout()
        ctl.setSpacing(4)
        self._start = MonoButton("▷ START")
        self._stop = MonoButton("□ STOP")
        ctl.addWidget(self._start)
        ctl.addWidget(self._stop)
        ctl.addStretch()
        root.addLayout(ctl)
        self._start.clicked.connect(self._on_start)
        self._stop.clicked.connect(self._on_stop)

        cad = QHBoxLayout()
        cad.setSpacing(4)
        self._interval_in = QLineEdit()
        self._interval_in.setPlaceholderText("interval s")
        self._cap_in = QLineEdit()
        self._cap_in.setPlaceholderText("cap/day")
        for w in (self._interval_in, self._cap_in):
            w.setFixedWidth(70)
            w.setProperty("panelInset", True)           # shared inset style (theme QSS)
            w.setStyleSheet("font-family:Consolas;font-size:9px;")
        self._apply = MonoButton("apply")
        cad.addWidget(self._interval_in)
        cad.addWidget(self._cap_in)
        cad.addWidget(self._apply)
        cad.addStretch()
        root.addLayout(cad)
        self._apply.clicked.connect(self._on_apply)

        # ── tabs: WAKES / ACTIVITY / THINKING ──
        self._tabs = QTabWidget()
        self._tabs.setProperty("panelInset", True)      # shared inset style (theme QSS)
        self._wakes = QTextEdit(); self._wakes.setReadOnly(True)
        self._activity = QTextEdit(); self._activity.setReadOnly(True)
        self._thinking = QTextEdit(); self._thinking.setReadOnly(True)
        # Feeds sit transparently ON the inset tab pane — no box-in-box fill.
        _feed_css = (f"background:transparent;color:{_s.FG_TEXT};border:none;"
                     f"font-family:Consolas;font-size:9px;")
        for w in (self._wakes, self._activity, self._thinking):
            w.setStyleSheet(_feed_css)
        self._tabs.addTab(self._wakes, "WAKES")
        self._tabs.addTab(self._activity, "ACTIVITY")
        self._tabs.addTab(self._thinking, "THINKING")
        root.addWidget(self._tabs, 1)

        self._last = {}

        self._foot = QLabel("—")
        self._foot.setStyleSheet(f"color:{_s.FG_DIM};font-family:Consolas;font-size:8px;")
        root.addWidget(self._foot)

        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self.refresh)

    # ── companion contract ──
    def bind_controller(self, controller) -> None:
        return  # read-only feed + direct daemon control; no conversation dependency

    def showEvent(self, event) -> None:
        self._timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    @staticmethod
    def _runner():
        from engine.self_maint_runner import get_runner
        return get_runner()

    # ── controls ──
    def _on_start(self) -> None:
        try:
            from engine.self_maint_runner import engine_is_busy
            r = self._runner()
            r.set_is_busy(lambda: engine_is_busy(self._ws))  # live-turn guard
            ok = r.start(force=True)
            self._foot.setText("started (observe-first)" if ok else "already running")
        except Exception as exc:  # noqa: BLE001
            self._foot.setText(f"start error: {exc}")
        self.refresh()

    def _on_stop(self) -> None:
        try:
            self._runner().stop(timeout=2.0)
        except Exception:
            pass
        self.refresh()

    def _on_apply(self) -> None:
        msgs = []
        iv = self._interval_in.text().strip()
        if iv:
            try:
                eff = self._runner().set_interval(int(iv))
                msgs.append(f"interval→{eff}s")
            except Exception:
                msgs.append("bad interval")
        cap = self._cap_in.text().strip()
        if cap:
            try:
                from core.self_maint_leash import set_cap_override
                set_cap_override(int(cap))
                msgs.append(f"cap→{int(cap)}/day")
            except Exception:
                msgs.append("bad cap")
        if msgs:
            self._foot.setText(" · ".join(msgs))
        self.refresh()

    def _set_text(self, widget: QTextEdit, text: str, key: str) -> None:
        if self._last.get(key) == text:
            return
        bar = widget.verticalScrollBar()
        follow = bar.value() >= max(0, bar.maximum() - 2)
        self._last[key] = text
        widget.setPlainText(text)
        bar.setValue(bar.maximum() if follow else min(bar.value(), bar.maximum()))

    # ── live render ──
    def refresh(self) -> None:
        try:
            snap = self._runner().snapshot()
        except Exception:
            return
        st = str(snap.get("status", "idle"))
        self._status.setText(f"● {st}")
        self._status.setStyleSheet(
            f"color:{_STATUS_COLOR.get(st, _s.FG_DIM)};font-family:Consolas;font-size:10px;font-weight:bold;")

        trig = "● on" if snap.get("trigger_on") else "○ off"
        if snap.get("apply_on"):
            self._mode.setText(f"TRIGGER {trig} · APPLY ● ON — actions are LIVE")
            self._mode.setStyleSheet(f"color:{_s.FG_WARN};font-family:Consolas;font-size:10px;")
        else:
            self._mode.setText(f"TRIGGER {trig} · APPLY ○ off — observe-first (logs only, applies nothing)")
            self._mode.setStyleSheet(f"color:{_s.FG_TEXT};font-family:Consolas;font-size:10px;")

        iv = int(snap.get("interval_s", 0) or 0)
        self._vitals.setText(
            f"wake {snap.get('wake', 0)} · every {iv}s (~{max(1, iv // 60)}m) · "
            f"faults {snap.get('fault_streak', 0)}")
        if not self._interval_in.text().strip():
            self._interval_in.setPlaceholderText(f"{iv}s")

        recent = snap.get("recent", []) or []
        wakes_text = "\n".join(_row_line(r) for r in reversed(recent)) or "(no wakes yet — first wake is ~one interval after start)"
        self._set_text(self._wakes, wakes_text, "wakes")

        activity = snap.get("activity", []) or []
        self._set_text(self._activity, "\n".join(reversed([str(a) for a in activity])) or "(idle)", "activity")

        last_think = ""
        for r in reversed(recent):
            if r.get("thinking"):
                last_think = f"› {r.get('turn_id', '')} ({_hhmm(r.get('ts', ''))})\n{r.get('thinking')}"
                if r.get("raw"):
                    last_think += f"\n\n— would do —\n{r.get('raw')}"
                break
        self._set_text(self._thinking, last_think or "(no reasoning captured yet)", "thinking")

        err = snap.get("last_error") or ""
        if err and "error" not in self._foot.text():
            self._foot.setText(f"⚠ {err[:170]}")
            self._foot.setStyleSheet(f"color:{_s.FG_ERROR};font-family:Consolas;font-size:8px;")
