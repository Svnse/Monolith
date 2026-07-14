"""ExpeditionPanel — feature-rich companion view for MonoExplore.

Lives in CompanionState.EXPEDITION. Observes the singleton ExpeditionRunner via
get_runner().snapshot() (read-only) and drives it (Generate / Pause / Stop). Shows
what the model is THINKING, the per-tick ACTIVITY, FINDINGS/referents (grounding
color-coded), the SELF (last lesson), and dev vitals: tick/budget, tools, world
ACUs, fault streak, ~token total, and the coherence dims. Polls only while shown.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

import core.style as _s
from ui.components.atoms import MonoButton

_VERDICT_COLOR = {"GREEN": _s.FG_OK, "YELLOW": _s.FG_WARN, "RED": _s.FG_ERROR}
_STATUS_COLOR = {"running": _s.FG_OK, "idle": _s.FG_DIM, "paused": _s.FG_WARN,
                 "halted": _s.FG_ERROR, "no-plan": _s.FG_WARN, "stopped": _s.FG_DIM}
_MONO = "font-family: Consolas; font-size: 9px;"
_GROUNDED = _s.FG_OK


class ExpeditionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # ── status row: badge · coherence chip (identity lives in the pane
        # header — UI_CONTRACT §2: status is content, identity is chrome) ──
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addStretch()
        self._status = QLabel("● idle")
        head.addWidget(self._status)
        self._chip = QLabel("[—]")
        head.addWidget(self._chip)
        root.addLayout(head)

        # ── goal / next move ──
        self._goal_lbl = QLabel("goal: —")
        self._goal_lbl.setWordWrap(True)
        self._goal_lbl.setStyleSheet(f"color:{_s.FG_TEXT};font-family:Consolas;font-size:10px;")
        self._next_lbl = QLabel("next: —")
        self._next_lbl.setWordWrap(True)
        self._next_lbl.setStyleSheet(f"color:{_s.FG_DIM};{_MONO}")
        root.addWidget(self._goal_lbl)
        root.addWidget(self._next_lbl)

        # ── controls: goal input + Generate / Pause / Stop ──
        self._goal_in = QLineEdit()
        self._goal_in.setPlaceholderText("goal…  (blank = seed from curiosity / bearing)")
        self._goal_in.setProperty("panelInset", True)   # shared inset style (theme QSS)
        self._goal_in.setStyleSheet("font-family:Consolas;font-size:10px;")
        root.addWidget(self._goal_in)
        btns = QHBoxLayout()
        btns.setSpacing(4)
        self._gen = MonoButton("⟫ GENERATE")
        self._pause = MonoButton("PAUSE")
        self._stop = MonoButton("STOP")
        for b in (self._gen, self._pause, self._stop):
            btns.addWidget(b)
        root.addLayout(btns)
        self._gen.clicked.connect(self._on_generate)
        self._pause.clicked.connect(self._on_pause)
        self._stop.clicked.connect(self._on_stop)

        opts = QHBoxLayout()
        opts.setContentsMargins(0, 0, 0, 0)
        self._clear_thinking_on_run = QCheckBox("clear thinking on run")
        self._clear_thinking_on_run.setChecked(True)
        self._clear_thinking_on_run.setStyleSheet(
            f"color:{_s.FG_DIM};font-family:Consolas;font-size:9px;"
        )
        opts.addWidget(self._clear_thinking_on_run)
        opts.addStretch()
        root.addLayout(opts)

        # ── vitals (dev) ──
        self._vitals = QLabel("—")
        self._vitals.setStyleSheet(f"color:{_s.FG_SECONDARY};{_MONO}")
        self._dims = QLabel("—")
        self._dims.setStyleSheet(f"color:{_s.FG_DIM};{_MONO}")
        root.addWidget(self._vitals)
        root.addWidget(self._dims)

        # ── tabs: THINKING / ACTIVITY / FINDINGS / SELF ──
        self._tabs = QTabWidget()
        self._tabs.setProperty("panelInset", True)      # shared inset style (theme QSS)
        self._thinking = QTextEdit(); self._thinking.setReadOnly(True)
        self._activity = QListWidget()
        self._findings = QListWidget()
        self._self = QTextEdit(); self._self.setReadOnly(True)
        # Feeds sit transparently ON the inset tab pane — no box-in-box fill.
        _feed_css = (f"background:transparent;color:{_s.FG_TEXT};border:none;"
                     f"font-family:Consolas;font-size:9px;")
        for w in (self._thinking, self._activity, self._findings, self._self):
            w.setStyleSheet(_feed_css)
        self._tabs.addTab(self._thinking, "THINKING")
        self._tabs.addTab(self._activity, "ACTIVITY")
        self._tabs.addTab(self._findings, "FINDINGS")
        self._tabs.addTab(self._self, "SELF")
        root.addWidget(self._tabs, 1)

        self._last_thinking_text: str | None = None
        self._last_activity_rows: tuple[str, ...] | None = None
        self._last_finding_rows: tuple[tuple[str, str, str], ...] | None = None
        self._last_self_text: str | None = None

        # ── footer: flag + refresh ──
        self._foot = QLabel("—")
        self._foot.setStyleSheet(f"color:{_s.FG_DIM};font-family:Consolas;font-size:8px;")
        root.addWidget(self._foot)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)

    # ── companion contract ──
    def bind_controller(self, controller) -> None:
        return

    def showEvent(self, event) -> None:  # poll only while visible
        self._timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    # ── runner access ──
    @staticmethod
    def _runner():
        from engine.expedition_runner import get_runner
        return get_runner()

    def _on_generate(self) -> None:
        # Non-blocking: seeding + decompose (an LLM call) run on the runner's
        # daemon thread (see ExpeditionRunner._bootstrap_and_run), so the UI
        # never freezes here.
        try:
            ok = self._runner().start(
                self._goal_in.text().strip(),
                clear_thinking=self._clear_thinking_on_run.isChecked(),
            )
            self._foot.setText("expedition starting…" if ok else "already running")
        except Exception as exc:
            self._foot.setText(f"start error: {exc}")
        self.refresh()

    def _on_pause(self) -> None:
        try:
            self._runner().pause()
        except Exception:
            pass
        self.refresh()

    def _on_stop(self) -> None:
        try:
            self._runner().stop()
        except Exception:
            pass
        self.refresh()

    @staticmethod
    def _restore_scroll(widget, value: int, *, follow_tail: bool) -> None:
        bar = widget.verticalScrollBar()
        if follow_tail:
            bar.setValue(bar.maximum())
        else:
            bar.setValue(max(0, min(value, bar.maximum())))

    def _set_text_if_changed(self, widget: QTextEdit, text: str, cache_attr: str) -> None:
        if getattr(self, cache_attr) == text:
            return
        bar = widget.verticalScrollBar()
        prev = bar.value()
        follow_tail = prev >= max(0, bar.maximum() - 2)
        setattr(self, cache_attr, text)
        widget.setPlainText(text)
        self._restore_scroll(widget, prev, follow_tail=follow_tail)

    def _set_activity_rows(self, rows: tuple[str, ...]) -> None:
        if self._last_activity_rows == rows:
            return
        bar = self._activity.verticalScrollBar()
        prev = bar.value()
        follow_tail = prev >= max(0, bar.maximum() - 2)
        self._last_activity_rows = rows
        self._activity.clear()
        for line in rows:
            self._activity.addItem(QListWidgetItem(line))
        self._restore_scroll(self._activity, prev, follow_tail=follow_tail)

    def _set_finding_rows(self, rows: tuple[tuple[str, str, str], ...]) -> None:
        if self._last_finding_rows == rows:
            return
        bar = self._findings.verticalScrollBar()
        prev = bar.value()
        follow_tail = prev >= max(0, bar.maximum() - 2)
        self._last_finding_rows = rows
        self._findings.clear()
        for kind, name, status in rows:
            item = QListWidgetItem(f"{kind}:{name}  [{status}]")
            item.setForeground(QColor(_GROUNDED if status == "observed" else _s.FG_DIM))
            self._findings.addItem(item)
        self._restore_scroll(self._findings, prev, follow_tail=follow_tail)

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
        coh = snap.get("coherence", {}) or {}
        verdict = str(coh.get("verdict", "—"))
        self._chip.setText(f"[{verdict}]")
        self._chip.setStyleSheet(
            f"color:{_VERDICT_COLOR.get(verdict, _s.FG_DIM)};font-family:Consolas;font-size:10px;font-weight:bold;")
        self._goal_lbl.setText(f"goal: {snap.get('goal') or '—'}")
        self._next_lbl.setText(f"next: {snap.get('next_move') or '—'}")
        self._vitals.setText(
            f"tick {snap.get('tick', 0)}/{snap.get('max_ticks', 0)} · "
            f"tools {snap.get('tools_total', 0)} · world+{snap.get('world_acus', 0)} · "
            f"faults {snap.get('fault_streak', 0)} · ~{snap.get('tokens', 0)} tok")
        dims = coh.get("dims", {}) or {}
        self._dims.setText(
            f"grounding {dims.get('grounding_ratio', '—')} · "
            f"progress {dims.get('progress', '—')} · drift {dims.get('drift_overlap', '—')}")

        thinking = snap.get("thinking", []) or []
        thinking_text = (
            "\n\n".join(
                f"› {t.get('turn', '')}\n{t.get('thinking') or '(no thinking captured)'}"
                + (f"\n— output —\n{t.get('output', '')[:600]}" if t.get('output') else "")
                for t in thinking
            ) or "(nothing yet — press Generate)"
        )
        self._set_text_if_changed(self._thinking, thinking_text, "_last_thinking_text")

        activity_rows = tuple(str(line) for line in reversed(snap.get("activity", []) or []))
        self._set_activity_rows(activity_rows)

        finding_rows = tuple(
            (str(r.get("kind") or ""), str(r.get("name") or ""), str(r.get("status") or ""))
            for r in snap.get("referents", []) or []
        )
        self._set_finding_rows(finding_rows)

        self_text = f"last lesson:\n{snap.get('last_lesson') or '(none yet)'}"
        self._set_text_if_changed(self._self, self_text, "_last_self_text")
        err = snap.get("last_error") or ""
        if err:
            self._foot.setText(f"⚠ {err[:170]}")
            self._foot.setStyleSheet(
                f"color:{_VERDICT_COLOR['RED']};font-family:Consolas;font-size:8px;")
        else:
            flag = "● on" if snap.get("flag_on") else "○ off (dark)"
            self._foot.setText(f"MONOLITH_MONOEXPLORE_V1 {flag}  ·  ⟳ 1.0s")
            self._foot.setStyleSheet(f"color:{_s.FG_DIM};font-family:Consolas;font-size:8px;")
