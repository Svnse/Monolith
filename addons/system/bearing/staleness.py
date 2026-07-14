"""Bearing staleness — the detector-agnostic closure spine.

The shared actuator that handles a staleness loop the way the rejection loop
does (see updater.py): persist a streak → escalate the model-facing nudge each
turn it persists → (optionally) emit a fault once at threshold → clear on
resolution.

It knows NOTHING about channels. A detector produces a `StalenessSignal`; the
compiler runs it through `evaluate()` and applies the returned `Decision`
(inject text / persist state / clear / maybe emit fault). A second detector
(turn-count age, other instance's workstream) plugs in by emitting its own
`StalenessSignal` — no change here.

ESCALATION IS PER-SIGNAL. `escalates`/`threshold` live on the signal because the
escalation policy belongs to the detector, not the actuator. The rejection loop
can escalate to a fault safely because a failed structural verify is
*unambiguous*. Channel-mismatch is a *precision heuristic with false positives*
(compiler.py:146-151) — a legitimate USER frame that mentions a peer trips it and
can only "clear" by self-degrading the frame. So V1 ships the channel signal
`escalates=False` (nudge-only, never a fault). Escalation is reserved for a
signal that opts in behind an acknowledgment-clear escape (V2).

Pure: no I/O, no env reads. The caller owns persistence (store), audit, any fault
emission, and the flag. See
docs/superpowers/specs/2026-06-06-bearing-staleness-loop-design.md.

Liveness signal: we never inspect whether the model emitted a <bearing_update>.
The detector re-firing next turn IS the proof the model did not resolve it; the
streak keys on `signal_id` (not frame text) so a cosmetic edit can't reset it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StalenessSignal:
    """One active staleness condition, produced by a detector.

    `signal_id` MUST be stable across turns for the same underlying staleness, or
    the streak resets every turn and never escalates (channel → the direction,
    e.g. "channel:user"; turn-count → a stable per-element id).

    `escalates`/`threshold` are the detector's escalation policy. `escalates=False`
    → nudge-only, `should_escalate` is never set (the safe default for
    false-positive-prone heuristics).
    """
    signal_id: str
    kind: str
    text: str
    detail: str = ""
    escalates: bool = True
    threshold: int = 3


@dataclass(frozen=True)
class Decision:
    text: str | None          # nudge to inject (None → inject nothing)
    next_state: dict | None    # {signal_id, streak, turn_id} to persist (None → nothing)
    cleared: bool              # prior staleness resolved this turn
    should_escalate: bool      # emit a fault this turn (fires once, on the advancing turn)
    advanced: bool = False     # a genuine new turn-step (vs a same-turn re-render)
    escalate_detail: str = ""


def _prior_streak(prior_state: dict | None, signal_id: str) -> int:
    if not prior_state or prior_state.get("signal_id") != signal_id:
        return 0
    try:
        return int(prior_state.get("streak", 0) or 0)
    except (TypeError, ValueError):
        return 0


def evaluate(
    active: StalenessSignal | None,
    prior_state: dict | None,
    turn_id: str = "",
) -> Decision:
    """Advance the staleness state machine. Pure.

    `turn_id` is the OUTER turn id (stable across a turn's tool-loop followup
    generations; the caller derives it as `_parent_turn_id or _turn_id`). The
    streak advances ONLY when it changes — a same-turn re-fire re-renders at the
    same streak and never escalates, so "streak == turns persisted" stays true
    and a mid-turn re-fire can't double-emit the fault.
    """
    if active is None:
        # No mismatch this turn. If we were tracking one, it resolved.
        if prior_state:
            return Decision(text=None, next_state=None, cleared=True,
                            should_escalate=False, advanced=False)
        return Decision(text=None, next_state=None, cleared=False,
                        should_escalate=False, advanced=False)

    same_signal = bool(prior_state) and prior_state.get("signal_id") == active.signal_id
    prior_turn = prior_state.get("turn_id") if same_signal else None
    # Same outer turn (a within-turn re-fire) → re-render, do NOT advance.
    same_turn = bool(turn_id) and prior_turn is not None and str(prior_turn) == str(turn_id)

    if same_turn:
        streak = _prior_streak(prior_state, active.signal_id)
        advanced = False
    else:
        streak = _prior_streak(prior_state, active.signal_id) + 1
        advanced = True

    next_state = {"signal_id": active.signal_id, "streak": streak, "turn_id": str(turn_id)}
    text = render_nudge(active, streak)
    # Fire ONCE, on the advancing turn that hits threshold, and only for opt-in signals.
    should_escalate = advanced and bool(active.escalates) and streak == max(1, active.threshold)
    detail = f"{active.detail} (streak={streak})" if active.detail else f"streak={streak}"
    return Decision(
        text=text,
        next_state=next_state,
        cleared=False,
        should_escalate=should_escalate,
        advanced=advanced,
        escalate_detail=detail,
    )


def render_nudge(active: StalenessSignal, streak: int) -> str:
    """Model-facing nudge text, escalating with the streak.

    streak 1                       → the detector's own text, verbatim (== flag-off behavior).
    non-escalating signal (≥2)     → an advisory line that ALWAYS names the false-positive
                                      escape ("no update is needed") + the original detail.
                                      Stays ignorable; never coercive.
    escalating signal, 1<streak<N  → a "still stale" line + the original detail.
    escalating signal, streak ≥ N  → a "persistent / now flagged" line + the original detail.

    Wording lives here as tunable strings — editable without touching logic.
    """
    if streak <= 1:
        return active.text

    if active.kind == "empty_bearing":
        return (
            f"[BEARING — still empty, turn {streak}] Your bearing has had no posture "
            f"for {streak} turns. Emit a one-line <frame>...</frame> at the end of "
            f"your reply to establish it.\n{active.text}"
        )

    if not active.escalates:
        return (
            f"[BEARING — staleness still noted, turn {streak}] This bearing flag has "
            f"stood for {streak} turns. If the frame still holds — e.g. you are "
            f"legitimately referencing a peer while talking to E — no update is needed; "
            f"otherwise emit a <bearing_update> to refresh current_frame.\n{active.text}"
        )

    if streak >= max(1, active.threshold):
        return (
            f"[BEARING — persistent staleness, turn {streak}, now flagged] This has held "
            f"for {streak} turns and is now recorded as a fault. Emit a <bearing_update> "
            f"to refresh current_frame, or explicitly confirm the frame still holds.\n"
            f"{active.text}"
        )
    return (
        f"[BEARING — still stale, turn {streak}] current_frame has read the same way for "
        f"{streak} turns running without a refresh. Emit a <bearing_update> or explicitly "
        f"confirm the frame still holds.\n{active.text}"
    )
