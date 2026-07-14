"""Bearing compiler — renders the [BEARING] block and injects it per turn.

Two surfaces:
  * `format_bearing_block(bearing, pending_rejection)` — pure formatter.
    Deterministic key order so the KV cache stays warm turn-to-turn.
  * `bearing_interceptor(messages, config)` — register_interceptor entrypoint.
    Direct-inject (NOT through ephemeral_coalescer) so BEARING lands BEFORE
    the coalesced RUNTIME STATE block in injection order.

Unlike continuity_interceptor (first-turn only), bearing_interceptor fires
every turn — Bearing is active state, not ambient. The model needs current
position, not just a single reminder at session start.

Empty Bearing renders a single-line marker so the model knows the substrate
exists and can populate it. The marker is the entry point — silently
omitting BEARING when empty would hide the substrate from the model.

Header explicitly tags the block as ACTIVE STATE, distinct from
CONTINUITY's "ambient state from prior sessions" framing. See system.md
ATTRIBUTION block.
"""
from __future__ import annotations

from typing import Any

from .schema import Bearing
from . import store
from . import kill_switch
from . import audit
from . import staleness

_TAG = "[BEARING]"
_CLOSE_TAG = "[/BEARING]"
_REJECTED_TAG = "[BEARING_UPDATE_REJECTED]"
_REJECTED_CLOSE_TAG = "[/BEARING_UPDATE_REJECTED]"
_SOURCE = "bearing"
_PLAN_VIEW_CACHE: dict = {}  # {"turn_id": str, "view": dict|None} — single-flight freeze


# ── formatting ──────────────────────────────────────────────────────


def _format_rejection_block(pending_rejection: dict[str, Any]) -> str:
    """Render the [BEARING_UPDATE_REJECTED] block from a pending_rejection dict.

    Tells the model: last turn's <bearing_update> envelope failed structural
    verification. Surfaces failed_rules, the verifier's `detail` string, and
    on parse_error the truncated offending body so the model can repair
    without needing the peer to mediate the diagnostic loop.
    """
    failed_rules = pending_rejection.get("failed_rules") or []
    turn_id = str(pending_rejection.get("turn_id", ""))
    detail = str(pending_rejection.get("detail", "") or "")
    evidence = str(pending_rejection.get("evidence", "") or "")
    lines = [
        f"{_REJECTED_TAG} — your prior <bearing_update> envelope was rejected. "
        "One repair attempt allowed this turn.",
    ]
    if turn_id:
        lines.append(f"prior_turn_id: {turn_id}")
    if failed_rules:
        lines.append("failed_rules: " + ", ".join(str(r) for r in failed_rules))
    if detail:
        lines.append(f"detail: {detail}")
    if evidence:
        # Evidence is the offending body (parse_error path); preserved
        # verbatim so the model can see exactly what it emitted.
        lines.append("offending_body:")
        lines.append(evidence)
    lines.append(_REJECTED_CLOSE_TAG)
    return "\n".join(lines)


def _render_updated_at(bearing: Bearing, current_turn_n: int | None) -> str | None:
    """Render the updated_at line. With a known current turn-count and a stamped
    count, show a readable age ("42 turns ago") so the model can self-judge frame
    staleness. Otherwise fall back to the UUID — byte-identical to pre-feature."""
    n = bearing.updated_at_turn_n
    if current_turn_n is not None and n > 0 and current_turn_n >= n:
        age = current_turn_n - n
        if age == 0:
            age_str = "this turn"
        elif age == 1:
            age_str = "1 turn ago"
        else:
            age_str = f"{age} turns ago"
        return f"updated_at_turn: {n} ({age_str})"
    if bearing.updated_at_turn:
        return f"updated_at_turn: {bearing.updated_at_turn}"
    return None


def format_bearing_block(
    bearing: Bearing,
    pending_rejection: dict[str, Any] | None = None,
    current_turn_n: int | None = None,
    plan_view: dict | None = None,
) -> str:
    """Build the [BEARING] block (deterministic key order). Always non-empty.

    Empty Bearing → single-line marker block.
    Full Bearing → multi-section block.
    pending_rejection (if any) → appended as separate [BEARING_UPDATE_REJECTED] block.
    """
    if bearing.is_empty():
        body = f"{_TAG} (no situational state established) — active state, NOT ambient context {_CLOSE_TAG}"
    else:
        lines = [
            f"{_TAG} — active state across turns; NOT ambient context, NOT this turn's request"
        ]
        if bearing.current_frame:
            lines.append(f"current_frame: {bearing.current_frame}")
        if bearing.active_goal:
            lines.append(f"active_goal: {bearing.active_goal}")
        traj = plan_view["trajectory"] if plan_view is not None else bearing.trajectory
        if traj:
            lines.append(f"trajectory: {traj}")
        if bearing.open_tensions:
            lines.append("open_tensions:")
            for t in bearing.open_tensions:
                lines.append(f"  - ({t.opened_at_turn}) {t.text}")
        if bearing.modal_branches:
            lines.append("modal_branches:")
            for b in bearing.modal_branches:
                lines.append(
                    f"  - [{b.status}] {b.text} — {b.reason} @{b.last_touched_turn}"
                )
        if bearing.referents:
            lines.append("referents:")
            for r in bearing.referents:
                lines.append(f"  - {r.kind}:{r.name} [{r.status}] @{r.grounded_at_turn}")
        if bearing.user_model is not None:
            um = bearing.user_model
            lines.append(
                f"user_model: register={um.register} confidence={um.confidence:.2f} — {um.intent_read}"
            )
        if bearing.stakes is not None:
            sk = bearing.stakes
            lines.append(
                f"stakes: reversibility={sk.reversibility} urgency={sk.urgency} — {sk.cost_if_wrong}"
            )
        nxt = plan_view["next_move"] if plan_view is not None else bearing.next_move
        if nxt:
            lines.append(f"next_move: {nxt}")
        updated_at_line = _render_updated_at(bearing, current_turn_n)
        if updated_at_line:
            lines.append(updated_at_line)
        lines.append(_CLOSE_TAG)
        body = "\n".join(lines)
    if pending_rejection:
        return body + "\n\n" + _format_rejection_block(pending_rejection)
    return body


# ── interceptor ─────────────────────────────────────────────────────


def _find_last_non_ephemeral_user_idx(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            return i
    return -1


def _already_injected(messages: list[dict]) -> bool:
    for msg in messages:
        if msg.get("source") == _SOURCE:
            return True
    return False


# High-precision only: proper names + multi-word phrases. Bare common words
# ("connect", "peer", "user", "direct") were removed — they false-fire on
# ordinary technical frames ("the user login flow", "connecting the DB"), which
# trains the model to ignore the nudge or spam updates to silence it.
_PEER_MARKERS = ("claude", "codex", "exchange with", "peer channel", "connect channel")
_USER_MARKERS = ("with e", " e ", "e's ", "user channel", "direct channel")


def _extract_channel_role(messages: list[dict]) -> str | None:
    """Extract the channel role from the latest user message's CHANNEL tag."""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        if msg.get("ephemeral"):
            continue
        content = str(msg.get("content", ""))
        if content.startswith("[CHANNEL:"):
            bracket_end = content.find("]")
            if bracket_end > 0:
                tag_body = content[9:bracket_end].strip()
                return tag_body.split(",")[0].strip().upper()
    return None


def _mismatch_nudge(frame: str, channel_role: str, prior_context: str) -> str:
    """Advisory, self-author nudge: names the action (<bearing_update>) and
    leaves an out, since the frame may still hold and bearing is model-authored."""
    return (
        f"[BEARING — possible staleness] current_frame reads \"{frame[:80]}\" "
        f"(looks set in a {prior_context} context) but this turn is on channel "
        f"{channel_role}. If your position has moved, emit a <bearing_update> to "
        f"refresh current_frame; if the frame still holds, no update is needed."
    )


def _detect_bearing_mismatch(bearing, messages: list[dict]) -> str | None:
    """Nudge if current_frame looks set in a different channel context than this
    turn. HIGH-PRECISION markers only (see _PEER_MARKERS/_USER_MARKERS): proper
    names and multi-word phrases, so common words in ordinary technical frames do
    not false-fire. Advisory only — the model self-authors any refresh via
    <bearing_update>; the runtime never writes posture itself."""
    frame = bearing.current_frame
    if not frame:
        return None
    channel_role = _extract_channel_role(messages)
    if not channel_role:
        return None
    frame_lower = frame.lower()
    if channel_role == "USER":
        if any(marker in frame_lower for marker in _PEER_MARKERS):
            return _mismatch_nudge(frame, channel_role, "peer/connect")
    elif channel_role.startswith("CONNECT"):
        if any(marker in frame_lower for marker in _USER_MARKERS):
            return _mismatch_nudge(frame, channel_role, "user/direct")
    return None


# ── staleness closure actuator (flag-gated, MONOLITH_BEARING_STALENESS_V2) ──
#
# Closes the channel-staleness loop the rejection loop already closes: persist a
# streak → vary the nudge so it stops being identical habituation-bait → clear on
# resolution. The channel signal is escalates=False (NEVER a fault): channel-
# mismatch is a precision heuristic with false positives (see _PEER_MARKERS note
# above and staleness.py), and escalating a false positive would force the model
# to self-degrade its frame to silence the runtime — the very pathology the
# high-precision markers protect against. Escalation waits for V2's
# acknowledgment-clear escape.


def _channel_signal_id(channel_role: str) -> str:
    if channel_role == "USER":
        return "channel:user"
    if channel_role.startswith("CONNECT"):
        return "channel:connect"
    return "channel:other"


def _channel_staleness_signal(bearing, messages: list[dict]):
    """Build the channel-mismatch StalenessSignal (or None) from the existing
    detector. escalates=False → nudge-only in V1."""
    mismatch = _detect_bearing_mismatch(bearing, messages)
    if not mismatch:
        return None
    role = _extract_channel_role(messages) or ""
    return staleness.StalenessSignal(
        signal_id=_channel_signal_id(role),
        kind="channel_mismatch",
        text=mismatch,
        detail=f"current_frame reads cross-channel on the {role or 'unknown'} channel",
        escalates=False,
        threshold=3,
    )


_FRAME_NUDGE_TEXT = (
    "[BEARING — empty] No situational posture is established this session. "
    "At the END of your reply, emit a single line: "
    "<frame>one sentence naming what you're working on right now</frame> "
    "— a lightweight heartbeat (no full <bearing_update> needed). "
    "It persists your frame across turns."
)


def _empty_bearing_signal() -> staleness.StalenessSignal:
    """Build the empty-bearing StalenessSignal.

    escalates=False → nudge-only (never a fault). The signal self-clears:
    once bearing is non-empty, this signal isn't selected and evaluate(None,...)
    clears the streak automatically."""
    return staleness.StalenessSignal(
        signal_id="bearing:empty",
        kind="empty_bearing",
        text=_FRAME_NUDGE_TEXT,
        detail="bearing is empty — no situational posture established",
        escalates=False,
        threshold=3,
    )


def _apply_staleness(block: str, bearing, messages: list[dict], config: dict) -> str:
    """Append the staleness nudge to the [BEARING] block.

    Both flags OFF (default): legacy single advisory — append the detector's
    nudge if any. Byte-identical to the pre-loop behavior.

    Either flag ON: run the closure actuator (staleness.evaluate) with ONE
    active signal selected by precedence:
      1. empty_bearing (frame_nudge ON + bearing.is_empty())
      2. channel_mismatch (staleness_v2 ON + non-empty bearing with mismatch)
    They are mutually exclusive by construction (empty_bearing requires
    is_empty(); channel_mismatch requires a non-empty current_frame).

    Defensive: any actuator/store error falls back to the un-augmented block,
    so a disk hiccup never drops the whole [BEARING] block."""
    frame_nudge = kill_switch.frame_nudge_is_enabled()
    staleness_v2 = kill_switch.staleness_is_enabled()

    if not staleness_v2 and not frame_nudge:
        # Legacy path — byte-identical to pre-loop behavior.
        mismatch = _detect_bearing_mismatch(bearing, messages)
        return block + "\n\n" + mismatch if mismatch else block

    try:
        # Select ONE active signal by precedence.
        active = _empty_bearing_signal() if (frame_nudge and bearing.is_empty()) else None
        if active is None and staleness_v2:
            active = _channel_staleness_signal(bearing, messages)
        prior = store.get_pending_staleness()
        # OUTER turn id: stable across a turn's tool-loop followup generations
        # (the interceptor re-fires per generation — engine/llm.py:1400/1429), so
        # the streak counts turns, not generations.
        outer_turn_id = ""
        if isinstance(config, dict):
            outer_turn_id = str(config.get("_parent_turn_id") or config.get("_turn_id") or "")
        decision = staleness.evaluate(active, prior, outer_turn_id)
        if decision.cleared:
            store.clear_pending_staleness()
            audit.append(
                "staleness_cleared",
                turn_id=outer_turn_id,
                signal_id=str((prior or {}).get("signal_id", "")),
            )
        elif decision.advanced and decision.next_state:
            # Persist + audit only on a genuine new turn-step; a same-turn re-fire
            # just re-renders the (unchanged) nudge into the rebuilt prompt.
            store.set_pending_staleness(decision.next_state)
            audit.append(
                "staleness_nudged",
                turn_id=outer_turn_id,
                signal_id=str(decision.next_state.get("signal_id", "")),
                streak=decision.next_state.get("streak"),
            )
        return block + "\n\n" + decision.text if decision.text else block
    except Exception:
        return block


def _resolve_current_turn_n(config: dict) -> int | None:
    """Current monotonic turn-count for age rendering — only when the feature
    flag is on AND the engine stamped config['_turn_n']. None → UUID fallback,
    keeping the flag-off render byte-identical to pre-feature."""
    try:
        from core import turn_counter
        if not turn_counter.enabled():
            return None
        raw = config.get("_turn_n")
        if raw is None:
            return None
        return int(raw)
    except Exception:
        return None


def _capture_plan_view() -> dict | None:
    """Snapshot the active plan into render strings, or None if no active plan."""
    try:
        from core import plans
        p = plans.get_active_plan()
        if not p:
            return None
        seg = " · ".join(
            f"[{s['status']}] {s['verb']} {s['target']}".strip() for s in (p.get("steps") or [])
        )
        trajectory = f"{p['goal']} → {seg}" if seg else p["goal"]
        ready = plans.next_ready_steps(p["plan_uid"])
        next_move = (
            f"step {ready[0]['seq']}: {ready[0]['verb']} {ready[0]['target']}".strip()
            if ready else ""
        )
        return {"trajectory": trajectory, "next_move": next_move}
    except Exception:
        return None


def _resolve_plan_view(config: dict) -> dict | None:
    """Active-plan render view, frozen per OUTER turn via single-flight module
    cache (mirrors core.recall_handles). None when the done-gate is off → keeps
    the flag-off [BEARING] render byte-identical. When on but no active plan →
    empty strings (so the model-authored trajectory/next_move are NOT shown — the
    dedup), while active_goal stays the model-authored seed."""
    try:
        from core import plans
        if not plans.done_gate_enabled():
            return None
        outer = ""
        if isinstance(config, dict):
            outer = str(config.get("_parent_turn_id") or config.get("_turn_id") or "")
        if outer and _PLAN_VIEW_CACHE.get("turn_id") == outer:
            return _PLAN_VIEW_CACHE.get("view")
        view = _capture_plan_view() or {"trajectory": "", "next_move": ""}
        _PLAN_VIEW_CACHE.clear()
        _PLAN_VIEW_CACHE.update({"turn_id": outer, "view": view})
        return view
    except Exception:
        return None


def _observe_drift(bearing, messages: list[dict], config: dict) -> None:
    """Flag-gated, log-only frame-drift observation. Never mutates, never raises.

    Writes the cheap overlap signal to the drift ledger so it can be calibrated
    on clean live data before any nudge is wired (see drift_observe). Observe
    only: it does not touch the [BEARING] block or the staleness path.
    """
    try:
        from . import drift_observe
        if not drift_observe.enabled():
            return
        outer = ""
        if isinstance(config, dict):
            outer = str(config.get("_parent_turn_id") or config.get("_turn_id") or "")
        drift_observe.record(outer, getattr(bearing, "current_frame", "") or "", messages)
    except Exception:
        pass


def _apply_correction_example(block: str, messages: list[dict]) -> str:
    """Append the nearest TRAINABLE human CorrectionCard as ONE worked example
    (MonoFrame v2, flag MONOLITH_MONOFRAME_V1). No-op when off or when no human
    card matches — so the chat path is byte-identical by default. Never raises:
    a store hiccup leaves the [BEARING] block untouched."""
    try:
        from . import correction_store
        if not correction_store.enabled():
            return block
        from .drift import recent_asks as _recent_asks
        asks = _recent_asks(messages)
        query = asks[-1] if asks else ""
        card = correction_store.nearest_human_card(query)
        example = correction_store.render_card_for_scaffold(card)
        return block + "\n\n" + example if example else block
    except Exception:
        return block


def _apply_frame_selection_contract(block: str) -> str:
    """Append the <frame_selection> commitment contract when the standing recorder
    is on (MONOLITH_MONOFRAME_V1) — the model commits its frame choice (candidates /
    selected / rejected runner-up / reason) before its answer, EVERY turn, so the
    recorder fires automatically (not only when asked). No-op when off: byte-identical."""
    try:
        from . import frame_selection
        if not frame_selection.enabled():
            return block
        contract = (
            "\n\n[FRAME-SELECTION CONTRACT] Before your answer this turn, emit one "
            "<frame_selection> block committing the frame you adopt:\n"
            "<frame_selection>\n"
            "CANDIDATES: <the frames you weighed, separated by | >\n"
            "SELECTED: <the one you adopt>\n"
            "REJECTED: <the runner-up you did not adopt>\n"
            "REASON: <why the runner-up lost>\n"
            "</frame_selection>\n"
            "Then answer. This is a commitment your answer will be judged against."
        )
        return block + contract
    except Exception:
        return block


def bearing_interceptor(messages: list[dict], config: dict) -> list[dict] | None:
    """Inject the [BEARING] block before the latest non-ephemeral user message.

    Includes a deterministic staleness check: if bearing.current_frame
    implies a different channel context than the actual CHANNEL tag,
    a [BEARING — possible staleness] nudge is appended to the block so the
    model can self-author a <bearing_update> if its position moved.

    Returns None when:
      - kill switch is off (MONOLITH_BEARING_V1=0)
      - block is already present (double-fire defense)
      - no non-ephemeral user message exists to insert before
    """
    if not kill_switch.is_enabled():
        return None
    if _already_injected(messages):
        return None
    last_user_idx = _find_last_non_ephemeral_user_idx(messages)
    if last_user_idx < 0:
        return None
    try:
        bearing = store.get_bearing()
        pending = store.get_pending_rejection()
    except Exception:
        return None
    current_turn_n = _resolve_current_turn_n(config)
    plan_view = _resolve_plan_view(config)
    block = format_bearing_block(bearing, pending, current_turn_n, plan_view)
    block = _apply_staleness(block, bearing, messages, config)
    block = _apply_correction_example(block, messages)
    block = _apply_frame_selection_contract(block)
    _observe_drift(bearing, messages, config)
    result = list(messages)
    result.insert(
        last_user_idx,
        {
            "role": "user",
            "content": block,
            "ephemeral": True,
            "source": _SOURCE,
        },
    )
    return result
