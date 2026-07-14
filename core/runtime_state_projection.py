"""Deterministic prose projection for the coalesced [RUNTIME STATE] block."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from core.runtime_state_lanes import LANE_ORDER, lead_phrase
from core.self_description import describe_self


_TAG = "[RUNTIME STATE]"
_END_TAG = "[/RUNTIME STATE]"
_IDENTITY_BODY = "the Monolith identity seed governs posture and commitments."
_CATEGORY_RANK = {"anchor": 0, "pending": 1, "lesson": 2}
_MAX_RECALL_CHARS = 600

# Relational-time lane: elapsed since last turn / since the previous session.
# Dark by default — ships off for first observation (when-plane primitive).
_RELATIVE_FLAG = "MONOLITH_RELATIVE_TIME_V1"


def _relative_enabled() -> bool:
    raw = str(os.environ.get(_RELATIVE_FLAG, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _humanize_span(seconds: float) -> str:
    """Coarse magnitude token for an elapsed span (no 'ago' suffix).

    Date/minute precision lives in the absolute temporal_context lane; this is
    deliberately bucketed so the relative grounding reads as orientation, not a
    stopwatch.
    """
    s = max(0.0, float(seconds))
    if s < 90:
        return "moments"
    if s < 3600:
        return f"{int(round(s / 60))}m"
    if s < 86400:
        return f"{int(s // 3600)}h"
    if s < 14 * 86400:
        return f"{int(s // 86400)}d"
    return f"{int(s // (7 * 86400))}w"


def render_identity_lane() -> str:
    """Render identity as operating law, not first-person runtime fact."""
    return f"{lead_phrase('identity_material')} {_IDENTITY_BODY}"


def _non_ephemeral_user_count(messages: list[dict]) -> int:
    return sum(
        1 for msg in messages
        if isinstance(msg, dict) and msg.get("role") == "user" and not msg.get("ephemeral")
    )


def _latest_non_ephemeral_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user" and not msg.get("ephemeral"):
            return str(msg.get("content", "") or "")
    return ""


def _short(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _pin_sort_key(pin: dict) -> tuple[int, int]:
    category = str(pin.get("category", "lesson"))
    try:
        pin_id = int(pin.get("id", 0) or 0)
    except (TypeError, ValueError):
        pin_id = 0
    return (_CATEGORY_RANK.get(category, 99), pin_id)


def _format_pin(pin: dict) -> str:
    category = str(pin.get("category", "lesson") or "lesson")
    pin_id = pin.get("id", "?")
    text = _short(pin.get("text", ""))
    if category == "pending":
        line = f"- pending({pin_id}), open promise: {text}"
    else:
        line = f"- {category}({pin_id}): {text}"
    evidence = _short(pin.get("evidence", ""), limit=80)
    if evidence:
        line += f" (evidence: {evidence})"
    return line


def render_continuity_lane(messages: list[dict]) -> str:
    """Render first-turn continuity pins and preserve working-memory clear."""
    from core import continuity

    user_count = _non_ephemeral_user_count(messages)
    if user_count == 1:
        continuity.clear_working_memory()
    if not continuity.is_continuity_enabled() or user_count != 1:
        return ""

    snapshot = continuity.read(include_retired=False)
    active = list(snapshot.get("active", []))
    if not active:
        return ""
    active.sort(key=_pin_sort_key)
    lines = [lead_phrase("continuity")]
    lines.extend(_format_pin(pin) for pin in active)
    return "\n".join(lines)


def render_recall_lane(messages: list[dict]) -> str:
    from core import acu_retrieval, recall_handles
    from core.acatalepsy.authority import compute_authority

    recall_handles.reset()  # handles are per-turn; never carry across turns
    prompt = _latest_non_ephemeral_user_text(messages).strip()
    if not prompt:
        return ""
    acus = acu_retrieval.retrieve_relevant_acus(prompt)
    if not acus:
        return ""
    acu_retrieval._write_recall_hit(acus)  # deference hit-log (the live recall path)
    lines = [lead_phrase("recall")]
    total = len(lines[0])
    shown = 0
    for acu in acus:
        canonical = str(acu.get("canonical", "")).strip()
        if not canonical:
            continue
        # Authority-derived label so AU3 truth-confirmed renders [VERIFIED] (the
        # old label_text read the dead `veracity` and never did). system.md's
        # POLICY-PRIORITY rule makes [LOCKED]/[VERIFIED] override the model's guess.
        label = acu_retrieval._deference_label(compute_authority(acu))
        # [handle] precedes [label] so the existing "[LABEL] canonical" substring is
        # preserved; the handle is the citable identity the grounded verdict resolves.
        handle = f"R{shown + 1}"
        line = f"- [{handle}] [{label}] {canonical}"
        if total + len(line) + 1 > _MAX_RECALL_CHARS:
            break
        recall_handles.register(handle, acu)
        shown += 1
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) if len(lines) > 1 else ""


def render_current_model_execution_lane(config: dict | None = None) -> str:
    payload = describe_self(config if isinstance(config, dict) else {})
    execution = payload.get("current_model_execution", {})
    if not isinstance(execution, dict):
        return ""

    backend = _short(execution.get("backend_kind"))
    location = _short(execution.get("execution_location"))
    provider = _short(execution.get("provider"))
    model = _short(execution.get("model"))
    context_window = execution.get("context_window")
    persistent = execution.get("persistent_process")
    stateless = execution.get("stateless_per_turn")

    if not any([backend and backend != "unknown", location and location != "unknown", provider, model, context_window]):
        return ""

    parts: list[str] = []
    if backend and backend != "unknown":
        parts.append(f"backend={backend}")
    if location and location != "unknown":
        parts.append(f"execution_location={location}")
    if provider:
        parts.append(f"provider={provider}")
    if model:
        parts.append(f"model={model}")
    if context_window:
        parts.append(f"context_window={context_window}")
    if persistent is not None:
        parts.append(f"persistent_process={str(bool(persistent)).lower()}")
    if stateless is not None:
        parts.append(f"stateless_per_turn={str(bool(stateless)).lower()}")
    return f"{lead_phrase('current_model_execution')} {'; '.join(parts)}."


def render_temporal_lane(now: datetime | None = None) -> str:
    from core import temporal_context

    if not temporal_context.is_temporal_enabled():
        return ""
    return f"{lead_phrase('temporal_context')} {temporal_context.format_temporal_value(now)}"


def render_relative_time_lane(messages: list[dict], now: datetime | None = None) -> str:
    """Elapsed grounding: 'last turn ~Nm ago' / '~Nd since the previous session'.

    Dark by default (``MONOLITH_RELATIVE_TIME_V1``). Reads the single continuity
    ``last_turn_at`` marker; on the first turn of a session that marker still
    holds the PRIOR session's last activity, so ``now - last_turn_at`` is the
    inter-session gap. Conservative: no marker or a corrupt one → render nothing
    (never a fabricated '0s', never a crash that would drop the whole block).
    """
    if not _relative_enabled():
        return ""
    from core import continuity

    last_iso = continuity.get_last_turn_at()
    if not last_iso:
        return ""
    try:
        last = datetime.fromisoformat(str(last_iso))
    except (ValueError, TypeError):
        return ""
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    span = _humanize_span((now_dt - last).total_seconds())
    if _non_ephemeral_user_count(messages) == 1:
        clause = f"{span} since the previous session"   # first turn → inter-session gap
    else:
        clause = f"last turn {span} ago"
    return f"{lead_phrase('temporal_relative')} {clause}."


def render_runtime_state(
    messages: list[dict],
    config: dict | None = None,
    *,
    now: datetime | None = None,
) -> str:
    """Render the full runtime-state envelope in fixed registry order."""
    renderers = {
        "identity_material": lambda: render_identity_lane(),
        "continuity": lambda: render_continuity_lane(messages),
        "recall": lambda: render_recall_lane(messages),
        "current_model_execution": lambda: render_current_model_execution_lane(config),
        "temporal_context": lambda: render_temporal_lane(now),
        "temporal_relative": lambda: render_relative_time_lane(messages, now),
    }
    lines = [f"{_TAG} - ambient runtime state; NOT this turn's request."]
    for lane_name in LANE_ORDER:
        text = renderers[lane_name]().strip()
        if text:
            lines.append(text)
    if len(lines) == 1:
        return ""
    lines.append(_END_TAG)
    return "\n".join(lines)


def contribute_section(messages: list[dict], config: dict):
    """Ephemeral coalescer section contributor."""
    from core.ephemeral_coalescer import SectionResult

    if _non_ephemeral_user_count(messages) <= 0:
        return None
    # TurnClock: derive both temporal render lanes from the ONE instant captured
    # once per outer turn (config['_now_iso']), instead of each lane reading the OS
    # clock independently. parse_local → local-aware (the absolute lane strftimes it
    # directly); the relative lane's aware subtraction is zone-agnostic, so one value
    # serves both. Absent/corrupt key → None → each lane falls through to the OS clock
    # exactly as before (flag-off byte-identical). Dark flag gates the STAMP, not here.
    from core import turn_clock
    now = turn_clock.parse_local((config or {}).get("_now_iso"))
    block = render_runtime_state(messages, config, now=now)
    if not block:
        return None
    # Record this turn's timestamp for the relative-time lane — but only via
    # on_commit, so the marker advances ONLY when the block actually lands
    # (when-plane fix #7). Gated by the same dark flag so nothing is written
    # until relative time is enabled. The render above reads the OLD marker;
    # this commit writes the new one → read-old-then-write-new is correct.
    on_commit = None
    if _relative_enabled():
        from core import continuity
        now_iso = continuity._now_iso()
        on_commit = lambda: continuity.set_last_turn_at(now_iso)  # noqa: E731
    return SectionResult(name="runtime_state", text=block, on_commit=on_commit)
