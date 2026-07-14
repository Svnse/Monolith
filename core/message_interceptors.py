from __future__ import annotations

from typing import Callable, Iterable

Message = dict
Interceptor = Callable[[list[Message], dict], list[Message] | None]

_INTERCEPTORS: list[Interceptor] = []


def register_interceptor(fn: Interceptor) -> None:
    """Register a message interceptor.

    Interceptors receive (messages, config) and can return a new list of messages
    or None to leave messages unchanged.
    """
    if fn in _INTERCEPTORS:
        return
    _INTERCEPTORS.append(fn)


def clear_interceptors() -> None:
    _INTERCEPTORS.clear()


def iter_interceptors() -> Iterable[Interceptor]:
    return list(_INTERCEPTORS)


def apply_interceptors(messages: list[Message], config: dict) -> list[Message]:
    """Run registered interceptors in order.

    Each interceptor receives (messages, config) and returns either a new list
    of messages or None (no change). Exceptions are caught — a bad interceptor
    must not break generation.

    Layer A trace emit: each interceptor call produces a StageTraceRecord
    when config["_turn_id"] is set. Trace failures never affect generation.
    """
    # Lazy import to avoid circular dependency at module load.
    from core import turn_trace as _tt
    from datetime import datetime, timezone

    turn_id = ""
    parent_turn_id: str | None = None
    if isinstance(config, dict):
        turn_id = str(config.get("_turn_id") or "")
        parent_raw = config.get("_parent_turn_id")
        if isinstance(parent_raw, str) and parent_raw:
            parent_turn_id = parent_raw

    current = messages
    seq = 0
    for fn in list(_INTERCEPTORS):
        stage_name = getattr(fn, "__name__", "anonymous_interceptor")
        entered_at = datetime.now(timezone.utc).isoformat()
        messages_in = len(current) if isinstance(current, list) else 0
        outcome = "ran"
        outcome_reason: str | None = None
        items_added: tuple[_tt.StageItem, ...] = ()
        metadata: dict = {}
        new_current = current
        try:
            updated = fn(current, config)
        except Exception as exc:
            outcome = "errored"
            outcome_reason = f"{type(exc).__name__}: {exc}"
            updated = None
        if isinstance(updated, list):
            new_current = updated
            added_msgs = _tt.diff_added_messages(current, updated)
            items_added = tuple(
                _tt.StageItem.added(
                    kind=("ephemeral_user" if m.get("ephemeral") else "message"),
                    content=str(m.get("content", "")),
                    source=str(m.get("source", "") or stage_name),
                )
                for m in added_msgs
            )
            current = new_current
            if not added_msgs:
                # Returned a list but added nothing — record as ran with empty items.
                pass
        elif outcome != "errored":
            outcome = "skipped"
            outcome_reason = "returned None"
        exited_at = datetime.now(timezone.utc).isoformat()
        messages_out = len(current) if isinstance(current, list) else messages_in

        if turn_id:
            try:
                record = _tt.StageTraceRecord(
                    turn_id=turn_id,
                    parent_turn_id=parent_turn_id,
                    seq=seq,
                    stage_name=stage_name,
                    stage_kind="interceptor",
                    entered_at=entered_at,
                    exited_at=exited_at,
                    outcome=outcome,
                    outcome_reason=outcome_reason,
                    messages_in=messages_in,
                    messages_out=messages_out,
                    items_added=items_added,
                    items_dropped=(),
                    metadata=metadata,
                )
                _tt.record_stage(record)
            except Exception:
                # Trace failures must not break generation (Q7).
                pass
        seq += 1
    return current
