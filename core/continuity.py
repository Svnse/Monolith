"""Continuity — first-person session-survival workspace for the model.

This module owns three things:
  * The pin store (CONFIG_DIR / "continuity.json"; single global file)
  * The legacy projection/interceptor that can render pins into a [CONTINUITY] block
  * The live continuity lane consumed by core.runtime_state_projection

Designed for self-curation: the model writes pins it wants future-self to see.
The user can edit the JSON directly to override; there is no approval gate.

Pin schema (per entry):
    id          : int                 # auto-increment
    text        : str                 # ≤200 chars (lesson|pending), ≤500 (anchor)
    category    : "anchor" | "pending" | "lesson"
    source      : "user_said" | "i_inferred" | "evidence"
    evidence    : str | None          # ≤120 chars; ONE concrete instance
    supersedes  : int | None          # predecessor pin id; predecessor auto-retires
    created_at  : ISO-8601

Bounds:
    8 active pins. On overflow: oldest non-anchor lesson auto-retires
    with reason="aged_out". Anchors are protected; pendings are protected
    over lessons; only as a last resort does the oldest anchor cascade out.

    16 retired entries kept in tail; older entries silently dropped.

Categories:
    anchor   — load-bearing context that must not decay
    pending  — open promise the model owes
    lesson   — calibration; what to keep doing or stop doing

Six ops via skills/scratchpad/executor.py: pin / retire / read / working_memory_set / working_memory_get / working_memory_clear.

Flag: MONOLITH_CONTINUITY_BOOT_V1 (default ON). Set =0 to disable continuity
lane/block rendering.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from core.paths import CONFIG_DIR

_STORE_PATH = CONFIG_DIR / "continuity.json"

_VALID_CATEGORIES = {"anchor", "pending", "lesson"}
_VALID_SOURCES = {"user_said", "i_inferred", "evidence"}
_VALID_RETIRE_REASONS = {"resolved", "wrong", "stale", "aged_out", "user_retired"}

_ACTIVE_CAP = 8
_RETIRED_CAP = 16
_TEXT_LIMIT_DEFAULT = 200
_TEXT_LIMIT_ANCHOR = 500
_EVIDENCE_LIMIT = 120

_FLAG_ENV = "MONOLITH_CONTINUITY_BOOT_V1"
_TAG = "[CONTINUITY]"


# ── helpers ──────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_continuity_enabled() -> bool:
    """Public flag check for cross-module consumers (describe_self, etc.).

    Mirrors _flag_enabled so the env-var + truthy-set semantics stay in one
    place; downstream callers should import this rather than re-reading the
    env var directly.
    """
    return _flag_enabled()


# ── relational-time marker (when-plane) ──────────────────────────────
# A single durable timestamp of the most recent turn. On the first turn of a
# new session it still holds the PRIOR session's last activity, so
# (now - last_turn_at) is the inter-session gap; on later turns it is the
# since-last-turn delta. Read by the temporal_relative runtime-state lane.


def get_last_turn_at() -> str | None:
    """Return the stored last-turn ISO timestamp, or None if never recorded."""
    return _load().get("last_turn_at")


def set_last_turn_at(now_iso: str) -> None:
    """Record the most recent turn's timestamp (additive; preserves pins)."""
    data = _load()
    data["last_turn_at"] = str(now_iso)
    _save(data)


def _empty_store() -> dict:
    return {"version": 1, "next_id": 1, "active": [], "retired": [], "last_turn_at": None}


def _load() -> dict:
    if not _STORE_PATH.exists():
        return _empty_store()
    try:
        with _STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return _empty_store()
    if not isinstance(data, dict):
        return _empty_store()
    data.setdefault("version", 1)
    data.setdefault("next_id", 1)
    data.setdefault("active", [])
    data.setdefault("retired", [])
    data.setdefault("last_turn_at", None)  # when-plane relational-time marker
    return data


def _save(data: dict) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STORE_PATH.with_name(_STORE_PATH.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, _STORE_PATH)


def _normalize_category(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in _VALID_CATEGORIES else "lesson"


def _normalize_source(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "user": "user_said",
        "model": "i_inferred",
        "inferred": "i_inferred",
        "agent": "i_inferred",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in _VALID_SOURCES else "i_inferred"


def _trim_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _trim_retired(data: dict) -> None:
    retired = data.get("retired", [])
    if len(retired) > _RETIRED_CAP:
        data["retired"] = retired[-_RETIRED_CAP:]


def _retire_one(data: dict, idx: int, reason: str) -> dict:
    """Pop active[idx], append to retired with reason and timestamp."""
    removed = data["active"].pop(idx)
    removed["retired_at"] = _now_iso()
    removed["retire_reason"] = reason
    data["retired"].append(removed)
    _trim_retired(data)
    return removed


def _auto_retire_if_overfull(data: dict) -> int | None:
    """If active count exceeds cap, retire one entry. Priority: oldest
    lesson, then oldest pending, then oldest anchor (last resort)."""
    if len(data.get("active", [])) <= _ACTIVE_CAP:
        return None
    for category in ("lesson", "pending", "anchor"):
        for i, p in enumerate(data["active"]):
            if p.get("category") == category:
                removed = _retire_one(data, i, "aged_out")
                return removed.get("id")
    return None


# ── public store API ─────────────────────────────────────────────────


def pin(
    text: str,
    category: str = "lesson",
    source: str = "i_inferred",
    evidence: str | None = None,
    supersedes: int | None = None,
) -> dict:
    """Add a new pin. Returns the new pin dict (with assigned id).

    If `supersedes` references an active pin, that pin is auto-retired
    with reason `superseded_by:<new_id>`. If active count exceeds the
    cap after the new pin lands, the oldest non-protected entry retires
    with reason `aged_out`.
    """
    data = _load()
    cat = _normalize_category(category)
    src = _normalize_source(source)
    text_limit = _TEXT_LIMIT_ANCHOR if cat == "anchor" else _TEXT_LIMIT_DEFAULT
    body = _trim_text(text, text_limit)
    if not body:
        raise ValueError("pin text is empty")
    pin_dict: dict[str, Any] = {
        "id": int(data["next_id"]),
        "text": body,
        "category": cat,
        "source": src,
        "created_at": _now_iso(),
    }
    if evidence is not None:
        ev = _trim_text(evidence, _EVIDENCE_LIMIT)
        if ev:
            pin_dict["evidence"] = ev
    if supersedes is not None:
        try:
            sup_id = int(supersedes)
            pin_dict["supersedes"] = sup_id
            for i, p in enumerate(data["active"]):
                if p.get("id") == sup_id:
                    _retire_one(data, i, f"superseded_by:{pin_dict['id']}")
                    break
        except (TypeError, ValueError):
            pass
    data["next_id"] = pin_dict["id"] + 1
    data["active"].append(pin_dict)
    _auto_retire_if_overfull(data)
    _save(data)
    return pin_dict


def retire(pin_id: int, reason: str) -> dict | None:
    """Move pin to retired tail with reason. Returns retired pin or None
    if id is not found in active. `reason` outside the known set is
    coerced to `user_retired`; `superseded_by:<id>` is allowed verbatim.
    """
    try:
        target_id = int(pin_id)
    except (TypeError, ValueError):
        return None
    data = _load()
    raw_reason = str(reason or "").strip().lower()
    if raw_reason.startswith("superseded_by:"):
        coerced_reason = raw_reason
    elif raw_reason in _VALID_RETIRE_REASONS:
        coerced_reason = raw_reason
    else:
        coerced_reason = "user_retired"
    for i, p in enumerate(data["active"]):
        if p.get("id") == target_id:
            removed = _retire_one(data, i, coerced_reason)
            _save(data)
            return removed
    return None


def read(*, include_retired: bool = False, retired_limit: int = 5) -> dict:
    """Return store snapshot.

    Always returns:
        {"active": [...pins...], "counts": {"active": N, "retired_total": M}}
    With include_retired=True:
        adds "retired": [...last `retired_limit` pins...]
    """
    data = _load()
    active = list(data.get("active", []))
    retired_all = list(data.get("retired", []))
    out: dict[str, Any] = {
        "active": active,
        "counts": {"active": len(active), "retired_total": len(retired_all)},
    }
    if include_retired:
        if retired_limit and retired_limit > 0:
            out["retired"] = retired_all[-retired_limit:]
        else:
            out["retired"] = retired_all
    return out


# ── projection ──────────────────────────────────────────────────────


_CATEGORY_RANK = {"anchor": 0, "pending": 1, "lesson": 2}


def _category_rank(cat: str) -> int:
    return _CATEGORY_RANK.get(cat, 99)


def render_continuity_block() -> str | None:
    """Build the [CONTINUITY] block string. Returns None when no active pins.

    The block is wrapped in an explicit [CONTINUITY ... ambient state]
    /// [/CONTINUITY] envelope so the model can tell it apart from the
    user's current-turn message. The runtime injects this with
    role="user" (single-turn channels don't support mid-conversation
    system messages reliably), so the envelope is the only structural
    signal the model has that the contents are NOT a fresh request.
    """
    data = _load()
    active = list(data.get("active", []))
    if not active:
        return None
    retired_count = len(data.get("retired", []))
    active.sort(key=lambda p: (_category_rank(p.get("category", "lesson")), p.get("id", 0)))
    pin_word = "pin" if len(active) == 1 else "pins"
    header = f"{_TAG} — {len(active)} {pin_word}"
    if retired_count:
        header += f", {retired_count} retired"
    header += " — ambient state from prior sessions; NOT this turn's request"
    lines = [header]
    for p in active:
        cat = p.get("category", "lesson")
        pid = p.get("id", "?")
        text = p.get("text", "")
        if cat == "pending":
            line = f"- pending({pid}) — open promise, acknowledge or defer this turn: {text}"
        else:
            line = f"- {cat}({pid}): {text}"
        ev = p.get("evidence")
        if ev:
            line += f" (evidence: {ev})"
        lines.append(line)
    lines.append("[/CONTINUITY]")
    return "\n".join(lines)


# ── working memory: per-session, per-execution-surface scratchpad ────


def get_working_memory() -> dict | None:
    """Return the current working_memory slot, or None if empty.

    Slot shape: {"text": str, "writer_model_id": str}. No other keys.
    """
    store = _load()
    slot = store.get("working_memory")
    if not isinstance(slot, dict):
        return None
    text = slot.get("text")
    writer = slot.get("writer_model_id")
    if not isinstance(text, str) or not isinstance(writer, str):
        return None
    return {"text": text, "writer_model_id": writer}


def set_working_memory(text: str, writer_model_id: str) -> None:
    """Overwrite the working_memory slot. Caller is responsible for validation.

    The scratchpad op layer does strip/empty/length validation before calling
    this. This function does no validation — it persists what it's given.
    """
    store = _load()
    store["working_memory"] = {
        "text": str(text),
        "writer_model_id": str(writer_model_id),
    }
    _save(store)


def clear_working_memory() -> None:
    """Null the working_memory slot. Idempotent — no-op if already null."""
    store = _load()
    store["working_memory"] = None
    _save(store)


# ── interceptor ──────────────────────────────────────────────────────


def continuity_interceptor(
    messages: list[dict], config: dict
) -> list[dict] | None:
    """First-turn-only [CONTINUITY] injection + working_memory clear.

    Fires when exactly one non-ephemeral user message exists — i.e. the user
    has just sent their first prompt of the session. Robust against /regen
    (which wipes assistant messages but retains user messages) and ephemeral
    assistant artifacts (which don't represent a real prior turn).

    On first turn, ALSO clears the working_memory slot (active
    session-boundary cleanup; the other clear site is lazy at
    read-time on writer_model_id mismatch in build_system_prompt).

    Skips when:
      - flag MONOLITH_CONTINUITY_BOOT_V1 is off
      - non-ephemeral user message count != 1 (not first turn, or no user yet)
      - store has no active pins
      - block is already present in the message list (defense vs double-fire)
    """
    user_count = sum(
        1 for msg in messages
        if msg.get("role") == "user" and not msg.get("ephemeral")
    )
    if user_count == 1:
        # Active session-boundary clear: fires on every new session,
        # regardless of whether the CONTINUITY block injection is enabled.
        # Lock 1 requires per-session lifetime, independent of the flag.
        clear_working_memory()
    if not _flag_enabled():
        return None
    if user_count != 1:
        return None
    block = render_continuity_block()
    if block is None:
        return None
    for msg in messages:
        if _TAG in str(msg.get("content", "")):
            return None
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            last_user_idx = i
            break
    if last_user_idx < 0:
        return None
    result = list(messages)
    result.insert(
        last_user_idx,
        {
            "role": "user",
            "content": block,
            "ephemeral": True,
            "source": "continuity",
        },
    )
    return result


def contribute_section(messages: list[dict], config: dict):
    """Section-contributor variant for the ephemeral_coalescer.

    Same gate as ``continuity_interceptor`` (first-turn-only, robust against
    /regen and ephemeral assistant artifacts), but returns just the rendered
    block — the coalescer owns the insertion point + budget.
    """
    from core.ephemeral_coalescer import SectionResult
    user_count = sum(
        1 for msg in messages
        if msg.get("role") == "user" and not msg.get("ephemeral")
    )
    if user_count == 1:
        # Active session-boundary clear: fires on every new session,
        # regardless of whether the CONTINUITY block injection is enabled.
        # Lock 1 requires per-session lifetime, independent of the flag.
        clear_working_memory()
    if not _flag_enabled():
        return None
    if user_count != 1:
        return None
    block = render_continuity_block()
    if block is None:
        return None
    return SectionResult(name="continuity", text=block)
