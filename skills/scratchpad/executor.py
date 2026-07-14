"""Scratchpad — first-person continuity workspace.

Storage and projection live in core.continuity. This file is the tool
dispatch surface only. Thirteen ops: pin / retire / read / working_memory_set / working_memory_get / working_memory_clear / propose_amendment / list_proposals / record_confidence / review_read / review_mark / observe / introspect.

The [CONTINUITY] block is auto-injected on the first user turn of each
session by core.continuity.continuity_interceptor. Mid-session, the
model uses op=pin to commit a new pin to the store.

This is a clean break from the previous flat append-text-log scratchpad.
Old text logs at ~/.monolith/scratch are ignored — they were not used as
continuity. The new structured store lives at CONFIG_DIR/continuity.json.

propose_amendment / list_proposals: amendment proposal queue stored at
CONFIG_DIR/proposals.json. Monolith writes; E reads and manually applies
or rejects. No automatic apply or rollback — explicit human gate.
"""
from __future__ import annotations

import re
from typing import Any

from core import continuity
from core import proposals as _proposals
from core import confidence_trajectory as _ct
from core import review_loop as _review


# Third-person self-reference patterns. When an anchor pin contains
# rule-language AND any of these, it has the pin-10 self-violation shape:
# the pin defines a first-person rule but speaks of Monolith in third
# person (e.g., "Monolith's current commitments" in a pin about how
# pins should be written). Refuse at write time with explicit override.
_THIRD_PERSON_SELF_TERMS = (
    "Monolith's",
    "Monolith writes",
    "Monolith updates",
    "Monolith clears",
    "Monolith does",
    "Monolith has",
    "Monolith is",
    "Monolith maintains",
    "Monolith holds",
    "past-you",
    "prior Monolith",
    "actor-character",
)
_THIRD_PERSON_SELF_PATTERNS = tuple(
    re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    for term in _THIRD_PERSON_SELF_TERMS
)


def _has_third_person_self(text: str) -> bool:
    hay = str(text or "")
    return any(p.search(hay) for p in _THIRD_PERSON_SELF_PATTERNS)


def _format_active_line(p: dict) -> str:
    cat = p.get("category", "lesson")
    pid = p.get("id", "?")
    text = p.get("text", "")
    line = f"  {cat}({pid}): {text}"
    ev = p.get("evidence")
    if ev:
        line += f" (evidence: {ev})"
    sup = p.get("supersedes")
    if sup:
        line += f" (supersedes #{sup})"
    return line


def _format_retired_line(p: dict) -> str:
    cat = p.get("category", "lesson")
    pid = p.get("id", "?")
    text = p.get("text", "")
    reason = p.get("retire_reason", "?")
    return f"  retired {cat}({pid}) [{reason}]: {text}"


def _op_pin(cmd: dict) -> str:
    text = str(cmd.get("text") or "").strip()
    if not text:
        return "[scratchpad: pin requires non-empty 'text']"
    category = str(cmd.get("category") or "lesson")
    source = str(cmd.get("source") or "i_inferred")
    evidence = cmd.get("evidence")
    if evidence is not None:
        evidence = str(evidence)
    supersedes = cmd.get("supersedes")
    if supersedes is not None:
        try:
            supersedes = int(supersedes)
        except (TypeError, ValueError):
            return "[scratchpad: 'supersedes' must be an integer pin id]"
    bypass = bool(cmd.get("bypass_self_violation_check", False))
    if (
        category == "anchor"
        and not bypass
        and _review._has_rule_language(text)
        and _has_third_person_self(text)
    ):
        return (
            "[scratchpad: pin REJECTED — anchor with rule-language uses "
            "third-person Monolith framing (pin-10 self-violation shape, "
            "2026-05-20 incident). Rewrite in first-person (I/my/me) or "
            "pass bypass_self_violation_check=true to force.]"
        )
    try:
        pinned = continuity.pin(
            text=text,
            category=category,
            source=source,
            evidence=evidence,
            supersedes=supersedes,
        )
    except ValueError as exc:
        return f"[scratchpad: {exc}]"
    suffix = f" superseding #{supersedes}" if supersedes else ""
    return f"[scratchpad: pinned {pinned['category']}({pinned['id']}){suffix}]"


def _op_retire(cmd: dict) -> str:
    raw_id = cmd.get("id")
    try:
        pin_id = int(raw_id)
    except (TypeError, ValueError):
        return "[scratchpad: retire requires integer 'id']"
    reason = str(cmd.get("reason") or "user_retired").strip()
    retired = continuity.retire(pin_id, reason)
    if retired is None:
        return f"[scratchpad: id={pin_id} not found in active pins]"
    return f"[scratchpad: retired #{pin_id} ({retired.get('retire_reason', '?')})]"


def _op_read(cmd: dict) -> str:
    include_retired = bool(cmd.get("include_retired", False))
    snap = continuity.read(include_retired=include_retired, retired_limit=5)
    counts = snap.get("counts", {})
    active = snap.get("active", [])
    lines = [
        f"[scratchpad: {counts.get('active', 0)} active, "
        f"{counts.get('retired_total', 0)} retired total]"
    ]
    if active:
        lines.append("Active pins:")
        for p in active:
            lines.append(_format_active_line(p))
    else:
        lines.append("(no active pins)")
    if include_retired:
        retired = snap.get("retired", [])
        if retired:
            lines.append(f"Last {len(retired)} retired:")
            for p in retired:
                lines.append(_format_retired_line(p))
    return "\n".join(lines)


def _op_working_memory_set(cmd: dict) -> str:
    raw = str(cmd.get("text") or "")
    stripped = raw.strip()
    if not stripped:
        return "[working_memory_set: text is empty]"
    if len(stripped) > 1000:
        return "[working_memory_set: text exceeds 1000-char cap]"
    from core import llm_config
    writer_id = llm_config.get_current_model_id()
    continuity.set_working_memory(stripped, writer_id)
    return f"[working_memory_set: {len(stripped)} chars written]"


def _op_working_memory_get(cmd: dict) -> str:
    slot = continuity.get_working_memory()
    if slot is None:
        return "[working_memory_get: empty]"
    text = slot["text"]
    writer = slot["writer_model_id"]
    return f"[working_memory_get: writer={writer}]\n{text}"


def _op_working_memory_clear(cmd: dict) -> str:
    continuity.clear_working_memory()
    return "[working_memory_clear: ok]"


def _op_propose_amendment(cmd: dict) -> str:
    target = str(cmd.get("target") or "").strip()
    section = str(cmd.get("section") or "").strip()
    current_text = str(cmd.get("current_text") or "").strip()
    proposed_text = str(cmd.get("proposed_text") or "").strip()
    rationale = str(cmd.get("rationale") or "").strip()
    from core import llm_config
    writer_model_id = llm_config.get_current_model_id()
    try:
        record = _proposals.propose_amendment(
            target=target,
            section=section,
            current_text=current_text,
            proposed_text=proposed_text,
            rationale=rationale,
            writer_model_id=writer_model_id,
        )
    except ValueError as exc:
        return f"[propose_amendment: {exc}]"
    return (
        f"[propose_amendment: queued as proposal id={record['id']} "
        f"targeting {record['target']}:{record['section']}]"
    )


def _op_list_proposals(cmd: dict) -> str:
    items = _proposals.list_proposals(limit=20)
    total = len(items)
    lines = [f"[list_proposals: {total} queued]"]
    for p in items:
        pid = p.get("id", "?")
        status = p.get("status", "pending")
        target = p.get("target", "?")
        section = p.get("section", "?")
        rationale = p.get("rationale", "")
        snippet = rationale[:80]
        lines.append(f"- #{pid} ({status}) {target} / {section} — {snippet}")
    return "\n".join(lines)


def _op_record_confidence(cmd: dict) -> str:
    raw_value = cmd.get("value")
    # Coerce string digits sent through JSON → int, but reject bools.
    if isinstance(raw_value, bool):
        return "[record_confidence: value must be an integer 0-100, got bool]"
    if isinstance(raw_value, str):
        try:
            raw_value = int(raw_value.strip())
        except (ValueError, TypeError):
            return "[record_confidence: value must be an integer 0-100]"
    claim = str(cmd.get("claim") or "").strip()
    premise = str(cmd.get("premise") or "").strip()
    from core import llm_config
    writer_model_id = llm_config.get_current_model_id()
    try:
        record = _ct.record_confidence(
            value=raw_value,
            claim=claim,
            premise=premise,
            writer_model_id=writer_model_id,
        )
    except ValueError as exc:
        return f"[record_confidence: {exc}]"
    snippet = record["claim"][:60]
    return f'[record_confidence: logged value={record["value"]} for "{snippet}"]'


def _format_review_line(item: dict) -> str:
    iid = item.get("id", "?")
    kind = item.get("kind", "?")
    subkind = item.get("subkind", "general")
    sev = item.get("effective_severity", item.get("severity", "?"))
    age = item.get("age_days", 0)
    summary = str(item.get("summary") or "")[:120]
    return f"- {iid} {kind}.{subkind} sev={sev} age={age}d — {summary}"


def _op_review_read(cmd: dict) -> str:
    kind = str(cmd.get("kind") or "").strip() or None
    subkind = str(cmd.get("subkind") or "").strip() or None
    try:
        limit = max(1, min(int(cmd.get("limit", 10) or 10), 50))
    except (TypeError, ValueError):
        limit = 10
    items = _review.list_review_items(kind=kind, subkind=subkind, limit=limit)
    lines = [f"[review_read: {len(items)} item(s)]"]
    if not items:
        lines.append("(no unresolved review items)")
        return "\n".join(lines)
    for item in items:
        lines.append(_format_review_line(item))
        reason = str(item.get("reason") or "")
        if reason:
            lines.append(f"  reason: {reason[:140]}")
    return "\n".join(lines)


def _op_review_mark(cmd: dict) -> str:
    item_id = str(cmd.get("item_id") or cmd.get("id") or "").strip()
    action = str(cmd.get("action") or "").strip().lower()
    if not item_id:
        return "[review_mark: item_id is required]"
    if not action:
        return "[review_mark: action is required]"
    note = cmd.get("note")
    raw_hours = cmd.get("snooze_hours", cmd.get("hours"))
    snooze_hours = None
    if raw_hours is not None:
        try:
            snooze_hours = float(raw_hours)
        except (TypeError, ValueError):
            return "[review_mark: snooze_hours must be numeric]"
    snoozed_until = cmd.get("snoozed_until")
    try:
        result = _review.review_mark(
            item_id,
            action,
            actor="monolith",
            note=str(note) if note is not None else None,
            snooze_hours=snooze_hours,
            snoozed_until=str(snoozed_until) if snoozed_until is not None else None,
        )
    except _review.ReviewAuthorizationError as exc:
        return f"[review_mark: unauthorized: {exc}]"
    except ValueError as exc:
        return f"[review_mark: {exc}]"
    return f"[review_mark: {result['action']} {result['item_id']} as {result['actor']}]"


def _op_observe(cmd: dict) -> str:
    summary = str(cmd.get("summary") or cmd.get("text") or "").strip()
    if not summary:
        return "[observe: summary/text is required]"
    reason = str(cmd.get("reason") or "").strip()
    subkind = str(cmd.get("subkind") or "general").strip() or "general"
    raw_severity = cmd.get("severity", 2)
    try:
        severity = int(raw_severity)
    except (TypeError, ValueError):
        return "[observe: severity must be an integer 1-5]"
    try:
        record = _review.record_observation(
            summary,
            reason=reason,
            severity=severity,
            subkind=subkind,
            source="monolith",
        )
    except ValueError as exc:
        return f"[observe: {exc}]"
    return f"[observe: recorded observation id={record['id']} severity={record['severity']}]"


def _op_introspect(cmd: dict) -> str:
    from core import subsystem_map as _smap
    data = _smap.read_subsystem_map()
    raw_kind = cmd.get("kind")
    kind = str(raw_kind).strip().lower() if raw_kind else None
    if kind == "all":
        kind = None
    raw_name = cmd.get("name")
    name_filter = str(raw_name).strip() if raw_name else None
    return _smap.format_subsystem_map(data, kind=kind, name_filter=name_filter)


def run(cmd: dict, ctx: Any) -> str:
    op = str(cmd.get("op") or "read").strip().lower()
    if op == "pin":
        return _op_pin(cmd)
    if op == "retire":
        return _op_retire(cmd)
    if op == "read":
        return _op_read(cmd)
    if op == "working_memory_set":
        return _op_working_memory_set(cmd)
    if op == "working_memory_get":
        return _op_working_memory_get(cmd)
    if op == "working_memory_clear":
        return _op_working_memory_clear(cmd)
    if op == "propose_amendment":
        return _op_propose_amendment(cmd)
    if op == "list_proposals":
        return _op_list_proposals(cmd)
    if op == "record_confidence":
        return _op_record_confidence(cmd)
    if op == "review_read":
        return _op_review_read(cmd)
    if op == "review_mark":
        return _op_review_mark(cmd)
    if op == "observe":
        return _op_observe(cmd)
    if op == "introspect":
        return _op_introspect(cmd)
    return (
        f"[scratchpad: unknown op '{op}'. Use pin / retire / read / "
        f"working_memory_set / working_memory_get / working_memory_clear / "
        f"propose_amendment / list_proposals / record_confidence / "
        f"review_read / review_mark / observe / introspect. "
        f"pin params: text, category (anchor|pending|lesson), source "
        f"(user_said|i_inferred|evidence), evidence?, supersedes?, "
        f"bypass_self_violation_check? . "
        f"retire params: id, reason. read params: include_retired? . "
        f"working_memory_set params: text (max 1000 chars, whitespace-stripped). "
        f"working_memory_get params: (none). "
        f"working_memory_clear params: (none). "
        f"propose_amendment params: target (identity.md|system.md), section, "
        f"current_text (≤2000 chars), proposed_text (≤2000 chars), rationale (≤800 chars). "
        f"list_proposals params: (none). "
        f"record_confidence params: value (int 0-100), claim (≤200 chars), premise (≤200 chars). "
        f"review_read params: kind?, subkind?, limit?. "
        f"review_mark params: item_id, action (resolve|dismiss|snooze|escalate), note?, snooze_hours?. "
        f"observe params: summary or text, reason?, severity?, subkind?. "
        f"introspect params: kind? (policies|planes|skills|interceptors|all), name? (substring filter).]"
    )
