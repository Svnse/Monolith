"""Review loop - attention routing for substrate signals.

This module aggregates existing source-of-truth stores into review items and
keeps only triage state locally. It must not mutate ACUs, proposals, or
continuity pins.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.paths import CONFIG_DIR


_STATE_PATH = CONFIG_DIR / "review_loop.json"
_OBSERVATIONS_PATH = CONFIG_DIR / "review_observations.json"

_SCHEMA_VERSION = 1
_OBS_SCHEMA_VERSION = 1

_DEFAULT_LIMIT = 20
_SOURCE_LIMIT = 200
_SNOOZE_HOURS_DEFAULT = 24

_THRESHOLD_HOURS: dict[tuple[str, str], int] = {
    ("proposal", "pending"): 48,
    ("continuity_pin", "pending"): 48,
    ("continuity_pin", "anchor_rule"): 24 * 7,
    ("continuity_pin", "evidence"): 24 * 7,
    ("audit_claim", "bug"): 24,
    ("audit_claim", "ontology"): 24 * 7,
    ("audit_claim", "architecture"): 24 * 7,
    ("audit_claim", "general"): 24 * 7,
    ("experiment_observation", "general"): 72,
}

_ACTIONS_ALL = ("resolve", "dismiss", "snooze", "escalate")
# Monolith is granted ONLY soft actions on every item kind (2026-06-19 chokepoint
# fix). resolve/dismiss silently remove a real claim/pin from E's queue — an
# un-checked judgment call that stays E's, enforced HERE at the mutator's authz
# (review_mark) so it holds on every caller, not just the self_maint actuator.
# A future re-verified-resolve (bug-claim re-check) re-introduces resolve behind
# an executable check, not via this open grant.
_ACTIONS_SOFT = ("snooze", "escalate")

_BUG_TERMS = (
    "not injected",
    "asserts",
    "broken",
    "fails",
    "does not",
    "missing",
    "runtime mismatch",
    "mismatch",
)
_ONTOLOGY_TERMS = (
    "identity",
    "selfhood",
    "local",
    "cloud",
    "runtime",
    "substrate",
    "contradiction",
)
_ARCHITECTURE_TERMS = (
    "queue",
    "routing",
    "governance",
    "schema",
    "cadence",
    "validator",
    "proposal",
    "review",
)
_RULE_TERMS = (
    "avoid",
    "do not",
    "must",
    "should",
    "use",
    "prefer",
    "never",
    "always",
    "the rule is",
    "pins are",
)
_MONOLITH_RUNTIME_EVIDENCE_TERMS = (
    "trace",
    "state",
    "continuity",
    "proposal",
    "acu store",
    "acatalepsy",
    "turn_trace",
    "/state",
)


def _compile_term_pattern(term: str) -> re.Pattern[str]:
    """Compile a regex matching `term` only at word-character boundaries.

    `"use"` matches `"use it"` and `"use."` but not `"user"` or `"abuse"`.
    Multi-word phrases like `"not injected"` and slash-prefixed terms like
    `"/state"` are handled by conditionally applying lookarounds based on
    whether the term's first/last char is a word char.
    """
    if not term:
        return re.compile(r"(?!)")
    parts: list[str] = []
    if term[0].isalnum() or term[0] == "_":
        parts.append(r"(?<!\w)")
    parts.append(re.escape(term))
    if term[-1].isalnum() or term[-1] == "_":
        parts.append(r"(?!\w)")
    return re.compile("".join(parts), re.IGNORECASE)


def _matches_any(hay: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(p.search(hay) for p in patterns)


_BUG_PATTERNS = tuple(_compile_term_pattern(t) for t in _BUG_TERMS)
_ONTOLOGY_PATTERNS = tuple(_compile_term_pattern(t) for t in _ONTOLOGY_TERMS)
_ARCHITECTURE_PATTERNS = tuple(_compile_term_pattern(t) for t in _ARCHITECTURE_TERMS)
_RULE_PATTERNS = tuple(_compile_term_pattern(t) for t in _RULE_TERMS)
_MONOLITH_RUNTIME_EVIDENCE_PATTERNS = tuple(
    _compile_term_pattern(t) for t in _MONOLITH_RUNTIME_EVIDENCE_TERMS
)


@dataclass(frozen=True)
class ReviewItem:
    id: str
    source: str
    source_id: str
    kind: str
    subkind: str
    severity: int
    effective_severity: int
    created_at: str
    age_days: float
    summary: str
    reason: str
    recommended_actions: list[str]
    allowed_actors: dict[str, list[str]]
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReviewAuthorizationError(PermissionError):
    """Raised when an actor attempts a review action it cannot take."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _age_hours(created_at: Any, *, now: datetime | None = None) -> float:
    dt = _parse_dt(created_at)
    if dt is None:
        return 0.0
    ref = now or _now()
    return max(0.0, (ref - dt).total_seconds() / 3600.0)


def _short(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "..."


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return dict(default)
    if not isinstance(data, dict):
        return dict(default)
    return data


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _empty_state() -> dict:
    return {"schema_version": _SCHEMA_VERSION, "items": {}}


def _load_state() -> dict:
    data = _load_json(_STATE_PATH, _empty_state())
    data.setdefault("schema_version", _SCHEMA_VERSION)
    data.setdefault("items", {})
    if not isinstance(data["items"], dict):
        data["items"] = {}
    return data


def _save_state(data: dict) -> None:
    _save_json(_STATE_PATH, data)


def _empty_observations() -> dict:
    return {"schema_version": _OBS_SCHEMA_VERSION, "next_id": 1, "observations": []}


def _load_observations() -> dict:
    data = _load_json(_OBSERVATIONS_PATH, _empty_observations())
    data.setdefault("schema_version", _OBS_SCHEMA_VERSION)
    data.setdefault("next_id", 1)
    data.setdefault("observations", [])
    if not isinstance(data["observations"], list):
        data["observations"] = []
    return data


def _save_observations(data: dict) -> None:
    _save_json(_OBSERVATIONS_PATH, data)


def classify_audit_subkind(text: str) -> str:
    """Classify audit claims with first-match priority.

    Priority is bug > ontology > architecture > general. Matches at
    word-character boundaries — `"use"` does not match `"user"`.
    """
    hay = str(text or "")
    if _matches_any(hay, _BUG_PATTERNS):
        return "bug"
    if _matches_any(hay, _ONTOLOGY_PATTERNS):
        return "ontology"
    if _matches_any(hay, _ARCHITECTURE_PATTERNS):
        return "architecture"
    return "general"


def _has_rule_language(text: str) -> bool:
    return _matches_any(str(text or ""), _RULE_PATTERNS)


def _is_monolith_owned_bug(*, origin_source: str, text: str) -> bool:
    if str(origin_source or "").strip().lower() == "auditor_monolith":
        return True
    return _matches_any(str(text or ""), _MONOLITH_RUNTIME_EVIDENCE_PATTERNS)


def _normalized_severity(kind: str, subkind: str, suggested: Any = None) -> int:
    try:
        raw = int(suggested)
    except (TypeError, ValueError):
        raw = 0
    if raw:
        return max(1, min(raw, 5))
    if kind == "proposal":
        return 4
    if kind == "continuity_pin" and subkind == "pending":
        return 4
    if kind == "audit_claim" and subkind == "bug":
        return 4
    if kind == "audit_claim" and subkind in {"ontology", "architecture"}:
        return 3
    if kind == "experiment_observation":
        return 2
    return 2


def _threshold_hours(kind: str, subkind: str) -> int:
    return _THRESHOLD_HOURS.get((kind, subkind), _THRESHOLD_HOURS.get((kind, "general"), 24 * 7))


def _record_for(state: dict, item_id: str) -> dict:
    items = state.setdefault("items", {})
    rec = items.get(item_id)
    if not isinstance(rec, dict):
        rec = {}
    return rec


def _status_for_record(record: dict) -> str:
    status = str(record.get("status") or "unresolved").strip().lower()
    return status if status else "unresolved"


def _is_hidden_by_state(record: dict, *, now: datetime) -> bool:
    status = _status_for_record(record)
    if status in {"resolved", "dismissed"}:
        return True
    if status == "snoozed":
        until = _parse_dt(record.get("snoozed_until"))
        if until is not None and until > now:
            return True
    return False


def _effective_severity(severity: int, record: dict) -> int:
    if bool(record.get("escalated")) or _status_for_record(record) == "escalated":
        return 5
    try:
        snooze_count = int(record.get("snooze_count", 0) or 0)
    except (TypeError, ValueError):
        snooze_count = 0
    return max(1, min(int(severity) + max(0, snooze_count), 5))


def _make_item(
    *,
    item_id: str,
    source: str,
    source_id: Any,
    kind: str,
    subkind: str,
    severity: int,
    created_at: str,
    summary: str,
    reason: str,
    allowed_actors: dict[str, tuple[str, ...] | list[str]],
    recommended_actions: tuple[str, ...] | list[str] = _ACTIONS_ALL,
    state: dict,
    now: datetime,
) -> ReviewItem:
    record = _record_for(state, item_id)
    allowed = {
        actor: [action for action in actions if action in _ACTIONS_ALL]
        for actor, actions in allowed_actors.items()
    }
    allowed_union = {a for actions in allowed.values() for a in actions}
    recommended = [a for a in recommended_actions if a in allowed_union]
    if not recommended:
        recommended = sorted(allowed_union)
    return ReviewItem(
        id=item_id,
        source=source,
        source_id=str(source_id),
        kind=kind,
        subkind=subkind,
        severity=max(1, min(int(severity), 5)),
        effective_severity=_effective_severity(max(1, min(int(severity), 5)), record),
        created_at=str(created_at or ""),
        age_days=round(_age_hours(created_at, now=now) / 24.0, 2),
        summary=_short(summary, 240),
        reason=_short(reason, 220),
        recommended_actions=list(recommended),
        allowed_actors=allowed,
        status=_status_for_record(record),
    )


def _acu_items(state: dict, *, now: datetime) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    try:
        from core.acu_store import ACUStore
        store = ACUStore()
        try:
            rows = store.retrieve(limit=_SOURCE_LIMIT)
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
    except Exception:
        return items

    for row in rows:
        canonical = str(row.get("canonical") or row.get("canonical_form") or "").strip()
        if not canonical:
            continue
        created = str(row.get("created_at") or row.get("last_seen") or "")
        subkind = classify_audit_subkind(canonical)
        if _age_hours(created, now=now) < _threshold_hours("audit_claim", subkind):
            continue
        acu_id = row.get("id")
        source_origin = str(row.get("source") or "")
        monolith_owned = _is_monolith_owned_bug(origin_source=source_origin, text=canonical)
        if subkind == "bug" and not monolith_owned:
            allowed = {"e": _ACTIONS_ALL, "monolith": _ACTIONS_SOFT}
        else:
            allowed = {"e": _ACTIONS_ALL, "monolith": _ACTIONS_SOFT}
        items.append(_make_item(
            item_id=f"acu:{acu_id}",
            source="acu",
            source_id=acu_id,
            kind="audit_claim",
            subkind=subkind,
            severity=_normalized_severity("audit_claim", subkind),
            created_at=created,
            summary=canonical,
            reason=f"active ACU older than {_threshold_hours('audit_claim', subkind)}h; subkind={subkind}",
            allowed_actors=allowed,
            recommended_actions=("escalate", "snooze", "resolve", "dismiss"),
            state=state,
            now=now,
        ))
    return items


def _proposal_items(state: dict, *, now: datetime) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    try:
        from core import proposals
        rows = proposals.list_proposals(limit=50)
    except Exception:
        return items
    for row in rows:
        if str(row.get("status") or "").strip().lower() != "pending":
            continue
        created = str(row.get("created_at") or "")
        if _age_hours(created, now=now) < _threshold_hours("proposal", "pending"):
            continue
        pid = row.get("id")
        target = str(row.get("target") or "?")
        section = str(row.get("section") or "?")
        rationale = str(row.get("rationale") or "")
        items.append(_make_item(
            item_id=f"proposal:{pid}",
            source="proposal",
            source_id=pid,
            kind="proposal",
            subkind="pending",
            severity=_normalized_severity("proposal", "pending"),
            created_at=created,
            summary=f"pending proposal #{pid} for {target}:{section}: {rationale}",
            reason=f"pending proposal older than {_threshold_hours('proposal', 'pending')}h",
            allowed_actors={"e": _ACTIONS_ALL, "monolith": _ACTIONS_SOFT},
            recommended_actions=("escalate", "snooze"),
            state=state,
            now=now,
        ))
    return items


def _continuity_items(state: dict, *, now: datetime) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    try:
        from core import continuity
        snap = continuity.read(include_retired=False)
        active = list(snap.get("active", []))
    except Exception:
        return items

    for pin in active:
        pid = pin.get("id")
        created = str(pin.get("created_at") or "")
        text = str(pin.get("text") or "")
        category = str(pin.get("category") or "lesson").strip().lower()
        reasons: list[str] = []
        subkind = category if category in {"pending", "anchor", "lesson"} else "lesson"
        severity = 2
        if category == "pending" and _age_hours(created, now=now) >= _threshold_hours("continuity_pin", "pending"):
            subkind = "pending"
            severity = max(severity, 4)
            reasons.append(f"pending pin older than {_threshold_hours('continuity_pin', 'pending')}h")
        if category == "anchor" and _has_rule_language(text) and _age_hours(created, now=now) >= _threshold_hours("continuity_pin", "anchor_rule"):
            subkind = "anchor_rule"
            severity = max(severity, 3)
            reasons.append(f"rule-like anchor older than {_threshold_hours('continuity_pin', 'anchor_rule')}h")
        if str(pin.get("evidence") or "").strip() and _age_hours(created, now=now) >= _threshold_hours("continuity_pin", "evidence"):
            if subkind not in {"pending", "anchor_rule"}:
                subkind = "evidence"
            severity = max(severity, 3)
            reasons.append(f"evidence-backed pin older than {_threshold_hours('continuity_pin', 'evidence')}h")
        if not reasons:
            continue
        items.append(_make_item(
            item_id=f"pin:{pid}",
            source="continuity",
            source_id=pid,
            kind="continuity_pin",
            subkind=subkind,
            severity=severity,
            created_at=created,
            summary=f"{category}({pid}): {text}",
            reason="; ".join(reasons),
            allowed_actors={"e": _ACTIONS_ALL, "monolith": _ACTIONS_SOFT},
            recommended_actions=("resolve", "snooze", "escalate", "dismiss"),
            state=state,
            now=now,
        ))
    return items


def _observation_items(state: dict, *, now: datetime) -> list[ReviewItem]:
    data = _load_observations()
    items: list[ReviewItem] = []
    for row in data.get("observations", []):
        if not isinstance(row, dict):
            continue
        created = str(row.get("created_at") or "")
        subkind = str(row.get("subkind") or "general").strip().lower() or "general"
        if _age_hours(created, now=now) < _threshold_hours("experiment_observation", subkind):
            continue
        oid = row.get("id")
        summary = str(row.get("summary") or "")
        reason = str(row.get("reason") or "")
        items.append(_make_item(
            item_id=f"observation:{oid}",
            source="observation",
            source_id=oid,
            kind="experiment_observation",
            subkind=subkind,
            severity=_normalized_severity("experiment_observation", subkind, row.get("severity")),
            created_at=created,
            summary=summary,
            reason=reason or f"experiment observation older than {_threshold_hours('experiment_observation', subkind)}h",
            allowed_actors={"e": _ACTIONS_ALL, "monolith": _ACTIONS_SOFT},
            recommended_actions=("resolve", "snooze", "escalate", "dismiss"),
            state=state,
            now=now,
        ))
    return items


def list_review_items(
    *,
    kind: str | None = None,
    subkind: str | None = None,
    limit: int | None = _DEFAULT_LIMIT,
    include_hidden: bool = False,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return unresolved review items, sorted by effective severity and age."""
    ref = now or _now()
    state = _load_state()
    generated: list[ReviewItem] = []
    for fn in (_acu_items, _proposal_items, _continuity_items, _observation_items):
        generated.extend(fn(state, now=ref))

    items: list[ReviewItem] = []
    for item in generated:
        record = _record_for(state, item.id)
        if not include_hidden and _is_hidden_by_state(record, now=ref):
            continue
        if kind and item.kind != kind:
            continue
        if subkind and item.subkind != subkind:
            continue
        items.append(item)

    items.sort(key=lambda i: (-i.effective_severity, -i.age_days, i.id))
    if limit is not None:
        try:
            max_items = max(1, int(limit))
        except (TypeError, ValueError):
            max_items = _DEFAULT_LIMIT
        items = items[:max_items]
    return [item.as_dict() for item in items]


def get_review_item(item_id: str, *, now: datetime | None = None) -> dict[str, Any] | None:
    target = str(item_id or "").strip()
    if not target:
        return None
    for item in list_review_items(limit=None, include_hidden=True, now=now):
        if item.get("id") == target:
            return item
    return None


def _normalize_actor(actor: str) -> str:
    raw = str(actor or "").strip().lower()
    if raw in {"e", "user", "user_e", "operator"}:
        return "e"
    if raw in {"monolith", "agent_monolith", "junior"} or raw.startswith("agent_"):
        return "monolith"
    return raw or "monolith"


def review_mark(
    item_id: str,
    action: str,
    *,
    actor: str = "monolith",
    note: str | None = None,
    snooze_hours: int | float | None = None,
    snoozed_until: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record review routing state for an item without mutating its source."""
    ref = now or _now()
    target = str(item_id or "").strip()
    act = str(action or "").strip().lower()
    if act not in _ACTIONS_ALL:
        raise ValueError(f"invalid review action {act!r}; valid: {', '.join(_ACTIONS_ALL)}")
    item = get_review_item(target, now=ref)
    if item is None:
        raise ValueError(f"review item {target!r} not found")

    normalized_actor = _normalize_actor(actor)
    actor_actions = item.get("allowed_actors", {}).get(normalized_actor, [])
    if act not in actor_actions:
        raise ReviewAuthorizationError(
            f"actor {normalized_actor!r} cannot {act} review item {target!r}"
        )

    state = _load_state()
    records = state.setdefault("items", {})
    rec = records.setdefault(target, {})
    rec["updated_at"] = _iso(ref)
    rec["updated_by"] = normalized_actor
    if note is not None:
        rec["notes"] = str(note)

    if act == "snooze":
        if snoozed_until:
            until = _parse_dt(snoozed_until)
            if until is None:
                raise ValueError("snoozed_until must be ISO-8601")
        else:
            try:
                hours = float(snooze_hours if snooze_hours is not None else _SNOOZE_HOURS_DEFAULT)
            except (TypeError, ValueError):
                hours = _SNOOZE_HOURS_DEFAULT
            until = ref + timedelta(hours=max(1.0, hours))
        rec["status"] = "snoozed"
        rec["snoozed_until"] = _iso(until)
        rec["snooze_count"] = int(rec.get("snooze_count", 0) or 0) + 1
        rec["escalated"] = bool(rec.get("escalated", False))
    elif act == "escalate":
        rec["status"] = "escalated"
        rec["escalated"] = True
        rec.pop("snoozed_until", None)
    elif act == "resolve":
        rec["status"] = "resolved"
        rec["resolved_at"] = _iso(ref)
        rec["escalated"] = bool(rec.get("escalated", False))
    elif act == "dismiss":
        rec["status"] = "dismissed"
        rec["dismissed_at"] = _iso(ref)
        rec["escalated"] = bool(rec.get("escalated", False))

    _save_state(state)
    return {"ok": True, "item_id": target, "action": act, "actor": normalized_actor, "state": dict(rec)}


def record_observation(
    summary: str,
    *,
    reason: str = "",
    severity: int = 2,
    subkind: str = "general",
    source: str = "monolith",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write an experiment observation into its source store."""
    body = _short(summary, 500)
    if not body:
        raise ValueError("observation summary is required")
    data = _load_observations()
    oid = int(data.get("next_id", 1) or 1)
    record = {
        "id": oid,
        "summary": body,
        "reason": _short(reason, 500),
        "severity": _normalized_severity("experiment_observation", subkind, severity),
        "subkind": str(subkind or "general").strip().lower() or "general",
        "source": str(source or "monolith").strip() or "monolith",
        "created_at": _iso(now or _now()),
    }
    data["next_id"] = oid + 1
    data.setdefault("observations", []).append(record)
    data["observations"] = data["observations"][-200:]
    _save_observations(data)
    return dict(record)


def review_summary(*, limit: int = 3, now: datetime | None = None) -> dict[str, Any]:
    items = list_review_items(limit=None, now=now)
    counts_by_kind: dict[str, int] = {}
    counts_by_subkind: dict[str, int] = {}
    oldest = 0.0
    for item in items:
        kind = str(item.get("kind") or "unknown")
        sub = f"{kind}.{item.get('subkind') or 'general'}"
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
        counts_by_subkind[sub] = counts_by_subkind.get(sub, 0) + 1
        try:
            oldest = max(oldest, float(item.get("age_days") or 0.0))
        except (TypeError, ValueError):
            pass
    top = sorted(
        items,
        key=lambda i: (-int(i.get("effective_severity", 0) or 0), -float(i.get("age_days", 0.0) or 0.0), str(i.get("id") or "")),
    )[: max(1, int(limit))]
    return {
        "unresolved_count": len(items),
        "counts_by_kind": counts_by_kind,
        "counts_by_subkind": counts_by_subkind,
        "oldest_unresolved_age_days": round(oldest, 2),
        "top": top,
    }


def _non_ephemeral_user_count(messages: list[dict]) -> int:
    return sum(
        1 for msg in messages
        if msg.get("role") == "user" and not msg.get("ephemeral")
    )


def _cadence_allows(messages: list[dict]) -> bool:
    count = _non_ephemeral_user_count(messages)
    if count <= 0:
        return False
    return count == 1 or (count - 1) % 15 == 0


def _format_review_queue(items: list[dict[str, Any]], total: int) -> str:
    top = items[:3]
    lines = [
        "[REVIEW QUEUE] attention-needed substrate items; NOT this turn's request",
    ]
    for item in top:
        lines.append(
            "- {id} {kind}.{subkind} sev={sev}: {summary} (reason: {reason})".format(
                id=item.get("id"),
                kind=item.get("kind"),
                subkind=item.get("subkind"),
                sev=item.get("effective_severity"),
                summary=_short(item.get("summary"), 120),
                reason=_short(item.get("reason"), 90),
            )
        )
    more = max(0, total - len(top))
    if more:
        lines.append(f"(+{more} more unresolved)")
    lines.append("[/REVIEW QUEUE]")
    return "\n".join(lines)


def contribute_section(messages: list[dict], config: dict):
    """Section-contributor variant for the ephemeral coalescer."""
    from core.ephemeral_coalescer import SectionResult
    if not _cadence_allows(messages):
        return None
    items = list_review_items(limit=None)
    eligible = [
        item for item in items
        if int(item.get("effective_severity", 0) or 0) >= 4
    ]
    if not eligible:
        return None
    block = _format_review_queue(eligible, len(items))
    return SectionResult(name="review_loop", text=block)
