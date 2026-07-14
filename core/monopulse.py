"""MonoPulse V0: a pull-only runtime attention layer.

MonoSearch stays the retrieval fabric. MonoPulse is the opinionated question:
"what should I look at now?" It composes recurrent failures, plan/reminder
stalls, drift signals, runtime health, recent traces, and investigations into a
small ranked report. It never injects prompt context by itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

_MODES = frozenset({"pulse", "hotspots", "stalled", "drift", "changed"})
_SEVERITY_RANK = {"fail": 3, "warn": 2, "info": 1}


@dataclass(frozen=True)
class PulseItem:
    kind: str
    severity: str
    title: str
    detail: str = ""
    source: str = ""
    ref: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PulseReport:
    mode: str
    generated_at: str
    items: tuple[PulseItem, ...]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "generated_at": self.generated_at,
            "items": [item.to_dict() for item in self.items],
            "summary": dict(self.summary),
        }


def available_modes() -> tuple[str, ...]:
    return tuple(sorted(_MODES))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_limit(value: int | str | None, default: int = 10, maximum: int = 50) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, maximum))


def _safe(fn: Callable[[], Any], fallback: Any) -> Any:
    try:
        return fn()
    except Exception:
        return fallback


def _safe_items(fn: Callable[[], Iterable[PulseItem]]) -> list[PulseItem]:
    return list(_safe(fn, []))


def _severity_key(item: PulseItem) -> tuple[int, float, str]:
    return (_SEVERITY_RANK.get(item.severity, 0), float(item.score or 0.0), item.title.lower())


def _dedupe(items: Iterable[PulseItem]) -> list[PulseItem]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[PulseItem] = []
    for item in items:
        key = (item.kind, item.source, item.ref, item.title)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _rank(items: Iterable[PulseItem], limit: int) -> tuple[PulseItem, ...]:
    ranked = _dedupe(items)
    ranked.sort(key=_severity_key, reverse=True)
    return tuple(ranked[:_coerce_limit(limit)])


def _summary(mode: str, items: tuple[PulseItem, ...]) -> dict[str, Any]:
    by_severity = {key: 0 for key in ("fail", "warn", "info")}
    by_kind: dict[str, int] = {}
    sources: dict[str, int] = {}
    for item in items:
        by_severity[item.severity] = by_severity.get(item.severity, 0) + 1
        by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
        if item.source:
            sources[item.source] = sources.get(item.source, 0) + 1
    if by_severity.get("fail", 0):
        status = "fail"
    elif by_severity.get("warn", 0):
        status = "warn"
    elif items:
        status = "info"
    else:
        status = "quiet"
    return {
        "mode": mode,
        "status": status,
        "count": len(items),
        "by_severity": by_severity,
        "by_kind": by_kind,
        "sources": sources,
    }


def _report(mode: str, items: Iterable[PulseItem], limit: int) -> PulseReport:
    ranked = _rank(items, limit)
    return PulseReport(
        mode=mode,
        generated_at=_now_iso(),
        items=ranked,
        summary=_summary(mode, ranked),
    )


def _init_monosearch() -> None:
    from core.monosearch.bootstrap import init_monosearch

    init_monosearch()


def _clean_key(key: object) -> str:
    text = str(key or "").strip()
    if not text:
        return "unknown"
    replacements = (
        ("stage_error:", "stage error: "),
        ("health:", "health: "),
        ("turn_shape:", "turn shape: "),
        ("investigation:", "investigation: "),
        ("reminder:", "reminder: "),
    )
    for old, new in replacements:
        if text.startswith(old):
            return new + text[len(old):].replace(":", " / ")
    if text == "due_reminder":
        return "due reminder"
    return text.replace("_", " ")


def _row_count(row: dict[str, Any]) -> int:
    try:
        return int(row.get("count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _row_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("salience", row.get("count", 0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _opaque_recurrence(row: dict[str, Any]) -> bool:
    source = str(row.get("source") or "")
    key = str(row.get("recurrence_key") or "")
    if source != "canonical_log":
        return False
    if key.startswith(("user_message|", "assistant_message|")):
        return True
    return len(key) >= 40 and all(ch in "0123456789abcdef" for ch in key.lower()[:40])


def _recurrence_item(
    row: dict[str, Any],
    *,
    kind: str,
    title_prefix: str,
    failure: bool = False,
) -> PulseItem | None:
    if _opaque_recurrence(row):
        return None
    key = str(row.get("recurrence_key") or "").strip()
    if not key:
        return None
    count = _row_count(row)
    score = _row_score(row)
    source = str(row.get("source") or "")
    severity = "fail" if failure and count >= 3 else "warn"
    return PulseItem(
        kind=kind,
        severity=severity,
        title=f"{title_prefix}: {_clean_key(key)}",
        detail=f"x{count}, salience {score:.2f}",
        source=source,
        ref=key,
        score=score,
        metadata=dict(row),
    )


def hotspots(limit: int = 10) -> PulseReport:
    """Recurring faults and other high-salience recurrence keys."""
    limit = _coerce_limit(limit)
    items: list[PulseItem] = []

    def _read() -> list[PulseItem]:
        from core.monosearch import service

        _init_monosearch()
        out: list[PulseItem] = []
        for row in service.failing(limit=limit):
            item = _recurrence_item(
                row,
                kind="hotspot",
                title_prefix="Recurring failure",
                failure=True,
            )
            if item is not None:
                out.append(item)
        for row in service.recurring(limit=max(limit * 2, 10)):
            if str(row.get("source") or "") == "fault_traces":
                continue
            item = _recurrence_item(row, kind="hotspot", title_prefix="Recurring signal")
            if item is not None:
                out.append(item)
        return out

    items.extend(_safe_items(_read))
    return _report("hotspots", items, limit)


def _step_label(step: dict[str, Any]) -> str:
    seq = step.get("seq")
    verb = str(step.get("verb") or "").strip()
    target = str(step.get("target") or "").strip()
    prefix = f"{seq}. " if seq is not None else ""
    label = " ".join(part for part in (verb, target) if part)
    return prefix + (label or "unnamed step")


def _stalled_plan_items(limit: int) -> list[PulseItem]:
    from core import plans

    items: list[PulseItem] = []
    active = plans.get_active_plan()
    if active is None:
        return items

    steps = list(active.get("steps") or [])
    plan_uid = str(active.get("plan_uid") or "")
    goal = str(active.get("goal") or "")
    status = str(active.get("status") or "")
    failed = [s for s in steps if s.get("status") == "failed"]
    pending = [s for s in steps if s.get("status") == "pending"]
    ready = plans.next_ready_steps(plan_uid) if plan_uid else []

    for step in failed[:limit]:
        items.append(
            PulseItem(
                kind="stalled",
                severity="fail",
                title=f"Plan step failed: {_step_label(step)}",
                detail=goal,
                source="plans",
                ref=plan_uid,
                score=95.0,
                metadata={"plan_uid": plan_uid, "step": dict(step), "status": status},
            )
        )

    if pending and not ready and not failed and status in {"active", "proposed"}:
        items.append(
            PulseItem(
                kind="stalled",
                severity="warn",
                title="Plan has no ready pending step",
                detail=goal,
                source="plans",
                ref=plan_uid,
                score=80.0,
                metadata={"plan_uid": plan_uid, "pending": len(pending), "status": status},
            )
        )
    for step in ready[: max(1, limit - len(items))]:
        items.append(
            PulseItem(
                kind="stalled",
                severity="info",
                title=f"Ready plan step: {_step_label(step)}",
                detail=goal,
                source="plans",
                ref=plan_uid,
                score=35.0,
                metadata={"plan_uid": plan_uid, "step": dict(step), "status": status},
            )
        )
    return items


def _stalled_reminder_items(limit: int) -> list[PulseItem]:
    from core import plan_reminders

    out: list[PulseItem] = []
    for row in plan_reminders.list_due_reminders(limit=limit):
        out.append(
            PulseItem(
                kind="stalled",
                severity="warn",
                title="Due reminder",
                detail=str(row.get("message") or ""),
                source="plan_reminders",
                ref=str(row.get("reminder_uid") or ""),
                score=85.0,
                metadata=dict(row),
            )
        )
    return out


def _stalled_investigation_items(limit: int) -> list[PulseItem]:
    from core import investigation_runs

    out: list[PulseItem] = []
    for run in investigation_runs.list_investigations(limit=limit):
        if run.status == "done" or run.source_refs:
            continue
        out.append(
            PulseItem(
                kind="stalled",
                severity="warn",
                title="Investigation has no sources yet",
                detail=run.goal,
                source="investigations",
                ref=run.run_id,
                score=70.0,
                metadata=run.to_dict(),
            )
        )
    return out


def stalled(limit: int = 10) -> PulseReport:
    """Open plans, due reminders, and investigations that need next action."""
    limit = _coerce_limit(limit)
    items: list[PulseItem] = []
    items.extend(_safe_items(lambda: _stalled_reminder_items(limit)))
    items.extend(_safe_items(lambda: _stalled_plan_items(limit)))
    items.extend(_safe_items(lambda: _stalled_investigation_items(limit)))
    return _report("stalled", items, limit)


def _coherence_items() -> list[PulseItem]:
    from core import monoexplore

    report = monoexplore.coherence_report()
    verdict = str(report.get("verdict") or "UNKNOWN").upper()
    if verdict == "GREEN":
        return []
    severity = "fail" if verdict == "RED" else "warn"
    return [
        PulseItem(
            kind="drift",
            severity=severity,
            title=f"MonoExplore coherence {verdict}",
            detail=str(report.get("reason") or ""),
            source="monoexplore",
            ref="coherence",
            score=90.0 if severity == "fail" else 75.0,
            metadata=dict(report),
        )
    ]


def _bearing_drift_items() -> list[PulseItem]:
    from addons.system.bearing import store as bearing_store

    pending = bearing_store.get_pending_rejection()
    streak = bearing_store.get_rejection_streak()
    if pending:
        failed_rules = pending.get("failed_rules") or []
        detail = str(pending.get("detail") or ", ".join(str(v) for v in failed_rules) or "pending rejection")
        return [
            PulseItem(
                kind="drift",
                severity="fail" if streak >= 3 else "warn",
                title="Bearing update rejection pending",
                detail=detail,
                source="bearing",
                ref=str(pending.get("turn_id") or ""),
                score=85.0 + min(float(streak), 10.0),
                metadata=dict(pending) | {"rejection_streak": streak},
            )
        ]
    if streak > 0:
        return [
            PulseItem(
                kind="drift",
                severity="warn",
                title="Bearing rejection streak active",
                detail=f"streak {streak}",
                source="bearing",
                ref="rejection_streak",
                score=60.0 + min(float(streak), 10.0),
                metadata={"rejection_streak": streak},
            )
        ]
    return []


def _identity_signal_items(limit: int) -> list[PulseItem]:
    from core.monosearch import service

    _init_monosearch()
    items: list[PulseItem] = []
    for rec in service.unresolved(limit=limit):
        items.append(
            PulseItem(
                kind="drift",
                severity="warn",
                title="Unresolved self-claim",
                detail=rec.text[:220],
                source=rec.source,
                ref=rec.namespaced_id,
                score=55.0,
                metadata=dict(rec.metadata),
            )
        )
    for rec in service.pulling(limit=limit):
        items.append(
            PulseItem(
                kind="drift",
                severity="info",
                title="Curiosity pull",
                detail=rec.text[:220],
                source=rec.source,
                ref=rec.namespaced_id,
                score=25.0,
                metadata=dict(rec.metadata),
            )
        )
    return items


def drift(limit: int = 10) -> PulseReport:
    """Coherence, bearing rejection, and unresolved identity signals."""
    limit = _coerce_limit(limit)
    items: list[PulseItem] = []
    items.extend(_safe_items(_coherence_items))
    items.extend(_safe_items(_bearing_drift_items))
    items.extend(_safe_items(lambda: _identity_signal_items(limit)))
    return _report("drift", items, limit)


def _health_change_items(limit: int) -> list[PulseItem]:
    from core import health

    items: list[PulseItem] = []
    runtime = health.get_runtime_health(probe_endpoint_now=False)
    for check in runtime.checks:
        if check.status == "ok":
            continue
        severity = "fail" if check.status == "fail" else "warn"
        items.append(
            PulseItem(
                kind="changed",
                severity=severity,
                title=f"Runtime health {check.status}: {check.name}",
                detail=check.message,
                source="runtime_health",
                ref=check.name,
                score=78.0 if severity == "fail" else 62.0,
                metadata=check.to_dict(),
            )
        )
    return items[:limit]


def _recent_turn_items(limit: int) -> list[PulseItem]:
    from core import turn_trace

    items: list[PulseItem] = []
    for row in turn_trace.list_recent_turns(limit=min(limit, 10)):
        errored = int(getattr(row, "errored_stage_count", 0) or 0)
        if errored <= 0:
            items.append(
                PulseItem(
                    kind="changed",
                    severity="info",
                    title="Recent turn recorded",
                    detail=f"{row.stage_count} stages, {row.total_chars} chars",
                    source="turn_trace",
                    ref=row.turn_id,
                    score=15.0,
                    metadata={
                        "turn_id": row.turn_id,
                        "captured_at": row.captured_at,
                        "backend": row.backend,
                        "stage_count": row.stage_count,
                        "errored_stage_count": row.errored_stage_count,
                    },
                )
            )
            continue
        items.append(
            PulseItem(
                kind="changed",
                severity="warn",
                title="Recent turn had stage errors",
                detail=f"{errored}/{row.stage_count} stages errored",
                source="turn_trace",
                ref=row.turn_id,
                score=65.0 + errored,
                metadata={
                    "turn_id": row.turn_id,
                    "captured_at": row.captured_at,
                    "backend": row.backend,
                    "stage_count": row.stage_count,
                    "errored_stage_count": row.errored_stage_count,
                },
            )
        )
    return items


def _investigation_change_items(limit: int) -> list[PulseItem]:
    from core import investigation_runs

    items: list[PulseItem] = []
    for run in investigation_runs.list_investigations(limit=limit):
        source_count = len(run.source_refs)
        severity = "warn" if run.status != "done" and source_count == 0 else "info"
        items.append(
            PulseItem(
                kind="changed",
                severity=severity,
                title=f"Investigation {run.status}: {run.goal}",
                detail=f"{source_count} source{'s' if source_count != 1 else ''}",
                source="investigations",
                ref=run.run_id,
                score=45.0 if severity == "warn" else 20.0,
                metadata=run.to_dict(),
            )
        )
    return items


def changed(limit: int = 10) -> PulseReport:
    """Recent runtime movement: health, traces, and investigations."""
    limit = _coerce_limit(limit)
    items: list[PulseItem] = []
    items.extend(_safe_items(lambda: _health_change_items(limit)))
    items.extend(_safe_items(lambda: _recent_turn_items(limit)))
    items.extend(_safe_items(lambda: _investigation_change_items(limit)))
    return _report("changed", items, limit)


def pulse(limit: int = 12) -> PulseReport:
    """The combined attention view across all MonoPulse modes."""
    limit = _coerce_limit(limit, default=12)
    sublimit = max(3, min(limit, 8))
    items: list[PulseItem] = []
    for fn in (hotspots, stalled, drift, changed):
        items.extend(fn(sublimit).items)
    return _report("pulse", items, limit)


def run(mode: str = "pulse", *, limit: int = 12) -> PulseReport:
    selected = str(mode or "pulse").strip().lower()
    if selected not in _MODES:
        raise ValueError(f"unknown MonoPulse mode {selected!r}; use one of: {', '.join(available_modes())}")
    if selected == "pulse":
        return pulse(limit)
    if selected == "hotspots":
        return hotspots(limit)
    if selected == "stalled":
        return stalled(limit)
    if selected == "drift":
        return drift(limit)
    return changed(limit)


def format_report(report: PulseReport) -> str:
    status = str(report.summary.get("status") or "quiet")
    if not report.items:
        return f"[monopulse:{report.mode} none status={status}]"
    lines = [f"[monopulse:{report.mode} count={len(report.items)} status={status}]"]
    for item in report.items:
        where = ""
        if item.source or item.ref:
            where = f" [{item.source}{':' if item.source and item.ref else ''}{item.ref}]"
        detail = f" - {item.detail}" if item.detail else ""
        lines.append(f"  {item.severity.upper()} {item.title}{detail}{where}")
    return "\n".join(lines)
