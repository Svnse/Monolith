from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeHealthCheck:
    name: str
    status: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeHealth:
    status: str
    checks: tuple[RuntimeHealthCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "checks": [check.to_dict() for check in self.checks]}


def _check(name: str, fn: Callable[[], RuntimeHealthCheck]) -> RuntimeHealthCheck:
    try:
        return fn()
    except Exception as exc:
        return RuntimeHealthCheck(name, "fail", str(exc))


def check_config() -> RuntimeHealthCheck:
    from core.config import get_config

    cfg = get_config()
    return RuntimeHealthCheck(
        "config",
        "ok",
        "config loaded",
        {"backend": cfg.llm.backend, "context_profile": cfg.llm.context_profile},
    )


def check_model_endpoint(*, probe: bool = False) -> RuntimeHealthCheck:
    from core.config import get_config
    from core.model_ops import probe_endpoint

    cfg = get_config().llm
    if not cfg.api_base:
        return RuntimeHealthCheck("model_endpoint", "warn", "no API base configured")
    if not probe:
        return RuntimeHealthCheck("model_endpoint", "ok", "API base configured", {"api_base": cfg.api_base})
    result = probe_endpoint(cfg.api_base, api_key=cfg.api_key, timeout=3)
    status = "ok" if result.reachable else "fail"
    return RuntimeHealthCheck(
        "model_endpoint",
        status,
        result.status if result.reachable else (result.error or result.status),
        result.to_dict(),
    )


def check_turn_trace() -> RuntimeHealthCheck:
    from core import turn_trace

    rows = turn_trace.list_recent_turns(limit=1)
    status = "ok" if rows else "stale"
    return RuntimeHealthCheck("turn_trace", status, "recent turns available" if rows else "no recent turns", {"count": len(rows)})


def check_acu_store() -> RuntimeHealthCheck:
    from core.acu_store import ACUStore

    count = ACUStore().count()
    status = "ok" if count > 0 else "stale"
    return RuntimeHealthCheck("acu_store", status, f"{count} ACU rows", {"count": count})


def check_planner() -> RuntimeHealthCheck:
    from core import plans

    active = plans.get_active_plan()
    if active is None:
        return RuntimeHealthCheck("planner", "stale", "no active plan")
    return RuntimeHealthCheck(
        "planner",
        "ok",
        f"active plan: {active.get('goal', '')}",
        {"plan_uid": active.get("plan_uid"), "steps": len(active.get("steps", []))},
    )


def check_monoexplore() -> RuntimeHealthCheck:
    try:
        from core import monoexplore
    except Exception as exc:
        return RuntimeHealthCheck("monoexplore", "warn", f"unavailable: {exc}")
    report = monoexplore.coherence_report()
    verdict = str(report.get("verdict", "UNKNOWN"))
    status = "ok" if verdict == "GREEN" else "warn"
    return RuntimeHealthCheck("monoexplore", status, str(report.get("reason", verdict)), report)


def check_rating_telemetry() -> RuntimeHealthCheck:
    from core import turn_trace

    summary = turn_trace.recent_ratings_summary(window=10)
    count = int(summary.get("count", 0) or 0)
    status = "ok" if count else "stale"
    return RuntimeHealthCheck("rating_telemetry", status, f"{count} recent ratings", summary)


def check_plan_reminders() -> RuntimeHealthCheck:
    from core import plan_reminders

    due = plan_reminders.list_due_reminders(limit=10)
    status = "warn" if due else "ok"
    return RuntimeHealthCheck("plan_reminders", status, f"{len(due)} due reminders", {"due": due})


def check_monosearch() -> RuntimeHealthCheck:
    from core.monosearch.bootstrap import init_monosearch
    from core.monosearch import registry, salience

    init_monosearch()
    salience.ensure_schema()
    adapters = [adapter.name for adapter in registry.all_adapters()]
    expected = {
        "fault_traces",
        "canonical_log",
        "turn_trace",
        "acatalepsy-acus",
        "continuity",
        "bearing",
        "identity_signals",
        "identity",
    }
    missing = sorted(expected - set(adapters))
    if missing:
        return RuntimeHealthCheck(
            "monosearch",
            "warn",
            f"{len(adapters)} adapters registered; missing {', '.join(missing)}",
            {"adapters": adapters, "missing": missing},
        )
    return RuntimeHealthCheck("monosearch", "ok", f"{len(adapters)} adapters registered", {"adapters": adapters})


def get_runtime_health(*, probe_endpoint_now: bool = False) -> RuntimeHealth:
    checks = (
        _check("config", check_config),
        _check("model_endpoint", lambda: check_model_endpoint(probe=probe_endpoint_now)),
        _check("turn_trace", check_turn_trace),
        _check("monosearch", check_monosearch),
        _check("acu_store", check_acu_store),
        _check("planner", check_planner),
        _check("monoexplore", check_monoexplore),
        _check("rating_telemetry", check_rating_telemetry),
        _check("plan_reminders", check_plan_reminders),
    )
    statuses = {check.status for check in checks}
    if "fail" in statuses:
        overall = "fail"
    elif "warn" in statuses:
        overall = "warn"
    elif "stale" in statuses:
        overall = "stale"
    else:
        overall = "ok"
    return RuntimeHealth(overall, checks)
