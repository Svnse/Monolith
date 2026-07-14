from __future__ import annotations

from core import health


def test_runtime_health_rolls_up_warn_before_stale(monkeypatch) -> None:
    monkeypatch.setattr(health, "check_config", lambda: health.RuntimeHealthCheck("config", "ok", "ok"))
    monkeypatch.setattr(health, "check_model_endpoint", lambda probe=False: health.RuntimeHealthCheck("model", "warn", "warn"))
    monkeypatch.setattr(health, "check_turn_trace", lambda: health.RuntimeHealthCheck("trace", "stale", "stale"))
    monkeypatch.setattr(health, "check_monosearch", lambda: health.RuntimeHealthCheck("monosearch", "ok", "ok"))
    monkeypatch.setattr(health, "check_acu_store", lambda: health.RuntimeHealthCheck("acu", "ok", "ok"))
    monkeypatch.setattr(health, "check_planner", lambda: health.RuntimeHealthCheck("planner", "ok", "ok"))
    monkeypatch.setattr(health, "check_monoexplore", lambda: health.RuntimeHealthCheck("monoexplore", "ok", "ok"))
    monkeypatch.setattr(health, "check_rating_telemetry", lambda: health.RuntimeHealthCheck("rating", "ok", "ok"))
    monkeypatch.setattr(health, "check_plan_reminders", lambda: health.RuntimeHealthCheck("reminders", "ok", "ok"))

    result = health.get_runtime_health()

    assert result.status == "warn"
    assert result.to_dict()["checks"][1]["status"] == "warn"
