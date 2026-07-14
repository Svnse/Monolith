from __future__ import annotations

from datetime import datetime, timedelta, timezone

import core.turn_trace as tt
from core.health import RuntimeHealth, RuntimeHealthCheck
from core.monosearch.record import EvidenceTier


def test_stage_trace_adapter_maps_errored_stages_to_recurrence_keys():
    from core.monosearch.adapters.stage_traces import StageTraceAdapter

    turn_id = "stage-turn-1"
    tt.record_frame(tt.FrameTraceRecord(
        turn_id=turn_id,
        captured_at="2026-06-03T15:00:00+00:00",
        backend="test",
        engine_key="ek",
        gen_id=1,
        final_messages=(),
        system_prompt_chars=10,
        user_prompt_chars=20,
        total_chars=30,
    ))
    tt.record_stage(tt.StageTraceRecord(
        turn_id=turn_id,
        seq=1,
        stage_name="runtime_state_projection",
        stage_kind="prompt_stage",
        entered_at="2026-06-03T15:00:00+00:00",
        exited_at="2026-06-03T15:00:01+00:00",
        outcome="errored",
        outcome_reason="fixture failure",
        messages_in=1,
        messages_out=1,
    ))

    records = StageTraceAdapter().search("runtime_state", {}, 10)

    assert records
    rec = records[0]
    assert rec.namespaced_id == f"stage:{turn_id}:1"
    assert rec.source == "stage_traces"
    assert rec.evidence_tier == EvidenceTier.TELEMETRY
    assert rec.recurrence_key == "stage_error:runtime_state_projection"


def test_plan_reminder_adapter_marks_due_pending_reminders(tmp_path):
    from core import plan_reminders
    from core.monosearch.adapters.plan_reminders import PlanReminderAdapter

    plan_reminders.set_db_path(tmp_path / "turn_trace.sqlite3")
    try:
        due = datetime.now(timezone.utc) - timedelta(minutes=5)
        uid = plan_reminders.create_reminder("resume the stuck plan", due)
        recs = PlanReminderAdapter().list({"due": True}, 10)
    finally:
        plan_reminders.set_db_path(None)

    rec = next(r for r in recs if r.namespaced_id == f"reminder:{uid}")
    assert rec.recurrence_key == "due_reminder"
    assert rec.metadata["due"] is True


def test_investigation_adapter_lists_active_runs(tmp_path, monkeypatch):
    from core import investigation_runs
    from core.monosearch.adapters.investigations import InvestigationAdapter

    monkeypatch.setattr(investigation_runs, "INVESTIGATION_DIR", tmp_path)
    run = investigation_runs.create_investigation("map dark systems")

    recs = InvestigationAdapter().list({}, 10)

    rec = next(r for r in recs if r.namespaced_id == f"investigation:{run.run_id}")
    assert rec.recurrence_key == "investigation:active"
    assert rec.metadata["source_count"] == 0


def test_lag_watch_adapter_maps_turn_shape_recurrence(tmp_path, monkeypatch):
    import core.monosearch.adapters.lag_watch as lag_mod
    from core.monosearch.adapters.lag_watch import LagWatchAdapter

    path = tmp_path / "lag_watch.jsonl"
    path.write_text(
        '{"ts":"2026-06-03T16:00:00+00:00","user_preview":"build it",'
        '"system_class":{"task_type":"build","effort_tier":"deep"},'
        '"llm_class":{"task_type":"build"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(lag_mod, "_LOG_PATH", path)

    rec = LagWatchAdapter().list({}, 10)[0]

    assert rec.namespaced_id == "lag:1"
    assert rec.recurrence_key == "turn_shape:build:deep"
    assert rec.metadata["system_class"]["task_type"] == "build"


def test_health_adapter_surfaces_non_ok_checks(monkeypatch):
    import core.monosearch.adapters.health as health_mod
    from core.monosearch.adapters.health import HealthAdapter

    runtime = RuntimeHealth(
        "warn",
        (
            RuntimeHealthCheck("config", "ok", "loaded"),
            RuntimeHealthCheck("planner", "stale", "no active plan"),
        ),
    )
    monkeypatch.setattr(health_mod, "get_runtime_health", lambda probe_endpoint_now=False: runtime)

    recs = HealthAdapter().list({}, 10)
    stale = next(r for r in recs if r.namespaced_id == "health:planner")

    assert stale.recurrence_key == "health:planner:stale"
    assert stale.metadata["status"] == "stale"
