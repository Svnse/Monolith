from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core import plan_reminders


@pytest.fixture
def reminder_store(tmp_path):
    plan_reminders.set_db_path(tmp_path / "turn_trace.sqlite3")
    yield
    plan_reminders.set_db_path(None)


def test_create_and_list_due_reminders(reminder_store) -> None:
    uid = plan_reminders.create_reminder(
        "reopen this",
        "2026-06-03T12:00:00+00:00",
        plan_uid="plan-1",
    )

    due = plan_reminders.list_due_reminders(now="2026-06-03T12:01:00+00:00")

    assert [item["reminder_uid"] for item in due] == [uid]
    assert due[0]["plan_uid"] == "plan-1"


def test_future_reminder_is_not_due(reminder_store) -> None:
    plan_reminders.create_reminder("later", "2026-06-04T12:00:00+00:00")

    assert plan_reminders.list_due_reminders(now="2026-06-03T12:00:00+00:00") == []


def test_mark_reminder_seen_hides_from_due(reminder_store) -> None:
    uid = plan_reminders.create_reminder("done", datetime(2026, 6, 3, tzinfo=timezone.utc))

    plan_reminders.mark_reminder_seen(uid, seen_at="2026-06-03T12:00:00+00:00")

    assert plan_reminders.list_due_reminders(now="2026-06-04T00:00:00+00:00") == []
    assert plan_reminders.list_reminders()[0]["status"] == "seen"
