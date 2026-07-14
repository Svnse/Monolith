from __future__ import annotations

from core.monopulse import PulseItem, PulseReport


def test_hotspots_filters_opaque_canonical_log_recurrence(monkeypatch):
    from core import monopulse
    from core.monosearch import service

    monkeypatch.setattr(monopulse, "_init_monosearch", lambda: None)
    monkeypatch.setattr(
        service,
        "failing",
        lambda limit=10: [
            {
                "recurrence_key": "tool_no_fire",
                "source": "fault_traces",
                "count": 3,
                "salience": 2.5,
            }
        ],
    )
    monkeypatch.setattr(
        service,
        "recurring",
        lambda limit=10: [
            {
                "recurrence_key": "user_message|abcdef1234567890",
                "source": "canonical_log",
                "count": 99,
                "salience": 99.0,
            },
            {
                "recurrence_key": "stage_error:runtime_state_projection",
                "source": "stage_traces",
                "count": 2,
                "salience": 2.0,
            },
        ],
    )

    report = monopulse.hotspots(limit=10)
    titles = [item.title for item in report.items]

    assert "Recurring failure: tool no fire" in titles
    assert "Recurring signal: stage error: runtime_state_projection" in titles
    assert all("canonical" not in item.source for item in report.items)
    assert report.items[0].severity == "fail"


def test_pulse_combines_modes_and_sorts_by_severity(monkeypatch):
    from core import monopulse

    def _report(mode: str, item: PulseItem) -> PulseReport:
        return PulseReport(mode=mode, generated_at="now", items=(item,), summary={})

    monkeypatch.setattr(
        monopulse,
        "hotspots",
        lambda limit=10: _report(
            "hotspots",
            PulseItem("hotspot", "warn", "warn item", score=50.0),
        ),
    )
    monkeypatch.setattr(
        monopulse,
        "stalled",
        lambda limit=10: _report(
            "stalled",
            PulseItem("stalled", "fail", "fail item", score=10.0),
        ),
    )
    monkeypatch.setattr(
        monopulse,
        "drift",
        lambda limit=10: _report(
            "drift",
            PulseItem("drift", "info", "info item", score=100.0),
        ),
    )
    monkeypatch.setattr(
        monopulse,
        "changed",
        lambda limit=10: _report(
            "changed",
            PulseItem("changed", "warn", "warn high", score=90.0),
        ),
    )

    report = monopulse.pulse(limit=3)

    assert [item.title for item in report.items] == ["fail item", "warn high", "warn item"]
    assert report.summary["status"] == "fail"


def test_format_report_is_compact_and_includes_source_refs():
    from core import monopulse

    report = PulseReport(
        mode="drift",
        generated_at="now",
        items=(
            PulseItem(
                kind="drift",
                severity="warn",
                title="Bearing update rejection pending",
                detail="D1 failed",
                source="bearing",
                ref="turn-1",
            ),
        ),
        summary={"status": "warn"},
    )

    text = monopulse.format_report(report)

    assert text.startswith("[monopulse:drift count=1 status=warn]")
    assert "WARN Bearing update rejection pending - D1 failed [bearing:turn-1]" in text


def test_unknown_mode_raises_clear_error():
    from core import monopulse

    try:
        monopulse.run("bogus")
    except ValueError as exc:
        assert "unknown MonoPulse mode" in str(exc)
    else:
        raise AssertionError("unknown MonoPulse mode did not raise")
