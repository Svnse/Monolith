"""Tests for addons.system.bearing.frame_observe (observable_frame_fastpath_v0).

Flag: MONOLITH_OBSERVABLE_FRAME_V0
Ledger: monkeypatched via frame_observe._LEDGER to tmp_path isolation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


# ---------------------------------------------------------------------------
# observe() — pure, no I/O
# ---------------------------------------------------------------------------

class TestObservePure:
    def test_detects_bearing_update_present(self):
        from addons.system.bearing import frame_observe
        result = frame_observe.observe("hello <bearing_update>...</bearing_update> world")
        assert result["has_bearing_update"] is True

    def test_detects_bearing_update_absent(self):
        from addons.system.bearing import frame_observe
        result = frame_observe.observe("just plain text")
        assert result["has_bearing_update"] is False

    def test_bearing_update_case_insensitive(self):
        from addons.system.bearing import frame_observe
        result = frame_observe.observe("text <BEARING_UPDATE>...</BEARING_UPDATE>")
        assert result["has_bearing_update"] is True

    def test_frame_extraction(self):
        from addons.system.bearing import frame_observe
        result = frame_observe.observe("before <frame>  my frame text  </frame> after")
        assert result["has_frame"] is True
        assert result["observed_frame"] == "my frame text"

    def test_frame_absent(self):
        from addons.system.bearing import frame_observe
        result = frame_observe.observe("no frame tag here")
        assert result["has_frame"] is False
        assert result["observed_frame"] == ""

    def test_frame_capped_at_600(self):
        from addons.system.bearing import frame_observe
        long_text = "x" * 700
        raw = f"<frame>{long_text}</frame>"
        result = frame_observe.observe(raw)
        assert result["has_frame"] is True
        assert len(result["observed_frame"]) == 600

    def test_raw_len_correct(self):
        from addons.system.bearing import frame_observe
        raw = "hello world"
        result = frame_observe.observe(raw)
        assert result["raw_len"] == len(raw)

    def test_empty_raw(self):
        from addons.system.bearing import frame_observe
        result = frame_observe.observe("")
        assert result["has_bearing_update"] is False
        assert result["has_frame"] is False
        assert result["observed_frame"] == ""
        assert result["raw_len"] == 0

    def test_none_raw_treated_as_empty(self):
        from addons.system.bearing import frame_observe
        # observe coerces None via `raw or ""`
        result = frame_observe.observe(None)  # type: ignore[arg-type]
        assert result["raw_len"] == 0
        assert result["has_bearing_update"] is False

    def test_frame_multiline(self):
        from addons.system.bearing import frame_observe
        raw = "<frame>\nline one\nline two\n</frame>"
        result = frame_observe.observe(raw)
        assert result["has_frame"] is True
        assert "line one" in result["observed_frame"]


# ---------------------------------------------------------------------------
# enabled() flag
# ---------------------------------------------------------------------------

class TestEnabledFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MONOLITH_OBSERVABLE_FRAME_V0", raising=False)
        from addons.system.bearing import frame_observe
        assert frame_observe.enabled() is False

    def test_on_with_1(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "1")
        from addons.system.bearing import frame_observe
        assert frame_observe.enabled() is True

    def test_on_with_true(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "true")
        from addons.system.bearing import frame_observe
        assert frame_observe.enabled() is True

    def test_on_with_yes(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "yes")
        from addons.system.bearing import frame_observe
        assert frame_observe.enabled() is True

    def test_on_with_on(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "on")
        from addons.system.bearing import frame_observe
        assert frame_observe.enabled() is True

    def test_off_with_0(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "0")
        from addons.system.bearing import frame_observe
        assert frame_observe.enabled() is False

    def test_off_with_false(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "false")
        from addons.system.bearing import frame_observe
        assert frame_observe.enabled() is False


# ---------------------------------------------------------------------------
# record() — flag OFF writes nothing
# ---------------------------------------------------------------------------

class TestRecordFlagOff:
    def test_flag_off_no_file_created(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MONOLITH_OBSERVABLE_FRAME_V0", raising=False)
        from addons.system.bearing import frame_observe
        ledger_path = tmp_path / "frame.ledger.jsonl"
        monkeypatch.setattr(frame_observe, "_LEDGER", ledger_path)
        frame_observe.record("tid-1", "some raw text", bu_outcome="none", current_frame="cf")
        assert not ledger_path.exists()

    def test_flag_off_empty_existing_file_unchanged(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MONOLITH_OBSERVABLE_FRAME_V0", raising=False)
        from addons.system.bearing import frame_observe
        ledger_path = tmp_path / "frame.ledger.jsonl"
        ledger_path.write_text("", encoding="utf-8")
        monkeypatch.setattr(frame_observe, "_LEDGER", ledger_path)
        frame_observe.record("tid-1", "some raw text", bu_outcome="none", current_frame="")
        assert ledger_path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# record() — flag ON, decision derivation
# ---------------------------------------------------------------------------

class TestRecordDecisions:
    def _setup(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "1")
        from addons.system.bearing import frame_observe
        ledger_path = tmp_path / "frame.ledger.jsonl"
        monkeypatch.setattr(frame_observe, "_LEDGER", ledger_path)
        return frame_observe, ledger_path

    def test_no_emission_plain_text(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t1", "plain text no tags", bu_outcome="none", current_frame="")
        rows = _read_ledger(ledger)
        assert len(rows) == 1
        assert rows[0]["decision"] == "no_emission"
        assert rows[0]["bu_outcome"] == "none"

    def test_bearing_update_unprocessed_not_labeled_silent(self, monkeypatch, tmp_path):
        # Advisor edge: a <bearing_update> tag IS present but the outcome is not
        # one of applied/rejected/parse_failed/noop (e.g. bearing kill-switch off
        # → "none"). It must NOT collapse into no_emission (self-contradictory).
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t-unproc", "x <bearing_update>{}</bearing_update> y",
                  bu_outcome="none", current_frame="")
        rows = _read_ledger(ledger)
        assert len(rows) == 1
        assert rows[0]["has_bearing_update"] is True
        assert rows[0]["decision"] == "bearing_update_unprocessed"
        assert rows[0]["decision"] != "no_emission"

    def test_frame_only(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t2", "<frame>some context</frame>", bu_outcome="none", current_frame="old frame")
        rows = _read_ledger(ledger)
        assert len(rows) == 1
        assert rows[0]["decision"] == "frame_only"
        assert rows[0]["has_frame"] is True
        assert rows[0]["has_bearing_update"] is False

    def test_bearing_update_applied(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        raw = "<bearing_update>{}</bearing_update>"
        fo.record("t3", raw, bu_outcome="applied", current_frame="frame text")
        rows = _read_ledger(ledger)
        assert len(rows) == 1
        assert rows[0]["decision"] == "bearing_update_applied"
        assert rows[0]["has_bearing_update"] is True

    def test_bearing_update_dropped_rejected(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        raw = "<bearing_update>{bad json}</bearing_update>"
        fo.record("t4", raw, bu_outcome="rejected", current_frame="")
        rows = _read_ledger(ledger)
        assert len(rows) == 1
        assert rows[0]["decision"] == "bearing_update_dropped"

    def test_bearing_update_dropped_parse_failed(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        raw = "<bearing_update>not json at all</bearing_update>"
        fo.record("t5", raw, bu_outcome="parse_failed", current_frame="")
        rows = _read_ledger(ledger)
        assert len(rows) == 1
        assert rows[0]["decision"] == "bearing_update_dropped"

    def test_bearing_update_noop(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        raw = "<bearing_update>{}</bearing_update>"
        fo.record("t6", raw, bu_outcome="noop", current_frame="my frame")
        rows = _read_ledger(ledger)
        assert len(rows) == 1
        assert rows[0]["decision"] == "bearing_update_noop"

    def test_row_contains_required_fields(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t7", "text", bu_outcome="none", current_frame="cf")
        rows = _read_ledger(ledger)
        row = rows[0]
        assert "ts" in row
        assert row["turn_id"] == "t7"
        assert "decision" in row
        assert "bu_outcome" in row
        assert "has_bearing_update" in row
        assert "has_frame" in row
        assert "observed_frame" in row
        assert "raw_len" in row
        assert "current_frame_len" in row

    def test_current_frame_len_recorded(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t8", "text", bu_outcome="none", current_frame="hello")
        rows = _read_ledger(ledger)
        assert rows[0]["current_frame_len"] == len("hello")

    def test_multiple_records_appended(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t1", "text", bu_outcome="none", current_frame="")
        fo.record("t2", "text", bu_outcome="none", current_frame="")
        fo.record("t3", "text", bu_outcome="none", current_frame="")
        rows = _read_ledger(ledger)
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# record() — never raises on bad ledger path
# ---------------------------------------------------------------------------

class TestRecordErrorSafety:
    def test_no_exception_on_unwritable_path(self, monkeypatch, tmp_path):
        """record() must not raise even when _LEDGER points at an unwritable path.

        Make a FILE at tmp_path/"blocker", then set _LEDGER to
        tmp_path/"blocker"/"frame.ledger.jsonl" so mkdir(parents=True) will
        fail because "blocker" is a file, not a directory.
        """
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "1")
        from addons.system.bearing import frame_observe
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file", encoding="utf-8")
        bad_ledger = blocker / "frame.ledger.jsonl"
        monkeypatch.setattr(frame_observe, "_LEDGER", bad_ledger)
        # Must not raise
        frame_observe.record("t_err", "raw text", bu_outcome="none", current_frame="")


# ---------------------------------------------------------------------------
# read_recent()
# ---------------------------------------------------------------------------

class TestReadRecent:
    def test_empty_when_ledger_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "1")
        from addons.system.bearing import frame_observe
        ledger_path = tmp_path / "frame.ledger.jsonl"
        monkeypatch.setattr(frame_observe, "_LEDGER", ledger_path)
        assert frame_observe.read_recent() == []

    def test_returns_rows(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "1")
        from addons.system.bearing import frame_observe
        ledger_path = tmp_path / "frame.ledger.jsonl"
        monkeypatch.setattr(frame_observe, "_LEDGER", ledger_path)
        for i in range(5):
            frame_observe.record(f"t{i}", "text", bu_outcome="none", current_frame="")
        rows = frame_observe.read_recent(limit=3)
        assert len(rows) == 3

    def test_skips_malformed_lines(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "1")
        from addons.system.bearing import frame_observe
        ledger_path = tmp_path / "frame.ledger.jsonl"
        monkeypatch.setattr(frame_observe, "_LEDGER", ledger_path)
        ledger_path.write_text('{"ok": true}\nnot json\n{"ok": true}\n', encoding="utf-8")
        rows = frame_observe.read_recent(limit=20)
        assert all(isinstance(r, dict) for r in rows)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Flag-off byte-identity proof:
# Call record() with flag OFF and confirm zero writes.
# This is the central invariant: no ledger file, no mutation.
# ---------------------------------------------------------------------------

class TestFlagOffByteIdentity:
    def test_no_side_effects_when_flag_off(self, monkeypatch, tmp_path):
        """With flag off: record() writes nothing — ledger absent after call."""
        monkeypatch.delenv("MONOLITH_OBSERVABLE_FRAME_V0", raising=False)
        from addons.system.bearing import frame_observe
        ledger_path = tmp_path / "frame.ledger.jsonl"
        monkeypatch.setattr(frame_observe, "_LEDGER", ledger_path)

        # Call with various inputs
        frame_observe.record("t1", "<bearing_update>{}</bearing_update>", bu_outcome="applied", current_frame="x")
        frame_observe.record("t2", "<frame>stuff</frame>", bu_outcome="none", current_frame="y")
        frame_observe.record("t3", "plain", bu_outcome="none", current_frame="")

        # File must not exist
        assert not ledger_path.exists(), (
            "flag-off: no ledger file should be created; found one"
        )


# ---------------------------------------------------------------------------
# Change 1 — hardened parse: last-match, fence-strip, quality flags
# ---------------------------------------------------------------------------

class TestHardenedParse:
    def test_last_frame_wins_over_prose_mention(self):
        """Prose <frame> closed pair before the real heartbeat → last is captured."""
        from addons.system.bearing import frame_observe
        raw = (
            "I am thinking about using <frame>prose mention of frame tag</frame> "
            "and here is the real one: <frame>actual heartbeat content</frame>"
        )
        result = frame_observe.observe(raw)
        assert result["has_frame"] is True
        assert result["observed_frame"] == "actual heartbeat content"
        assert result["frame_count"] == 2

    def test_code_fence_stripped_from_observed_frame(self):
        """Leading/trailing backtick noise removed from captured frame text."""
        from addons.system.bearing import frame_observe
        raw = "<frame>```the real frame```</frame>"
        result = frame_observe.observe(raw)
        assert result["observed_frame"] == "the real frame"

    def test_inline_backtick_stripped(self):
        """Single backticks stripped from frame text."""
        from addons.system.bearing import frame_observe
        raw = "<frame>`heartbeat text`</frame>"
        result = frame_observe.observe(raw)
        assert result["observed_frame"] == "heartbeat text"

    def test_frame_count_single(self):
        """Single clean frame: count == 1."""
        from addons.system.bearing import frame_observe
        raw = "<frame>clean content</frame>"
        result = frame_observe.observe(raw)
        assert result["frame_count"] == 1

    def test_frame_count_zero(self):
        """No frame tags: count == 0."""
        from addons.system.bearing import frame_observe
        raw = "no frame here at all"
        result = frame_observe.observe(raw)
        assert result["frame_count"] == 0

    def test_frame_count_multiple(self):
        """Three frame pairs: count == 3."""
        from addons.system.bearing import frame_observe
        raw = "<frame>a</frame> text <frame>b</frame> more <frame>c</frame>"
        result = frame_observe.observe(raw)
        assert result["frame_count"] == 3

    def test_frame_multiline_false_for_single_line(self):
        """Single-line heartbeat: frame_multiline is False."""
        from addons.system.bearing import frame_observe
        raw = "<frame>one liner heartbeat</frame>"
        result = frame_observe.observe(raw)
        assert result["frame_multiline"] is False

    def test_frame_multiline_true_for_multiline_content(self):
        """Multi-line frame content: frame_multiline is True."""
        from addons.system.bearing import frame_observe
        raw = "<frame>line1\nline2</frame>"
        result = frame_observe.observe(raw)
        assert result["frame_multiline"] is True

    def test_has_frame_equals_frame_count_gt_zero(self):
        """has_frame is derived from frame_count > 0."""
        from addons.system.bearing import frame_observe
        result_present = frame_observe.observe("<frame>x</frame>")
        result_absent = frame_observe.observe("nothing")
        assert result_present["has_frame"] is True
        assert result_absent["has_frame"] is False

    def test_frame_multiline_false_when_no_frame(self):
        """No frame at all: frame_multiline is False."""
        from addons.system.bearing import frame_observe
        result = frame_observe.observe("no frame tag")
        assert result["frame_multiline"] is False


# ---------------------------------------------------------------------------
# Change 2 — disparity() pure function
# ---------------------------------------------------------------------------

class TestDisparity:
    def test_no_frame_returns_no_frame(self):
        from addons.system.bearing import frame_observe
        verdict = frame_observe.disparity("", "some bearing", has_frame=False)
        assert verdict == "no_frame"

    def test_empty_bearing_returns_empty_bearing(self):
        """has_frame True + current_frame empty → empty_bearing (cold-start revive signal)."""
        from addons.system.bearing import frame_observe
        verdict = frame_observe.disparity("model stated frame", "", has_frame=True)
        assert verdict == "empty_bearing"

    def test_blank_bearing_returns_empty_bearing(self):
        """Whitespace-only current_frame treated as empty."""
        from addons.system.bearing import frame_observe
        verdict = frame_observe.disparity("some frame", "   ", has_frame=True)
        assert verdict == "empty_bearing"

    def test_match_case_insensitive(self):
        """Normalized comparison ignores case."""
        from addons.system.bearing import frame_observe
        verdict = frame_observe.disparity("My Frame Text", "my frame text", has_frame=True)
        assert verdict == "match"

    def test_match_whitespace_insensitive(self):
        """Extra spaces collapse in normalization."""
        from addons.system.bearing import frame_observe
        verdict = frame_observe.disparity("hello  world", "hello world", has_frame=True)
        assert verdict == "match"

    def test_differ_when_both_present_and_unequal(self):
        from addons.system.bearing import frame_observe
        verdict = frame_observe.disparity("current topic A", "old topic B", has_frame=True)
        assert verdict == "differ"

    def test_no_frame_overrides_empty_bearing(self):
        """has_frame=False must return no_frame even when current_frame is also empty."""
        from addons.system.bearing import frame_observe
        verdict = frame_observe.disparity("", "", has_frame=False)
        assert verdict == "no_frame"


# ---------------------------------------------------------------------------
# Change 2 — record() includes disparity + quality fields; flag-OFF clean
# ---------------------------------------------------------------------------

class TestRecordDisparityFields:
    def _setup(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "1")
        from addons.system.bearing import frame_observe
        ledger_path = tmp_path / "frame.ledger.jsonl"
        monkeypatch.setattr(frame_observe, "_LEDGER", ledger_path)
        return frame_observe, ledger_path

    def test_row_includes_frame_vs_bearing(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t-disp-1", "<frame>heartbeat</frame>", bu_outcome="none", current_frame="old frame")
        rows = _read_ledger(ledger)
        assert len(rows) == 1
        assert "frame_vs_bearing" in rows[0]

    def test_row_includes_frame_count(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t-fc", "<frame>x</frame>", bu_outcome="none", current_frame="x")
        rows = _read_ledger(ledger)
        assert "frame_count" in rows[0]
        assert rows[0]["frame_count"] == 1

    def test_row_includes_frame_multiline(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t-fm", "<frame>x</frame>", bu_outcome="none", current_frame="x")
        rows = _read_ledger(ledger)
        assert "frame_multiline" in rows[0]

    def test_disparity_differ_in_row(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t-diff", "<frame>new content</frame>", bu_outcome="none", current_frame="old content")
        rows = _read_ledger(ledger)
        assert rows[0]["frame_vs_bearing"] == "differ"

    def test_disparity_match_in_row(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t-match", "<frame>same text</frame>", bu_outcome="none", current_frame="same text")
        rows = _read_ledger(ledger)
        assert rows[0]["frame_vs_bearing"] == "match"

    def test_disparity_empty_bearing_in_row(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t-empty", "<frame>something</frame>", bu_outcome="none", current_frame="")
        rows = _read_ledger(ledger)
        assert rows[0]["frame_vs_bearing"] == "empty_bearing"

    def test_disparity_no_frame_in_row(self, monkeypatch, tmp_path):
        fo, ledger = self._setup(monkeypatch, tmp_path)
        fo.record("t-noframe", "plain text", bu_outcome="none", current_frame="bearing text")
        rows = _read_ledger(ledger)
        assert rows[0]["frame_vs_bearing"] == "no_frame"

    def test_flag_off_no_writes_with_new_inputs(self, monkeypatch, tmp_path):
        """Flag OFF: record() still writes nothing even with inputs that exercise new fields."""
        monkeypatch.delenv("MONOLITH_OBSERVABLE_FRAME_V0", raising=False)
        from addons.system.bearing import frame_observe
        ledger_path = tmp_path / "frame.ledger.jsonl"
        monkeypatch.setattr(frame_observe, "_LEDGER", ledger_path)
        frame_observe.record("t-off", "<frame>x</frame><frame>y</frame>", bu_outcome="none", current_frame="")
        assert not ledger_path.exists()
