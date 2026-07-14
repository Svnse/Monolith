"""Tests for core/history_search.py — chat-archive full-text search.

Focus: a matched message must be dated by its OWN `time` field, not the
archive file's `updated_at` (which tracks the last edit to the whole
conversation and otherwise mis-dates old messages as recent).
"""
from __future__ import annotations

import json

from core.history_search import search_archives


def _write_archive(path, *, updated_at, messages):
    path.write_text(
        json.dumps({"meta": {"title": "Chat", "updated_at": updated_at}, "messages": messages}),
        encoding="utf-8",
    )


def test_search_dates_match_by_message_time_not_file_updated_at(tmp_path):
    """When-plane fix: the result date comes from the matched message's `time`,
    not the file-level updated_at."""
    _write_archive(
        tmp_path / "c1.json",
        updated_at="2026-06-01T10:00:00+00:00",  # whole-file last edit (recent)
        messages=[
            {"role": "user", "text": "tell me about widgets",
             "time": "2026-01-15T09:00:00+00:00"},  # the message is OLD
        ],
    )
    results = search_archives("widgets", tmp_path)
    assert len(results) == 1
    assert results[0].date == "2026-01-15"  # message's own date, not 2026-06-01


def test_search_falls_back_to_file_date_when_message_has_no_time(tmp_path):
    """A message lacking a `time` field falls back to the file updated_at."""
    _write_archive(
        tmp_path / "c2.json",
        updated_at="2026-06-01T10:00:00+00:00",
        messages=[{"role": "user", "text": "legacy widgets row"}],  # no `time`
    )
    results = search_archives("widgets", tmp_path)
    assert len(results) == 1
    assert results[0].date == "2026-06-01"
