"""Tests for engine.agent_server._process_frame_correction — POST /frame core.

Mirrors the _process_rating pattern: a unit-testable core, no running server, no
live model (runner injected). Drives the MonoFrame v2 /frame command for BOTH
surfaces: app /frame (source=human, trains) and trainer --frame (source=
claude_candidate, logged only). Source defaults to candidate (never accidentally train).
"""
from __future__ import annotations

import pytest


def test_missing_better_frame_is_400():
    from engine.agent_server import _process_frame_correction
    payload, status = _process_frame_correction({"source": "human"}, runner=lambda *a, **k: None)
    assert status == 400
    assert payload["ok"] is False


def test_human_source_maps_and_calls_runner():
    from engine.agent_server import _process_frame_correction
    from addons.system.bearing import correction_card as cc
    captured = {}

    def runner(turn_id, **kwargs):
        captured["turn_id"] = turn_id
        captured.update(kwargs)
        return object()

    payload, status = _process_frame_correction(
        {"better_frame": "answering the mutex question", "source": "human",
         "bad_frame": "vehicle ethics", "recent_asks": ["what is a mutex?"]},
        runner=runner,
    )
    assert status == 200
    assert payload["ok"] is True
    assert payload["source"] == "human"
    assert captured["source"] is cc.Source.HUMAN
    assert captured["better_frame"] == "answering the mutex question"
    assert captured["bad_frame"] == "vehicle ethics"


def test_default_source_is_candidate():
    from engine.agent_server import _process_frame_correction
    from addons.system.bearing import correction_card as cc
    captured = {}
    _process_frame_correction(
        {"better_frame": "f"}, runner=lambda turn_id, **k: captured.update(k),
    )
    assert captured["source"] is cc.Source.CLAUDE_CANDIDATE


def test_unknown_source_falls_back_to_candidate():
    from engine.agent_server import _process_frame_correction
    from addons.system.bearing import correction_card as cc
    captured = {}
    _process_frame_correction(
        {"better_frame": "f", "source": "nonsense"},
        runner=lambda turn_id, **k: captured.update(k),
    )
    assert captured["source"] is cc.Source.CLAUDE_CANDIDATE
