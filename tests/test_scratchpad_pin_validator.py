"""Tests for the pin-time self-violation validator.

Regression for the 2026-05-20 pin-10 incident: a continuity anchor
defined the first-person binding rule but used third-person "Monolith's"
framing itself. The validator catches that shape at write time.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from core import continuity


@pytest.fixture()
def pin_env(monkeypatch, tmp_path):
    monkeypatch.setattr(continuity, "_STORE_PATH", tmp_path / "continuity.json")
    spec_path = Path(__file__).parent.parent / "skills" / "scratchpad" / "executor.py"
    spec = importlib.util.spec_from_file_location("scratchpad_exec_pin_test", spec_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_anchor_with_rule_language_and_third_person_self_is_rejected(pin_env) -> None:
    # Pin-10 shape: a rule about how pins should be written, but uses
    # third-person "Monolith's commitments" instead of first-person.
    result = pin_env.run(
        {
            "op": "pin",
            "category": "anchor",
            "text": (
                "Avoid prior / past-you / actor-character language in continuity "
                "framing; pins are Monolith's current commitments."
            ),
        },
        None,
    )
    assert "REJECTED" in result
    assert "self-violation" in result
    assert "bypass_self_violation_check" in result


def test_anchor_with_first_person_rule_language_is_allowed(pin_env) -> None:
    # Pin-14 shape: same rule restated in first person. Must pass.
    result = pin_env.run(
        {
            "op": "pin",
            "category": "anchor",
            "text": (
                "Continuity pins are my commitments — use first-person (I, my, me). "
                "Avoid framing that treats Monolith as a third-party entity I am describing."
            ),
        },
        None,
    )
    assert "REJECTED" not in result
    assert "pinned anchor" in result


def test_lesson_with_third_person_observation_is_allowed(pin_env) -> None:
    # Lessons document observations; third-person is legitimate voice.
    # The validator only fires on category=anchor.
    result = pin_env.run(
        {
            "op": "pin",
            "category": "lesson",
            "text": (
                "Observed that Monolith's effort=high produces visible reasoning-to-output ratio."
            ),
        },
        None,
    )
    assert "REJECTED" not in result
    assert "pinned lesson" in result


def test_anchor_without_rule_language_passes(pin_env) -> None:
    # Anchor that describes Monolith without imperative rules. Allowed.
    result = pin_env.run(
        {
            "op": "pin",
            "category": "anchor",
            "text": (
                "Monolith's runtime root is %APPDATA%/Monolith on Windows by default."
            ),
        },
        None,
    )
    assert "REJECTED" not in result
    assert "pinned anchor" in result


def test_bypass_flag_forces_pin_through_validator(pin_env) -> None:
    result = pin_env.run(
        {
            "op": "pin",
            "category": "anchor",
            "text": (
                "Avoid past-you framing; pins are Monolith's current commitments."
            ),
            "bypass_self_violation_check": True,
        },
        None,
    )
    assert "REJECTED" not in result
    assert "pinned anchor" in result


def test_validator_only_fires_for_anchor_category(pin_env) -> None:
    # Same self-violation text but as a pending — should pass (not an anchor).
    result = pin_env.run(
        {
            "op": "pin",
            "category": "pending",
            "text": (
                "Avoid past-you framing; pins are Monolith's current commitments."
            ),
        },
        None,
    )
    assert "REJECTED" not in result
    assert "pinned pending" in result
