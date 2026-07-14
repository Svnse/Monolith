"""Tests for core/effort_resolver.py — M2: effort governs generation depth.

The classifier emits an effort_tier every turn; this resolves the effective
tier and maps it to the one non-performative, KV-stable behavioral lever the
2026-05-09 smart-spec reserved for /effort: the backend reasoning mode
(enable_thinking). Pure functions — the testable core of M2.
See docs/superpowers/specs/2026-06-02-execution-plane-sync-design.md.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.effort_resolver import (
    effort_enables_thinking,
    effort_governance_enabled,
    resolve_effort_tier,
    resolve_thinking,
)


def test_effort_governance_default_on(monkeypatch):
    monkeypatch.delenv("MONOLITH_EFFORT_V1", raising=False)
    assert effort_governance_enabled() is True


def test_effort_governance_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MONOLITH_EFFORT_V1", "0")
    assert effort_governance_enabled() is False


def test_high_tiers_enable_thinking():
    assert effort_enables_thinking("high") is True
    assert effort_enables_thinking("xhigh") is True
    assert effort_enables_thinking("ultimate") is True


def test_low_and_med_tiers_do_not_enable_thinking():
    assert effort_enables_thinking("low") is False
    assert effort_enables_thinking("med") is False
    assert effort_enables_thinking("unknown") is False


def test_resolve_thinking_effort_upgrades_hard_turn():
    # User left thinking off (or unset); a high-effort turn engages it anyway.
    assert resolve_thinking(manual=False, tier="high") is True
    assert resolve_thinking(manual=None, tier="ultimate") is True


def test_resolve_thinking_manual_on_always_wins():
    # An explicit manual 'thinking on' is never overridden, even on an easy turn.
    assert resolve_thinking(manual=True, tier="low") is True


def test_resolve_thinking_passthrough_on_easy_turn():
    # Effort never FORCES thinking off — it passes the manual value through.
    assert resolve_thinking(manual=False, tier="med") is False
    assert resolve_thinking(manual=None, tier="low") is None


def test_resolve_effort_tier_reads_classifier_shape_object():
    config = {"_turn_shape": SimpleNamespace(effort_tier="high")}
    assert resolve_effort_tier(config) == "high"
    assert config["_resolved_effort_tier"] == "high"


def test_resolve_effort_tier_reads_dict_shape():
    config = {"_turn_shape": {"effort_tier": "ultimate"}}
    assert resolve_effort_tier(config) == "ultimate"
    assert config["_resolved_effort_tier"] == "ultimate"


def test_resolve_effort_tier_defaults_med_when_unclassified():
    config = {}
    assert resolve_effort_tier(config) == "med"
    assert config["_resolved_effort_tier"] == "med"


def test_resolve_effort_tier_rejects_garbage_tier():
    config = {"_turn_shape": SimpleNamespace(effort_tier="banana")}
    assert resolve_effort_tier(config) == "med"
