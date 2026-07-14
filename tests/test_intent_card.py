"""Tests for core/intent_card — the Monoline sidecar enrichment.

The card NEVER hard-fails the organ: any bad/empty/malformed output → None →
predict runs on the floor. `move` is coerced into the turn_classifier vocab.
"""
from __future__ import annotations

from core import intent_card as ic


def test_invoke_card_default_none():
    """Until wired to the live runtime, the card no-ops (floor carries)."""
    assert ic._invoke_card("answer", "msg") is None


def test_read_intent_none_when_card_unavailable():
    assert ic.read_intent("some answer", "some msg") is None


def test_coerce_valid_json(monkeypatch):
    raw = ('{"intent_read": "wants impact analysis", '
           '"directions": [{"move": "analysis", "referent": "blast radius"}], '
           '"referents": ["blast", "radius", "downstream"]}')
    monkeypatch.setattr(ic, "_invoke_card", lambda a, m: raw)
    out = ic.read_intent("ans", "msg")
    assert out is not None
    assert out["intent_read"] == "wants impact analysis"
    assert out["directions"][0]["move"] == "analysis"
    assert "downstream" in out["referents"]


def test_coerce_drops_unknown_move(monkeypatch):
    """A move outside the shared turn_classifier vocab is dropped (keeps
    membership apples-to-apples)."""
    raw = ('{"directions": [{"move": "telepathy", "referent": "x"}, '
           '{"move": "plan", "referent": "auth"}], "referents": ["auth"]}')
    monkeypatch.setattr(ic, "_invoke_card", lambda a, m: raw)
    out = ic.read_intent("ans", "msg")
    moves = [d["move"] for d in out["directions"]]
    assert "telepathy" not in moves and "plan" in moves


def test_coerce_malformed_returns_none(monkeypatch):
    for bad in ("not json at all", "", "   ", "[1,2,3]", '{"directions": "nope"}'):
        monkeypatch.setattr(ic, "_invoke_card", lambda a, m, _b=bad: _b)
        assert ic.read_intent("ans", "msg") is None


def test_coerce_tolerates_json_fence(monkeypatch):
    raw = '```json\n{"referents": ["token"], "intent_read": "wants tokens"}\n```'
    monkeypatch.setattr(ic, "_invoke_card", lambda a, m: raw)
    out = ic.read_intent("ans", "msg")
    assert out is not None and out["referents"] == ["token"]
