"""Tests for addons.system.bearing.correction_synthesis — the two LLM calls.

  synthesize()  — call 3: {bad, better, stateless-control} -> a candidate
                  CorrectionCard (process_move + anchor + anchor_error + aperture).
  advise()      — the promotion GATE: a critic that attacks the card on 5 tests.
  gate()        — promote iff human-sourced AND the advisor passed.

Parsers are pure; orchestration takes an injected `generate` (no live model).
"""
from __future__ import annotations

import pytest


_SYNTH_TEXT = """
PROCESS_MOVE: drop the carried topic; anchor the live ask, not the loud prior noun
ANCHOR: implied_task
ANCHOR_ERROR: mirrored_loud_noun (kept the AV-ethics noun from a prior turn)
APERTURE: collapse
""".strip()


class TestParseSynthesis:
    def test_parses_labeled_fields(self):
        from addons.system.bearing import correction_synthesis as cs, correction_card as cc
        out = cs.parse_synthesis(_SYNTH_TEXT)
        assert out["process_move"].startswith("drop the carried topic")
        assert out["anchor_type"] is cc.AnchorType.IMPLIED_TASK
        assert out["aperture"] is cc.Aperture.COLLAPSE
        assert "mirrored_loud_noun" in out["anchor_error"]

    def test_unknown_anchor_falls_back_to_explicit_noun(self):
        from addons.system.bearing import correction_synthesis as cs, correction_card as cc
        out = cs.parse_synthesis("PROCESS_MOVE: x\nANCHOR: gibberish\nAPERTURE: diffuse")
        assert out["anchor_type"] is cc.AnchorType.EXPLICIT_NOUN  # conservative default

    def test_tolerant_to_missing_fields(self):
        from addons.system.bearing import correction_synthesis as cs
        out = cs.parse_synthesis("nothing structured here")
        assert out["process_move"] == ""


class TestSynthesize:
    def test_builds_human_candidate_card(self, monkeypatch):
        from addons.system.bearing import correction_synthesis as cs, correction_card as cc
        card = cs.synthesize(
            bad_frame="reasoning about vehicle ethics",
            better_frame="answering the mutex question",
            stateless_control="explaining a mutex",
            source=cc.Source.HUMAN,
            generate=lambda msgs: _SYNTH_TEXT,
        )
        assert isinstance(card, cc.CorrectionCard)
        assert card.source is cc.Source.HUMAN
        assert card.promoted is False                # synthesis never promotes
        assert card.anchor_type is cc.AnchorType.IMPLIED_TASK
        assert card.stateless_control == "explaining a mutex"


_ADVISOR_PASS = """
HUMAN_GROUNDED: yes
PROCESS_SHAPED: yes
NOT_OVERFIT: yes
SIGN_PRESERVED: yes
REAL_ANCHOR: yes
""".strip()

_ADVISOR_FAIL = _ADVISOR_PASS.replace("REAL_ANCHOR: yes", "REAL_ANCHOR: no")


class TestAdvise:
    def test_parses_pass_verdict(self):
        from addons.system.bearing import correction_synthesis as cs
        v = cs.parse_advisor(_ADVISOR_PASS)
        assert v.passed() is True

    def test_parses_fail_verdict(self):
        from addons.system.bearing import correction_synthesis as cs
        v = cs.parse_advisor(_ADVISOR_FAIL)
        assert v.real_anchor is False
        assert v.passed() is False

    def test_advise_runs_critic(self):
        from addons.system.bearing import correction_synthesis as cs, correction_card as cc
        card = cc.CorrectionCard(
            bad_frame="x", better_frame="y", process_move="z",
            anchor_type=cc.AnchorType.IMPLIED_TASK, anchor_error="",
            aperture=cc.Aperture.COLLAPSE, stateless_control="c",
            source=cc.Source.HUMAN,
        )
        v = cs.advise(card, generate=lambda msgs: _ADVISOR_PASS)
        assert v.passed() is True


class TestGate:
    def test_promotes_human_card_that_passed(self):
        from addons.system.bearing import correction_synthesis as cs, correction_card as cc
        card = cc.CorrectionCard(
            bad_frame="x", better_frame="y", process_move="z",
            anchor_type=cc.AnchorType.IMPLIED_TASK, anchor_error="",
            aperture=cc.Aperture.COLLAPSE, stateless_control="c",
            source=cc.Source.HUMAN,
        )
        v = cs.parse_advisor(_ADVISOR_PASS)
        gated = cs.gate(card, v)
        assert gated.promoted is True
        assert gated.is_trainable() is True

    def test_does_not_promote_claude_candidate_even_on_pass(self):
        from addons.system.bearing import correction_synthesis as cs, correction_card as cc
        card = cc.CorrectionCard(
            bad_frame="x", better_frame="y", process_move="z",
            anchor_type=cc.AnchorType.IMPLIED_TASK, anchor_error="",
            aperture=cc.Aperture.COLLAPSE, stateless_control="c",
            source=cc.Source.CLAUDE_CANDIDATE,
        )
        v = cs.parse_advisor(_ADVISOR_PASS)
        gated = cs.gate(card, v)
        assert gated.promoted is False
        assert gated.is_trainable() is False

    def test_does_not_promote_on_advisor_fail(self):
        from addons.system.bearing import correction_synthesis as cs, correction_card as cc
        card = cc.CorrectionCard(
            bad_frame="x", better_frame="y", process_move="z",
            anchor_type=cc.AnchorType.IMPLIED_TASK, anchor_error="",
            aperture=cc.Aperture.COLLAPSE, stateless_control="c",
            source=cc.Source.HUMAN,
        )
        v = cs.parse_advisor(_ADVISOR_FAIL)
        gated = cs.gate(card, v)
        assert gated.promoted is False
