"""Tests for addons.system.bearing.correction_card — MonoFrame v2 CorrectionCards.

A correction is an EXAMPLE-bound lesson, not a number and not a standalone rule:
  bad_frame + better_frame + process_move + anchor/aperture error.
Only HUMAN corrections train; Claude-proposed frames are logged as candidates.
The advisor is a promotion GATE (attacks the card on 5 tests) — a card only
becomes trainable once it passes. Invariants emerge later, from clustering.
"""
from __future__ import annotations

import pytest


class TestSource:
    def test_human_source_is_eligible_to_train(self):
        from addons.system.bearing import correction_card as cc
        card = cc.CorrectionCard(
            bad_frame="reasoning about vehicle ethics",
            better_frame="answering the mutex question",
            process_move="drop the carried topic; anchor the live ask",
            anchor_type=cc.AnchorType.IMPLIED_TASK,
            anchor_error="mirrored_loud_noun",
            aperture=cc.Aperture.COLLAPSE,
            stateless_control="explaining a mutex",
            source=cc.Source.HUMAN,
        )
        assert card.source is cc.Source.HUMAN

    def test_claude_candidate_never_trains_even_if_promoted(self):
        from addons.system.bearing import correction_card as cc
        card = cc.CorrectionCard(
            bad_frame="x", better_frame="y", process_move="z",
            anchor_type=cc.AnchorType.EXPLICIT_NOUN, anchor_error="",
            aperture=cc.Aperture.DIFFUSE, stateless_control="c",
            source=cc.Source.CLAUDE_CANDIDATE,
            promoted=True,  # even a promoted candidate must not train
        )
        assert card.is_trainable() is False


class TestPromotionGate:
    def test_unpromoted_human_card_is_not_trainable(self):
        from addons.system.bearing import correction_card as cc
        card = cc.CorrectionCard(
            bad_frame="x", better_frame="y", process_move="z",
            anchor_type=cc.AnchorType.LIVE_CONSTRAINT, anchor_error="",
            aperture=cc.Aperture.COLLAPSE, stateless_control="c",
            source=cc.Source.HUMAN, promoted=False,
        )
        assert card.is_trainable() is False

    def test_promoted_human_card_is_trainable(self):
        from addons.system.bearing import correction_card as cc
        card = cc.CorrectionCard(
            bad_frame="x", better_frame="y", process_move="z",
            anchor_type=cc.AnchorType.CONTINUITY, anchor_error="",
            aperture=cc.Aperture.DIFFUSE, stateless_control="c",
            source=cc.Source.HUMAN, promoted=True,
        )
        assert card.is_trainable() is True


class TestAnchorRouting:
    def test_anchor_candidate_set_is_the_five_typed_routes(self):
        from addons.system.bearing import correction_card as cc
        names = {a.name for a in cc.AnchorType}
        assert names == {
            "EXPLICIT_NOUN", "IMPLIED_TASK", "LIVE_CONSTRAINT",
            "USER_STATE", "CONTINUITY",
        }


class TestAdvisorVerdict:
    def test_advisor_verdict_carries_the_five_gate_checks(self):
        from addons.system.bearing import correction_card as cc
        v = cc.AdvisorVerdict(
            human_grounded=True, process_shaped=True, not_overfit=True,
            sign_preserved=True, real_anchor=True,
        )
        assert v.passed() is True

    def test_advisor_verdict_fails_if_any_check_fails(self):
        from addons.system.bearing import correction_card as cc
        v = cc.AdvisorVerdict(
            human_grounded=True, process_shaped=True, not_overfit=False,
            sign_preserved=True, real_anchor=True,
        )
        assert v.passed() is False
