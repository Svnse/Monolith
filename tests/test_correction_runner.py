"""Tests for addons.system.bearing.correction_runner — the /frame orchestrator.

process_correction ties one correction through the full pipeline:
  stateless control (call 2) -> synthesize (call 3) -> advisor gate -> store.
Model calls are one injected `generate` that discriminates by prompt. Human +
advisor-pass -> promoted + stored + trainable; Claude candidate -> never trains.
"""
from __future__ import annotations

import pytest

_SYNTH_TEXT = (
    "PROCESS_MOVE: drop the carried topic; anchor the live ask\n"
    "ANCHOR: implied_task\nANCHOR_ERROR: mirrored_loud_noun\nAPERTURE: collapse"
)
_ADVISOR_PASS = (
    "HUMAN_GROUNDED: yes\nPROCESS_SHAPED: yes\nNOT_OVERFIT: yes\n"
    "SIGN_PRESERVED: yes\nREAL_ANCHOR: yes"
)


def _fake_generate(msgs):
    sysmsg = msgs[0]["content"]
    if "re-deriving a one-sentence situational frame" in sysmsg:
        return "explaining a mutex"           # the stateless control
    if "comparing three one-sentence" in sysmsg:
        return _SYNTH_TEXT                     # synthesis
    if "PROMOTION GATE" in sysmsg:
        return _ADVISOR_PASS                   # advisor gate
    return ""


class TestProcessCorrection:
    def test_human_correction_is_promoted_and_stored(self, monkeypatch, tmp_path):
        from addons.system.bearing import correction_runner as cr, correction_store as st, correction_card as cc
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
        card = cr.process_correction(
            "turn-1",
            bad_frame="reasoning about vehicle ethics",
            better_frame="answering the mutex question",
            recent_asks=["what is a mutex?"],
            base_config={},
            source=cc.Source.HUMAN,
            generate=_fake_generate,
        )
        assert card.is_trainable() is True
        assert card.stateless_control == "explaining a mutex"
        # stored + retrievable as the nearest human card
        near = st.nearest_human_card("mutex")
        assert near is not None and near["better_frame"] == "answering the mutex question"

    def test_claude_candidate_logged_not_trained(self, monkeypatch, tmp_path):
        from addons.system.bearing import correction_runner as cr, correction_store as st, correction_card as cc
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
        card = cr.process_correction(
            "turn-2",
            bad_frame="b", better_frame="claude guess frame",
            recent_asks=["x"], base_config={}, source=cc.Source.CLAUDE_CANDIDATE,
            generate=_fake_generate,
        )
        assert card.is_trainable() is False
        assert st.nearest_human_card("claude") is None   # candidates never injected
        assert len(st.read_cards()) == 1                 # but it IS logged

    def test_never_raises_on_generation_failure(self, monkeypatch, tmp_path):
        from addons.system.bearing import correction_runner as cr, correction_store as st, correction_card as cc
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")

        def boom(msgs):
            raise RuntimeError("model down")

        card = cr.process_correction(
            "t", bad_frame="b", better_frame="y", recent_asks=["x"],
            base_config={}, source=cc.Source.HUMAN, generate=boom,
        )
        assert card is not None  # degrades, does not raise
