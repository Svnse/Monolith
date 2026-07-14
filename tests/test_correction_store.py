"""Tests for addons.system.bearing.correction_store — card storage + retrieval + render.

Stores cards to CONFIG_DIR/correction_cards.jsonl (flag MONOLITH_MONOFRAME_V1).
The live scaffold injects ONE nearest HUMAN card (trainable), never a rule pile,
and only trainable (human + promoted) cards are eligible for injection.
"""
from __future__ import annotations

import pytest


def _human_card(better, *, promoted=True, bad="reasoning about vehicle ethics"):
    from addons.system.bearing import correction_card as cc
    return cc.CorrectionCard(
        bad_frame=bad, better_frame=better, process_move="anchor the live ask",
        anchor_type=cc.AnchorType.IMPLIED_TASK, anchor_error="mirrored_loud_noun",
        aperture=cc.Aperture.COLLAPSE, stateless_control="ctrl",
        source=cc.Source.HUMAN, promoted=promoted,
    )


class TestStore:
    def test_store_and_read_roundtrip(self, monkeypatch, tmp_path):
        from addons.system.bearing import correction_store as st
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
        st.store_card(_human_card("answering the mutex question"))
        rows = st.read_cards()
        assert len(rows) == 1
        assert rows[0]["better_frame"] == "answering the mutex question"
        assert rows[0]["source"] == "human"
        assert rows[0]["promoted"] is True

    def test_noop_when_disabled(self, monkeypatch, tmp_path):
        from addons.system.bearing import correction_store as st
        monkeypatch.delenv("MONOLITH_MONOFRAME_V1", raising=False)
        store = tmp_path / "cards.jsonl"
        monkeypatch.setattr(st, "_STORE", store)
        st.store_card(_human_card("x"))
        assert not store.exists()


class TestAdvisorVerdictSerialization:
    def test_stores_verdict_with_failing_check_and_reason(self, monkeypatch, tmp_path):
        from addons.system.bearing import correction_store as st, correction_card as cc
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
        v = cc.AdvisorVerdict(
            human_grounded=True, process_shaped=True, not_overfit=False,
            sign_preserved=True, real_anchor=True,
        )
        card = cc.CorrectionCard(
            bad_frame="b", better_frame="y", process_move="m",
            anchor_type=cc.AnchorType.IMPLIED_TASK, anchor_error="",
            aperture=cc.Aperture.COLLAPSE, stateless_control="c",
            source=cc.Source.HUMAN, promoted=False, advisor_verdict=v,
        )
        st.store_card(card)
        row = st.read_cards()[0]
        assert row["advisor_verdict"]["not_overfit"] is False
        assert row["advisor_verdict"]["passed"] is False
        assert row["advisor_verdict"]["real_anchor"] is True

    def test_none_verdict_serializes_as_none(self, monkeypatch, tmp_path):
        from addons.system.bearing import correction_store as st
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
        st.store_card(_human_card("x"))
        assert st.read_cards()[0]["advisor_verdict"] is None


class TestNearestCard:
    def test_returns_nearest_trainable_human_card(self, monkeypatch, tmp_path):
        from addons.system.bearing import correction_store as st
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
        st.store_card(_human_card("answering the mutex locking question"))
        st.store_card(_human_card("explaining bloom filter hashing"))
        near = st.nearest_human_card("how does a mutex lock work")
        assert near is not None
        assert "mutex" in near["better_frame"]

    def test_ignores_unpromoted_and_candidate_cards(self, monkeypatch, tmp_path):
        from addons.system.bearing import correction_store as st, correction_card as cc
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
        st.store_card(_human_card("mutex answer", promoted=False))   # not promoted
        cand = cc.CorrectionCard(
            bad_frame="b", better_frame="mutex candidate", process_move="m",
            anchor_type=cc.AnchorType.IMPLIED_TASK, anchor_error="",
            aperture=cc.Aperture.COLLAPSE, stateless_control="c",
            source=cc.Source.CLAUDE_CANDIDATE, promoted=False,
        )
        st.store_card(cand)
        assert st.nearest_human_card("mutex") is None  # neither is trainable

    def test_none_when_no_cards(self, monkeypatch, tmp_path):
        from addons.system.bearing import correction_store as st
        monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
        assert st.nearest_human_card("anything") is None


class TestRenderForScaffold:
    def test_renders_single_card_as_example_not_rule(self):
        from addons.system.bearing import correction_store as st
        block = st.render_card_for_scaffold({
            "bad_frame": "reasoning about vehicle ethics",
            "better_frame": "answering the mutex question",
            "process_move": "drop the carried topic; anchor the live ask",
        })
        assert "answering the mutex question" in block
        assert "drop the carried topic" in block
        # one example, not a rule pile / numbered list
        assert "1." not in block

    def test_empty_when_no_card(self):
        from addons.system.bearing import correction_store as st
        assert st.render_card_for_scaffold(None) == ""
